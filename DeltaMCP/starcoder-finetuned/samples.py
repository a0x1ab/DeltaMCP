import os
import sys
import ray
import json
import subprocess
import shutil
import ast
import textwrap
import logging
import traceback
from datetime import datetime
from pathlib import Path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
try:
    import helpers
except ImportError:
    sys.path.append(str(script_dir.parent))

def load_automcp_runner(script_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_automcp", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    workspace_root = str(Path(script_path).parent.parent.resolve())
    return module.AutoMCPRunner(workspace_root=workspace_root)

def generate_stub(runner, spec_file, out_dir, version_name, service_name):
    stub_name = f"{service_name}_{version_name}_stub"
    tmp_stub = out_dir / stub_name
    
    spec_file_path = Path(spec_file)
    try:
        relative_spec_file = spec_file_path.relative_to(runner.root)
    except ValueError:
        return None
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{runner.root}:/workspace",
        "-e", "WINEDEBUG=-all",
        "automcp-wine", "wine",
        "AutoMCP/automcp.exe",
        "--input", str(relative_spec_file),
        "--output", str(tmp_stub.relative_to(runner.root))
    ]
    
    try:
        result = subprocess.run(cmd, cwd=runner.root, capture_output=True, text=True, timeout=300, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    server_stub_file = tmp_stub / "server_stub.py"
    if not server_stub_file.exists():
        return None
    
    return server_stub_file

def extract_changed_tool_functions(stub_file, changed_paths):
    if not stub_file or not stub_file.exists():
        return {}
        
    with open(stub_file, 'r') as f:
        source_code = f.read()
        source_lines = source_code.split('\n')
        
    try:
        tree = ast.parse(source_code)
    except Exception:
        return {}
    
    extracted_functions = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            tool_name = None
            has_tool_decorator = False
            
            if node.decorator_list:
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Call) and 
                        isinstance(decorator.func, ast.Attribute) and
                        decorator.func.attr == 'tool'):
                        
                        has_tool_decorator = True
                        for keyword in decorator.keywords:
                            if keyword.arg == 'name':
                                if isinstance(keyword.value, ast.Constant):
                                    tool_name = keyword.value.value
                        
                        if not tool_name:
                            tool_name = node.name
                        break
            
            if has_tool_decorator and tool_name:
                if is_tool_affected_by_changes(tool_name, changed_paths):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        start_line = node.lineno - 1  # Convert to 0-based indexing
                        end_line = node.end_lineno if node.end_lineno else start_line + 1
                        
                        decorator_start = start_line
                        for dec in node.decorator_list:
                            if hasattr(dec, 'lineno'):
                                decorator_start = min(decorator_start, dec.lineno - 1)
                        
                        func_source = '\n'.join(source_lines[decorator_start:end_line])
                        func_source = textwrap.dedent(func_source)
                        extracted_functions[tool_name] = func_source.strip()
    return extracted_functions

def is_tool_affected_by_changes(tool_name, changed_paths):
    if not changed_paths:
        return True
    
    tool_name_lower = tool_name.lower()
    if isinstance(changed_paths, dict):
        paths_to_check = []
        if 'paths' in changed_paths:
            for path, changes in changed_paths['paths'].items():
                paths_to_check.append(path)
        else:
            for key, value in changed_paths.items():
                if isinstance(value, (str, list)):
                    paths_to_check.extend([key] + (value if isinstance(value, list) else [value]))
    elif isinstance(changed_paths, list):
        paths_to_check = changed_paths
    else:
        paths_to_check = [str(changed_paths)]
    
    for path in paths_to_check:
        path_str = str(path).lower()
        path_parts = path_str.replace('/', ' ').replace('-', ' ').replace('_', ' ').split()
        
        for part in path_parts:
            if len(part) > 2 and part in tool_name_lower:
                return True
    return False

def convert_function_to_ast(func_source):
    try:
        tree = ast.parse(func_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return ast.dump(node, indent=2)
        return None
    except Exception:
        return None

@ray.remote
def process_version_pair(args):
    import sys
    import logging
    import traceback
    from datetime import datetime
    from pathlib import Path
    
    script_dir = Path(__file__).resolve().parent
    log_file = script_dir / "training_data" / "processing_errors.log"
    
    logging.basicConfig(
        filename=str(log_file),
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    
    sys.path.insert(0, str(script_dir.parent))
    import helpers
    
    service, version_a, version_b = args
    
    workspace_root = Path(__file__).resolve().parent.parent.parent
    specs_dir = workspace_root / "azure-rest-api-specs" / "specification"
    data_dir = workspace_root / "DeltaMCP" / "starcoder-finetuned" / "training_data"
    data_dir.mkdir(exist_ok=True)
    automcp_script_path = workspace_root / "AutoMCP" / "run_automcp.py"
    
    runner = load_automcp_runner(str(automcp_script_path))
    stub_a = None
    stub_b = None
    
    service_dir = specs_dir / service / "resource-manager"
    provider_dirs = [p for p in service_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    if not provider_dirs:
        return
    
    provider_dir = provider_dirs[0]
    stable_dir = provider_dir / "stable"
    
    version_a_dir = stable_dir / version_a
    version_b_dir = stable_dir / version_b
    
    spec_a_files = list(version_a_dir.glob("*.json"))
    spec_b_files = list(version_b_dir.glob("*.json"))
    
    if not spec_a_files or not spec_b_files:
        return
    
    try:
        diff = helpers.compare_specs(str(spec_a_files[0]), str(spec_b_files[0]))
        
        # Skip if oasdiff failed
        if isinstance(diff, dict) and 'error' in diff:
            logging.warning(f"Skipping {service} {version_a}->{version_b}: oasdiff error - {diff['error'][:100]}...")
            return
        
        stub_a = generate_stub(runner, str(spec_a_files[0]), data_dir, version_a, service)
        stub_b = generate_stub(runner, str(spec_b_files[0]), data_dir, version_b, service)
        
        if not stub_a or not stub_b:
            return
        
        changed_paths = []
        if isinstance(diff, dict) and 'paths' in diff:
            for change_type, paths_data in diff['paths'].items():
                if isinstance(paths_data, dict):
                    changed_paths.extend(paths_data.keys())
        
        functions_a = extract_changed_tool_functions(stub_a, changed_paths)
        functions_b = extract_changed_tool_functions(stub_b, changed_paths)
        
        ast_functions_a = {}
        ast_functions_b = {}
        
        for tool_name, func_source in functions_a.items():
            ast_dump = convert_function_to_ast(func_source)
            if ast_dump:
                ast_functions_a[tool_name] = ast_dump
        
        for tool_name, func_source in functions_b.items():
            ast_dump = convert_function_to_ast(func_source)
            if ast_dump:
                ast_functions_b[tool_name] = ast_dump
        
        training_sample = {
            "oasdiff": diff,
            "tools_a": ast_functions_a,
            "tools_b": ast_functions_b
        }
        
        logging.info(f"Generated {service} {version_a}->{version_b}: {len(ast_functions_a)} tools A, {len(ast_functions_b)} tools B")
        
        output_file = data_dir / f"{service}_{version_a}_{version_b}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_sample, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        error_msg = f"Failed processing {service} {version_a}->{version_b}: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
    finally:
        if stub_a and stub_a.parent.exists():
            shutil.rmtree(stub_a.parent, ignore_errors=True)
        if stub_b and stub_b.parent.exists():
            shutil.rmtree(stub_b.parent, ignore_errors=True)

def main():
    workspace_root = Path(__file__).resolve().parent.parent.parent
    specs_dir = workspace_root / "azure-rest-api-specs" / "specification"
    
    log_dir = Path(__file__).resolve().parent / "training_data"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "processing_errors.log"
    
    if log_file.exists():
        log_file.unlink()
    
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    logging.info(f"Starting training data generation at {datetime.now()}")
    
    ray.init(ignore_reinit_error=True)
    
    version_pairs = []
    
    for service_dir in specs_dir.iterdir():
        if not service_dir.is_dir() or service_dir.name.startswith('.'):
            continue
        
        service_name = service_dir.name
        resource_manager_dir = service_dir / "resource-manager"
        
        if not resource_manager_dir.exists():
            continue
        
        provider_dirs = [p for p in resource_manager_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
        if not provider_dirs:
            continue
        
        provider_dir = provider_dirs[0]
        stable_dir = provider_dir / "stable"
        
        if not stable_dir.exists():
            continue
        
        versions = sorted([v for v in stable_dir.iterdir() if v.is_dir()], key=lambda x: x.name)
        
        for i in range(len(versions) - 1):
            version_pairs.append((service_name, versions[i].name, versions[i + 1].name))
    
    if version_pairs:
        logging.info(f"Processing {len(version_pairs)} version pairs in parallel")
        futures = [process_version_pair.remote(pair) for pair in version_pairs]
        ray.get(futures)
        logging.info("Processing completed")
    else:
        logging.warning("No version pairs found to process")
    
    ray.shutdown()
    logging.info(f"Training data generation finished at {datetime.now()}")

if __name__ == "__main__":
    main()