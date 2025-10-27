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
import copy
import re
from datetime import datetime
from pathlib import Path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
try:
    import helpers
except ImportError:
    sys.path.append(str(script_dir.parent))

_KEYMAP = {
    "oasdiff": "od",
    "paths": "p",
    "added": "a",
    "deleted": "d",
    "modified": "m",
    "operations": "ops",
    "description": "desc",
    "operationId": "opId",
    "tags": "t",
    "x-ms-examples": "ex",
    "x-ms-long-running-operation": "lro",
    "x-ms-pageable": "page",
    "responses": "resp",
    "parameters": "params",
    "schema": "sch",
    "$ref": "r",
    "extensions": "ext",
    "properties": "prop",
    "oldValue": "ov",
    "value": "v",
    "path": "pth",
    "op": "o",
    "from": "f",
}

_PATH_REPLACEMENTS = [
    (r"/subscriptions/\{subscriptionId\}", "/subs/{s}"),
    (r"/resourceGroups/\{resourceGroupName\}", "/rg/{rg}"),
    (r"/providers/Microsoft\.Storage", "/prov/MS.Storage"),
    (r"/providers/Microsoft\.Subscription", "/prov/MS.Subscription"),
]

def _shorten_keys(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            short_key = _KEYMAP.get(k, k)
            new_obj[short_key] = _shorten_keys(v)
        return new_obj
    elif isinstance(obj, list):
        return [_shorten_keys(i) for i in obj]
    return obj

def _flatten_ops(o):
    if not isinstance(o, dict) or "ops" not in o:
        return o

    new_ops = {}
    for change_type in ("a", "d", "m"):
        if change_type in o["ops"]:
            for method, val in o["ops"][change_type].items():
                new_ops[f"{method}+{change_type}"] = val
    if not new_ops:
        for method, val in o["ops"].items():
            new_ops[method] = val
    o["ops"] = new_ops
    return o

def _apply_path_shortcuts(p):
    if not isinstance(p, str):
        return p
    for pattern, repl in _PATH_REPLACEMENTS:
        p = re.sub(pattern, repl, p)
    return p

def compress_oasdiff(oasdiff_obj):
    obj = copy.deepcopy(oasdiff_obj)
    obj = _shorten_keys(obj)

    if "od" in obj and "p" in obj["od"]:
        paths = obj["od"]["p"]
        new_paths = {}
        for change_type, path_block in paths.items():
            short_type = _KEYMAP.get(change_type, change_type)
            new_block = {}
            for path, path_val in path_block.items():
                short_path = _apply_path_shortcuts(path)
                if isinstance(path_val, dict):
                    path_val = _flatten_ops(path_val)
                new_block[short_path] = path_val
            new_paths[short_type] = new_block
        return {"p": new_paths}
    elif "paths" in obj:
        paths = obj["paths"]
        new_paths = {}
        for change_type, path_block in paths.items():
            short_type = _KEYMAP.get(change_type, change_type)
            new_block = {}
            for path, path_val in path_block.items():
                short_path = _apply_path_shortcuts(path)
                if isinstance(path_val, dict):
                    path_val = _flatten_ops(path_val)
                new_block[short_path] = path_val
            new_paths[short_type] = new_block
        return {"p": new_paths}

    return obj

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
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if node.end_lineno else start_line + 1
                    
                    decorator_start = start_line
                    for dec in node.decorator_list:
                        if hasattr(dec, 'lineno'):
                            decorator_start = min(decorator_start, dec.lineno - 1)
                    
                    func_source = '\n'.join(source_lines[decorator_start:end_line])
                    func_source = textwrap.dedent(func_source)
                    
                    affected_result = is_tool_affected_by_changes(func_source, changed_paths)
                    if affected_result:
                        is_affected, relevant_paths = affected_result
                        extracted_functions[tool_name] = {
                            'function_source': func_source.strip(),
                            'relevant_paths': relevant_paths
                        }
    return extracted_functions

def is_tool_affected_by_changes(func_source, changed_paths):
    if not changed_paths:
        return True
    
    relevant_paths = {}
    func_source_lower = func_source.lower()
    
    for path in changed_paths:
        path_lower = path.lower()
        if path_lower in func_source_lower:
            relevant_paths[path] = True
                
    if relevant_paths:
        return (True, relevant_paths)
    return (False, {})

def parse_version_date(version_str):
    try:
        return datetime.strptime(version_str, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(version_str, "%Y-%m-%d-preview")
        except ValueError:
            return datetime.min

def get_consecutive_version_pairs(versions):
    if len(versions) < 2:
        return []
    sorted_versions = sorted(versions, key=parse_version_date)
    pairs = []
    for i in range(len(sorted_versions) - 1):
        pairs.append((sorted_versions[i], sorted_versions[i + 1]))
    return pairs

def has_valid_oasdiff(diff):
    if not isinstance(diff, dict):
        return False
    if "error" in diff:
        return False
    if not diff.get("paths") and not diff.get("extensions"):
        return False
    return True

def extract_http_method_from_function(function_source):
    if not function_source:
        return None
    
    try:
        tree = ast.parse(function_source)
        
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'requests'):
                
                method = node.func.attr.upper()
                if method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
                    return method
        
        return None
    except Exception as e:
        logging.warning(f"Failed to extract HTTP method from function: {e}")
        return None

def filter_oasdiff_by_operation(oasdiff, http_method, relevant_paths):
    if not isinstance(oasdiff, dict) or 'paths' not in oasdiff:
        return oasdiff
    
    if not http_method or not relevant_paths:
        return oasdiff
    
    filtered_diff = {"paths": {}}
    
    for change_type, paths_data in oasdiff['paths'].items():
        if not isinstance(paths_data, dict):
            continue
            
        filtered_paths = {}
        
        for path, path_data in paths_data.items():
            if path not in relevant_paths:
                continue
                
            if isinstance(path_data, dict) and 'operations' in path_data:
                filtered_operations = {}
                
                for op_change_type, operations_data in path_data['operations'].items():
                    http_method_lower = http_method.lower() if http_method else None
                    http_method_upper = http_method.upper() if http_method else None
                    
                    if isinstance(operations_data, dict):
                        method_key = None
                        if http_method_lower and http_method_lower in operations_data:
                            method_key = http_method_lower
                        elif http_method_upper and http_method_upper in operations_data:
                            method_key = http_method_upper
                        
                        if method_key:
                            if op_change_type not in filtered_operations:
                                filtered_operations[op_change_type] = {}
                            filtered_operations[op_change_type][method_key] = operations_data[method_key]
                
                if filtered_operations:
                    filtered_path_data = path_data.copy()
                    filtered_path_data['operations'] = filtered_operations
                    filtered_paths[path] = filtered_path_data
            else:
                filtered_paths[path] = path_data
        
        if filtered_paths:
            filtered_diff['paths'][change_type] = filtered_paths
    
    return filtered_diff if any(filtered_diff['paths'].values()) else {"paths": {}}

def extract_operation_details_from_spec(spec_file, path, http_method=None):
    if not spec_file:
        return None
        
    try:
        import json
        with open(spec_file, 'r') as f:
            spec = json.load(f)
        
        paths = spec.get('paths', {})
        if path not in paths:
            return None
            
        path_data = paths[path]
        
        if http_method and http_method.lower() in path_data:
            return {http_method.lower(): path_data[http_method.lower()]}
        
        return path_data
        
    except Exception as e:
        logging.warning(f"Failed to extract operation details from {spec_file}: {e}")
        return None

def convert_ast_to_code(source):
    if isinstance(source, str) and source.startswith('FunctionDef('):
        try:
            safe_dict = {
                'FunctionDef': ast.FunctionDef,
                'arguments': ast.arguments,
                'arg': ast.arg,
                'Name': ast.Name,
                'Load': ast.Load,
                'Store': ast.Store,
                'Call': ast.Call,
                'Attribute': ast.Attribute,
                'keyword': ast.keyword,
                'Constant': ast.Constant,
                'Expr': ast.Expr,
                'Assign': ast.Assign,
                'JoinedStr': ast.JoinedStr,
                'FormattedValue': ast.FormattedValue,
                'Subscript': ast.Subscript,
                'Dict': ast.Dict,
                'List': ast.List,
                'If': ast.If,
                'Compare': ast.Compare,
                'UnaryOp': ast.UnaryOp,
                'Not': ast.Not,
                'Is': ast.Is,
                'IsNot': ast.IsNot,
                'In': ast.In,
                'BinOp': ast.BinOp,
                'Add': ast.Add,
                'Try': ast.Try,
                'ExceptHandler': ast.ExceptHandler,
                'For': ast.For,
                'Tuple': ast.Tuple,
                'Return': ast.Return,
                'Raise': ast.Raise,
                'Slice': ast.Slice,
            }
            ast_node = eval(source, {"__builtins__": {}}, safe_dict)
            return ast.unparse(ast_node)
        except Exception as e:
            return source
    return source

def create_training_sample(change_type, tool_name, paths_data, full_diff, tool_a_data, tool_b_data, service, version_a, version_b, api_type, spec_a_file=None, spec_b_file=None):
    tool_a_source = tool_a_data.get('function_source', '')
    tool_a_paths = tool_a_data.get('relevant_paths', {})
    
    tool_b_source = tool_b_data.get('function_source', '')
    tool_b_paths = tool_b_data.get('relevant_paths', {})
    
    http_method = None
    if tool_a_source:
        http_method = extract_http_method_from_function(tool_a_source)
    elif tool_b_source:
        http_method = extract_http_method_from_function(tool_b_source)
    
    all_relevant_paths = {}
    all_relevant_paths.update(tool_a_paths)
    all_relevant_paths.update(tool_b_paths)
    
    if change_type == 'modified':
        if not all_relevant_paths:
            return None
            
        filtered_diff = {"paths": {"modified": {
            path: data for path, data in paths_data.items() 
            if path in all_relevant_paths
        }}}
        
        if http_method and filtered_diff.get('paths', {}).get('modified'):
            filtered_diff = filter_oasdiff_by_operation(filtered_diff, http_method, all_relevant_paths)
            
    elif change_type == 'added':
        relevant_added_paths = [path for path in paths_data if path in all_relevant_paths]
        if not relevant_added_paths:
            return None
        
        added_operations = {}
        for path in relevant_added_paths:
            op_details = extract_operation_details_from_spec(spec_b_file, path, http_method)
            if op_details:
                added_operations[path] = {"operations": {"added": op_details}}
        
        if added_operations:
            filtered_diff = {"paths": {"added": added_operations}}
        else:
            filtered_diff = {"paths": {"added": relevant_added_paths}}
        
    elif change_type == 'deleted':
        relevant_deleted_paths = [path for path in paths_data if path in all_relevant_paths]
        if not relevant_deleted_paths:
            return None
        
        deleted_operations = {}
        for path in relevant_deleted_paths:
            op_details = extract_operation_details_from_spec(spec_a_file, path, http_method)
            if op_details:
                deleted_operations[path] = {"operations": {"deleted": op_details}}
        
        if deleted_operations:
            filtered_diff = {"paths": {"deleted": deleted_operations}}
        else:
            filtered_diff = {"paths": {"deleted": relevant_deleted_paths}}
    
    else:
        return None
    
    if not any(filtered_diff.get('paths', {}).values()):
        return None
    
    tool_a_final = convert_ast_to_code(tool_a_source) if tool_a_source else ''
    tool_b_final = convert_ast_to_code(tool_b_source) if tool_b_source else ''
    
    tools_a_dict = {tool_name: tool_a_final} if tool_a_final else {}
    tools_b_dict = {tool_name: tool_b_final} if tool_b_final else {}
    
    if not is_valid_training_sample(filtered_diff, tools_a_dict, tools_b_dict, tool_name):
        return None
    
    compressed_oasdiff = compress_oasdiff(filtered_diff)
    
    result = {
        "tools_a": tools_a_dict,
        "tools_b": tools_b_dict
    }
    result.update(compressed_oasdiff)
    return result

def is_valid_training_sample(oasdiff, tools_a, tools_b, tool_name):
    if not isinstance(oasdiff, dict) or 'paths' not in oasdiff:
        return True
    
    tool_in_a = bool(tools_a.get(tool_name))
    tool_in_b = bool(tools_b.get(tool_name))
    
    paths = oasdiff['paths']
    
    for change_type in paths:
        if change_type == 'modified':
            if not (tool_in_a and tool_in_b):
                return False
        elif change_type == 'added':
            if not (not tool_in_a and tool_in_b):
                return False
        elif change_type == 'deleted':
            if not (tool_in_a and not tool_in_b):
                return False
    
    return True

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
    
    service, version_a, version_b, api_type = args
    
    workspace_root = Path(__file__).resolve().parent.parent.parent
    specs_dir = workspace_root / "azure-rest-api-specs" / "specification"
    data_dir = workspace_root / "DeltaMCP" / "llm-finetuned" / "training_data"
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
    api_dir = provider_dir / api_type
    
    version_a_dir = api_dir / version_a
    version_b_dir = api_dir / version_b
    
    spec_a_files = list(version_a_dir.glob("*.json"))
    spec_b_files = list(version_b_dir.glob("*.json"))
    
    if not spec_a_files or not spec_b_files:
        return
    
    try:
        diff = helpers.compare_specs(str(spec_a_files[0]), str(spec_b_files[0]))
        
        if not has_valid_oasdiff(diff):
            logging.warning(f"Skipping {service} {version_a}->{version_b} ({api_type}): invalid oasdiff")
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
                elif isinstance(paths_data, list):
                    changed_paths.extend(paths_data)
        
        if not changed_paths:
            logging.info(f"Skipping {service} {version_a}->{version_b} ({api_type}): no changed paths")
            return
        
        samples_created = 0
        
        for change_type, paths_data in diff['paths'].items():
            if change_type == 'modified' and isinstance(paths_data, dict):
                functions_a = extract_changed_tool_functions(stub_a, list(paths_data.keys()))
                functions_b = extract_changed_tool_functions(stub_b, list(paths_data.keys()))
                all_tool_names = set(functions_a.keys()) | set(functions_b.keys())
                
                for tool_name in all_tool_names:
                    sample = create_training_sample(
                        change_type, tool_name, paths_data, diff,
                        functions_a.get(tool_name, {}), functions_b.get(tool_name, {}),
                        service, version_a, version_b, api_type,
                        str(spec_a_files[0]), str(spec_b_files[0])
                    )
                    if sample:
                        output_file = data_dir / f"{service}_{version_a}_{version_b}_{api_type}_{tool_name}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(sample, f, separators=(',', ':'), ensure_ascii=False)
                        samples_created += 1
                        
            elif change_type == 'added' and isinstance(paths_data, list):
                functions_b = extract_changed_tool_functions(stub_b, paths_data)
                
                for tool_name in functions_b.keys():
                    sample = create_training_sample(
                        change_type, tool_name, paths_data, diff,
                        {}, functions_b.get(tool_name, {}),
                        service, version_a, version_b, api_type,
                        str(spec_a_files[0]), str(spec_b_files[0])
                    )
                    if sample:
                        output_file = data_dir / f"{service}_{version_a}_{version_b}_{api_type}_{tool_name}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(sample, f, separators=(',', ':'), ensure_ascii=False)
                        samples_created += 1
                        
            elif change_type == 'deleted' and isinstance(paths_data, list):
                functions_a = extract_changed_tool_functions(stub_a, paths_data)
                
                for tool_name in functions_a.keys():
                    sample = create_training_sample(
                        change_type, tool_name, paths_data, diff,
                        functions_a.get(tool_name, {}), {},
                        service, version_a, version_b, api_type,
                        str(spec_a_files[0]), str(spec_b_files[0])
                    )
                    if sample:
                        output_file = data_dir / f"{service}_{version_a}_{version_b}_{api_type}_{tool_name}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(sample, f, separators=(',', ':'), ensure_ascii=False)
                        samples_created += 1
        
        logging.info(f"Generated {service} {version_a}->{version_b} ({api_type}): {samples_created} individual tool samples")
            
    except Exception as e:
        error_msg = f"Failed processing {service} {version_a}->{version_b} ({api_type}): {str(e)}"
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
        
        for api_type in ['stable', 'preview']:
            api_dir = provider_dir / api_type
            
            if not api_dir.exists():
                continue
            
            version_dirs = [v for v in api_dir.iterdir() if v.is_dir() and not v.name.startswith('.')]
            version_names = [v.name for v in version_dirs]
            
            if len(version_names) < 2:
                continue
            
            consecutive_pairs = get_consecutive_version_pairs(version_names)
            
            for version_a, version_b in consecutive_pairs:
                version_pairs.append((service_name, version_a, version_b, api_type))
    
    if version_pairs:
        logging.info(f"Processing {len(version_pairs)} version pairs in parallel")
        futures = [process_version_pair.remote(pair) for pair in version_pairs]
        ray.get(futures)
        logging.info("Processing completed")
    else:
        logging.warning("No version pairs found to process")
    
    ray.shutdown()
    logging.info(f"Training data generation finished at {datetime.now()}")
    logging.info("Creating training JSONL file...")
    create_training_jsonl()
    logging.info("Training JSONL file created successfully")

def create_training_jsonl():
    script_dir = Path(__file__).resolve().parent
    output_jsonl = script_dir / "training_data.jsonl"
    samples_dir = script_dir / "training_data"

    all_files = list(samples_dir.glob("*.json"))
    all_files.sort()

    action_summaries = {
        "a": "New operation added to the service.",
        "added": "New operation added to the service.",
        "m": "Operation modified to reflect spec changes.",
        "modified": "Operation modified to reflect spec changes.",
        "d": "Operation removed from the service.",
        "deleted": "Operation removed from the service."
    }

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for file_path in all_files:
            with open(file_path, "r", encoding="utf-8") as f_in:
                sample = json.load(f_in)
                tools_a = sample.get("tools_a", {})
                tools_b = sample.get("tools_b", {})
                p = sample.get("p", {})

                if not tools_a:
                    tools_a_str = "<none>"
                else:
                    blocks = []
                    for code in tools_a.values():
                        sanitized = textwrap.dedent(code).strip()
                        blocks.append(f"```python\n{sanitized}\n```")
                    tools_a_str = "\n\n".join(blocks)

                deleted_lines = []
                for path, path_val in p.get("d", {}).items():
                    for op_key, op_val in path_val.get("ops", {}).items():
                        ctype = op_key.split("+")[1] if "+" in op_key else "deleted"
                        if ctype.lower() == "d":
                            tool_name = op_val.get("opId")
                            if tool_name:
                                deleted_lines.append(f"# REMOVE {tool_name}")
                deleted_text = "\n".join(deleted_lines)

                if tools_b:
                    added_segments = []
                    for code in tools_b.values():
                        added_segments.append(f"```python\n{textwrap.dedent(code).strip()}\n```")
                    added_text = "\n\n".join(added_segments)
                else:
                    added_text = ""
                assistant_text = "\n".join([deleted_text, added_text]).strip()
                if not assistant_text:
                    assistant_text = "# No tools generated"

                diff_lines = []
                for change_type, path_data in p.items():
                    if isinstance(path_data, dict):
                        for path, path_val in path_data.items():
                            if isinstance(path_val, dict) and "ops" in path_val:
                                for op_key, op_val in path_val.get("ops", {}).items():
                                    if "+" in op_key:
                                        method, ctype = op_key.split("+")
                                    else:
                                        method = op_key
                                        ctype = change_type
                                    opId = op_val.get("opId", "")
                                    desc = op_val.get("desc", "")
                                    summary = desc.strip() if isinstance(desc, str) else ""
                                    if not summary:
                                        summary = action_summaries.get(ctype.lower(), "Operation updated to match upstream spec.")
                                    diff_lines.append(f"- {ctype.capitalize()} operation:\n  {method.upper()} {path}\n  OperationId: {opId}\n  Summary: {summary}")
                    elif isinstance(path_data, list):
                        for path in path_data:
                            summary = action_summaries.get(change_type.lower(), f"Path {change_type}.")
                            diff_lines.append(f"- {change_type.capitalize()} operation:\n  {path}\n  Summary: {summary}")
                diff_summary = "\n".join(diff_lines) if diff_lines else "<none>"

                user_sections = [
                    "User: Update tools based on diff while preserving tool metadata and request structure.",
                    f"Existing implementation(s):\n{tools_a_str}",
                    f"Changes to apply:\n{diff_summary}"
                ]
                user_text = "\n\n".join(user_sections)

                hf_sample = {"text": f"{user_text}\n\nAssistant:\n{assistant_text}"}
                f_out.write(json.dumps(hf_sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "jsonl":
        print("Creating training JSONL file...")
        create_training_jsonl()
        print("Training JSONL file created successfully!")
    else:
        main()