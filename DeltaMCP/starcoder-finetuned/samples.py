import shutil, sys, os, json, ast, subprocess, importlib.util
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import convert_to_ast, generate_diff

def get_services(specs_dir):
    return [f.name for f in specs_dir.iterdir() if f.is_dir()]

def get_versions(service_dir):
    versions = []
    for provider_path in service_dir.iterdir():
        if not provider_path.is_dir(): continue
        for stability_path in provider_path.iterdir():
            if not stability_path.is_dir() or stability_path.name.endswith(('.md', '.yaml')): continue
            if stability_path.name in ['stable', 'preview']:
                for version_path in stability_path.iterdir():
                    if version_path.is_dir() and not version_path.name.endswith('.md'):
                        versions.append(version_path)
    
    def parse_date(path):
        try:
            name = path.name.replace('-preview', '')
            return datetime.strptime(name, '%Y-%m-%d')
        except: return datetime.min
    
    return sorted(versions, key=parse_date)

def load_automcp_runner(script_path):
    spec = importlib.util.spec_from_file_location("run_automcp", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AutoMCPRunner(workspace_root=str(Path(script_path).parent.parent))

def ast_to_dict(node):
    if not isinstance(node, ast.AST): return node
    result = {'type': node.__class__.__name__}
    for field, value in ast.iter_fields(node):
        result[field] = [ast_to_dict(item) for item in value] if isinstance(value, list) else ast_to_dict(value)
    return result

def generate_code_and_ast(runner, json_file_path, output_dir, version_suffix):
    try:
        temp_dir_name = f"AutoMCP/generated/temp_{version_suffix}"
        cmd = ["docker", "run", "--rm", "-v", f"{runner.root}:/workspace", "-e", "WINEDEBUG=-all", 
               "automcp-wine", "wine", "AutoMCP/automcp.exe", 
               "--input", str(json_file_path.relative_to(runner.root)), "--output", temp_dir_name]
        
        result = subprocess.run(cmd, cwd=runner.root, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            temp_dir = runner.root / temp_dir_name
            if temp_dir.exists():
                for file in temp_dir.iterdir():
                    if file.is_file() and file.suffix == '.py':
                        dest_file = output_dir / f"{file.stem}_{version_suffix}.py"
                        shutil.copy(file, dest_file)
                        
                        temp_pkl = output_dir / "temp_ast.pkl"
                        ast_tree = convert_to_ast(str(dest_file), str(temp_pkl))
                        
                        ast_file = output_dir / f"{file.stem}_{version_suffix}.json"
                        with open(ast_file, 'w') as f:
                            json.dump(ast_to_dict(ast_tree), f, indent=2)
                        
                        if temp_pkl.exists(): temp_pkl.unlink()
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                return True
    except: pass
    return False

def create_version_pairs_training_data(specs_dir, data_dir, services, runner):
    training_pairs = []
    
    for service in tqdm(services, desc="Creating version pairs"):
        service_dir = specs_dir / service / "resource-manager"
        if not service_dir.exists(): continue
            
        versions = get_versions(service_dir)
        if len(versions) < 2: continue
        
        for i in range(len(versions) - 1):
            version_a_path, version_b_path = versions[i], versions[i + 1]
            version_a, version_b = version_a_path.name, version_b_path.name
            
            pair_dir = data_dir / service / f"{version_a}_{version_b}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            spec_a = version_a_path / f"{service}.json"
            spec_b = version_b_path / f"{service}.json"
            
            if not spec_a.exists():
                json_files = list(version_a_path.glob("*.json"))
                spec_a = json_files[0] if json_files else None
            
            if not spec_b.exists():
                json_files = list(version_b_path.glob("*.json"))
                spec_b = json_files[0] if json_files else None
            
            if not spec_a or not spec_b: continue
                
            shutil.copy(spec_a, pair_dir / f"spec_{version_a}.json")
            shutil.copy(spec_b, pair_dir / f"spec_{version_b}.json")
            
            if generate_code_and_ast(runner, spec_a, pair_dir, version_a) and \
               generate_code_and_ast(runner, spec_b, pair_dir, version_b):
                
                code_a_files = list(pair_dir.glob(f"*_{version_a}.py"))
                if code_a_files:
                    try:
                        diff_result = generate_diff(str(spec_a), str(spec_b), str(code_a_files[0]))
                        with open(pair_dir / "diff_output.json", 'w') as f:
                            json.dump(diff_result, f, indent=2)
                        
                        training_file = create_final_training_file(pair_dir, version_a, version_b, service)
                        if training_file:
                            training_pairs.append({'service': service, 'version_a': version_a, 'version_b': version_b, 'pair_dir': str(pair_dir)})
                    except Exception as e:
                        print(f"Failed diff for {service} {version_a}->{version_b}: {e}")
            
            for pkl_file in pair_dir.glob("*.pkl"): pkl_file.unlink()
        
        service_training_dir = data_dir / service
        if service_training_dir.exists():
            training_files = reorganize_service_files(service_training_dir)
            print(f"Service '{service}' completed: {len(training_files)} training samples created")
    
    return training_pairs

def cleanup_intermediate_files(pair_dir, training_file_path):
    try:
        for file_path in pair_dir.iterdir():
            if file_path.is_file() and file_path != training_file_path:
                file_path.unlink()
    except Exception as e:
        print(f"Failed to cleanup intermediate files in {pair_dir}: {e}")

def reorganize_service_files(service_dir):
    try:
        training_files = []
        
        for subdir in service_dir.iterdir():
            if subdir.is_dir():
                for file_path in subdir.iterdir():
                    if file_path.is_file() and file_path.name.startswith("training_sample_"):
                        dest_path = service_dir / file_path.name
                        shutil.move(str(file_path), str(dest_path))
                        training_files.append(dest_path)
                
                shutil.rmtree(subdir, ignore_errors=True)
        
        return training_files
        
    except Exception as e:
        print(f"Failed to reorganize service files in {service_dir}: {e}")
        return []

def create_final_training_file(pair_dir, version_a, version_b, service_name):
    try:
        diff_file = pair_dir / "diff_output.json"
        if not diff_file.exists(): return None
            
        with open(diff_file, 'r') as f:
            oas_diff = json.load(f)
        
        ast_a_files = [f for f in pair_dir.glob(f"*_{version_a}.json") if not f.name.startswith("spec_")]
        ast_b_files = [f for f in pair_dir.glob(f"*_{version_b}.json") if not f.name.startswith("spec_")]
        
        if not ast_a_files or not ast_b_files: return None
            
        with open(ast_a_files[0], 'r') as f:
            stub_ast_a = json.load(f)
        with open(ast_b_files[0], 'r') as f:
            stub_ast_b = json.load(f)
        
        training_data = {
            "input": {"oas_diff": oas_diff, "stub_ast_a": stub_ast_a},
            "output": {"stub_ast_b": stub_ast_b}
        }
        
        training_filename = f"training_sample_{service_name}_{version_a}_{version_b}.json"
        training_file_path = pair_dir / training_filename
        
        with open(training_file_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        cleanup_intermediate_files(pair_dir, training_file_path)
        
        return training_file_path
        
    except Exception as e:
        print(f"Failed training file for {pair_dir}: {e}")
        return None

def create_training_dataset(
    specs_dir_path="/Users/adi/Documents/Uni/incremental-mcp/azure-rest-api-specs/specification",
    data_dir_path="training_data",
    automcp_script_path="/Users/adi/Documents/Uni/incremental-mcp/AutoMCP/run_automcp.py",
    services=None
):
    data_dir, specs_dir = Path(data_dir_path), Path(specs_dir_path)
    data_dir.mkdir(exist_ok=True)
    
    if services is None: services = get_services(specs_dir)
    
    runner = load_automcp_runner(automcp_script_path)
    training_pairs = create_version_pairs_training_data(specs_dir, data_dir, services, runner)
    
    with open(data_dir / "training_summary.json", 'w') as f:
        json.dump(training_pairs, f, indent=2)
    
    return training_pairs

if __name__ == "__main__":
    specs_dir = Path("/Users/adi/Documents/Uni/incremental-mcp/azure-rest-api-specs/specification")
    training_pairs = create_training_dataset()
    print(f"Created {len(training_pairs)} training pairs")
