import json
import os
import argparse
import subprocess

def preprocess_spec_file(spec_file_path):
    with open(spec_file_path) as f:
        data = json.load(f)
    
    file_cache = {}
    processed_refs = set()
    stack = [data]
    
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref = obj['$ref']
                
                if ref in processed_refs:
                    continue
                processed_refs.add(ref)
                
                try:
                    if ref.startswith('#/'):
                        result = data
                        for part in ref[2:].split('/'):
                            result = result[part]
                    else:
                        external_file = ref.split('#/')[0]
                        external_path = os.path.join(os.path.dirname(spec_file_path), external_file)
                        
                        if external_path not in file_cache:
                            try:
                                with open(external_path) as f:
                                    file_cache[external_path] = json.load(f)
                            except FileNotFoundError:
                                continue
                        
                        result = file_cache[external_path]
                        
                        if '#/' in ref:
                            for part in ref.split('#/')[1].split('/'):
                                result = result[part]
                    
                    obj.clear()
                    obj.update(result)
                    stack.append(obj)
                    
                except (KeyError, TypeError, FileNotFoundError):
                    continue
            else:
                stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj)
    
    return data
def diff_spec_files(old_spec_path, new_spec_path):
    result = subprocess.run(['openapi-diff', old_spec_path, new_spec_path, '--format', 'json'], 
                          capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else None