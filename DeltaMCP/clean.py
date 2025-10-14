import json
import os
import argparse
import subprocess

def preprocess_spec_file(spec_file_path):
    with open(spec_file_path) as f:
        data = json.load(f)
    
    files = {}
    
    for iteration in range(10):
        refs_found = 0
        
        def process_object(obj):
            nonlocal refs_found
            
            if isinstance(obj, dict):
                if '$ref' in obj:
                    ref = obj['$ref']
                    refs_found += 1
                    
                    try:
                        if ref.startswith('#/'):
                            content = data
                            for part in ref[2:].split('/'):
                                content = content[part]
                        else:
                            file_path = ref.split('#/')[0]
                            full_path = os.path.join(os.path.dirname(spec_file_path), file_path)
                            
                            if full_path not in files:
                                with open(full_path) as f:
                                    files[full_path] = json.load(f)
                            
                            content = files[full_path]
                            if '#/' in ref:
                                for part in ref.split('#/')[1].split('/'):
                                    content = content[part]
                        
                        obj.clear()
                        obj.update(content)
                    except:
                        pass
                
                for value in list(obj.values()):
                    process_object(value)
            
            elif isinstance(obj, list):
                for item in obj:
                    process_object(item)
        
        process_object(data)
        
        if refs_found == 0:
            break
    
    return data

def diff_spec_files(old_spec_path, new_spec_path):
    result = subprocess.run(['openapi-diff', old_spec_path, new_spec_path, '--format', 'json'], 
                          capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else None