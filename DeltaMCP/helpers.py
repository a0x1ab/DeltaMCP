import json
import sys
import subprocess
from prance import ResolvingParser
import os
import ast
import pickle

def load_spec(spec_path):
    try:
        return ResolvingParser(spec_path, recursion_limit=5000).specification
    except Exception as e:
        raise ValueError(f"Failed to load specification from {spec_path}: {e}")

def compare_specs(old_spec_path, new_spec_path):
    old, new = load_spec(old_spec_path), load_spec(new_spec_path)
    for n, s in zip(['old.json', 'new.json'], [old, new]):
        json.dump(s, open(n, 'w'))
    r = subprocess.run(['oasdiff', 'diff', '--format', 'json', 'old.json', 'new.json'], capture_output=True, text=True)
    [os.remove(f) for f in ['old.json', 'new.json'] if os.path.exists(f)]
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {"error": r.stderr or "No differences found"}

def convert_to_ast(original_stub, ast_file="ast_output.pkl"):
    with open(original_stub, "r") as f:
        source_code = f.read()
    tree = ast.parse(source_code)
    with open(ast_file, "wb") as f:
        pickle.dump(tree, f)
    return tree

def convert_from_ast(ast_file="ast_output.pkl", output_stub="upgraded_server.py"):
    with open(ast_file, "rb") as f:
        tree = pickle.load(f)
    python_code = ast.unparse(tree)
    with open(output_stub, "w") as f:
        f.write(python_code)
    return python_code

def get_range_of_lines(ast_tree, func_name):
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return (node.lineno, node.end_lineno)
    return None

def generate_diff(old_spec_path, new_spec_path, original_stub):
    differences = compare_specs(old_spec_path, new_spec_path)
    ast_tree = convert_to_ast(original_stub)
    paths = differences.get('paths', {})
    for section, changes in paths.items():
        if isinstance(changes, dict):
            for path in changes:
                func_name = next(
                    (node.name for node in ast.walk(ast_tree)
                     if isinstance(node, ast.FunctionDef) and path.strip('/') in ast.unparse(node)),
                    None
                )
                if func_name:
                    changes[path]['function_name'] = func_name
                    changes[path]['line_range'] = get_range_of_lines(ast_tree, func_name)
    if os.path.exists("ast_output.pkl"):
        os.remove("ast_output.pkl")
    with open("diff_output.json", "w") as f:
        json.dump(differences, f, indent=2)
    return differences