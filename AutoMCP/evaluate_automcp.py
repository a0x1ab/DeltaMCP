
import subprocess
import json
import time
import psutil
import threading
import re
from pathlib import Path
from run_automcp import AutoMCPRunner


class AutoMCPEvaluator:
    def __init__(self):
        self.root = Path.cwd().parent
        self.automcp_dir = Path.cwd()
        self.data_dir = self.root / "data"
        self.generated_dir = self.automcp_dir / "generated"
        self.process_stats = {}
        self.runner = AutoMCPRunner(self.root)

    def get_versions(self):
        return self.runner.get_versions()

    def monitor_process(self, process_pid, version):
        self.process_stats[version] = {"memory": [], "cpu": []}
        try:
            proc = psutil.Process(process_pid)
            while proc.is_running():
                memory = proc.memory_info().rss / 1024 / 1024
                cpu = proc.cpu_percent()
                self.process_stats[version]["memory"].append(memory)
                self.process_stats[version]["cpu"].append(cpu)
                time.sleep(0.1)
        except:
            pass

    def run_generation(self, version):
        start_time = time.time()
        
        def start_monitoring(process_pid):
            monitor_thread = threading.Thread(target=self.monitor_process, args=(process_pid, version))
            monitor_thread.start()
        
        try:
            self.runner.run_version(version, monitor_callback=start_monitoring)
            generation_time = time.time() - start_time
            return {
                "success": True,
                "generation_time": generation_time
            }
        except Exception as e:
            generation_time = time.time() - start_time
            return {
                "success": False,
                "generation_time": generation_time,
                "error": str(e)
            }

    def endpoint_coverage_score(self, version):
        spec_file = self.data_dir / version / "storage.json"
        generated_dir = self.generated_dir / version
        
        if not spec_file.exists() or not generated_dir.exists():
            return 0.0
        
        with open(spec_file) as f:
            spec = json.load(f)
        
        total_endpoints = 0
        operation_ids = set()
        
        for path, methods in spec.get("paths", {}).items():
            for method in methods.keys():
                if method in ["get", "post", "put", "delete", "patch"]:
                    total_endpoints += 1
                    if "operationId" in methods[method]:
                        operation_ids.add(methods[method]["operationId"])
        
        if total_endpoints == 0:
            return 0.0
        
        covered_endpoints = 0
        for py_file in generated_dir.glob("**/*.py"):
            with open(py_file) as f:
                content = f.read()
                for operation_id in list(operation_ids):
                    if operation_id in content:
                        covered_endpoints += 1
                        operation_ids.remove(operation_id)
        
        return covered_endpoints / total_endpoints

    def generation_score(self, version):
        generated_dir = self.generated_dir / version
        py_files = list(generated_dir.glob("**/*.py"))
        
        if not py_files:
            return 0.0
        
        total_score = 0.0
        for py_file in py_files:
            try:
                with open(py_file) as f:
                    content = f.read()
                    
                file_score = 0.0
                if "def " in content:
                    file_score += 0.3
                if "class " in content:
                    file_score += 0.3
                if "import " in content:
                    file_score += 0.2
                if len(content.strip()) > 100:
                    file_score += 0.2
                
                total_score += file_score
            except:
                continue
        
        return min(total_score / len(py_files), 1.0)

    def downtime_score(self, version):
        generated_dir = self.generated_dir / version
        
        server_files = ["server.py", "server_stub.py", "main.py"]
        server_file = None
        
        for filename in server_files:
            candidate = generated_dir / filename
            if candidate.exists():
                server_file = candidate
                break
        
        if not server_file:
            return float('inf')
        
        start_time = time.time()
        try:
            python_exe = self.root / ".venv" / "bin" / "python"
            if not python_exe.exists():
                python_exe = "python3"
            
            process = subprocess.Popen(
                [str(python_exe), str(server_file)],
                cwd=generated_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(0.5)
            
            if process.poll() is None:
                downtime = time.time() - start_time
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                return downtime
            else:
                stdout, stderr = process.communicate()
                if b"ModuleNotFoundError" in stderr or b"ImportError" in stderr:
                    return -1.0
                else:
                    return float('inf')
        except Exception:
            return float('inf')

    def generation_memory_usage(self, version):
        if version not in self.process_stats or not self.process_stats[version]["memory"]:
            return 0.0
        return max(self.process_stats[version]["memory"])

    def generation_cpu_usage(self, version):
        if version not in self.process_stats or not self.process_stats[version]["cpu"]:
            return 0.0
        cpu_values = [c for c in self.process_stats[version]["cpu"] if c > 0]
        return sum(cpu_values) / len(cpu_values) if cpu_values else 0.0

    def count_tools_in_file(self, file_path):
        """Count the number of @mcp.tool decorators in a Python file."""
        if not file_path.exists():
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tool_count = len(re.findall(r'@mcp\.tool\(', content))
            return tool_count
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 0

    def tools_generated_count(self, version):
        """Get the total number of tools generated for a specific experiment version."""
        generated_dir = self.generated_dir / version
        
        if not generated_dir.exists():
            return 0
        
        total_tools = 0
        
        server_stub_path = generated_dir / "server_stub.py"
        total_tools += self.count_tools_in_file(server_stub_path)
        
        oauth_server_path = generated_dir / "oauth_login_server.py"
        total_tools += self.count_tools_in_file(oauth_server_path)
        
        return total_tools

    def evaluate_version(self, version):
        print(f"Evaluating {version}")
        
        generation_result = self.run_generation(version)
        
        return {
            "version": version,
            "generation_success": generation_result["success"],
            "generation_time_seconds": generation_result["generation_time"],
            "endpoint_coverage_ratio": self.endpoint_coverage_score(version),
            "generation_score_ratio": self.generation_score(version),
            "downtime_seconds": self.downtime_score(version),
            "memory_usage_mb": self.generation_memory_usage(version),
            "cpu_usage_percent": self.generation_cpu_usage(version),
            "tools_generated": self.tools_generated_count(version)
        }

    def run_all_evaluations(self):
        versions = self.get_versions()
        results = []
        
        for version in versions:
            result = self.evaluate_version(version)
            results.append(result)
        
        with open(self.root / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    evaluator = AutoMCPEvaluator()
    results = evaluator.run_all_evaluations()
    
    for result in results:
        print(f"{result['version']}: Success={result['generation_success']}, "
              f"Coverage={result['endpoint_coverage_ratio']:.2f}, "
              f"Tools={result['tools_generated']}")


if __name__ == "__main__":
    main()