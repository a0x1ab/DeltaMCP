import subprocess
import argparse
from pathlib import Path
from datetime import datetime


class AutoMCPRunner:
    def __init__(self, workspace_root=None):
        self.root = Path(workspace_root) if workspace_root else Path.cwd()
        self.data_dir = self.root / "data"
        self.automcp_dir = self.root / "AutoMCP"

    def get_versions(self):
        if not self.data_dir.exists():
            return []
        return sorted(
            [
                item.name
                for item in self.data_dir.iterdir()
                if item.is_dir() and (item / "storage.json").exists()
            ]
        )

    def run_version(self, version, monitor_callback=None):
        spec_file = self.data_dir / version / "storage.json"
        output_dir = self.automcp_dir / "generated" / version

        if not spec_file.exists():
            raise FileNotFoundError(f"No spec: {spec_file}")

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running {version}...")

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.root}:/workspace",
            "-e",
            "WINEDEBUG=-all",
            "automcp-wine",
            "wine",
            "AutoMCP/automcp.exe",
            "--input",
            f"data/{version}/storage.json",
            "--output",
            f"AutoMCP/generated/{version}",
        ]

        try:
            if monitor_callback:
                process = subprocess.Popen(
                    cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                monitor_callback(process.pid)
                stdout, stderr = process.communicate(timeout=300)
                result_code = process.returncode
            else:
                result = subprocess.run(
                    cmd, cwd=self.root, capture_output=True, text=True, timeout=300
                )
                result_code = result.returncode
                stderr = result.stderr
            
            if result_code != 0:
                print(f"Failed: {stderr}")
                raise RuntimeError(f"Generation failed: {stderr}")
            else:
                print(f"Completed successfully")
        except Exception as e:
            print(f"Error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Run AutoMCP experiments")
    parser.add_argument("--version", help="Run on specific version")
    parser.add_argument("--list", action="store_true", help="List available versions")
    args = parser.parse_args()

    runner = AutoMCPRunner()

    if args.list:
        versions = runner.get_versions()
        print(f"Available versions ({len(versions)}):")
        for v in versions:
            print(f"  {v}")
    elif args.version:
        runner.run_version(args.version)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
