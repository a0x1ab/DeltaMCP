import subprocess
import argparse
from pathlib import Path
from datetime import datetime


class AutoMCPRunner:
    def __init__(self, workspace_root=None):
        self.root = Path(workspace_root) if workspace_root else Path.cwd()
        self.data_dir = self.root / "data"

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

    def run_version(self, version):
        spec_file = self.data_dir / version / "storage.json"
        output_dir = self.root / version

        if not spec_file.exists():
            raise FileNotFoundError(f"No spec: {spec_file}")

        output_dir.mkdir(exist_ok=True)
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
            version,
        ]

        try:
            result = subprocess.run(
                cmd, cwd=self.root, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                print(f"Failed: {result.stderr}")
            else:
                print(f"Completed successfully")
        except Exception as e:
            print(f"Error: {e}")


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
