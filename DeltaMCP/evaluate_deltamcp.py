import json
import os
import re
import subprocess
import sys
import threading
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from DeltaMCP.main import DeltaMCPGenerator


class InstrumentedDeltaMCPGenerator(DeltaMCPGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.llm_metrics = {
            "calls": 0,
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time": 0.0,
        }

    def _record_llm_usage(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_ids: torch.Tensor,
        duration: float,
    ) -> None:
        input_cpu = input_ids.detach().cpu()
        if attention_mask is not None:
            attn_cpu = attention_mask.detach().cpu()
            prompt_lengths = attn_cpu.sum(dim=1).tolist()
        else:
            prompt_lengths = [input_cpu.shape[1]] * input_cpu.shape[0]
        output_cpu = output_ids.detach().cpu()

        for idx in range(output_cpu.shape[0]):
            prompt_len = int(prompt_lengths[idx])
            total_len = int(output_cpu[idx].shape[0])
            prompt_len = max(0, min(prompt_len, total_len))
            generated_len = max(total_len - prompt_len, 0)
            self.llm_metrics["total_prompt_tokens"] += prompt_len
            self.llm_metrics["total_generated_tokens"] += generated_len

        self.llm_metrics["calls"] += output_cpu.shape[0]
        self.llm_metrics["total_time"] += duration

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        do_sample: bool,
    ) -> torch.Tensor:
        start = time.time()
        output_ids = super()._generate_tokens(input_ids, attention_mask, do_sample)
        duration = time.time() - start
        self._record_llm_usage(input_ids, attention_mask, output_ids, duration)
        return output_ids

    def _batched_generate(self, prompts: List[str], batch_size: int = 2) -> List[str]:
        if not prompts:
            return []

        results: List[str] = []
        batch_size = max(1, batch_size)

        for start_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start_idx:start_idx + batch_size]
            enc = self.tokenizer(batch_prompts, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            target_device = next(self.model.parameters()).device
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)

            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": 2048,
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            start = time.time()
            with self._inference_semaphore:
                with torch.no_grad():
                    output_ids = self.model.generate(**gen_kwargs)
            duration = time.time() - start
            self._record_llm_usage(input_ids, attention_mask, output_ids, duration)

            attention_cpu = attention_mask.detach().cpu()
            output_cpu = output_ids.detach().cpu()

            for idx in range(len(batch_prompts)):
                prompt_len = int(attention_cpu[idx].sum().item())
                prompt_len = max(prompt_len, 0)
                prompt_len = min(prompt_len, int(output_cpu[idx].shape[0]))
                generated_ids = output_cpu[idx][prompt_len:]
                decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                cleaned = self._clean_response(decoded) or decoded.strip()
                cleaned = textwrap.dedent(cleaned).strip()
                results.append(cleaned)

        return results


class DeltaMCPEvaluator:
    def __init__(self) -> None:
        self.delta_dir = Path(__file__).resolve().parent
        self.root = self.delta_dir.parent
        self.spec_dir = (
            self.root
            / "azure-rest-api-specs"
            / "specification"
            / "storage"
            / "resource-manager"
            / "Microsoft.Storage"
            / "stable"
        )
        self.auto_generated_dir = self.root / "AutoMCP" / "generated"
        self.delta_generated_dir = self.delta_dir / "generated"
        self.results_dir = self.delta_dir / "results"
        self.delta_generated_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.generator = InstrumentedDeltaMCPGenerator()

    def get_version_pairs(self) -> List[Tuple[str, str]]:
        available = sorted(
            [p.name for p in self.spec_dir.iterdir() if p.is_dir() and (p / "storage.json").exists()]
        )
        if "2015-06-15" not in available:
            return []
        start_index = available.index("2015-06-15")
        pairs: List[Tuple[str, str]] = []
        for idx in range(start_index, len(available) - 1):
            current_version = available[idx]
            next_version = available[idx + 1]
            auto_stub = self.auto_generated_dir / next_version / "server_stub.py"
            if not auto_stub.exists():
                break
            pairs.append((current_version, next_version))
        return pairs

    def _start_resource_monitor(self) -> Tuple[threading.Event, threading.Thread, List[float], List[float]]:
        process = psutil.Process(os.getpid())
        stop_event = threading.Event()
        cpu_samples: List[float] = []
        memory_samples: List[float] = []

        def monitor() -> None:
            try:
                process.cpu_percent(interval=None)
            except psutil.Error:
                return
            while not stop_event.is_set():
                try:
                    memory_samples.append(process.memory_info().rss / (1024 * 1024))
                    cpu_samples.append(process.cpu_percent(interval=None))
                except psutil.Error:
                    break
                time.sleep(0.1)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return stop_event, thread, cpu_samples, memory_samples

    @staticmethod
    def _endpoint_coverage(spec_path: Path, stub_path: Path) -> float:
        if not spec_path.exists() or not stub_path.exists():
            return 0.0
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            with open(stub_path, "r", encoding="utf-8") as f:
                stub_content = f.read()
        except Exception:
            return 0.0

        operation_ids: List[str] = []
        for methods in spec.get("paths", {}).values():
            if not isinstance(methods, dict):
                continue
            for method_data in methods.values():
                if not isinstance(method_data, dict):
                    continue
                op_id = method_data.get("operationId")
                if isinstance(op_id, str):
                    operation_ids.append(op_id)

        total = len(operation_ids)
        if total == 0:
            return 0.0
        remaining = set(operation_ids)
        for op_id in list(remaining):
            if op_id in stub_content:
                remaining.discard(op_id)
        covered = total - len(remaining)
        return covered / total

    @staticmethod
    def _generation_score(stub_path: Path) -> float:
        if not stub_path.exists():
            return 0.0
        try:
            with open(stub_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return 0.0

        if not content.strip():
            return 0.0
        score = 0.0
        if "def " in content:
            score += 0.3
        if "class " in content:
            score += 0.3
        if "import " in content:
            score += 0.2
        if len(content.strip()) > 100:
            score += 0.2
        return min(score, 1.0)

    def _downtime_seconds(self, output_dir: Path) -> float:
        candidate_files = ["server.py", "server_stub.py", "main.py"]
        server_file: Optional[Path] = None
        for name in candidate_files:
            path = output_dir / name
            if path.exists():
                server_file = path
                break
        if not server_file:
            return float("inf")

        python_exe = self.root / ".venv" / "bin" / "python"
        if not python_exe.exists():
            python_exe = Path("python3")

        start = time.time()
        try:
            process = subprocess.Popen(
                [str(python_exe), str(server_file)],
                cwd=server_file.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.5)
            if process.poll() is None:
                downtime = time.time() - start
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                return downtime
            stdout, stderr = process.communicate()
            if b"ModuleNotFoundError" in stderr or b"ImportError" in stderr:
                return -1.0
            return float("inf")
        except Exception:
            return float("inf")

    @staticmethod
    def _tools_generated(stub_path: Path) -> int:
        if not stub_path.exists():
            return 0
        try:
            with open(stub_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return 0
        return len(re.findall(r"@mcp\\.tool\\(", content))

    def evaluate(self) -> List[Dict]:
        pairs = self.get_version_pairs()
        if not pairs:
            print("No version pairs discovered; exiting.")
            return []

        results: List[Dict] = []
        current_stub = self.auto_generated_dir / pairs[0][0] / "server_stub.py"
        if not current_stub.exists():
            print(f"Initial stub missing: {current_stub}")
            return []

        for index, (spec_a_version, spec_b_version) in enumerate(pairs, start=1):
            spec_a_path = self.spec_dir / spec_a_version / "storage.json"
            spec_b_path = self.spec_dir / spec_b_version / "storage.json"
            auto_stub_b = self.auto_generated_dir / spec_b_version / "server_stub.py"
            output_dir = self.delta_generated_dir / spec_b_version
            output_dir.mkdir(parents=True, exist_ok=True)
            delta_stub_path = output_dir / "server_stub.py"

            self.generator.reset_metrics()
            stop_event, monitor_thread, cpu_samples, memory_samples = self._start_resource_monitor()
            start_time = time.time()
            success = True
            error_message: Optional[str] = None
            try:
                self.generator.run(
                    str(current_stub),
                    str(spec_a_path),
                    str(spec_b_path),
                    str(delta_stub_path),
                )
            except Exception as exc:
                success = False
                error_message = str(exc)
            finally:
                total_time = time.time() - start_time
                stop_event.set()
                monitor_thread.join(timeout=1)

            memory_usage = max(memory_samples) if memory_samples else 0.0
            cpu_values = [value for value in cpu_samples if value > 0]
            cpu_usage = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0

            llm_metrics = self.generator.llm_metrics
            llm_time = llm_metrics.get("total_time", 0.0) or 0.0
            generated_tokens = llm_metrics.get("total_generated_tokens", 0)
            total_tokens = (
                generated_tokens + llm_metrics.get("total_prompt_tokens", 0)
            )
            throughput = (
                generated_tokens / llm_time if llm_time > 0 and generated_tokens > 0 else 0.0
            )

            delta_coverage = self._endpoint_coverage(spec_b_path, delta_stub_path)
            auto_coverage = self._endpoint_coverage(spec_b_path, auto_stub_b)
            generation_score = self._generation_score(delta_stub_path)
            downtime = self._downtime_seconds(output_dir) if success else float("inf")
            tools_generated = self._tools_generated(delta_stub_path)

            result = {
                "experiment": index,
                "spec_a_version": spec_a_version,
                "spec_b_version": spec_b_version,
                "input_stub": str(current_stub.relative_to(self.root)),
                "output_stub": str(delta_stub_path.relative_to(self.root)) if delta_stub_path.exists() else None,
                "generation_success": success,
                "error": error_message,
                "generation_time_seconds": total_time,
                "correctness": success,
                "delta_endpoint_coverage_ratio": delta_coverage,
                "auto_endpoint_coverage_ratio": auto_coverage,
                "generation_score_ratio": generation_score,
                "downtime_seconds": downtime,
                "memory_usage_mb": memory_usage,
                "cpu_usage_percent": cpu_usage,
                "tools_generated": tools_generated,
                "llm_calls": llm_metrics.get("calls", 0),
                "llm_total_tokens": total_tokens,
                "llm_generated_tokens": generated_tokens,
                "llm_prompt_tokens": llm_metrics.get("total_prompt_tokens", 0),
                "llm_throughput_tokens_per_second": throughput,
                "llm_time_seconds": llm_time,
            }

            results.append(result)

            if success and delta_stub_path.exists():
                current_stub = delta_stub_path
            else:
                print(f"Stopping after experiment {index} due to failure.")
                break

        results_path = self.results_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {results_path}")

        self._generate_charts(results)
        return results

    def _generate_charts(self, results: List[Dict]) -> None:
        if not results or plt is None:
            if plt is None:
                print("matplotlib not available; skipping charts.")
            return

        versions = [item["spec_b_version"] for item in results]
        delta_cov = [item["delta_endpoint_coverage_ratio"] for item in results]
        auto_cov = [item["auto_endpoint_coverage_ratio"] for item in results]

        plt.figure(figsize=(10, 5))
        plt.plot(versions, delta_cov, marker="o", label="DeltaMCP")
        plt.plot(versions, auto_cov, marker="o", label="AutoMCP")
        plt.title("Endpoint Coverage Progression")
        plt.xlabel("Spec Version")
        plt.ylabel("Coverage Ratio")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        coverage_path = self.results_dir / "coverage_progression.png"
        plt.savefig(coverage_path)
        plt.close()

        tokens = [item["llm_generated_tokens"] for item in results]
        gen_time = [item["generation_time_seconds"] for item in results]
        plt.figure(figsize=(8, 5))
        plt.scatter(tokens, gen_time, c="tab:blue")
        plt.title("Generation Time vs Generated Tokens")
        plt.xlabel("Generated Tokens")
        plt.ylabel("Generation Time (s)")
        plt.tight_layout()
        time_tokens_path = self.results_dir / "generation_time_vs_tokens.png"
        plt.savefig(time_tokens_path)
        plt.close()

        throughput = [item["llm_throughput_tokens_per_second"] for item in results]
        memory = [item["memory_usage_mb"] for item in results]
        plt.figure(figsize=(8, 5))
        plt.scatter(memory, throughput, c="tab:green")
        plt.title("Throughput vs Peak Memory")
        plt.xlabel("Peak Memory (MB)")
        plt.ylabel("Throughput (tokens/s)")
        plt.tight_layout()
        throughput_path = self.results_dir / "throughput_vs_memory.png"
        plt.savefig(throughput_path)
        plt.close()

        cpu_usage = [item["cpu_usage_percent"] for item in results]
        plt.figure(figsize=(8, 5))
        plt.scatter(cpu_usage, memory, c="tab:red")
        plt.title("CPU vs Memory Usage")
        plt.xlabel("Average CPU (%)")
        plt.ylabel("Peak Memory (MB)")
        plt.tight_layout()
        cpu_memory_path = self.results_dir / "cpu_vs_memory.png"
        plt.savefig(cpu_memory_path)
        plt.close()

        print("Charts saved to", self.results_dir)


def main() -> None:
    evaluator = DeltaMCPEvaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
