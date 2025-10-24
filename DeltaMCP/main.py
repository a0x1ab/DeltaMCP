import os
import sys
import json
import ast
import argparse
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "llm-finetuned"))
from helpers import compare_specs, convert_to_ast, get_range_of_lines
from samples import (
    compress_oasdiff, extract_changed_tool_functions, is_tool_affected_by_changes,
    extract_http_method_from_function, filter_oasdiff_by_operation,
    has_valid_oasdiff, convert_ast_to_code
)


class DeltaMCPGenerator:
    
    def __init__(self, model_name: str = "DeltaMCP-Phi-3-24Oct", device: str = "auto"):
        self.model_path = str(Path(__file__).parent / "llm-finetuned" / model_name)
        self.device = device
        self.tokenizer = None
        self.model = None
        self.max_workers = max(1, min((os.cpu_count() or 1), 4))
        self.original_tools: Dict[str, str] = {}
        self._progress_state = None
        self._tag_comments = {
            "added": "#added with DeltaMCP",
            "updated": "#updated with DeltaMCP"
        }
        self._load_model()
    
    def _load_model(self):
        print(f"Loading model from: {self.model_path}")
        try:
            adapter_config_path = Path(self.model_path) / "adapter_config.json"
            if adapter_config_path.exists():
                print("Loading LoRA adapter model...")
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config["base_model_name_or_path"]
                print(f"Base model: {base_model_name}")
                tokenizer_source = self.model_path if (Path(self.model_path) / "tokenizer_config.json").exists() else base_model_name
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map=self.device,
                    torch_dtype='auto'
                )
                if base_model.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
                    base_model.resize_token_embeddings(len(self.tokenizer))
                self.model = PeftModel.from_pretrained(base_model, self.model_path).eval()
            else:
                print("Loading full model...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    torch_dtype='auto'
                ).eval()
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @staticmethod
    def _clean_response(text: str) -> str:
        if "@mcp.tool" in text:
            text = text[text.index("@mcp.tool"):]
        text = text.replace("<|im_end|>", "")
        for prefix in ("Assistant:", "A:"):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
        return text.strip()

    def _generate_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, do_sample: bool):
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 2048,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        if do_sample:
            gen_kwargs.update({"temperature": 0.7, "top_p": 0.9})

        with torch.no_grad():
            return self.model.generate(**gen_kwargs)
    
    def extract_tools_from_stub(self, stub_file: str) -> Dict[str, str]:
        if not os.path.exists(stub_file):
            print(f"Stub file not found: {stub_file}")
            return {}
        
        try:
            with open(stub_file, 'r') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            tools = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    has_mcp_decorator = any(
                        isinstance(decorator, ast.Call) and
                        isinstance(decorator.func, ast.Attribute) and
                        decorator.func.attr == 'tool'
                        for decorator in node.decorator_list
                    )
                    
                    if has_mcp_decorator:
                        start_line = node.lineno - 1
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                        
                        source_lines = source_code.split('\n')
                        func_source = '\n'.join(source_lines[start_line:end_line])
                        tools[node.name] = textwrap.dedent(func_source)
            
            print(f"Extracted {len(tools)} tools from stub file")
            return tools
            
        except Exception as e:
            print(f"Error extracting tools from stub: {e}")
            return {}
    
    def generate_training_samples(self, stub_a: str, spec_a: str, spec_b: str) -> List[Dict]:
        print("Generating training samples...")
        
        tools_a = self.extract_tools_from_stub(stub_a)
        self.original_tools = tools_a.copy()
        if not tools_a:
            print("No tools found in stub A")
            return []
        
        try:
            diff = compare_specs(spec_a, spec_b)
            if not has_valid_oasdiff(diff):
                print("No valid differences found between specifications")
                return []
            
            print(f"Found differences between specifications")
        except Exception as e:
            print(f"Error comparing specs: {e}")
            return []
        
        changed_paths = []
        if 'paths' in diff:
            for change_type, paths_data in diff['paths'].items():
                if isinstance(paths_data, dict):
                    changed_paths.extend(paths_data.keys())
                elif isinstance(paths_data, list):
                    changed_paths.extend(paths_data)
        
        if not changed_paths:
            print("No changed paths found")
            return []
        
        print(f"Found {len(changed_paths)} changed paths")
        
        samples = []

        future_to_tool = {}
        tools_list = list(tools_a.items())
        self._start_progress("Preparing samples", len(tools_list))
        if not tools_list:
            self._finish_progress()
            return samples
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for tool_name, tool_source in tools_list:
                future = executor.submit(
                    self._prepare_sample,
                    tool_name,
                    tool_source,
                    diff,
                    changed_paths
                )
                future_to_tool[future] = tool_name

            prepared: Dict[str, Optional[Dict]] = {}
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    prepared[tool_name] = future.result()
                except Exception as exc:
                    print(f"Failed to prepare sample for {tool_name}: {exc}")
                    prepared[tool_name] = None
                finally:
                    self._advance_progress()

        for tool_name, _ in tools_list:
            sample = prepared.get(tool_name)
            if sample:
                samples.append(sample)

        self._finish_progress()
        
        print(f"Generated {len(samples)} training samples")
        return samples
    
    def _create_training_sample(self, tool_name: str, tool_source: str, 
                               diff: Dict, relevant_paths: Dict) -> Optional[Dict]:
        try:
            compressed_diff = compress_oasdiff(diff)
            
            tools_a = {tool_name: tool_source}
            
            sample = {
                "tool_name": tool_name,
                "tools_a": tools_a,
                "relevant_paths": relevant_paths,
                "diff": compressed_diff,
                "action": "update"
            }
            
            return sample
            
        except Exception as e:
            print(f"Error creating training sample for {tool_name}: {e}")
            return None
    
    def create_prompt_from_sample(self, sample: Dict) -> str:
        tools_a = sample.get("tools_a", {})
        diff_data = sample.get("diff", {})
        
        tools_a_str = "<none>" if not tools_a else "\n".join(f"- {name}(...)" for name in tools_a.keys())
        
        diff_lines = []
        p = diff_data.get("p", {})
        
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
                            diff_lines.append(
                                f"- {ctype.capitalize()} operation:\n  {method.upper()} {path}\n"
                                f"  OperationId: {opId}\n  Summary: {desc}"
                            )
            elif isinstance(path_data, list):
                for path in path_data:
                    diff_lines.append(
                        f"- {change_type.capitalize()} operation:\n  {path}\n"
                        f"  Summary: Path {change_type}"
                    )
        
        diff_summary = "\n".join(diff_lines) if diff_lines else "<none>"
        user_text = f"User: Update tools based on diff. Existing tools: {tools_a_str}. Changes: {diff_summary}"
        return f"{user_text}\n\nAssistant:\n"
    
    def generate_response(self, prompt: str) -> str:
        try:
            enc = self.tokenizer(prompt, return_tensors='pt')
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

            target_device = next(self.model.parameters()).device
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)

            output_ids = self._generate_tokens(input_ids, attention_mask, do_sample=False)

            response = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            cleaned = self._clean_response(response)

            if "@mcp.tool" not in cleaned:
                output_ids = self._generate_tokens(input_ids, attention_mask, do_sample=True)
                response = self.tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                cleaned = self._clean_response(response)
            
            return cleaned or response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "# Error generating response"
    
    def _prepare_sample(self, tool_name: str, tool_source: str, diff: Dict, changed_paths: List[str]) -> Optional[Dict]:
        affected_result = is_tool_affected_by_changes(tool_source, changed_paths)

        if not affected_result or not affected_result[0]:
            return None

        _, relevant_paths = affected_result
        http_method = extract_http_method_from_function(tool_source)
        filtered_diff = filter_oasdiff_by_operation(diff, http_method, relevant_paths)

        if not any(filtered_diff.get('paths', {}).values()):
            return None

        action = self._determine_action(filtered_diff)

        if action == "remove":
            return {
                "tool_name": tool_name,
                "action": "remove",
                "relevant_paths": relevant_paths,
                "http_method": http_method
            }

        sample = self._create_training_sample(tool_name, tool_source, filtered_diff, relevant_paths)
        return sample

    def _determine_action(self, filtered_diff: Dict) -> str:
        paths = filtered_diff.get("paths", {}) if isinstance(filtered_diff, dict) else {}
        has_delete = False
        has_update = False

        for change_type, path_data in paths.items():
            normalized = str(change_type).lower()
            if normalized in {"deleted", "removed"}:
                has_delete = True
            elif normalized in {"modified", "changed", "added", "updated"}:
                has_update = True

            if isinstance(path_data, dict):
                for data in path_data.values():
                    if not isinstance(data, dict):
                        continue
                    operations = data.get("operations", {})
                    if isinstance(operations, dict):
                        for op_change_type in operations.keys():
                            norm_op = str(op_change_type).lower()
                            if norm_op in {"deleted", "removed"}:
                                has_delete = True
                            elif norm_op in {"modified", "changed", "added", "updated"}:
                                has_update = True

        if has_delete and not has_update:
            return "remove"
        return "update"

    def _upsert_tool_in_stub(self, stub_content: str, tool_name: str, generated_block: str) -> Tuple[str, str]:
        new_block = generated_block.strip()
        if not new_block:
            return stub_content, "skipped"

        stub_lines = stub_content.splitlines()
        new_lines = new_block.splitlines()

        try:
            tree = ast.parse(stub_content)
        except SyntaxError:
            tree = None

        tag_comment = self._tag_comments["updated"]

        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == tool_name:
                    decorator_lines = [getattr(dec, 'lineno', node.lineno) for dec in node.decorator_list]
                    start_line = min(decorator_lines) if decorator_lines else node.lineno
                    end_line = node.end_lineno

                    start_idx = max(start_line - 1, 0)
                    end_idx = end_line

                    while start_idx > 0 and stub_lines[start_idx - 1].strip() in self._tag_comments.values():
                        start_idx -= 1

                    replacement = [tag_comment] + new_lines
                    stub_lines = stub_lines[:start_idx] + replacement + stub_lines[end_idx:]
                    updated = "\n".join(stub_lines).rstrip() + "\n"
                    return updated, "updated"

        tag_comment = self._tag_comments["added"]
        block_with_comment = "\n".join([tag_comment] + new_lines)
        separator = "\n\n" if stub_content.rstrip("\n") else ""
        updated_stub = stub_content.rstrip("\n") + separator + block_with_comment.rstrip() + "\n"
        return updated_stub, "added"

    def _remove_tool_from_stub(self, stub_content: str, tool_name: str) -> Tuple[str, bool]:
        try:
            tree = ast.parse(stub_content)
        except SyntaxError:
            return stub_content, False

        stub_lines = stub_content.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == tool_name:
                decorator_lines = [getattr(dec, 'lineno', node.lineno) for dec in node.decorator_list]
                start_line = min(decorator_lines) if decorator_lines else node.lineno
                end_line = node.end_lineno

                start_idx = max(start_line - 1, 0)
                end_idx = end_line

                while start_idx > 0 and stub_lines[start_idx - 1].strip() in self._tag_comments.values():
                    start_idx -= 1

                stub_lines = stub_lines[:start_idx] + stub_lines[end_idx:]
                updated = "\n".join(stub_lines).rstrip() + "\n"
                return updated, True

        return stub_content, False

    def _apply_removal_footer(self, stub_content: str, has_removals: bool) -> str:
        stripped = stub_content.rstrip("\n")
        lines = stripped.splitlines()

        removal_tag = "#removed with DeltaMCP"

        while lines and lines[-1].strip() == "":
            lines.pop()

        if has_removals:
            if not lines or lines[-1].strip() != removal_tag:
                lines.append(removal_tag)
        else:
            while lines and lines[-1].strip() == removal_tag:
                lines.pop()

        if not lines:
            return ""

        return "\n".join(lines) + "\n"

    def _start_progress(self, label: str, total: int):
        self._progress_state = {
            "label": label,
            "total": max(total, 0),
            "current": 0
        }
        if total <= 0:
            return
        self._render_progress()

    def _advance_progress(self):
        if not self._progress_state or self._progress_state["total"] <= 0:
            return
        self._progress_state["current"] = min(
            self._progress_state["current"] + 1,
            self._progress_state["total"]
        )
        self._render_progress()

    def _finish_progress(self):
        if not self._progress_state or self._progress_state["total"] <= 0:
            self._progress_state = None
            return
        self._progress_state["current"] = self._progress_state["total"]
        self._render_progress(end=True)
        self._progress_state = None

    def _render_progress(self, end: bool = False):
        state = self._progress_state
        if not state or state["total"] <= 0:
            return
        total = state["total"]
        current = state["current"]
        bar_length = 40
        filled = int(bar_length * current / total) if total else bar_length
        bar = "#" * filled + "-" * (bar_length - filled)
        message = f"\r{state['label']}: [{bar}] {current}/{total}"
        sys.stdout.write(message)
        sys.stdout.flush()
        if end:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def process_samples(self, samples: List[Dict], stub_path: str, output_file: str):
        print(f"Processing {len(samples)} samples...")

        try:
            with open(stub_path, 'r', encoding='utf-8') as f:
                updated_stub = f.read()
        except Exception as exc:
            print(f"Failed to read stub file {stub_path}: {exc}")
            return

        updated_tools: List[str] = []
        added_tools: List[str] = []
        removed_tools: List[str] = []

        total = len(samples)
        self._start_progress("Applying changes", total)

        for index, sample in enumerate(samples, 1):
            tool_name = sample.get('tool_name', 'Unknown')

            if sample.get("action") == "remove":
                updated_stub, removed = self._remove_tool_from_stub(updated_stub, tool_name)
                if removed:
                    removed_tools.append(tool_name)
                else:
                    print(f"Could not remove tool {tool_name}; tool not found in stub")
                self._advance_progress()
                continue

            prompt = self.create_prompt_from_sample(sample)
            response = self.generate_response(prompt)

            if '@mcp.tool' not in response:
                print(f"Skipping {tool_name}: model response did not include a tool definition")
                self._advance_progress()
                continue

            updated_stub, status = self._upsert_tool_in_stub(updated_stub, tool_name, response)

            if status == "updated":
                updated_tools.append(tool_name)
            elif status == "added":
                added_tools.append(tool_name)

            self._advance_progress()

        self._finish_progress()

        if not (updated_tools or added_tools or removed_tools):
            print("No tools were updated. Writing original stub to output path.")

        try:
            updated_stub = self._apply_removal_footer(updated_stub, bool(removed_tools))
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(updated_stub)
        except Exception as exc:
            print(f"Failed to write updated stub to {output_file}: {exc}")
            return

        print(f"Updated stub saved to {output_file}")
    
    def run(self, stub_a: str, spec_a: str, spec_b: str, output_file: str):
        print("Starting DeltaMCP generation...")
        print(f"Input stub: {stub_a}")
        print(f"Spec A: {spec_a}")
        print(f"Spec B: {spec_b}")
        print(f"Output file: {output_file}")
        
        samples = self.generate_training_samples(stub_a, spec_a, spec_b)
        
        if not samples:
            print("No training samples generated. Copying original stub to output.")
            try:
                shutil.copyfile(stub_a, output_file)
                print(f"Original stub copied to {output_file}")
            except Exception as exc:
                print(f"Failed to copy stub to {output_file}: {exc}")
            return
        
        self.process_samples(samples, stub_a, output_file)
        
        print("DeltaMCP generation completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate training samples using fine-tuned model")
    parser.add_argument("--stub-a", required=True, help="Path to input stub file (Python)")
    parser.add_argument("--spec-a", required=True, help="Path to specification A (JSON)")
    parser.add_argument("--spec-b", required=True, help="Path to specification B (JSON)")
    parser.add_argument("--output", help="Output Python stub file (auto-generated if not specified)")
    parser.add_argument("--model", default="DeltaMCP-Phi-3-24Oct", choices=["DeltaMCP-Phi-3-24Oct", "DeltaMCP-StarCoder-Q-24Oct"], help="Model to use")
    parser.add_argument("--device", default="auto", help="Device for model inference (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    for file_path, name in [(args.stub_a, "stub-a"), (args.spec_a, "spec-a"), (args.spec_b, "spec-b")]:
        if not os.path.exists(file_path):
            print(f"Error: {name} file not found: {file_path}")
            sys.exit(1)
    
    if not args.output:
        spec_b_name = Path(args.spec_b).stem
        output_path = Path(args.stub_a).parent / f"server_b.{spec_b_name}.py"
        args.output = str(output_path)
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        generator = DeltaMCPGenerator(args.model, args.device)
        
        generator.run(args.stub_a, args.spec_a, args.spec_b, args.output)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
