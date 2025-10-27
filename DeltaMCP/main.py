import os
import sys
import json
import ast
import argparse
import shutil
import textwrap
import threading
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
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


def discover_trained_model_dirs(models_root: Path) -> List[Path]:
    base_path = Path(models_root)
    if not base_path.exists():
        return []

    try:
        entries = list(base_path.iterdir())
    except Exception:
        return []

    candidates: List[Path] = []
    for entry in entries:
        if not entry.is_dir():
            continue

        has_adapter = (entry / "adapter_model.safetensors").exists()
        has_full_model = (entry / "pytorch_model.bin").exists() or (entry / "model.safetensors").exists()
        has_config = (entry / "config.json").exists() or (entry / "adapter_config.json").exists()

        if has_config and (has_adapter or has_full_model):
            candidates.append(entry)
            continue

        if any((entry / name).is_dir() for name in ("checkpoint", "checkpoints", "final")):
            candidates.append(entry)

    return sorted(candidates, key=lambda item: item.name.lower())


class DeltaMCPGenerator:

    HTTP_METHODS: Set[str] = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
    
    def __init__(
        self,
        model_name: str = "DeltaMCP-Phi-3-24Oct",
        device: str = "auto",
        max_workers: Optional[int] = None,
        prepare_workers: Optional[int] = None,
        apply_workers: Optional[int] = None,
        inference_workers: Optional[int] = None
    ):
        self.model_path = str(Path(__file__).parent / "llm-finetuned" / model_name)
        self.device = device
        self.tokenizer = None
        self.model = None

        cpu_count = os.cpu_count() or 1

        def resolve_workers(value: Optional[int], fallback: int) -> int:
            if value is None:
                return max(1, fallback)
            return max(1, min(value, cpu_count))

        default_pool = min(cpu_count, 8) if cpu_count > 4 else cpu_count
        self.max_workers = resolve_workers(max_workers, default_pool)
        self.prepare_workers = resolve_workers(prepare_workers, self.max_workers)
        self.apply_workers = resolve_workers(apply_workers, self.max_workers)

        if inference_workers is None:
            default_inference = max(1, min(self.apply_workers, max(1, cpu_count // 2)))
        else:
            default_inference = inference_workers
        self.inference_workers = resolve_workers(default_inference, default_inference)
        self._inference_semaphore = threading.Semaphore(self.inference_workers)

        self.original_tools: Dict[str, str] = {}
        self._progress_state: Optional[Dict[str, int]] = None
        self._progress_lock = threading.Lock()
        self._tag_comments = {
            "added": "#added with DeltaMCP",
            "updated": "#updated with DeltaMCP"
        }
        self.spec_a_path: Optional[str] = None
        self.spec_b_path: Optional[str] = None
        self._spec_b_data: Optional[Dict] = None
        self._http_methods_lower = {method.lower() for method in self.HTTP_METHODS}
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
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
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
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    torch_dtype='auto'
                ).eval()
            try:
                self.tokenizer.padding_side = "left"
            except Exception:
                pass
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @staticmethod
    def _clean_response(text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n")
        text = text.replace("<|im_end|>", "")
        text = text.strip()

        for prefix in ("Assistant:", "assistant:", "A:", "a:"):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()

        code_block = ""
        fenced_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
        for block in fenced_blocks:
            if "@mcp.tool" in block:
                code_block = block
                break

        if code_block:
            text = code_block
        elif "@mcp.tool" in text:
            text = text[text.index("@mcp.tool") :]

        stop_tokens = (
            "\n```",
            "\nUser:",
            "\nExisting implementation",
            "\nChanges to apply",
            "\nAssistant:",
            "\nassistant:",
        )
        for token in stop_tokens:
            idx = text.find(token)
            if idx != -1:
                text = text[:idx]

        text = text.replace("```", "")
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
        with ThreadPoolExecutor(max_workers=self.prepare_workers) as executor:
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

        if tools_a:
            existing_blocks: List[str] = []
            for name, code in tools_a.items():
                sanitized = code.strip()
                existing_blocks.append(
                    f"Tool `{name}` current implementation:\n```python\n{sanitized}\n```"
                )
            tools_a_str = "\n\n".join(existing_blocks)
        else:
            tools_a_str = "<none>"

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
        guidance = (
            "Follow these mandatory rules:\n"
            "1. Preserve the existing `@mcp.tool` decorator metadata (name, description, parameter list).\n"
            "2. Keep the HTTP verb and request construction identical unless the diff explicitly requires a change.\n"
            "3. Do not add or remove path/query parameters unless mandated by the diff.\n"
            "4. Retain the authentication, headers, and error handling blocks exactly as shown.\n"
            "5. Only adjust request/response payload handling per the diff.\n"
            "6. Respond with valid Python code starting at the decorator, no commentary."
        )

        user_text = (
            "User: Update the tool implementation to reflect the diff while following the rules.\n"
            f"Existing implementation(s):\n{tools_a_str}\n\n"
            f"Changes to apply:\n{diff_summary}\n\n"
            f"{guidance}"
        )
        return f"{user_text}\n\nAssistant:\n"
    
    def generate_response(self, prompt: str) -> str:
        try:
            enc = self.tokenizer(prompt, return_tensors='pt')
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

            target_device = next(self.model.parameters()).device
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)

            with self._inference_semaphore:
                output_ids = self._generate_tokens(input_ids, attention_mask, do_sample=False)

            response = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            cleaned = self._clean_response(response)

            if "@mcp.tool" not in cleaned:
                with self._inference_semaphore:
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

    @staticmethod
    def _detect_indent(lines: List[str]) -> str:
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("@mcp.tool"):
                return line[: len(line) - len(stripped)]
        for line in lines:
            if line.startswith("    "):
                return "    "
            if line.startswith("\t"):
                return "\t"
        return "    "

    @staticmethod
    def _apply_indent(lines: List[str], indent: str) -> List[str]:
        if not indent:
            return lines
        return [f"{indent}{line}" if line else "" for line in lines]

    def _upsert_tool_in_stub(self, stub_content: str, tool_name: str, generated_block: str) -> Tuple[str, str]:
        new_block = generated_block.strip()
        if not new_block:
            return stub_content, "skipped"

        stub_lines = stub_content.splitlines()
        new_lines = textwrap.dedent(new_block).splitlines()

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

                    indent = re.match(r"^\s*", stub_lines[start_idx]).group(0) if stub_lines else ""
                    if not indent:
                        indent = self._detect_indent(stub_lines)
                    indented_block = self._apply_indent(new_lines, indent)
                    replacement = [f"{indent}{tag_comment}" if indent else tag_comment] + indented_block
                    stub_lines = stub_lines[:start_idx] + replacement + stub_lines[end_idx:]
                    updated = "\n".join(stub_lines).rstrip() + "\n"
                    updated = self._deduplicate_tool_definitions(updated, tool_name)
                    return updated, "updated"

        tag_comment = self._tag_comments["added"]
        indent = self._detect_indent(stub_lines)
        indented_block = self._apply_indent(new_lines, indent)
        block_with_comment = "\n".join([f"{indent}{tag_comment}" if indent else tag_comment] + indented_block)
        separator = "\n\n" if stub_content.rstrip("\n") else ""
        updated_stub = stub_content.rstrip("\n") + separator + block_with_comment.rstrip() + "\n"
        updated_stub = self._deduplicate_tool_definitions(updated_stub, tool_name)
        return updated_stub, "added"

    def _deduplicate_tool_definitions(self, stub_content: str, tool_name: str) -> str:
        try:
            tree = ast.parse(stub_content)
        except SyntaxError:
            return stub_content

        lines = stub_content.splitlines()
        occurrences: List[Tuple[int, int, bool]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == tool_name:
                decorator_lines = [getattr(dec, 'lineno', node.lineno) for dec in node.decorator_list]
                start_line = min(decorator_lines) if decorator_lines else node.lineno
                end_line = node.end_lineno

                start_idx = max(start_line - 1, 0)
                end_idx = end_line

                while start_idx > 0 and lines[start_idx - 1].strip() in self._tag_comments.values():
                    start_idx -= 1

                has_tag = False
                if 0 <= start_idx < len(lines):
                    has_tag = lines[start_idx].strip() in self._tag_comments.values()

                occurrences.append((start_idx, end_idx, has_tag))

        if len(occurrences) <= 1:
            return stub_content

        occurrences.sort(key=lambda item: (0 if item[2] else 1, item[0]))
        keep_start, keep_end, _ = occurrences[0]

        new_lines = lines[:]
        for start_idx, end_idx, _ in sorted(occurrences[1:], key=lambda item: item[0], reverse=True):
            del new_lines[start_idx:end_idx]
            while start_idx < len(new_lines) and new_lines[start_idx].strip() == "" and (start_idx == 0 or new_lines[start_idx - 1].strip() == ""):
                del new_lines[start_idx]

        return "\n".join(new_lines).rstrip() + "\n"

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

    def _load_spec_data(self, spec_path: Optional[str]) -> Optional[Dict]:
        if not spec_path:
            return None
        try:
            with open(spec_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as exc:
            print(f"Warning: unable to load spec data from {spec_path}: {exc}")
            return None

    def _get_methods_for_path(self, spec_data: Optional[Dict], path: str) -> Set[str]:
        methods: Set[str] = set()
        if not spec_data or not path:
            return methods
        paths = spec_data.get("paths")
        if not isinstance(paths, dict):
            return methods
        path_item = paths.get(path)
        if not isinstance(path_item, dict):
            return methods
        for key in path_item.keys():
            if isinstance(key, str) and key.lower() in self._http_methods_lower:
                methods.add(key.upper())
        return methods

    def _determine_expected_http_method(self, tool_name: str, sample: Dict) -> Optional[str]:
        original_source = self.original_tools.get(tool_name)
        original_method = extract_http_method_from_function(original_source) if original_source else None

        relevant = sample.get("relevant_paths") or {}
        if isinstance(relevant, dict):
            paths = list(relevant.keys())
        elif isinstance(relevant, list):
            paths = relevant
        else:
            paths = []

        if not paths:
            return original_method.upper() if original_method else None

        aggregated_methods: Set[str] = set()
        for path in paths:
            path_methods = self._get_methods_for_path(self._spec_b_data, path)
            aggregated_methods.update(path_methods)
            if original_method and original_method.upper() in path_methods:
                return original_method.upper()

        if not aggregated_methods:
            return original_method.upper() if original_method else None
        if len(aggregated_methods) == 1:
            return next(iter(aggregated_methods))
        return original_method.upper() if original_method else None

    def _rewrite_http_method(self, tool_name: str, generated_block: str, expected_method: str) -> str:
        if not generated_block or not expected_method:
            return generated_block

        current_method = extract_http_method_from_function(generated_block)
        if current_method and current_method.upper() == expected_method.upper():
            return generated_block

        expected_lower = expected_method.lower()

        pattern = re.compile(r"requests\.(?P<verb>[a-zA-Z_][a-zA-Z0-9_]*)")

        def replace_first(match: re.Match) -> str:
            verb = match.group('verb')
            lower = verb.lower()
            if lower == expected_lower:
                return match.group(0)
            if lower in self._http_methods_lower:
                print(f"Adjusting HTTP method for {tool_name}: {verb.upper()} -> {expected_method.upper()}")
                return f"requests.{expected_lower}"
            return match.group(0)

        new_block, count = pattern.subn(replace_first, generated_block, count=1)
        return new_block if count else generated_block

    def _enforce_expected_http_method(self, tool_name: str, sample: Dict, generated_block: str) -> str:
        expected_method = self._determine_expected_http_method(tool_name, sample)
        if not expected_method:
            return generated_block
        return self._rewrite_http_method(tool_name, generated_block, expected_method)

    def _batched_generate(self, prompts: List[str], batch_size: int = 2) -> List[str]:
        if not prompts:
            return []

        results: List[str] = []
        batch_size = max(1, batch_size)

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            enc = self.tokenizer(batch_prompts, return_tensors='pt', padding=True)
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
                "pad_token_id": self.tokenizer.eos_token_id
            }

            with self._inference_semaphore:
                with torch.no_grad():
                    output_ids = self.model.generate(**gen_kwargs)

            for idx in range(len(batch_prompts)):
                prompt_len = int(attention_mask[idx].sum().item())
                generated_ids = output_ids[idx][prompt_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned = self._clean_response(decoded) or decoded.strip()
        cleaned = textwrap.dedent(cleaned).strip()
        results.append(cleaned)

        return results

    def _start_progress(self, label: str, total: int):
        with self._progress_lock:
            self._progress_state = {
                "label": label,
                "total": max(total, 0),
                "current": 0
            }
        if total <= 0:
            return
        self._render_progress()

    def _advance_progress(self):
        with self._progress_lock:
            if not self._progress_state or self._progress_state["total"] <= 0:
                return
            self._progress_state["current"] = min(
                self._progress_state["current"] + 1,
                self._progress_state["total"]
            )
        self._render_progress()

    def _finish_progress(self):
        with self._progress_lock:
            if not self._progress_state or self._progress_state["total"] <= 0:
                self._progress_state = None
                return
            self._progress_state["current"] = self._progress_state["total"]
        self._render_progress(end=True)
        with self._progress_lock:
            self._progress_state = None

    def _render_progress(self, end: bool = False):
        with self._progress_lock:
            state = self._progress_state
            if not state or state["total"] <= 0:
                return
            total = state["total"]
            current = state["current"]
            label = state["label"]
        bar_length = 40
        filled = int(bar_length * current / total) if total else bar_length
        bar = "#" * filled + "-" * (bar_length - filled)
        message = f"\r{label}: [{bar}] {current}/{total}"
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

        removal_samples = [sample for sample in samples if sample.get("action") == "remove"]
        update_samples = [sample for sample in samples if sample.get("action") != "remove"]

        total = len(removal_samples) + len(update_samples)
        self._start_progress("Applying changes", total)

        for sample in removal_samples:
            tool_name = sample.get('tool_name', 'Unknown')
            updated_stub, removed = self._remove_tool_from_stub(updated_stub, tool_name)
            if removed:
                removed_tools.append(tool_name)
            else:
                print(f"Could not remove tool {tool_name}; tool not found in stub")
            self._advance_progress()

        if update_samples:
            prompts = [self.create_prompt_from_sample(sample) for sample in update_samples]
            responses = self._batched_generate(prompts, batch_size=2)

            for idx, sample in enumerate(update_samples):
                tool_name = sample.get('tool_name', 'Unknown')
                response = responses[idx] if idx < len(responses) else ""

                if '@mcp.tool' not in response:
                    print(f"Retrying {tool_name}: batched response missing tool definition")
                    fallback = self.generate_response(prompts[idx])
                    fallback = textwrap.dedent(fallback).strip()
                    if '@mcp.tool' not in fallback:
                        print(f"Skipping {tool_name}: model response did not include a tool definition")
                        self._advance_progress()
                        continue
                    response = fallback

                response = self._enforce_expected_http_method(tool_name, sample, response)

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
        
        self.spec_a_path = spec_a
        self.spec_b_path = spec_b
        self._spec_b_data = self._load_spec_data(spec_b)

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
    parser.add_argument("--device", default="auto", help="Device for model inference (auto, cuda, cpu, mps)")
    parser.add_argument("--workers", type=int, help="Deprecated: sets both prepare and apply worker pools")
    parser.add_argument("--prepare-workers", type=int, help="Worker pool size for sample preparation")
    parser.add_argument("--apply-workers", type=int, help="Worker pool size for response generation")
    parser.add_argument("--inference-workers", type=int, help="Maximum concurrent inference calls")
    
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
        generator = DeltaMCPGenerator(
            args.model,
            args.device,
            max_workers=args.workers,
            prepare_workers=args.prepare_workers,
            apply_workers=args.apply_workers,
            inference_workers=args.inference_workers
        )
        
        generator.run(args.stub_a, args.spec_a, args.spec_b, args.output)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
