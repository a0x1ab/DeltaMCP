import json
import random
from pathlib import Path


def format_prompt(oasdiff, tools_a):
    prompt_parts = []
    prompt_parts.append("# Task")
    prompt_parts.append("Generate the updated tool implementation (as AST) based on the OpenAPI diff:")
    prompt_parts.append("")
    
    prompt_parts.append("# OpenAPI Specification Diff")
    prompt_parts.append(json.dumps(oasdiff, indent=2))
    prompt_parts.append("")
    
    if tools_a:
        prompt_parts.append("# Existing Tool Implementation (AST)")
        for tool_name, tool_ast in tools_a.items():
            prompt_parts.append(f"## {tool_name}")
            prompt_parts.append(tool_ast)
            prompt_parts.append("")
    else:
        prompt_parts.append("# No existing tool implementation")
        prompt_parts.append("")
    
    return "\n".join(prompt_parts)


def format_completion(tools_b):
    if not tools_b:
        return "# No tool implementation needed (tool deleted)"
    
    completion_parts = []
    for tool_name, tool_ast in tools_b.items():
        completion_parts.append(f"# {tool_name}")
        completion_parts.append(tool_ast)
    
    return "\n".join(completion_parts)


def load_training_samples(data_dir):
    samples = []
    json_files = list(data_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "processing_errors.log"]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            if all(key in sample for key in ["oasdiff", "tools_a", "tools_b"]):
                samples.append(sample)
        except:
            continue
    
    return samples


def create_training_examples(samples):
    examples = []
    
    for sample in samples:
        try:
            prompt = format_prompt(sample["oasdiff"], sample["tools_a"])
            completion = format_completion(sample["tools_b"])
            examples.append({"prompt": prompt, "completion": completion})
        except:
            continue
    
    return examples


def split_train_test(examples):
    random.seed(42)
    shuffled_examples = examples.copy()
    random.shuffle(shuffled_examples)
    test_size = int(len(shuffled_examples) * 0.2)
    return shuffled_examples[test_size:], shuffled_examples[:test_size]


def save_jsonl(examples, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "training_data"
    
    samples = load_training_samples(data_dir)
    examples = create_training_examples(samples)
    train_examples, test_examples = split_train_test(examples)
    
    save_jsonl(train_examples, data_dir / "train.jsonl")
    save_jsonl(test_examples, data_dir / "test.jsonl")


if __name__ == "__main__":
    main()