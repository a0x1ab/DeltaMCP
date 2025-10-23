import json
import random
from pathlib import Path


def format_chat_text(oasdiff, tools_a, tools_b):
    """Format the data into a single chat text format."""
    # Format the user message
    user_message = "Generate the updated tool implementation as required in the form of an AST based on the OpenAPI diff."
    user_message += f" OpenAPI Specification Diff: {json.dumps(oasdiff)}"
    
    if tools_a:
        # Convert tools_a dict to a more readable format
        tools_a_str = ""
        for tool_name, tool_ast in tools_a.items():
            tools_a_str += f"{tool_name}: {tool_ast}; "
        user_message += f" and the existing tool implementation as an AST is: {tools_a_str.rstrip('; ')}"
    else:
        user_message += " and there is no existing tool implementation"
    
    # Format the assistant response
    if not tools_b:
        assistant_message = "No tool implementation needed (tool deleted)"
    else:
        assistant_parts = []
        for tool_name, tool_ast in tools_b.items():
            assistant_parts.append(f"{tool_name}: {tool_ast}")
        assistant_message = "; ".join(assistant_parts)
    
    # Combine into chat format
    chat_text = f"User: {user_message}\nAssistant: {assistant_message}"
    
    return chat_text


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
            chat_text = format_chat_text(sample["oasdiff"], sample["tools_a"], sample["tools_b"])
            examples.append({"text": chat_text})
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