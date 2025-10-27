from typing import  Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from peft import PeftModel
import json
import os


class EndpointHandler():
    def __init__(self, path=""):
        base_model_path = json.load(open(os.path.join(path, "training_params.json")))["model"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)
        if parameters is not None:
            prediction = self.pipeline(inputs, **parameters)
        else:
            prediction = self.pipeline(inputs)
        return prediction