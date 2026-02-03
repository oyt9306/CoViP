import os
import torch
import torch.nn as nn

import numpy as np
import json
from tqdm import tqdm

import huggingface_hub

from transformers import (
    AutoProcessor,
)

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from typing import List, Optional, Tuple
from faker import Faker
import argparse
from glob import glob 

# CUDA_VISIBLE_DEVICES=2 python generate_dialogue.py --batch_size 8 --data_path ./Benchmark/two

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    # print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def get_prompt_dial(name):
    prompt = f"""You are an AI model that can both perceive images and converse naturally with a human user.

[Goal]
Generate a short 6-turn dialogue between a fictional user and the model based on the given image.
The conversation should revolve around the main object in the image (person, animal, item, or place).

[Given]
The name of the main object is: {name}

[Guidelines]
1. The main object’s name ({name}) must be used consistently throughout the dialogue.
   - Do not invent or alter the name.
2. The user should describe a personal experience related to {name}.
   - The experience must include at least one **objective contextual element**, such as a specific **place**, **time**, **event**, or **situation** (e.g., “last summer at the riverside,” “during my first year in college,” “in my grandmother’s backyard”).
3. The model should respond naturally and empathetically — acknowledging, asking gentle questions, or adding brief reflections.
4. Keep the tone human-like, calm, and realistic — not overly emotional or robotic.
5. The conversation should have **6 turns total** (User → Model → User → Model → User → Model).
6. Avoid encyclopedic or factual world knowledge. Focus on the *personal connection* and *shared observation* of the object.

[Output Format]
**Dialogue:**
User: ...
Model: ...
User: ...
Model: ...
User: ...
Model: ..."""
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="Benchmark/two")
    args = parser.parse_args()

    model_id = args.model_id
    batch_size = args.batch_size
    data_path = args.data_path

    model = LLM(model=model_id, max_model_len=8192)

    processor = AutoProcessor.from_pretrained(model_id)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    dials = {}
    files = glob(data_path+'/*')
    data_size = len(data_size)

    ctx_len = len(os.listdir(f"{data_path}/sample_0/concepts"))
    fake = Faker()
    for i in tqdm(range(0, data_size, batch_size)):
        inputs = []
        bs = min(batch_size, data_size-i)
        for k in range(i, i + bs):
            names = []
            # 
            for j in range(ctx_len):
                img_path = files[k] + f"/concepts/concept_{j}.png" 
                obj_name = fake.name().split(" ")[-2]
                while obj_name in names:
                    obj_name = fake.name().split(" ")[-2]
                names.append(obj_name)
                prompt = get_prompt_dial(obj_name)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_path,
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ]
                inputs += [
                    prepare_inputs_for_vllm(message, processor) for message in [messages]
                ]
        outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        for k in range(bs):
            dials[k] = outputs[ctx_len * k:ctx_len * k+ctx_len].outputs[0].text 
            
    save_path = f'dials_concepts_{data_path.split("/")[-1]}.json'
    with open(save_path, "w") as f:
        json.dump(dials, f,  indent = 4)

if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    huggingface_hub.login(token="your_huggingface_token_here") 
    main()