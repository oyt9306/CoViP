# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_forward
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import yaml
import json
import random
import math

from faker import Faker


fake_app = Faker(['ko_KR', 'es_ES'])

def return_random_prompt():
    name = fake_app.name()
    naming_prompts = [
        f"Is {name} visible in this picture?",
        f"Is {name} in this image?",
        f"Do you see {name} in the photo?",
        f"Is {name} present in this photograph?",
        f"Can you identify if {name} is captured in this picture?",
        f"Is {name} depicted in this image?",
        f"Does the picture feature {name}?",
        f"Can you confirm if {name} appears in this photo?",
        f"Is {name} included in this shot?",
        f"Is {name} shown in this image?",
        f"Can you detect {name} in the photo?",
        f"Is {name} captured in this image?",
        f"Do you recognize {name} in this picture?"
    ]
    return name, random.choice(naming_prompts)



def return_prompt_vision(name, naming_prompts):
    prompt = f"As an evaluation expert, your task is to verify whether the object identified as {name} in the first image is also present in the second image. {naming_prompts}"
    return prompt

def return_prompt_caption():
    prompt_captioning = """You are an AI model that can perceive multiple past dialogues and use them as memory to personalize your description of a new image.

[Context]
You have been given several past dialogues.
Each dialogue contains an image and a corresponding conversation between a user and you.
These conversations describe specific objects (people, animals, items, or places) along with contextual details such as names, locations, times, and experiences.
This entire context represents your prior shared experiences with the user.

[Task]
Now, you are given a **new image** that may include one or more of the same objects mentioned in the previous dialogues.
Your goal is to describe this new image **by integrating relevant information from the context**.

Follow these rules carefully:

1. **Recall and reuse details** from the previous dialogues (object names, appearances, places, times, and relationships).
   - Treat the previous dialogues as your long-term memory.
   - If an object in the new image appears similar to one mentioned in the past, refer to it using the same name and contextual background.

2. **Ground your description in the new image’s visual content.**
   - Accurately describe what you see: composition, setting, lighting, and object state.
   - Then integrate remembered details from the context naturally (e.g., “This looks like Pino again, perhaps older than in the park photo from Busan Station.”).

3. Keep your tone natural and human-like — as if you’re describing something familiar to the same user.

4. Do not restate previous dialogues verbatim. Instead, synthesize and extend them with new image-grounded observations.

5. Write in paragraph form, not in a dialogue format.

6. **Use only relevant memories.**
   - If an object or scene described in the previous dialogues does **not appear in the new image**, ignore it completely.
   - Include contextual information **only for the objects that actually appear** in the new image.
   - Avoid bringing up unrelated names, locations, or events from the past context."""
    return prompt_captioning


def get_multi_vision_qa():
    prompt_multi_choice = """You are given multiple images.
Each image corresponds, in order, to a specific concept.

Your task is to identify which concept images are present in the final query image.
Include the numbers (indices) of the concept images that appear in the query image.

The query image may contain up to four concept images randomly selected from the given concepts.
However, some of the concept images are irrelevant to the query.
Do NOT include those irrelevant images in your answer.

[Answering Rules]

Observe all the concept images and the final query image carefully.

Determine which concept images appear in the query image.

For each question:

You may briefly explain your reasoning in natural language.

Then, on a separate line, output the final choice in the exact format:

Answer: \\boxed{X}

where X is something like [0, 1, 3, 5] (for example, if the query image contains the 0th, 1st, 3rd, and 5th concept images).

Inside \boxed{} you must include only the indices of the selected concept images, written as a list in square brackets, with no extra text."""
    return prompt_multi_choice


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
                random.shuffle(self.list_data_dict)

        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        def make_conversation_vis(example, problem):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": problem},
                        ],
                    },
                ],
            }
        
        def make_conversation_caps(example, problem):
            def return_text(idx):
                return {"type": "text", "text": f"===== Dialogue {idx} =====\n"}

            def return_dialogue(d):
                return {"type": "text", "text": d + "\n\n"}

            empty_image = {"type": "image"}
            template = {
                "prompt": [
                    {
                        "role": "user",
                        "content": [],
                    },
                ],
            }

            len_concept = example['c_num']

            for idx in range(len_concept):
                empty_text = return_text(idx + 1)
                empty_diag = return_dialogue(example['dialogues'][idx])

                template["prompt"][0]['content'].append(empty_text)
                template["prompt"][0]['content'].append(empty_image.copy())
                template["prompt"][0]['content'].append(empty_diag)

            template["prompt"][-1]['content'].append(empty_image.copy())
            template["prompt"][-1]['content'].append(
                {"type": "text", "text": problem}
            )

            return template

        def make_mul_vision_conversation(example, problem):
            def return_text(idx):
                return {"type": "text", "text": f"===== Concept {idx} =====\n"}

            empty_image = {"type": "image"}
            template = {
                "prompt": [
                    {
                        "role": "user",
                        "content": [],
                    },
                ],
            }

            len_concept = example['c_num']

            for idx in range(len_concept):
                empty_text = return_text(idx)
                template["prompt"][0]['content'].append(empty_text)
                template["prompt"][0]['content'].append(empty_image.copy())

            template["prompt"][-1]['content'].append(
                {"type": "text", "text": problem}
            )
            template["prompt"][-1]['content'].append(empty_image.copy())

            return template
        
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        name, naming_prompts = return_random_prompt()

        if isinstance(example['images'], list):
            image_path = [img for img in example['images']]
            image = [Image.open(PATH).convert("RGB") for PATH in image_path]

        if example['accu_reward_method'] == 'vision':
            make_conv = make_conversation_vis
            problem = return_prompt_vision(name, naming_prompts)
            solution = example['answer']
            c_num = 2
        elif example['accu_reward_method'] == 'caption':
            make_conv = make_conversation_caps
            problem = return_prompt_caption()
            solution = [example['pos_qas'], example['neg_qas']] 
            c_num = example['c_num']
        elif example['accu_reward_method'] == 'multi_vision':
            make_conv = make_mul_vision_conversation
            problem = get_multi_vision_qa()
            solution = example['answer']
            c_num = example['c_num']
            
        return {    
            'image': image,
            'c_num': c_num,
            'problem': problem,
            'solution': solution,
            'prompt': make_conv(example, problem)['prompt'],
            'accu_reward_method': example['accu_reward_method'],
        }
        
def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen3VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    elif "glm" in model_name_or_path.lower():
        return GLMVModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.accuracy_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
