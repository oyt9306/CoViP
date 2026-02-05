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

from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from typing import List, Optional, Tuple
import re
import argparse


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


def extract_boxed_answer(text):
    """Extract answer from //boxed{answer} format"""
    # Try to find //boxed{...}
    pattern = r"//boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        # Return the last match (final answer)
        return matches[-1].strip()
    elif "//" in text:
        if text.split("//")[-1].strip() != "":
            return text.split("//")[-1].strip()
        else:
            return text.split("//")[-2].strip()
    return ""


def calculate_f1_score(prediction, ground_truth):
    """Calculate token-level F1 score between prediction and ground truth"""
    # Tokenize by splitting on whitespace and converting to lowercase
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # Calculate overlap
    common_tokens = pred_tokens & gt_tokens

    # Calculate precision and recall
    precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common_tokens) / len(gt_tokens) if len(gt_tokens) > 0 else 0

    # Calculate F1
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


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

prompt_personalization = """You are an AI model that can perceive multiple past dialogues and use them as memory to personalize your understanding of a new image.

[Context]
You have been given several past dialogues.
Each dialogue contains an image and a corresponding conversation between a user and you.
These conversations describe specific objects (people, animals, items, or places) along with contextual details such as names, locations, times, and experiences.
This entire context represents your prior shared experiences with the user.

[Task]
Now, you are given a **new image** that may includ e one or more of the same objects mentioned in the previous dialogues.
Your goal is to interpret this new image by integrating relevant information from the context.

Follow these rules carefully:

1. **Recall and reuse details** from the previous dialogues (object names, appearances, places, times, and relationships).
   - Treat the previous dialogues as your long-term memory.
   - If an object in the new image appears similar to one mentioned in the past, refer to it with the same name or contextual background.

2. **Ground your understanding in the new image’s visual content.**
   - Accurately recognize what you see: composition, setting, lighting, actions, and object state.
   - Then integrate relevant remembered details naturally (e.g., “This looks like Pino again, now indoors instead of the park near Busan Station.”).

3. Keep your tone natural and human-like — as if you’re interpreting something familiar to the same user.

4. Do not restate previous dialogues verbatim. Instead, reason by synthesizing memory with the current image content.

5. Write in paragraph form, not in a dialogue format.

6. **Use only relevant memories.**
   - If an object or context from past dialogues does **not appear in the new image**, ignore it completely.
   - Add contextual information **only when it helps understanding** of the visible content.
   - Avoid mentioning unrelated names, locations, or experiences."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token", type=str, default="your_huggingface_token_here"
    )
    parser.add_argument(
        "--model_id", type=str, default="Yeongtak/CoViP-Qwen3-VL-8B-GSPO"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="Yeongtak/lar")
    args = parser.parse_args()

    hf_token = args.hf_token
    model_id = args.model_id
    batch_size = args.batch_size
    data_path = args.data_path

    huggingface_hub.login(token=hf_token)

    model = LLM(
        model=model_id,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    dataset = load_dataset(data_path)
    data_size = len(dataset)
    anss = []
    for ex in tqdm(dataset):
        anss.append(ex["answer"])

    print("Generating captions...")
    captions = []
    for i in tqdm(range(0, data_size, batch_size)):
        inputs = []
        bs = min(batch_size, data_size - i)
        for j in range(i, i + bs):
            user_content = []
            lar_imgs = dataset[j]["images_context"]
            lar_dials = dataset[j]["dialogues_context"]
            for k in range(len(lar_imgs)):
                user_content.append(
                    {
                        "type": "text",
                        "text": f"===== Dialogue {k+1} =====\n",
                    }
                )
                user_content.append(
                    {
                        "type": "image",
                        "image": lar_imgs[k],
                    }
                )
                user_content.append(
                    {
                        "type": "text",
                        "text": f"{lar_dials[k]}\n\n",
                    }
                )
            user_content.append(
                {
                    "type": "text",
                    "text": f"===== New Image =====\n",
                }
            )
            user_content.append(
                {
                    "type": "image",
                    "image": dataset[j]["query_image"],
                }
            )
            user_content.append({"type": "text", "text": prompt_captioning})
            messages = [
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
            inputs += [
                prepare_inputs_for_vllm(message, processor) for message in [messages]
            ]
        outputs = model.generate(
            inputs, sampling_params=sampling_params, use_tqdm=False
        )
        for j in range(bs):
            captions.append(outputs[j].outputs[0].text.strip())

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        top_k=20,
        top_p=0.8,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        stop_token_ids=[],
    )

    print("Generating responses for LAR task...")
    responses = []
    for i in tqdm(range(0, data_size, batch_size)):
        inputs = []
        bs = min(batch_size, data_size - i)
        for j in range(i, i + bs):
            user_content = []
            lar_imgs = dataset[j]["images_context"]
            lar_dials = dataset[j]["dialogues_context"]
            for k in range(len(lar_imgs)):
                user_content.append(
                    {
                        "type": "text",
                        "text": f"===== Dialogue {k+1} =====\n",
                    }
                )
                user_content.append(
                    {
                        "type": "image",
                        "image": lar_imgs[k],
                    }
                )
                user_content.append(
                    {
                        "type": "text",
                        "text": f"{lar_dials[k]}\n\n",
                    }
                )
            user_content.append(
                {
                    "type": "text",
                    "text": f"===== New Image =====\n",
                }
            )
            user_content.append(
                {
                    "type": "image",
                    "image": dataset[j]["query_image"],
                }
            )
            messages = [
                {
                    "role": "system",
                    "content": prompt_captioning,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": captions[j],
                },
                {
                    "role": "user",
                    "content": "What was I doing the last time I told you about my most recent experience with the one in the new image?",
                },
            ]
            inputs += [
                prepare_inputs_for_vllm(message, processor) for message in [messages]
            ]
        outputs = model.generate(
            inputs, sampling_params=sampling_params, use_tqdm=False
        )
        for j in range(bs):
            responses.append(outputs[j].outputs[0].text.strip())

    save_path = f"lar_responses_{model_id.split('/')[-1]}.json"
    with open(save_path, "w") as f:
        json.dump(responses, f)


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
