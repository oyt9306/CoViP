import os
import torch
import torch.nn as nn

import numpy as np
import json
from tqdm import tqdm

import huggingface_hub

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig,
    AutoProcessor,
)

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from typing import List, Optional, Tuple
import argparse

# CUDA_VISIBLE_DEVICES=2 python generate_caption.py --batch_size 32 --data_path ./Benchmark/two

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    )
    parser.add_argument("--batch_size", type=int, default=32)
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

    with open(f"dials_concepts_{data_path.split("/")[-1]}.json") as f:
        dials = json.load(f)

    data_size = len(os.listdir((data_path)))
    ctx_len = len(os.listdir(f"{data_path}/sample_0/concepts"))
    captions = []
    for i in tqdm(range(0, data_size, batch_size)):
        inputs = []
        for k in range(i, min(i + batch_size, len(dials))):
            user_content = []
            for j in range(ctx_len):
                user_content.append(
                    {
                        "type": "text",
                        "text": f"===== Dialogue {j} =====\n",
                    }
                )
                user_content.append(
                    {
                        "type": "image",
                        "image": f"{data_path}/sample_{k}/concepts/concept_{j}.png",
                    }
                )
                user_content.append(
                    {
                        "type": "text",
                        "text": dials[k][j] + "\n\n",
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
                    "image": f"{data_path}/sample_{k}/query.png",
                }
            )
            user_content.append(
                {
                    "type": "text",
                    "text": prompt_captioning,
                }
            )
            messages = [
                {
                    "role": "user",
                    "content": user_content,
                }
            ]

            inputs += [
                prepare_inputs_for_vllm(message, processor) for message in [messages]
            ]
        outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        for k in range(len(outputs)):
            captions.append(outputs[k].outputs[0].text)

    save_path = f"captions_{model_id.split("/")[-1]}_concepts_{data_path.split("/")[-1]}.json"
    with open(save_path, "w") as f:
        json.dump(captions, f)

if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    HF_TOKEN = "your_huggingface_token_here"
    huggingface_hub.login(token=HF_TOKEN)
    main()