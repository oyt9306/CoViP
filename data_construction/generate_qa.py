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

from typing import List, Optional, Tuple
import random
import copy
import argparse


def shuffle_qa_options_abc(qa):
    """
    Shuffle only the text values of options A/B/C while keeping key order (A,B,C,D) fixed
    and D constant. Also update correct_answer accordingly.
    """
    qa_shuffled = copy.deepcopy(qa)

    for q in qa_shuffled["qa"]:
        options = q["options"]

        # Extract only the texts of A/B/C (in order)
        abc_texts = [options["A"], options["B"], options["C"]]
        random.shuffle(abc_texts)  # shuffle only texts

        # D text stays fixed
        d_text = options["D"]

        # Build new options in fixed order A,B,C,D
        new_options = {
            "A": abc_texts[0],
            "B": abc_texts[1],
            "C": abc_texts[2],
            "D": d_text,
        }

        # Determine new correct answer key by matching text
        correct_text = options[q["correct_answer"]]
        for key, value in new_options.items():
            if value == correct_text:
                q["correct_answer"] = key
                break

        q["options"] = new_options

    return qa_shuffled


prompt_qagen = """You are an AI model that creates factual multiple-choice questions and answers.

[Input]
You are given a conversation between a user and an AI model about a specific object (person, animal, item, or place). The conversation contains objective details such as the object’s name, location, time, or the user’s related experiences.

[Goal]
Generate 3 multiple-choice QA pairs that could later be answered by someone who only has access to a caption describing a new image of the same object (the original conversation will NOT be shown at evaluation time).

[Guidelines]
1) Each question must target an objective detail present in the conversation (e.g., name, place, time, habit/action).
2) Avoid emotions, opinions, or meta-dialogue.
3) Each question must have exactly 3 options: A, B, C.
4) Exactly one option is correct among A, B, C.
5) Make the wrong options (A/B/C except the correct one) plausible but clearly incorrect.
6) Do NOT require external/world knowledge; answers must come from the conversation content.
7) Output MUST be valid JSON, no additional text, no trailing commas.

[JSON Output Schema]
{
  "qa": [
    {
      "id": "Q1",
      "question": "<string>",
      "options": {
        "A": "<string>",
        "B": "<string>",
        "C": "<string>",
      },
      "correct_answer": "A" | "B" | "C"
    },
    {
      "id": "Q2",
      "question": "<string>",
      "options": { ... },
      "correct_answer": "A" | "B" | "C"
    },
    {
      "id": "Q3",
      "question": "<string>",
      "options": { ... },
      "correct_answer": "A" | "B" | "C"
    }
  ]
}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="Benchmark/two")
    args = parser.parse_args()

    model_id = args.model_id
    batch_size = args.batch_size
    data_path = args.data_path

    model = LLM(model=model_id, max_model_len=32768)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    samp_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=16384,
    )

    with open(f"dials_concepts_{data_path.split("/")[-1]}.json") as f:
        dials = json.load(f)

    data_size = len(os.listdir((data_path)))
    ctx_len = len(os.listdir(f"{data_path}/sample_0/concepts"))
    qa_text = []
    qa_json = []
    for i in tqdm(range(0, len(dials), batch_size)):
        bs = min(batch_size, len(dials) - i)
        prompts = []
        for j in range(i, i + bs):
            for k in range(ctx_len):
                messages = [
                    {
                        "role": "user",
                        "content": f"{prompt_qagen}\n\n[Input Conversation]\n{dials[j][k]}",
                    }
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
        outputs = model.generate(prompts, sampling_params=samp_params, use_tqdm=False)
        for j in range(bs):
            qa_text.append([])
            qa_json.append([])
            for k in range(ctx_len):
                qa_text[-1].append(outputs[ctx_len * j + k].outputs[0].text)
                try:
                    qa_json[-1].append(
                        json.loads(json.dumps(outputs[ctx_len * j + k].outputs[0].text))
                    )
                    for l in range(3):
                        qa_json[-1][-1]["qa"][l]["options"][
                            "D"
                        ] = "The answer cannot be determined"
                        if list(qa_json[-1][-1]["qa"][l].keys()) != [
                            "id",
                            "question",
                            "options",
                            "correct_answer",
                        ]:
                            qa_json[-1][-1]["qa"][l]["correct_answer"] = qa_json[-1][-1][
                                "qa"
                            ][l]["options"]["correct_answer"]
                            del qa_json[-1][-1]["qa"][l]["options"]["correct_answer"]
                except Exception:
                    qa_json[-1].append("Error")

    qa_json_final = []
    for i in range(len(qa_json)):
        qa_json_final.append([])
        for j in range(len(qa_json[i])):
            try:
                qa_json_final[-1].append(shuffle_qa_options_abc(qa_json[i][j]))
            except Exception:
                qa_json_final[-1].append("Error")

    save_path = f"qa_concepts_{data_path.split("/")[-1]}.json"
    with open(save_path, "w") as f:
        json.dump(qa_json_final, f)

if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    HF_TOKEN = "your_huggingface_token_here"
    huggingface_hub.login(token=HF_TOKEN)
    main()