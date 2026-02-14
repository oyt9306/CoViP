# from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoModelForImageTextToText, AutoProcessor

from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
from datetime import datetime

from faker import Faker
import random 
from openai import OpenAI
import re 
import ast 
import os  

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

fake_app = Faker(['ko_KR', 'es_ES'])


class RePICVLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        model_cls = AutoModelForImageTextToText # only for Qwen3 Models
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output



    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):

        # -----------------------------
        # Constants (tuning parameters)
        # -----------------------------
        MIN_CONTENT_LEN = 150          # Minimum output length to consider reward valid
        IOU_THRESHOLD   = 0.5            # IoU threshold for correct answer
        # BBOX_REGEX      = r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]"  # Regex pattern to match [x1, y1, x2, y2]
        BBOX_REGEX = r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        # -----------------------------
        # 1) Verifiable reward for OCT
        # -----------------------------
        @torch.no_grad()
        def yes_no_reward(content: str, sol: str, **kwargs) -> float:
            """
            OCT case: Checks if both ground truth and predicted output are 'yes' or 'no' and match.
            - Ignores case sensitivity.
            - Extracts 'yes' or 'no' if present anywhere in the string.
            - Returns 1.0 if they match, otherwise 0.0.
            """
            gt = (sol or "").lower()
            out = (content or "").lower()

            gt_m = re.search(r"(yes|no)", gt, flags=re.IGNORECASE)
            out_m = re.search(r"(yes|no)", out, flags=re.IGNORECASE)

            gt_ans = (gt_m.group(1).lower() if gt_m else "")
            out_ans = (out_m.group(1).lower() if out_m else "")

            return 1.0 if gt_ans and out_ans and (gt_ans == out_ans) else 0.0


        # --------------------------------
        # 2) Verifiable reward for single-ICT
        # --------------------------------
        @torch.no_grad()
        def yes_no_answer_reward(content: str, sol: str, **kwargs) -> float:
            """
            ICT (single-answer) case: Checks if the exact ground truth string exists in the model output.
            - Applies length regularization: output must be at least MIN_CONTENT_LEN characters.
            - Prevents reward hacking: if the ground truth starts with '<', the closing '</' tag 
            should not also be included in the prediction.
            - Uses exact string matching with re.escape to avoid regex special character issues.
            """
            if not isinstance(content, str) or not isinstance(sol, str):
                return 0.0

            # if len(content) < MIN_CONTENT_LEN:
            #     return 0.0

            found = re.search(re.escape(sol), content) is not None

            if sol.startswith("<"):  # Prevents reward hacking for tags like <name>
                hacked = re.search(re.escape(sol.replace("<", "</")), content) is not None
                return 1.0 if (found and not hacked) else 0.0

            return 1.0 if found else 0.0


        # --------------------------------
        # 3) Verifiable reward for multi-ICT
        # --------------------------------
        @torch.no_grad()
        def yes_no_answer_reward_multi(content: str, sol: Sequence[str], **kwargs) -> float:
            """
            ICT (multi-answer) case: Checks if each answer token in 'sol' exists in the output.
            - Returns the average success rate across all tokens.
            - Applies length regularization: output must be at least MIN_CONTENT_LEN characters.
            - Uses exact string matching with re.escape.
            """
            if not isinstance(content, str) or not isinstance(sol, (list, tuple)):
                return 0.0

            # if len(content) < MIN_CONTENT_LEN:
            #     return 0.0

            if len(sol) == 0:
                return 0.0

            hits = [(re.search(re.escape(s), content) is not None) for s in sol]
            return float(sum(1 for h in hits if h)) / float(len(hits))


        # -----------------------------
        # 4) Verifiable reward for IOU
        # -----------------------------
        def _iou(box1: Sequence[int], box2: Sequence[int]) -> float:
            """
            Calculates IoU between two bounding boxes.
            - Boxes are in the format [x1, y1, x2, y2] with inclusive coordinates.
            - Returns IoU value between 0.0 and 1.0.
            """
            x11, y11, x12, y12 = box1
            x21, y21, x22, y22 = box2

            inter_x1 = max(x11, x21)
            inter_y1 = max(y11, y21)
            inter_x2 = min(x12 - 1, x22 - 1)
            inter_y2 = min(y12 - 1, y22 - 1)

            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
            else:
                inter = 0

            area1 = max(0, (x12 - x11)) * max(0, (y12 - y11))
            area2 = max(0, (x22 - x21)) * max(0, (y22 - y21))
            union = max(1, area1 + area2 - inter)  # Avoid division by zero

            return float(inter) / float(union)


        @torch.no_grad()
        def iou_reward(content: str, sol: Sequence[int], **kwargs) -> float:
            """
            IoU case: Extracts the predicted bounding box from model output and compares it to the ground truth.
            - The predicted box is extracted using a regex pattern.
            - Returns 1.0 if IoU is above IOU_THRESHOLD, else 0.0.
            """
            if not isinstance(content, str) or not (isinstance(sol, (list, tuple)) and len(sol) == 4):
                return 0.0

            m = re.search(BBOX_REGEX, content)
            if not m:
                return 0.0

            pred = [int(m.group(i)) for i in range(1, 5)]
            return 1.0 if _iou(pred, sol) >= IOU_THRESHOLD else 0.0
    
        """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        # print(contents, solution, kwargs, kwargs.get("accu_reward_method"))
        for content, sol, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
            # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
            if accu_reward_method == "yes_no":
                reward = yes_no_reward(content, str(sol))
            elif accu_reward_method == "yes_no_name_multi":
                reward = yes_no_answer_reward_multi(content, sol if isinstance(sol, (list, tuple)) else [str(sol)])
            elif accu_reward_method == "iou":
                reward = iou_reward(content, sol if isinstance(sol, (list, tuple)) else [])
            elif accu_reward_method == "yes_no_name":
                reward = yes_no_answer_reward(content, str(sol))
            else:
                reward = 0.0
            # print(accu_reward_method, reward)
            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"accu_reward_method: {accu_reward_method}\n")
                    f.write(f"image_path: {image_path}\n")
                    # for stage-2
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards

