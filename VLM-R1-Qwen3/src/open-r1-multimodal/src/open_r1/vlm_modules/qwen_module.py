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
from transformers import AutoTokenizer

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

fake_app = Faker(['ko_KR', 'es_ES'])

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

def return_answer(a1, a):
    if a in a1:
        return 1 
    else:
        return 0

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

@torch.no_grad()
def return_result(messages):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        messages=messages,
        temperature=0.7,
        top_p=0.8,
        extra_body={"top_k": 20, "min_p":0.0},
    )
    result = response.choices[0].message.content.strip() 
    return result


def return_prompt_vision(name, naming_prompts):
    prompt = f"As an evaluation expert, your task is to verify whether the object identified as {name} in the first image is also present in the second image. {naming_prompts}"
    return prompt

def get_prompt_qatask(ex, question):
    prompt_qatask = """You are given a description about an object. This description may or may not contain enough information to answer a multiple-choice question.
    You must answer the question using only the information in the description below.
    - Do NOT use any external knowledge.
    - Do NOT assume facts that are not clearly supported by the description.

    [Answering Rules]
    1. Read the description carefully.
    2. For each question, choose the SINGLE best option:
    - If one of A/B/C is explicitly or clearly supported by the description, choose that option.
    - If NONE of A/B/C can be confirmed from the description, choose D.
    3. You MUST ignore any information that is not in the description.
    4. For each question:
    - You may briefly explain your reasoning in natural language.
    - Then, on a separate line, output the final choice in the exact format:

        Answer: \\boxed{X}

    where X is one of A, B, C, or D.
    5. Inside \\boxed{} there must be EXACTLY ONE letter (A/B/C/D), no extra text.
    """
    prompt_qatask += f"""
    [Description]
    {ex}

    [Question]
    {question}"""
    return prompt_qatask

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

@torch.no_grad()
def return_VRs_final_neg(caption, answer):
    def vr_neg(q, a):
        reward = 0
        for idx in range(len(q)):
            if return_answer(q[idx], a[idx]) != 1:
                reward += -1 / len(q)
        return reward

    neg_QAs = answer[1]
    matches, correct, judges = return_caption_matches(caption, neg_QAs, ['D'] * len(neg_QAs))
    return vr_neg(matches, correct), judges

@torch.no_grad()
def return_VRs_final_pos(caption, answer):
    def vr_pos(q, a):
        reward = 0
        for idx in range(len(q)):
            if return_answer(q[idx], a[idx]) == 1:
                reward += 1 / 3 
        return reward

    pos_QAs = answer[0]
    pos_QAs = [x for sublist in pos_QAs for x in sublist]
    pos_ans = [pos_QAs[aa]['correct_answer'] for aa in range(len(pos_QAs))]
        
    matches, correct, judges = return_caption_matches(caption, pos_QAs, pos_ans)
    return vr_pos(matches, correct), judges

def return_caption_matches(caption, QAs, ans):
    import re 
    matches, correct= [], ans
    judges = []
    for tmp in QAs:
        QA      = generate_qa_prompt(tmp)
        user    = get_prompt_qatask(caption, QA)
        messages = [
            {
                "role": "user",
                "content": user,
            }
        ]
        result = return_result(messages)
        judges.append(result)
        try:
            matches.append(re.findall(r"\\boxed\{(.*?)\}", result, flags=re.DOTALL)[0])
        except:
            matches.append(['D'])
    return matches, correct, judges


def generate_qa_prompt(qa_data):
    question = qa_data["question"]
    choices = ["A", "B", "C", "D"]
    prompt = f"Question: {question}\n"
    for choice in choices:
        prompt += f"{choice}. {qa_data['options'][choice]}\n"
    return prompt

class Qwen3VLModule(VLMBaseModule):
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
        from collections import Counter

        def has_heavy_repetition(
            text,
            sent_dup_ratio=0.3,
            ngram_n=5,
            ngram_dup_ratio=0.3,
            chunk_sizes=(10, 20, 30),
            chunk_dup_ratio=0.2,
        ) -> bool:
            """
            text에 의미 없는 반복이 심한지를 판단.

            - sent_dup_ratio: 문장 중복 비율 기준
            - ngram_dup_ratio: n-gram 중복 비율 기준
            - chunk_sizes: 반복 패턴을 잡기 위한 연속 토큰 길이 후보들
            - chunk_dup_ratio: chunk 중복 비율 기준
            """
            # 소문자 + 공백 정리
            clean = re.sub(r"\s+", " ", text.strip().lower())

            # --- 1) 문장 단위 중복 검사 ---
            sentences = re.split(r"[.!?]+", clean)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) >= 4:
                total_sents = len(sentences)
                unique_sents = len(set(sentences))
                dup_sents = total_sents - unique_sents
                if total_sents > 0 and dup_sents / total_sents >= sent_dup_ratio:
                    return True

            # --- 2) n-gram 단위 중복 검사 ---
            words = clean.split()
            if len(words) >= ngram_n * 4:
                ngrams = [" ".join(words[i:i+ngram_n])
                        for i in range(len(words) - ngram_n + 1)]
                total_ngrams = len(ngrams)
                unique_ngrams = len(set(ngrams))
                dup_ngrams = total_ngrams - unique_ngrams
                if total_ngrams > 0 and dup_ngrams / total_ngrams >= ngram_dup_ratio:
                    return True

            # --- 3) 큰 덩어리(chunk) 반복 검사 (새로 추가) ---
            # 예: 10/20/30단어씩 끊어서 어떤 덩어리가 여러 번 등장하는지 확인
            for size in chunk_sizes:
                if len(words) < size * 2:
                    # 텍스트가 너무 짧으면 이 길이의 chunk 반복은 어차피 없음
                    continue

                chunks = [
                    " ".join(words[i:i+size])
                    for i in range(0, len(words) - size + 1)
                ]
                counter = Counter(chunks)

                # 2번 이상 등장하는 chunk들만 고려
                repeated_chunks = {c: cnt for c, cnt in counter.items() if cnt > 1}
                if not repeated_chunks:
                    continue

                # 전체 chunk 개수 대비 "중복된 chunk"가 차지하는 비율
                total_chunks = len(chunks)
                repeated_positions = sum(cnt for cnt in repeated_chunks.values())
                # (중복으로 세지만, ratio를 보려는 용도라 대략값이면 충분)

                if total_chunks > 0 and repeated_positions / total_chunks >= chunk_dup_ratio:
                    return True

            return False


        def yes_no_reward_caps(completions, answer, **kwargs) -> float:
            """
            OCT case: Checks if both ground truth and predicted output are 'yes' or 'no' and match.
            - Ignores case sensitivity.
            - Extracts 'yes' or 'no' if present anywhere in the string.
            - Returns 1.0 if they match, otherwise 0.0.
            """
            if has_heavy_repetition(completions):
                return -1.0
            if len(tokenizer.encode(completions)) > 511:
                return -1.0
            vr_pos, judges =  return_VRs_final_pos(completions, answer)
            vr_neg, judges =  return_VRs_final_neg(completions, answer)
            return vr_pos + vr_neg
            
        def yes_no_reward_multi_vis(completions, answer, **kwargs) -> float:
            out = completions
            gt = answer  # 예: [5, 8, 7] 또는 0, "0" 등 단일 값일 수도 있음

            # 🔹 0, "0" 같은 단일 값도 리스트로 통일
            def to_list(x):
                if isinstance(x, (list, tuple, set)):
                    return list(x)
                else:
                    return [x]

            # 1) 예측값 추출
            try:
                # \boxed{...} 안의 내용을 찾음
                matches = re.findall(r"\\boxed\{(.*?)\}", out, flags=re.DOTALL)[0]
                pred = ast.literal_eval(matches)  # "0" -> 0, "[5, 8]" -> [5, 8]

                # 🔹 단일 값이면 리스트로 변환
                pred_list = to_list(pred)
                gt_list = to_list(gt)

                pred_set = set(pred_list)
                gt_set = set(gt_list)

            except Exception:
                return 0.0

            # 2) TP / FP / FN 계산
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            # 3) Precision, Recall 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # 4) F1 score 계산
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            return f1

    
        def yes_no_reward_vis(completions, answer, **kwargs) -> float:
            """
            OCT case: Checks if both ground truth and predicted output are 'yes' or 'no' and match.
            - Ignores case sensitivity.
            - Extracts 'yes' or 'no' if present anywhere in the string.
            - Returns 1.0 if they match, otherwise 0.0.
            """
            gt = (answer or "").lower()
            out = (completions or "").lower()

            gt_m = re.search(r"(yes|no)", gt, flags=re.IGNORECASE)
            out_m = re.search(r"(yes|no)", out, flags=re.IGNORECASE)

            gt_ans = (gt_m.group(1).lower() if gt_m else "")
            out_ans = (out_m.group(1).lower() if out_m else "")

            return 1.0 if gt_ans and out_ans and (gt_ans == out_ans) else 0.0
    
        """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        # print(contents, solution, kwargs, kwargs.get("accu_reward_method"))
        for content, sol, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
            # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
            # print(content, sol)
            if accu_reward_method == 'vision':
                reward = yes_no_reward_vis(content, sol)
            elif accu_reward_method == 'caption':
                reward = yes_no_reward_caps(content, sol)
            elif accu_reward_method == 'multi_vision':
                reward = yes_no_reward_multi_vis(content, sol)
            else:
                reward = 0.0
            # print(accu_reward_method, reward)
            rewards.append(reward)
    
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                c_num = kwargs.get("c_num")[0]
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"accu_reward_method: {accu_reward_method}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"c_num: {c_num}\n")
                    # for stage-2
                    # f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards

