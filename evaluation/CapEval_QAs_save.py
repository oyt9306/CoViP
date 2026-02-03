from openai import OpenAI
import torch
import json
import os
import re
import copy 
import ast
import random 
from tqdm import tqdm 
from pathlib import Path

        
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

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

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

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

def return_caption_matches(caption, QAs, ans, batch_size=4):
    import re
    from concurrent.futures import ThreadPoolExecutor

    if len(QAs) != len(ans):
        raise ValueError(
            f"len(QAs) ({len(QAs)}) and len(ans) ({len(ans)}) must match."
        )

    num_q = len(QAs)

    matches = [None] * num_q   
    correct = [None] * num_q   
    judges  = [None] * num_q   

    prompts = []
    for qa in QAs:
        QA_text = generate_qa_prompt(qa)
        user    = get_prompt_qatask(caption, QA_text)
        prompts.append(user)

    tasks = [(idx, prompts[idx], ans[idx]) for idx in range(num_q)]

    def _call_api(task):
        idx, user_text, corr = task
        messages = [
            {
                "role": "user",
                "content": user_text,
            }
        ]
        result = return_result(messages)

        try:
            pred = re.findall(r"\\boxed\{(.*?)\}", result, flags=re.DOTALL)[0]
        except Exception:
            pred = "D" 

        return idx, pred, corr, result

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for idx, pred, corr, result in executor.map(_call_api, tasks):
            matches[idx] = pred   
            correct[idx] = corr  
            judges[idx]  = result 

    return matches, correct, judges


def generate_qa_prompt(qa_data):
    question = qa_data["question"]
    choices = ["A", "B", "C", "D"]
    prompt = f"Question: {question}\n"
    for choice in choices:
        prompt += f"{choice}. {qa_data['options'][choice]}\n"
    return prompt

def return_answer(a1, a):
    if a in a1:
        return 1 
    else:
        return 0

@torch.no_grad()
def return_accs_final_neg(caption, answer, batch=32):
    def acc_neg(q, a):
        count = []
        for idx in range(len(q)):
            if return_answer(q[idx], a[idx]) != 1:
                count.append(0)
            else:
                count.append(1)
        return count

    neg_QAs = answer[1]
    # matches, correct, judges = return_caption_matches(caption, neg_QAs, ['D'] * len(neg_QAs))
    matches, correct, judges = return_caption_matches(caption, neg_QAs, ['D'] * len(neg_QAs), batch_size=batch)
    return acc_neg(matches, correct)

@torch.no_grad()
def return_accs_final_pos(caption, answer, batch=32):
    def acc_pos(q, a):
        count = []
        for idx in range(len(q)):
            if return_answer(q[idx], a[idx]) == 1:
                count.append(1)
            else:
                count.append(0)
        return count

    pos_QAs = answer[0]
    pos_QAs = [x for sublist in pos_QAs for x in sublist]
    pos_ans = [pos_QAs[aa]['correct_answer'] for aa in range(len(pos_QAs))]
        
    # matches, correct, judges = return_caption_matches(caption, pos_QAs, pos_ans)
    matches, correct, judges = return_caption_matches(caption, pos_QAs, pos_ans, batch_size=batch)

    return acc_pos(matches, correct)

def return_qas(qa, gt):
    gt_names = [f'concept_{i}' for i in gt]
    neg_names = list(qa.keys())
    _ = [neg_names.remove(i) for i in gt_names]

    pos_qas = [qa[i]['qa'] for i in gt_names]
    neg_qas = [qa[i]['qa'] for i in neg_names]
    neg_qas = [x for sublist in neg_qas for x in sublist]
    return pos_qas, neg_qas

def return_concept_num(gt):
    if len(gt) == 1:
        return 'one'
    elif len(gt) == 2:
        return 'two'
    elif len(gt) == 3:
        return 'three'
    else:
        return 'four'
    

from glob import glob 

caption_load = ['./path_to_your_caption.json'] 

def save_to_txt(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        
with open(f"./QAS_GT_test.json") as f:
    qas = json.load(f)    
    
for load_cap in caption_load:
    print(load_cap)
    with open(load_cap) as f:
        caption = json.load(f)

    txt_name = os.path.basename(load_cap).replace(".json", ".txt")
    txt_path = f"./{txt_name}"
    sample_names = list(qas.keys())


    accs_gt = {'one':[], 'two':[], 'three':[], 'four':[]}
    accs_neg =  {'one':[], 'two':[], 'three':[], 'four':[]}

    for idx in tqdm(range(len(sample_names))):
        cap = caption[sample_names[idx]]['caption'] # for gemini
        # cap = caption[sample_names[idx]] # for qwen

        qa  = qas[sample_names[idx]]['qas']
        gt  = qas[sample_names[idx]]['gts']
        qa_split = return_qas(qa, gt)
        name = return_concept_num(gt)
        try:
            acc_gt  = return_accs_final_pos(cap, qa_split)
            acc_neg = return_accs_final_neg(cap, qa_split)

            accs_gt[name] += acc_gt
            accs_neg[name] += acc_neg
            
            if idx % 100 == 0:
                print(idx)
                for name in list(accs_gt.keys()):
                    accuary_1 = torch.tensor(accs_gt[name]).float().mean()
                    accuary_2 = torch.tensor(accs_neg[name]).float().mean()

                    print(f"{name} Accuracy for GT concepts: {round(accuary_1.item(), 3)}")
                    print(f"{name} Accuracy for noisy concepts: {round(accuary_2.item(), 3)}")
        except:
            pass
        
    save_to_txt(txt_path, "\n===== FINAL SUMMARY =====")
    print("\n===== FINAL SUMMARY =====")

    for name2 in list(accs_gt.keys()):
        acc_1 = torch.tensor(accs_gt[name2]).float().mean().item()
        acc_2 = torch.tensor(accs_neg[name2]).float().mean().item()

        line1 = f"{name2} Accuracy (GT concepts): {round(acc_1, 3)}"
        line2 = f"{name2} Accuracy (noisy concepts): {round(acc_2, 3)}"

        print(load_cap)
        print(line1); print(line2)

        save_to_txt(txt_path, load_cap)
        save_to_txt(txt_path, line1)
        save_to_txt(txt_path, line2)

    save_to_txt(txt_path, "===== END =====\n")