from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
import random
import json
from datetime import datetime
import re
from glob import glob 
from natsort import natsorted 

app = Flask(__name__)
app.secret_key = "your_secret_key"  # 세션 사용을 위한 비밀키 (그냥 두면 됨)

# 이미지 폴더 경로
IMAGE_FOLDER = "static"
RESULTS_FOLDER = "results"
NUM_SELECTION = 10 # baseline 당 비교 횟수

PROMPT_gpt = "prompt_gpt.json"
PROMPT_gemini = "prompt_gemini.json"
PROMPT_ours = "prompt_ours.json"
PROMPT_zero = "prompt_zeroshot.json"

chatgpt = "chatgpt"
gemini = "gemini"
base = "base"
OURS = "ours"

"""
metric names;
index.html에 있는 id 및 name과 맞춰야 함
"""
context_groundness = "context_groundness"
new_image_description  = "new_image_description"

"""
evaluation results;
RESULTS_FOLDER 아래에 evaluation 1회당 file 1개로 저장 f"{일}_{시간}_{이름}.json"

Result format
  {baseline_name}: {
    {metric_name}: {
      "win": 1, # {baseline_name}이 Ours에게 이긴 횟수
      "lose": 5, # {baseline_name}이 Ours에게 진 횟수
      "tie": 9
    }
}
"""



def select_baseline(session):
    while True:
        baseline = [chatgpt, base, gemini][random.randint(0, 2)]
        if session["count"][baseline] < NUM_SELECTION:
            break
    return baseline


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        name = request.form.get("name")
        if name:
            session["name"] = name  # 세션에 이름 저장
            session["count"] = {chatgpt: 0, base: 0, gemini: 0}  # 선택 횟수 초기화
            session["total_count"] = 1
            session["time"] = datetime.now().strftime("%d_%H%M%S")
            session["selected"] = []
            return redirect(url_for("index"))
    return render_template("home.html")


@app.route("/index")
def index():
    if "name" not in session:  # 이름이 없는 경우 홈으로 리다이렉트
        return redirect(url_for("home"))

    if (
        session["count"][gemini] >= NUM_SELECTION
        and session["count"][base] >= NUM_SELECTION
        and session["count"][chatgpt] >= NUM_SELECTION
    ):
        return redirect(url_for("finish"))

    baseline = select_baseline(session)

    concept1 = natsorted(glob('./static/concept1/*'))
    concept2 = natsorted(glob('./static/concept2/*'))
    query    = natsorted(glob('./static/query/*'))

    if baseline == 'chatgpt':
        Prompt = PROMPT_gpt
    elif baseline == 'gemini':
        Prompt = PROMPT_gemini
    elif baseline == 'base':
        Prompt = PROMPT_zero
        
    with open(f"./static/{Prompt}", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    with open(f"./static/prompt_ours.json", "r", encoding="utf-8") as f:
        prompts_ours = json.load(f)
    
    with open(f"./static/diag_concept1.json", "r", encoding="utf-8") as f:
        dialogues1 = json.load(f)

    with open(f"./static/diag_concept2.json", "r", encoding="utf-8") as f:
        dialogues2 = json.load(f)
    
    while True:
        tot_ranges = list(range(51))
        avoid_indexes = [5, 20, 23, 30, 50]
        tot_ranges = [x for x in tot_ranges if x not in avoid_indexes]
        index = random.choice(tot_ranges)

        if index not in session["selected"]:
            break

    session["selected"] = session["selected"] + [index]
    session["prompt"] = prompts_ours[index]
    session["baseline"] = baseline

    img1 = concept1[index].removeprefix("./static/")
    img2 = concept2[index].removeprefix("./static/")
    img3 = query[index].removeprefix("./static/")
    print(img1, img2, img3)
    desc1 = dialogues1[index]
    desc2 = dialogues2[index]
    print(index)
    print(f"Prompt: {session['prompt']}")

    idx = random.randint(0,1)
    if idx == 0:
        pairs = [prompts[index], prompts_ours[index]]
    else:
        pairs = [prompts_ours[index], prompts[index]]
    session["index"] = idx

    # return render_template("index.html", pair1=pairs[0], pair2=pairs[1])
    return render_template(
        "index.html",
        img1=img1,
        img2=img2,
        img3=img3,
        desc1=desc1,
        desc2=desc2,
        line3=pairs[0],
        line4=pairs[1],
    )

@app.route("/vote", methods=["POST"])
def vote():
    if "name" not in session:
        return redirect(url_for("home"))

    # --- result file 준비 ---
    result_file = os.path.join(RESULTS_FOLDER, f"{session['time']}_{session['name']}.json")

    if not os.path.exists(result_file):
        init = {
            "chatgpt": {
                "context_groundedness": {"win": 0, "lose": 0, "tie": 0},
                "new_image_description": {"win": 0, "lose": 0, "tie": 0},
            },
            "gemini": {
                "context_groundedness": {"win": 0, "lose": 0, "tie": 0},
                "new_image_description": {"win": 0, "lose": 0, "tie": 0},
            },
            "base": {
                "context_groundedness": {"win": 0, "lose": 0, "tie": 0},
                "new_image_description": {"win": 0, "lose": 0, "tie": 0},
            },
        }
        with open(result_file, "w") as f:
            json.dump(init, f, indent=2)

    with open(result_file, "r") as f:
        results = json.load(f)

    baseline = session["baseline"]  # 예: "chatgpt" / "gemini" / "base"
    print(baseline)

    # --- 선택 결과를 win/lose/tie로 변환하는 헬퍼 ---
    def apply_vote(metric_key: str):
        picked = request.form.get(metric_key)  # "line3###line4" or "line4###line3" or "tie"
        if picked is None:
            raise ValueError(f"Missing form field: {metric_key}")

        # tie (index.html은 "tie"로 보냄)
        if picked.lower() == "tie":
            results[baseline][metric_key]["tie"] += 1
            return

        # "line3###line4" => Prompt A 선택, "line4###line3" => Prompt B 선택
        if session["index"] == 0:
            if picked.startswith("line3###line4"):
                results[baseline][metric_key]["lose"] += 1
            elif picked.startswith("line4###line3"):
                results[baseline][metric_key]["win"] += 1
        else:
            if picked.startswith("line3###line4"):
                results[baseline][metric_key]["win"] += 1
            elif picked.startswith("line4###line3"):
                results[baseline][metric_key]["lose"] += 1

    # --- 두 metric 처리 (HTML name과 정확히 일치) ---
    apply_vote("context_groundedness")
    apply_vote("new_image_description")

    # --- rate 계산하여 JSON에 저장 ---
    for metric_key in ["context_groundedness", "new_image_description"]:
        w = results[baseline][metric_key]["win"]
        l = results[baseline][metric_key]["lose"]
        t = results[baseline][metric_key]["tie"]
        total = w + l + t

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    # 진행 카운트
    session["count"][baseline] += 1
    session["total_count"] += 1
    session["count"] = session["count"]  # session reassignment

    return redirect(url_for("index"))


@app.route("/finish")
def finish():
    if "name" not in session:
        return redirect(url_for("home"))

    name = session["name"]
    session.clear()
    return render_template("finish.html", name=name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
