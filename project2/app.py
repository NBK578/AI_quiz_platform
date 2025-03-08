import os
import json
import time
import random
import re
import concurrent.futures
import logging

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename

import openai
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- Flask-Dance를 통한 구글 OAuth 도입 ---
from flask_dance.contrib.google import make_google_blueprint, google

google_bp = make_google_blueprint(
    client_id="YOUR_GOOGLE_CLIENT_ID",
    client_secret="YOUR_GOOGLE_CLIENT_SECRET",
    scope=["profile", "email"],
    redirect_url="/google_login"
)
app = Flask(__name__)
app.secret_key = "비밀키를_여기에_입력하세요"
app.register_blueprint(google_bp, url_prefix="/login")
# ------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# OpenAI API 및 Tesseract 경로 설정
openai.api_key = #공개 설정으로 인한 비활성화

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\USER\Desktop\2024\tesseract.exe"

# 파일 업로드 관련 설정
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 사용 가능한 문제 유형
AVAILABLE_TYPES = ["객관식", "빈칸 채우기", "OX문제", "주관식", "서술형"]

# ------------------------------
# 고유 파일명 생성 함수
# ------------------------------
def get_unique_filename(base_name, tag="_문제", extension=".json", directory=UPLOAD_FOLDER):
    candidate = f"{base_name}{tag}{extension}"
    counter = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base_name}{tag}({counter}){extension}"
        counter += 1
    return candidate

# ------------------------------
# 중복 문제 제거 함수
# ------------------------------
def remove_duplicate_questions(questions):
    unique = {}
    for q in questions:
        text = q.get("question_data", {}).get("question", "").strip()
        if text and text not in unique:
            unique[text] = q
    return list(unique.values())

# ------------------------------
# 공통 API 호출 및 JSON 파싱 함수
# ------------------------------
def call_openai_chat_completion(messages, model="gpt-4", max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            return response
        except openai.error.RateLimitError:
            logging.warning(f"[RateLimitError] 재시도 {attempt+1}/{max_retries} - {delay}초 후 재시도")
        except Exception as e:
            logging.error(f"[API Error] {e} - 재시도 {attempt+1}/{max_retries}")
        time.sleep(delay)
    raise Exception("OpenAI API 호출 최대 재시도 횟수를 초과하였습니다.")

def generate_question_with_prompt(prompt: str) -> dict:
    messages = [{"role": "user", "content": prompt}]
    response = call_openai_chat_completion(messages)
    content = response["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"원본": content}

def parse_multiple_json(text: str):
    decoder = json.JSONDecoder()
    pos = 0
    results = []
    text = text.strip()
    while pos < len(text):
        try:
            obj, index = decoder.raw_decode(text, pos)
            results.append(obj)
            pos = index
            while pos < len(text) and text[pos].isspace():
                pos += 1
        except json.JSONDecodeError:
            break
    return results

# ------------------------------
# PDF 및 이미지 처리 함수
# ------------------------------
def extract_text_from_pdf_parallel(pdf_path: str, page_range: tuple = None) -> str:
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        start_page, end_page = (0, total_pages) if not page_range else (page_range[0]-1, page_range[1])
        if start_page < 0 or end_page > total_pages:
            logging.warning("페이지 범위가 전체 페이지 수를 벗어났습니다. 전체 페이지 사용.")
            start_page, end_page = 0, total_pages
        texts = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_extract_text_from_page, pdf_path, i)
                       for i in range(start_page, end_page)]
            for future in concurrent.futures.as_completed(futures):
                texts.append(future.result())
        combined_text = "\n".join(texts).strip()
        if not combined_text:
            raise Exception("PDF에서 추출된 텍스트가 없습니다.")
        return combined_text
    except Exception as e:
        logging.error(f"PDF 텍스트 추출 실패: {e}")
        raise

def _extract_text_from_page(pdf_path: str, page_index: int) -> str:
    doc = fitz.open(pdf_path)
    text = doc[page_index].get_text()
    doc.close()
    return text or ""

def analyze_image_content(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        ocr_text = pytesseract.image_to_string(image, lang="eng+kor").strip()
    except Exception as e:
        logging.error(f"이미지 OCR 실패: {e}")
        ocr_text = ""
    if ocr_text:
        prompt = (
            "아래 OCR 결과를 기반으로, 이 이미지의 주요 내용, 분위기, 사용된 기법, 그리고 작가의 의도를 자세하게 설명하는 캡션을 작성해 주세요.\n\n"
            f"OCR 결과:\n{ocr_text}"
        )
        response = generate_question_with_prompt(prompt)
        caption = response.get("원본", response.get("caption", ""))
        return caption if caption else ocr_text
    else:
        base = os.path.splitext(os.path.basename(image_path))[0]
        return f"이 이미지는 '{base}'라는 제목을 가진 이미지입니다. 추가적인 분석이 필요합니다."

def truncate_text(text: str, max_word_count: int = 1500) -> str:
    words = text.split()
    return " ".join(words[:max_word_count]) if len(words) > max_word_count else text

# ------------------------------
# 질문 생성 관련 함수 (공통 템플릿 사용)
# ------------------------------
def generate_questions(prompt_template: str, source_text: str, count: int, difficulty: str, data_scope: str, qtype: str) -> list:
    source_for_prompt = truncate_text(source_text, 500) if data_scope == "PDF" else source_text
    source_label = "PDF 내용" if data_scope == "PDF" else "이미지 캡션"
    prompt = prompt_template.format(
        count=count,
        difficulty=difficulty,
        source_label=source_label,
        data_scope=data_scope,
        source_for_prompt=source_for_prompt
    )
    result = generate_question_with_prompt(prompt)
    questions = []
    if isinstance(result, list):
        questions = result
    elif isinstance(result, dict):
        if "원본" in result:
            parsed = parse_multiple_json(result["원본"])
            questions = parsed if parsed else [result]
        else:
            questions = [result]
    for q in questions:
        q["유형"] = qtype
        q.setdefault("page", None)
    return questions

# 템플릿별 프롬프트 정의
PROMPT_TEMPLATES = {
    "객관식": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, 객관식 문제를 총 {count}개 생성해 주세요.\n"
        "난이도: {difficulty}\n"
        "각 문제의 선택지는 반드시 'A. 보기1', 'B. 보기2', 'C. 보기3', 'D. 보기4' 형식으로 표현되어야 하며, 정답은 해당 알파벳으로 표기되어야 합니다.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"객관식\", \"question_data\": {{\"question\": \"주제는?\", \"choices\": [\"A. 옵션1\", \"B. 옵션2\", \"C. 옵션3\", \"D. 옵션4\"], \"answer\": \"A\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
    "빈칸 채우기": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, 난이도 {difficulty}의 빈칸 채우기 문제를 총 {count}개 생성해 주세요.\n"
        "문제는 간단한 단어 또는 짧은 구절 위주로 작성되어야 하며, 보기는 제공하지 말고 정답은 직접 입력하는 형태로 작성해 주세요.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"빈칸 채우기\", \"question_data\": {{\"question\": \"주요 기술은 ______이다.\", \"answer\": \"옵션1\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
    "OX문제": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, OX문제를 총 {count}개 생성해 주세요.\n"
        "난이도: {difficulty}\n"
        "각 문제의 선택지는 반드시 [\"O\", \"X\"] 형식으로 제공되어야 하며, 정답은 O 또는 X로 표기되어야 합니다.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"OX문제\", \"question_data\": {{\"question\": \"내용 일관성?\", \"choices\": [\"O\", \"X\"], \"answer\": \"O\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
    "주관식": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, 주관식 문제를 총 {count}개 생성해 주세요.\n"
        "난이도: {difficulty}\n"
        "주관식 문제는 정답이 한 단어 또는 짧은 구절로 작성되어야 합니다.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"주관식\", \"question_data\": {{\"question\": \"핵심 용어는?\", \"answer\": \"용어\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
    "서술형": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, 서술형 문제를 총 {count}개 생성해 주세요.\n"
        "난이도: {difficulty}\n"
        "서술형 문제는 정답이 긴 문장이나 문단 형태로 작성되어야 합니다.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"서술형\", \"question_data\": {{\"question\": \"문서의 내용과 구조를 서술하시오.\", \"answer\": \"내용...\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    )
}

def generate_questions_from_pdf(pdf_text: str, total_questions: int, difficulty: str, selected_types: list) -> list:
    all_questions = []
    num_types = len(selected_types)
    base_count = total_questions // num_types
    remainder = total_questions % num_types
    distribution = {ptype: base_count for ptype in selected_types}
    if remainder:
        extra = random.sample(selected_types, remainder)
        for p in extra:
            distribution[p] += 1
    for ptype in selected_types:
        count = distribution[ptype]
        template = PROMPT_TEMPLATES.get(ptype)
        qs = generate_questions(template, pdf_text, count, difficulty, "PDF", ptype)
        all_questions.extend(qs)
    return remove_duplicate_questions(all_questions)[:total_questions]

def generate_questions_from_image(image_path: str, total_questions: int, difficulty: str, selected_types: list) -> list:
    caption = analyze_image_content(image_path)
    all_questions = []
    num_types = len(selected_types)
    base_count = total_questions // num_types
    remainder = total_questions % num_types
    distribution = {ptype: base_count for ptype in selected_types}
    if remainder:
        extra = random.sample(selected_types, remainder)
        for p in extra:
            distribution[p] += 1
    for ptype in selected_types:
        count = distribution[ptype]
        template = PROMPT_TEMPLATES.get(ptype)
        qs = generate_questions(template, caption, count, difficulty, image_path, ptype)
        all_questions.extend(qs)
    return remove_duplicate_questions(all_questions)[:total_questions]

# ------------------------------
# 채점 관련 보조 함수
# ------------------------------
def extract_choice_letter(ans: str) -> str:
    ans = ans.strip()
    match = re.match(r'^([A-Za-z])[\.\)]', ans)
    if match:
        return match.group(1).lower()
    return ans.lower() if ans and ans.isalpha() else ans.lower()

def normalize_objective_answer(ans: str) -> str:
    return re.sub(r'^[a-z][\.\)]\s*', '', ans.strip().lower())

def is_objective_answer_correct(user_ans: str, ref_ans: str) -> bool:
    if extract_choice_letter(user_ans) == extract_choice_letter(ref_ans):
        return True
    user_text = normalize_objective_answer(user_ans)
    ref_text = normalize_objective_answer(ref_ans)
    if user_text == ref_text:
        return True
    if len(ref_text) < 10 and (user_text in ref_text or ref_text in user_text):
        return True
    return False

def grade_problem(question: dict, user_answer: str) -> dict:
    q_type = question.get("유형", "")
    q_data = question.get("question_data", {})
    problem_text = q_data.get("question", "")
    reference_answer = q_data.get("answer", "")
    explanation = q_data.get("explanation", "")
    
    if q_type in ["객관식", "OX문제", "빈칸 채우기"]:
        if is_objective_answer_correct(user_answer, reference_answer):
            feedback = "정답입니다."
            if explanation:
                feedback += " (" + explanation + ")"
            return {"result": "정답", "feedback": feedback, "score": 100}
        else:
            feedback = f"오답입니다. 정답은 '{reference_answer}' 입니다."
            if explanation:
                feedback += " " + explanation
            else:
                feedback += " (정답에 대한 추가 설명이 제공되지 않았습니다.)"
            return {"result": "오답", "feedback": feedback, "score": 0}
    else:
        prompt = (
            "너는 튜터 역할의 채점자입니다. 아래 문제와 참고 해설을 바탕으로, "
            "사용자 답안이 참고 해설과 얼마나 유사한지 평가하여 70% 이상의 유사도면 정답, 아니면 오답으로 채점하라. "
            "자세한 피드백과 0에서 100 사이의 점수를 JSON 형식으로 출력하라.\n\n"
            f"문제: {problem_text}\n"
            f"참고 해설: {explanation}\n"
            f"사용자 답안: {user_answer}\n\n"
            "출력 예시:\n"
            "{{\"result\": \"정답\" 또는 \"오답\", \"feedback\": \"피드백 내용\", \"score\": 85}}"
        )
        result = generate_question_with_prompt(prompt)
        if isinstance(result, dict) and "result" in result:
            if "정답" not in result.get("feedback", "") and reference_answer:
                result["feedback"] += f" (정답: {reference_answer})"
            return result
        return {"result": "오답", "feedback": f"채점 결과를 판단할 수 없습니다. 정답: {reference_answer if reference_answer else 'N/A'}", "score": 0}

# ------------------------------
# 구글 로그인 후 사용자 정보 저장 라우트
# ------------------------------
@app.route("/google_login")
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("구글 사용자 정보를 가져올 수 없습니다.")
        return redirect(url_for("index"))
    user_info = resp.json()
    session["user"] = user_info
    flash("구글 로그인이 완료되었습니다.")
    return redirect(url_for("index"))

# ------------------------------
# 단일 파일 업로드 (기존) 라우트
# ------------------------------
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html", available_types=AVAILABLE_TYPES)

@app.route("/create")
def create():
    return render_template("create.html", available_types=AVAILABLE_TYPES)

@app.route("/generate", methods=["POST"])
def generate():
    if 'file' not in request.files:
        flash("파일이 선택되지 않았습니다.")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == "":
        flash("파일 이름이 없습니다.")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        flash("지원되지 않는 파일 형식입니다.")
        return redirect(request.url)
    
    try:
        total_questions = int(request.form.get("total_questions", "9"))
    except ValueError:
        total_questions = 9
    difficulty = request.form.get("difficulty", "보통")
    selected_types = request.form.getlist("question_types")
    if not selected_types:
        selected_types = AVAILABLE_TYPES.copy()
    page_range_str = request.form.get("page_range", "").strip()
    
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            page_range = None
            if page_range_str:
                try:
                    start, end = map(int, page_range_str.split("-"))
                    page_range = (start, end)
                except ValueError:
                    logging.warning("페이지 범위 입력 형식 오류. 전체 페이지 사용.")
            pdf_text = extract_text_from_pdf_parallel(filepath, page_range)
            questions = generate_questions_from_pdf(pdf_text, total_questions, difficulty, selected_types)
        else:
            questions = generate_questions_from_image(filepath, total_questions, difficulty, selected_types)
    except Exception as e:
        flash(str(e))
        return redirect(url_for("index"))
    
    session["questions"] = questions
    base_name = os.path.splitext(filename)[0]
    session["generated_file"] = get_unique_filename(base_name)
    session["answers"] = {}  # 단계별 풀이용 초기화
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], session["generated_file"])
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    
    return redirect(url_for("solve", q=0))

# ------------------------------
# 다중 파일 업로드 및 문제 생성 라우트
# ------------------------------
@app.route("/multi_create")
def multi_create():
    return render_template("multi_create.html", available_types=AVAILABLE_TYPES)

@app.route("/multi_generate", methods=["POST"])
def multi_generate():
    files = request.files.getlist("files")
    if not files or len(files) == 0:
        flash("파일이 선택되지 않았습니다.")
        return redirect(request.url)
    texts = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            ext = os.path.splitext(filepath)[1].lower()
            try:
                if ext == ".pdf":
                    text = extract_text_from_pdf_parallel(filepath)
                    texts.append(text)
                else:
                    text = analyze_image_content(filepath)
                    texts.append(text)
            except Exception as e:
                flash(f"{filename} 처리 중 오류: {e}")
                continue
        else:
            flash("지원되지 않는 파일 형식입니다.")
    if not texts:
        flash("업로드된 파일에서 추출된 텍스트가 없습니다.")
        return redirect(url_for("multi_create"))
    combined_text = "\n".join(texts)
    try:
        total_questions = int(request.form.get("total_questions", "9"))
    except ValueError:
        total_questions = 9
    difficulty = request.form.get("difficulty", "보통")
    selected_types = request.form.getlist("question_types")
    if not selected_types:
        selected_types = AVAILABLE_TYPES.copy()
    questions = generate_questions_from_pdf(combined_text, total_questions, difficulty, selected_types)
    session["questions"] = questions
    session["answers"] = {}  # 단계별 풀이용 초기화
    base_name = f"multi_{int(time.time())}"
    session["generated_file"] = get_unique_filename(base_name)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], session["generated_file"])
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    return redirect(url_for("solve", q=0))

# ------------------------------
# 저장된 문제 세트 선택 및 재문제 생성 라우트
# ------------------------------
@app.route("/recreate", methods=["GET", "POST"])
def recreate():
    if request.method == "GET":
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith("_문제.json")]
        return render_template("recreate.html", files=files, available_types=AVAILABLE_TYPES)
    else:
        selected_files = request.form.getlist("selected_files")
        if not selected_files:
            flash("선택된 파일이 없습니다.")
            return redirect(url_for("recreate"))
        combined_texts = []
        for filename in selected_files:
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for question in data:
                        q_text = question.get("question_data", {}).get("question", "")
                        if q_text:
                            combined_texts.append(q_text)
            except Exception as e:
                flash(f"{filename} 처리 중 오류: {e}")
                continue
        if not combined_texts:
            flash("선택된 파일들에서 추출할 질문이 없습니다.")
            return redirect(url_for("recreate"))
        combined_text = "\n".join(combined_texts)
        # 프롬프트에 새로운 관점 지시 추가
        combined_text = "아래 문제들을 참고하여, 동일한 내용이 중복되지 않고 새로운 관점에서 문제를 생성해 주세요.\n" + combined_text
        try:
            total_questions = int(request.form.get("total_questions", "9"))
        except ValueError:
            total_questions = 9
        difficulty = request.form.get("difficulty", "보통")
        selected_types = request.form.getlist("question_types")
        if not selected_types:
            selected_types = AVAILABLE_TYPES.copy()
        new_questions = generate_questions_from_pdf(combined_text, total_questions, difficulty, selected_types)
        session["questions"] = new_questions
        session["answers"] = {}  # 단계별 풀이용 초기화
        base_name = f"recreate_{int(time.time())}"
        session["generated_file"] = get_unique_filename(base_name)
        output_path = os.path.join(UPLOAD_FOLDER, session["generated_file"])
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_questions, f, ensure_ascii=False, indent=4)
        return redirect(url_for("solve", q=0))

# ------------------------------
# 단계별 문제 풀이 라우트 (한 문제씩 풀기)
# ------------------------------
@app.route("/solve", methods=["GET", "POST"])
def solve():
    questions = session.get("questions", [])
    if not questions:
        flash("문제가 존재하지 않습니다.")
        return redirect(url_for("index"))
    try:
        q_index = int(request.args.get("q", 0))
    except ValueError:
        q_index = 0
    total = len(questions)
    if request.method == "POST":
        answer = request.form.get("answer", "")
        answers = session.get("answers", {})
        answers[str(q_index)] = answer
        session["answers"] = answers
        q_index += 1
        if q_index >= total:
            return redirect(url_for("grade_step"))
        else:
            return redirect(url_for("solve", q=q_index))
    current_question = questions[q_index]
    progress = f"문제 {q_index+1}/{total}"
    return render_template("solve.html", question=current_question, progress=progress, q_index=q_index, total=total)

# ------------------------------
# 단계별 채점 결과 라우트
# ------------------------------
@app.route("/grade_step")
def grade_step():
    answers = session.get("answers", {})
    questions = session.get("questions", [])
    grading_results = []
    for i, q in enumerate(questions):
        user_ans = answers.get(str(i), "")
        result = grade_problem(q, user_ans)
        grading_results.append({
            "question": q.get("question_data", {}).get("question", ""),
            "your_answer": user_ans,
            "result": result.get("result", ""),
            "feedback": result.get("feedback", ""),
            "score": result.get("score", 0)
        })
    return render_template("grade.html", grading_results=grading_results)

# ------------------------------
# 결과, 다운로드, 전체 채점 라우트 (기존)
# ------------------------------
@app.route("/results")
def results():
    questions = session.get("questions", [])
    generated_file = session.get("generated_file", "문제.json")
    return render_template("results.html", questions=questions, generated_file=generated_file)

@app.route("/download")
def download_file():
    generated_file = session.get("generated_file", None)
    if generated_file:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], generated_file), as_attachment=True)
    flash("다운로드할 파일이 없습니다.")
    return redirect(url_for("results"))

@app.route("/grade", methods=["GET", "POST"])
def grade():
    questions = session.get("questions", [])
    if request.method == "POST":
        grading_results = []
        for i, q in enumerate(questions):
            answer = request.form.get(f"answer_{i}", "")
            result = grade_problem(q, answer)
            grading_results.append({
                "question": q.get("question_data", {}).get("question", ""),
                "your_answer": answer,
                "result": result.get("result", ""),
                "feedback": result.get("feedback", ""),
                "score": result.get("score", 0)
            })
        return render_template("grade.html", grading_results=grading_results)
    return render_template("grade.html", questions=questions, grading_results=None)

if __name__ == "__main__":
    app.run(debug=True)
