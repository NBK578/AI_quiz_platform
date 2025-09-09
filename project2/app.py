import os
import json
import time
import random
import re
import concurrent.futures
import logging
from functools import wraps
from pathlib import Path

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, send_from_directory, current_app
)
from werkzeug.utils import secure_filename

# ✅ models.py에서 만든 '하나뿐인' db 객체 import (새로 만들지 않음)
from models import db, User, Quiz, Question, Choice, UserAnswer, Video

import openai
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv

# 로드 .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

# 로깅
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# OAuth / 기타 블루프린트
from flask_dance.contrib.google import make_google_blueprint, google
from youtube_routes import youtube_bp
from youtube_utils import youtube_to_pdf
from debate_routes import debate_bp

# 설정 클래스
from config import DevConfig, ProdConfig

# -----------------------------
# Flask 앱 생성 및 설정
# -----------------------------
load_dotenv()

pwd = os.getenv("DB_PASSWORD")
print("[DB CHECK]",
      "USER=", repr(os.getenv("DB_USER")),
      "HOST=", repr(os.getenv("DB_HOST")),
      "PORT=", repr(os.getenv("DB_PORT")),
      "NAME=", repr(os.getenv("DB_NAME")),
      "PWD_LEN=", 0 if pwd is None else len(pwd),
      "PWD_TAIL=", "" if not pwd else repr(pwd[-3:]))  # 마지막 3글자만 확인

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace_me")

# 환경 선택
env_name = os.getenv("FLASK_ENV", "development").lower()
Config = ProdConfig if env_name.startswith("prod") else DevConfig
app.config.from_object(Config)

# 커넥션 풀 옵션(선택)
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_size": Config.POOL_SIZE,
    "pool_timeout": Config.POOL_TIMEOUT,
    "pool_recycle": Config.POOL_RECYCLE,
    "pool_pre_ping": True,
}

# ✅ 여기서 models.py의 db 인스턴스를 '이 app'에 붙임
db.init_app(app)

# 최초 테이블 생성(개발용)
with app.app_context():
    db.create_all()

# -----------------------------
# 블루프린트 등록
# -----------------------------
google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ],
    redirect_to="google_login",
)
app.register_blueprint(google_bp, url_prefix="/login")
app.register_blueprint(youtube_bp)
app.register_blueprint(debate_bp)

# -----------------------
# Google Login Route
# -----------------------
@app.route("/google_login")
def google_login():
    """Google OAuth 콜백 후 사용자 정보 조회 → 사용자 생성/조회 → 세션 저장."""
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("/oauth2/v2/userinfo")
    if not resp or not resp.ok:
        flash("구글 사용자 정보를 가져올 수 없습니다.", "danger")
        return redirect(url_for("index"))

    info = resp.json() or {}
    google_uid = info.get("id") or info.get("sub")
    if not google_uid:
        flash("구글 사용자 ID를 확인할 수 없습니다.", "danger")
        return redirect(url_for("index"))

    user = User.query.filter_by(google_id=google_uid).first()
    if not user:
        user = User(
            google_id=google_uid,
            email=info.get("email"),
            name=info.get("name") or info.get("email"),
        )
        db.session.add(user)
        db.session.commit()

    session.permanent = True
    session["user_id"] = user.user_id
    session["user_name"] = user.name
    flash(f"{user.name}님, 로그인되었습니다.", "success")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.pop("google_oauth_token", None)
    session.pop("user_id", None)
    session.pop("user_name", None)
    flash("로그아웃되었습니다.", "info")
    return redirect(url_for("index"))


# 템플릿 어디서나 current_user 사용
@app.context_processor
def inject_current_user():
    uid = session.get("user_id")
    user = User.query.get(uid) if uid else None
    return {"current_user": user}


# 로그인 필요 데코레이터
def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("로그인이 필요합니다.", "warning")
            return redirect(url_for("google.login"))
        return view(*args, **kwargs)
    return wrapper


# 디버그
@app.route("/auth/debug")
def auth_debug():
    try:
        authed = bool(google.authorized)
        info = None
        if authed:
            r = google.get("/oauth2/v2/userinfo")
            if r and r.ok:
                info = r.json()
        return {
            "authorized": authed,
            "session_user_id": session.get("user_id"),
            "userinfo": info,
        }, 200
    except Exception as e:
        return {"authorized": False, "error": str(e)}, 500

    

# -----------------------------
# OpenAI / Tesseract
# -----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY", "")
# 환경에 맞춰 경로 조정
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", r"C:\tesseract\tesseract.exe")


# -----------------------------
# 업로드/출력 경로
# -----------------------------
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
OUTPUT_DIR = os.path.join(app.static_folder, "output")
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 사용 가능한 문제 유형(표시용 라벨)
AVAILABLE_TYPES = ["객관식", "빈칸 채우기", "OX문제", "주관식", "서술형"]

# 한글 라벨 ↔ DB 코드 매핑
TYPE_MAP = {
    "객관식": "multiple_choice",
    "빈칸 채우기": "fill_blank",
    "OX문제": "true_false",
    "주관식": "short_answer",
    "서술형": "descriptive",
}
REV_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}
ALLOWED_DB_TYPES = set(TYPE_MAP.values())

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
def get_unique_filename(base_name, tag="_문제", extension=".json", directory=UPLOAD_FOLDER):
    candidate = f"{base_name}{tag}{extension}"
    counter = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base_name}{tag}({counter}){extension}"
        counter += 1
    return candidate


def remove_duplicate_questions(questions):
    unique = {}
    for q in questions:
        text = q.get("question_data", {}).get("question", "").strip()
        if text and text not in unique:
            unique[text] = q
    return list(unique.values())


def clean_question_text(q_list):
    pattern = re.compile(r"^(?:PDF 내용을? (?:바탕으로|에 따르면),?\s*)+", re.IGNORECASE)
    for q in q_list:
        txt = q["question_data"]["question"]
        cleaned = pattern.sub("", txt)
        q["question_data"]["question"] = cleaned
    return q_list


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def map_type_to_db(label: str) -> str:
    db_type = TYPE_MAP.get(label)
    if not db_type:
        raise ValueError(f"허용되지 않은 문제 유형: {label}")
    return db_type


def db_type_to_label(db_type: str) -> str:
    return REV_TYPE_MAP.get(db_type, db_type)


def to_int_or_none(v):
    if v in (None, "", "null"):
        return None
    try:
        return int(v)
    except Exception:
        return None


def normalize_ox_choice(text: str) -> str:
    t = str(text).strip().lower()
    t = t.replace("옳", "o").replace("그", "x")
    if t in ("o", "x"):
        return t.upper()
    return text.strip()

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
def _extract_text_from_page(pdf_path: str, page_index: int) -> str:
    doc = fitz.open(pdf_path)
    text = doc[page_index].get_text()
    doc.close()
    return text or ""


def extract_text_from_pdf_parallel(pdf_path_or_text: str, page_range: tuple = None) -> str:
    """
    pdf_path_or_text가 실제 파일 경로면 PDF에서 추출,
    순수 텍스트면 그대로 반환.
    """
    if not os.path.isfile(pdf_path_or_text):
        return pdf_path_or_text  # 이미 텍스트

    pdf_path = pdf_path_or_text
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        start_page, end_page = (0, total_pages)
        if page_range:
            start_page = max(0, page_range[0] - 1)
            end_page = min(total_pages, page_range[1])
        if start_page < 0 or end_page > total_pages or start_page >= end_page:
            logging.warning("페이지 범위가 유효하지 않아 전체 사용")
            start_page, end_page = 0, total_pages

        texts = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_extract_text_from_page, pdf_path, i)
                for i in range(start_page, end_page)
            ]
            for future in concurrent.futures.as_completed(futures):
                texts.append(future.result())
        combined_text = "\n".join(texts).strip()
        if not combined_text:
            raise Exception("PDF에서 추출된 텍스트가 없습니다.")
        return combined_text
    except Exception as e:
        logging.error(f"PDF 텍스트 추출 실패: {e}")
        raise


def analyze_image_content(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        ocr_text = pytesseract.image_to_string(image, lang="eng+kor").strip()
    except Exception as e:
        logging.error(f"이미지 OCR 실패: {e}")
        ocr_text = ""
    if ocr_text:
        prompt = (
            "아래 OCR 결과를 기반으로, 이 이미지의 주요 내용, 분위기, 사용된 기법, "
            "그리고 작가의 의도를 자세하게 설명하는 캡션을 작성해 주세요.\n\n"
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
        source_for_prompt=source_for_prompt,
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


# 템플릿별 프롬프트 (객관식 예시 JSON 수정됨)
PROMPT_TEMPLATES = {
    "객관식": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, 객관식 문제를 총 {count}개 생성해 주세요.\n"
        "난이도: {difficulty}\n"
        "각 문제의 선택지는 반드시 'A. 보기1', 'B. 보기2', 'C. 보기3', 'D. 보기4', 'E. 보기5' 형식으로 표현하고, 정답은 해당 알파벳으로 표기하세요.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{"
        "\"page\": null, \"유형\": \"객관식\", "
        "\"question_data\": {{"
        "\"question\": \"주제는?\", "
        "\"choices\": [\"A. 옵션1\", \"B. 옵션2\", \"C. 옵션3\", \"D. 옵션4\", \"E. 옵션5\"], "
        "\"answer\": \"A\", "
        "\"explanation\": \"설명.\""
        "}}"
        "}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
    "빈칸 채우기": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, 난이도 {difficulty}의 빈칸 채우기 문제를 총 {count}개 생성해 주세요.\n"
        "보기는 제공하지 말고 정답은 직접 입력하는 형태로 작성하세요.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"빈칸 채우기\", "
        "\"question_data\": {{\"question\": \"주요 기술은 ______이다.\", \"answer\": \"옵션1\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
    "OX문제": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, OX문제를 총 {count}개 생성해 주세요.\n"
        "난이도: {difficulty}\n"
        "선택지는 반드시 [\"O\",\"X\"]이고 정답은 \"O\" 또는 \"X\"만 허용합니다.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"OX문제\", "
        "\"question_data\": {{\"question\": \"내용 일관성?\", \"choices\": [\"O\",\"X\"], \"answer\": \"O\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
    "주관식": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, 주관식 문제를 총 {count}개 생성해 주세요.\n"
        "난이도: {difficulty}\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"주관식\", "
        "\"question_data\": {{\"question\": \"핵심 용어는?\", \"answer\": \"용어\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
    "서술형": (
        "아래 {source_label}(자료 범위: {data_scope})를 바탕으로, 서술형 문제를 총 {count}개 생성해 주세요.\n"
        "난이도: {difficulty}\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"서술형\", "
        "\"question_data\": {{\"question\": \"문서의 내용과 구조를 서술하시오.\", \"answer\": \"내용...\", \"explanation\": \"설명.\"}}}}]\n"
        "{source_label}:\n---\n{source_for_prompt}\n---"
    ),
}


def generate_questions_from_pdf(pdf_path: str, total_questions: int, difficulty: str, selected_types: list) -> list:
    pdf_text = extract_text_from_pdf_parallel(pdf_path)

    # 선택 타입을 라운드로빈
    types_cycle = [selected_types[i % len(selected_types)] for i in range(total_questions)]

    all_questions = []
    for ptype in types_cycle:
        template = PROMPT_TEMPLATES[ptype]
        qs = generate_questions(
            template, pdf_text, count=1, difficulty=difficulty, data_scope="PDF", qtype=ptype
        )
        if qs:
            all_questions.append(qs[0])

    unique = remove_duplicate_questions(all_questions)
    return unique[:total_questions]


def generate_questions_from_image(image_path: str, total_questions: int, difficulty: str, selected_types: list) -> list:
    caption = analyze_image_content(image_path)
    all_questions = []
    num_types = len(selected_types)
    base_count = total_questions // num_types
    remainder = total_questions % num_types
    distribution = {ptype: base_count for ptype in selected_types}
    if remainder:
        for p in random.sample(selected_types, remainder):
            distribution[p] += 1
    for ptype in selected_types:
        count = distribution[ptype]
        template = PROMPT_TEMPLATES.get(ptype)
        qs = generate_questions(template, caption, count, difficulty, "IMAGE", ptype)
        all_questions.extend(qs)
    return remove_duplicate_questions(all_questions)[:total_questions]

# ------------------------------
# 채점 관련 보조 함수
# ------------------------------
def extract_choice_letter(ans: str) -> str:
    ans = ans.strip()
    match = re.match(r"^([A-Za-z])[\.\)]", ans)
    if match:
        return match.group(1).lower()
    return ans.lower() if ans and ans.isalpha() else ans.lower()


def normalize_objective_answer(ans: str) -> str:
    return re.sub(r"^[a-z][\.\)]\s*", "", ans.strip().lower())


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
    """
    question: {'유형': <한글 라벨 또는 DB코드>, 'question_data': {...}}
    """
    q_type_raw = question.get("유형", "")
    # DB코드가 들어와도 동작하도록 라벨로 보정
    q_type = db_type_to_label(q_type_raw)
    q_data = question.get("question_data", {})
    reference_answer = (q_data.get("answer") or "").strip()
    explanation = (q_data.get("explanation") or "").strip()

    # 객관식/빈칸/OX
    if q_type in ["객관식", "빈칸 채우기", "OX문제"]:
        ua = user_answer.strip().lower()
        ra = reference_answer.lower()

        if q_type == "OX문제":
            ua = ua.replace("옳", "o").replace("그", "x")
            ra = ra.replace("옳", "o").replace("그", "x")

        is_correct = (ua == ra)
        return {
            "question_data": q_data,
            "user_ans": user_answer,
            "result": "정답" if is_correct else "오답",
            "feedback": f"정답: {reference_answer}" if not is_correct else "정답입니다!",
            "score": 100 if is_correct else 0,
            "is_correct": is_correct,
        }

    # 주관식/서술형: AI 채점
    prompt = (
        "너는 튜터 역할의 채점자입니다. 아래 문제와 참고 해설을 바탕으로, "
        "사용자 답안이 참고 해설과 얼마나 유사한지 평가하여 70% 이상의 유사도면 정답, "
        "아니면 오답으로 채점하라. 자세한 피드백과 0에서 100 사이의 점수를 JSON 형식으로 출력하라.\n\n"
        f"문제: {q_data.get('question','')}\n"
        f"참고 해설: {explanation}\n"
        f"사용자 답안: {user_answer}\n\n"
        "출력 예시:\n"
        "{\"result\": \"정답\", \"feedback\": \"피드백 내용\", \"score\": 85}"
    )
    ai_result = generate_question_with_prompt(prompt)

    if isinstance(ai_result, dict) and ai_result.get("result") in ["정답", "오답"]:
        is_correct = ai_result["result"] == "정답"
        ai_score = ai_result.get("score", 0)
        return {
            "question_data": q_data,
            "user_ans": user_answer,
            "result": ai_result["result"],
            "feedback": (ai_result.get("feedback") or "").strip(),
            "score": ai_score,
            "is_correct": is_correct,
        }

    return {
        "question_data": q_data,
        "user_ans": user_answer,
        "result": "오답",
        "feedback": f"정답: {reference_answer}" if reference_answer else "오답",
        "score": 0,
        "is_correct": False,
    }

# ------------------------------
# 라우트 (화면)
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html", available_types=AVAILABLE_TYPES)


@app.route("/create")
def create():
    return render_template("create.html", available_types=AVAILABLE_TYPES)


# ------------------------------
# DB 저장용 정규화
# ------------------------------
def normalize_question_for_db(q: dict) -> dict:
    """
    입력: {'유형': '빈칸 채우기', 'page': '18', 'question_data': {...}}
    출력: {'type': 'fill_blank', 'page': 18, 'question_text': ..., 'answer': ..., 'explanation': ..., 'choices': [...]}
    """
    if not q or "question_data" not in q:
        raise ValueError("문항 데이터 형식이 올바르지 않습니다.")

    data = q["question_data"]
    q_type_db = map_type_to_db(q.get("유형"))
    page = to_int_or_none(q.get("page"))

    question_text = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    explanation = data.get("explanation") or None

    if q_type_db in ("multiple_choice", "true_false"):
        choices = data.get("choices") or []
        if q_type_db == "true_false":
            choices = [normalize_ox_choice(c) for c in choices] or ["O", "X"]
            answer = normalize_ox_choice(answer)
        if not choices:
            raise ValueError("객관식/OX는 choices가 필요합니다.")
    else:
        choices = []

    if not question_text:
        raise ValueError("문항 본문이 비었습니다.")
    if q_type_db != "descriptive" and not answer:
        raise ValueError("정답이 비었습니다.")

    return {
        "type": q_type_db,
        "page": page,
        "question_text": question_text,
        "answer": answer,
        "explanation": explanation,
        "choices": choices,
    }



# ------------------------------
# 단일 파일 업로드 → 생성 → DB 저장
# ------------------------------
@app.route("/generate", methods=["POST"])
@login_required
def generate():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("파일이 선택되지 않았습니다.", "warning")
        return redirect(request.url)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        flash("지원되지 않는 파일 형식입니다.", "danger")
        return redirect(request.url)

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Parse form inputs
    total_q = int(request.form.get("total_questions", 9))
    difficulty = request.form.get("difficulty", "보통")
    selected = request.form.getlist("question_types") or AVAILABLE_TYPES.copy()

    # Generate questions
    try:
        if ext == ".pdf":
            questions = generate_questions_from_pdf(filepath, total_q, difficulty, selected)
        else:
            questions = generate_questions_from_image(filepath, total_q, difficulty, selected)
    except Exception as e:
        flash(f"문제 생성 실패: {e}", "danger")
        return redirect(url_for("index"))

    # === 트랜잭션 저장 ===
    try:
        quiz = Quiz(user_id=session.get("user_id"), name=filename)
        db.session.add(quiz)
        db.session.flush()  # quiz_id 확보

        question_ids = []
        for q in questions:
            nq = normalize_question_for_db(q)  # ★ 한글 라벨 → DB 코드 + 검증

            q_rec = Question(
                quiz_id=quiz.quiz_id,
                type=nq["type"],
                page=nq["page"],
                question_text=nq["question_text"],
                answer=nq["answer"],
                explanation=nq["explanation"],
            )
            db.session.add(q_rec)
            db.session.flush()
            question_ids.append(q_rec.question_id)

            # 선택지 저장 (객관식/OX)
            for choice_text in nq["choices"]:
                raw = choice_text.strip()
                # "A. 보기1" / "A) 보기1" → A
                letter = raw.split(".")[0].split(")")[0].strip().upper()
                if len(letter) != 1 or not letter.isalpha():
                    letter = None  # DB 제약에 맞게 필요 시 Nullable이어야 함
                db.session.add(Choice(question_id=q_rec.question_id, letter=letter, text=raw))

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("저장 중 오류")
        flash(f"저장 중 오류: {e}", "danger")
        return redirect(url_for("index"))

    # 풀이 플로우를 DB 기반으로 진행
    session["quiz_id"] = quiz.quiz_id
    session["question_ids"] = question_ids
    # 결과 JSON도 세션에 저장(선택)
    base = os.path.splitext(filename)[0]
    gen_file = get_unique_filename(base)
    session["generated_file"] = gen_file
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], gen_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

    return redirect(url_for("solve", q=0))


# ------------------------------
# YouTube → PDF → 생성 (세션 기반)
# ------------------------------
@app.route("/youtube_generate", methods=["POST"])
def youtube_generate():
    youtube_url = request.form.get("youtube_url", "").strip()
    if not youtube_url:
        flash("유튜브 링크를 입력해주세요.", "warning")
        return redirect(url_for("create"))

    try:
        pdf_path, pdf_filename = youtube_to_pdf(youtube_url)
        questions = generate_questions_from_pdf(
            pdf_path,
            int(request.form.get("total_questions", 9)),
            request.form.get("difficulty", "보통"),
            request.form.getlist("question_types") or AVAILABLE_TYPES.copy(),
        )
        session["questions"] = questions
        base = os.path.splitext(pdf_filename)[0]
        gen_file = get_unique_filename(base)
        session["generated_file"] = gen_file
        session["answers"] = {}
        out_path = os.path.join(app.config["UPLOAD_FOLDER"], gen_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
        return redirect(url_for("solve", q=0))
    except Exception as e:
        flash(f"YouTube 기반 질문 생성 실패: {e}", "danger")
        return redirect(url_for("create"))
    
# ------------------------------
# 다중 파일 업로드 (세션 기반)
# ------------------------------
@app.route("/multi_create")
def multi_create():
    return render_template("multi_create.html", available_types=AVAILABLE_TYPES)


@app.route("/multi_generate", methods=["POST"])
def multi_generate():
    files = request.files.getlist("files")
    if not files:
        flash("파일이 선택되지 않았습니다.", "warning")
        return redirect(request.url)

    texts = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            ext = os.path.splitext(filepath)[1].lower()
            try:
                if ext == ".pdf":
                    text = extract_text_from_pdf_parallel(filepath)
                else:
                    text = analyze_image_content(filepath)
                texts.append(text)
            except Exception as e:
                flash(f"{filename} 처리 중 오류: {e}", "danger")
        else:
            flash("지원되지 않는 파일 형식입니다.", "warning")

    if not texts:
        flash("업로드된 파일에서 추출된 텍스트가 없습니다.", "warning")
        return redirect(url_for("multi_create"))

    combined_text = "\n".join(texts)
    try:
        total_questions = int(request.form.get("total_questions", "9"))
    except ValueError:
        total_questions = 9
    difficulty = request.form.get("difficulty", "보통")
    selected_types = request.form.getlist("question_types") or AVAILABLE_TYPES.copy()

    questions = generate_questions_from_pdf(combined_text, total_questions, difficulty, selected_types)
    session["questions"] = questions
    session["answers"] = {}
    base_name = f"multi_{int(time.time())}"
    session["generated_file"] = get_unique_filename(base_name)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], session["generated_file"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    return redirect(url_for("solve", q=0))


# ------------------------------
# 재문제 생성 (세션 기반)
# ------------------------------
@app.route("/recreate", methods=["GET", "POST"])
def recreate():
    if request.method == "GET":
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith("_문제.json")]
        return render_template("recreate.html", files=files, available_types=AVAILABLE_TYPES)
    else:
        selected_files = request.form.getlist("selected_files")
        if not selected_files:
            flash("선택된 파일이 없습니다.", "warning")
            return redirect(url_for("recreate"))

        combined_texts = []
        for filename in selected_files:
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for question in data:
                        q_text = question.get("question_data", {}).get("question", "")
                        if q_text:
                            combined_texts.append(q_text)
            except Exception as e:
                flash(f"{filename} 처리 중 오류: {e}", "danger")

        if not combined_texts:
            flash("선택된 파일들에서 추출할 질문이 없습니다.", "warning")
            return redirect(url_for("recreate"))

        combined_text = "다음 문제들을 참고하여, 동일한 내용이 중복되지 않고 새로운 관점에서 문제를 생성해 주세요.\n" + "\n".join(combined_texts)

        try:
            total_questions = int(request.form.get("total_questions", "9"))
        except ValueError:
            total_questions = 9
        difficulty = request.form.get("difficulty", "보통")
        selected_types = request.form.getlist("question_types") or AVAILABLE_TYPES.copy()

        source_text = extract_text_from_pdf_parallel(combined_text)
        new_questions = generate_questions_from_pdf(source_text, total_questions, difficulty, selected_types)
        new_questions = clean_question_text(new_questions)
        session["questions"] = new_questions
        session["answers"] = {}

        base_name = f"recreate_{int(time.time())}"
        session["generated_file"] = get_unique_filename(base_name)
        output_path = os.path.join(UPLOAD_FOLDER, session["generated_file"])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_questions, f, ensure_ascii=False, indent=4)

        return redirect(url_for("solve", q=0))


# ------------------------------
# 단계별 문제 풀이 (세션 기반)
# ------------------------------
@app.route("/solve", methods=["GET", "POST"])
def solve():
    session.pop("grading_results", None)
    # 세션 기반 풀이(YouTube/다중 업로드/재생성 플로우)
    questions = session.get("questions", [])
    if not questions and session.get("question_ids"):
        # DB 저장 플로우에서 바로 solve로 온 경우: DB에서 불러서 표시용 구조로 변환
        q_ids = session.get("question_ids", [])
        loaded = []
        for qid in q_ids:
            q = Question.query.get(qid)
            if not q:
                continue
            loaded.append({
                "유형": db_type_to_label(q.type),   # DB코드 → 한글 라벨
                "page": q.page,
                "question_data": {
                    "question": q.question_text,
                    "answer": q.answer,
                    "explanation": q.explanation,
                    "choices": [c.text for c in q.choices],
                }
            })
        questions = loaded
        session["questions"] = questions

    if not questions:
        flash("문제가 존재하지 않습니다.", "info")
        return redirect(url_for("index"))

    total = len(questions)
    try:
        q_index = int(request.args.get("q", 0))
    except ValueError:
        q_index = 0

    if request.method == "POST":
        answer = request.form.get("answer", "").strip()
        answers = session.get("answers", {})
        answers[str(q_index)] = answer
        session["answers"] = answers

        q_index += 1
        if q_index >= total:
            return redirect(url_for("grade"))
        else:
            return redirect(url_for("solve", q=q_index))

    current_question = questions[q_index]
    progress = f"문제 {q_index+1}/{total}"
    return render_template(
        "solve.html",
        question=current_question,
        progress=progress,
        q_index=q_index,
        total=total,
        hide_sidebar=True,
    )


# ------------------------------
# 전체 채점 (DB 기반 저장 플로우)
# ------------------------------
@app.route("/grade", methods=["GET"])
def grade():
    q_ids = session.get("question_ids", [])
    answers = session.get("answers", {})
    results = []

    for idx, qid in enumerate(q_ids):
        q = Question.query.get(qid)
        if not q:
            continue
        user_ans = answers.get(str(idx), "")
        question_data = {
            "유형": db_type_to_label(q.type),  # DB코드 → 한글 라벨
            "question_data": {
                "question": q.question_text,
                "answer": q.answer,
                "explanation": q.explanation,
                "choices": [c.text for c in q.choices],
            },
        }
        res = grade_problem(question_data, user_ans)
        results.append(res)

        ua = UserAnswer(
            user_id=session.get("user_id"),
            question_id=qid,
            user_answer=user_ans,
            is_correct=res["is_correct"],
            feedback=res["feedback"],
            score=res["score"],
        )
        db.session.add(ua)
    db.session.commit()

    return render_template("grade.html", grading_results=results, total=len(q_ids), hide_sidebar=True)


# ------------------------------
# 쇼츠 생성 / 관리
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
        return send_file(os.path.join(app.config["UPLOAD_FOLDER"], generated_file), as_attachment=True)
    flash("다운로드할 파일이 없습니다.", "warning")
    return redirect(url_for("results"))


@app.route("/make_short", methods=["POST"])
def make_short():
    quiz_id = session.get("quiz_id")
    json_fname = f"uploads/{session.get('generated_file')}"
    try:
        video_paths = create_shorts_from_json(json_fname)
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("results"))

    for path in video_paths:
        vid = Video(quiz_id=quiz_id, file_path=os.path.basename(path))
        db.session.add(vid)
    db.session.commit()

    session["shorts"] = [os.path.basename(p) for p in video_paths]
    flash(f"{len(video_paths)}개의 쇼츠 영상이 생성되었습니다.", "success")
    return redirect(url_for("results"))


@app.route("/shorts", methods=["GET", "POST"])
def shorts():
    json_files = sorted([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith("_문제.json")])

    if request.method == "POST":
        selected = request.form.getlist("selected_files")
        if not selected:
            flash("선택된 JSON 파일이 없습니다.", "warning")
            return redirect(url_for("shorts"))

        video_paths = []
        for fname in selected:
            json_path = os.path.join(UPLOAD_FOLDER, fname)
            try:
                paths = create_shorts_from_json(json_path)
                video_paths.extend(paths)
            except Exception as e:
                flash(f"{fname} 처리 중 오류: {e}", "danger")

        shorts_list = sorted([os.path.basename(p) for p in video_paths if os.path.isfile(p)])
        session["shorts"] = shorts_list
        flash(f"{len(shorts_list)}개의 쇼츠가 생성되었습니다.", "success")
        return redirect(url_for("shorts"))

    existing = session.get("shorts")
    if existing is None:
        existing = sorted([
            f for f in os.listdir(os.path.join(app.static_folder, "output"))
            if f.lower().endswith(".mp4")
        ])

    return render_template("shorts.html", files=json_files, shorts=existing)


@app.route("/output")
def output_list():
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))])
    return render_template("output_list.html", files=files)


@app.route("/json_list")
def json_list():
    files = sorted(
        f for f in os.listdir(UPLOAD_FOLDER)
        if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)) and f.endswith(".json")
    )
    return render_template("json_list.html", files=files)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)