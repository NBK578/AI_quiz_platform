import os
import json
import time
import random
import re
import concurrent.futures
import logging
import re
import os
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename
from models import db, User, Quiz, Question, Choice, UserAnswer, Video

import openai
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- Flask-Dance를 통한 구글 OAuth 도입 ---
from flask_dance.contrib.google import make_google_blueprint, google
from youtube_routes import youtube_bp
from youtube_utils import youtube_to_pdf            # ← 이 줄 추가

from debate_routes import debate_bp 


google_bp = make_google_blueprint(
    client_id="Your_Client_ID",
    client_secret="Your_Client_Secret",
       scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ],
    redirect_to="google_login"
)
app = Flask(__name__)
app.secret_key = "비밀키를_여기에_입력하세요"

app.register_blueprint(youtube_bp)
app.register_blueprint(debate_bp)         # ← Debate 탭 등록

app.register_blueprint(google_bp, url_prefix="/login")
# ------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# Database Configuration
# -----------------------
# PostgreSQL connection URI (update with actual credentials)
app.config['SQLALCHEMY_DATABASE_URI']        = "your_information"
# Disable track modifications for performance
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Initialize SQLAlchemy with app
db.init_app(app)
# Create tables if they don't exist (only for development)
with app.app_context():
    db.create_all()


# OpenAI API 및 Tesseract 경로 설정
openai.api_key = "Your_secret_number"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\USER\Desktop\2024\tesseract.exe"



# 업로드된 JSON 파일들
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
# 쇼츠 동영상들이 저장된 디렉터리
OUTPUT_DIR   = os.path.join(app.static_folder, 'output')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)




ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 사용 가능한 문제 유형
AVAILABLE_TYPES = ["객관식", "빈칸 채우기", "OX문제", "주관식", "서술형"]

from generate_quiz_video import create_shorts_from_json

# -----------------------
# Google Login Route
# -----------------------
@app.route("/google_login")
def google_login():
    """
    Google OAuth 콜백 후 사용자 정보 조회 → 사용자 생성/조회 → 세션 저장.
    """
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("/oauth2/v2/userinfo")
    if not resp or not resp.ok:
        flash("구글 사용자 정보를 가져올 수 없습니다.")
        return redirect(url_for("index"))

    info = resp.json() or {}
    google_uid = info.get("id") or info.get("sub")
    if not google_uid:
        flash("구글 사용자 ID를 확인할 수 없습니다.")
        return redirect(url_for("index"))

    # Lookup existing user by Google ID
    user = User.query.filter_by(google_id=google_uid).first()
    # If not found, create a new User record
    if not user:
        user = User(
            google_id=google_uid,
            email=info.get('email'),
            name=info.get('name')
        )
        db.session.add(user)
        db.session.commit()

    # Store user ID in session for later use
    session['user_id'] = user.user_id
    return redirect(url_for("index"))

# 템플릿 어디서나 current_user 사용 가능하도록 주입
@app.context_processor
def inject_current_user():
    uid = session.get("user_id")
    user = User.query.get(uid) if uid else None
    return {"current_user": user}

# 로그아웃: 토큰/세션 정리
@app.route("/logout")
def logout():
    session.pop("google_oauth_token", None)
    session.pop("user_id", None)
    return redirect(url_for("index"))

# 로그인 필요 데코레이터

def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("로그인이 필요합니다.")
            return redirect(url_for("google.login"))
        return view(*args, **kwargs)
    return wrapper

# 로그인 상태 확인용 디버그 엔드포인트
@app.route("/auth/debug")
def auth_debug():
    try:
        authed = bool(google.authorized)
        info = None
        if authed:
            r = google.get("/oauth2/v2/userinfo")
            if r and r.ok:
                info = r.json()
        return {"authorized": authed, "session_user_id": session.get("user_id"), "userinfo": info}, 200
    except Exception as e:
        return {"authorized": False, "error": str(e)}, 500



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

def clean_question_text(q_list):
    """
    q_list: [{'question_data': {'question': 'PDF 내용에 따르면…'}, …}, …]
    """
    pattern = re.compile(r'^(?:PDF 내용을? (?:바탕으로|에 따르면),?\s*)+', re.IGNORECASE)
    for q in q_list:
        txt = q['question_data']['question']
        # 접두사 매칭되는 부분 전부 삭제
        cleaned = pattern.sub('', txt)
        q['question_data']['question'] = cleaned
    return q_list

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
def extract_text_from_pdf_parallel(pdf_path_or_text: str, page_range: tuple = None) -> str:
    """
    PDF 파일 경로이거나 이미 추출된 텍스트를 받아 처리합니다.
    - 인자가 실제 파일 경로인 경우: fitz로 PDF를 열고 병렬로 페이지 텍스트를 추출
    - 그 외(이미 텍스트가 전달된 경우): 그대로 반환
    """
    # 1) 순수 텍스트가 넘어온 경우 그대로 반환sssssss
    if not os.path.isfile(pdf_path_or_text):
        return pdf_path_or_text

    # 2) 실제 파일 경로인 경우 기존 병렬 추출 로직 실행
    pdf_path = pdf_path_or_text
    try:
        # 전체 페이지 수 확인
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        # 페이지 범위 계산 (1-indexed 입력 → 0-indexed 내부 처리)
        start_page, end_page = (0, total_pages)
        if page_range:
            start_page = max(0, page_range[0] - 1)
            end_page   = min(total_pages, page_range[1])

        if start_page < 0 or end_page > total_pages or start_page >= end_page:
            logging.warning("페이지 범위가 전체 페이지 수를 벗어났습니다. 전체 페이지 사용합니다.")
            start_page, end_page = 0, total_pages

        # 병렬로 페이지별 추출
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
        "각 문제의 선택지는 반드시 'A. 보기1', 'B. 보기2', 'C. 보기3', 'D. 보기4' 'E. 보기5'형식으로 표현되어야 하며, 정답은 해당 알파벳으로 표기되어야 합니다.\n"
        "출력은 반드시 JSON 배열 형식이어야 합니다.\n"
        "형식 예시:\n"
        "[{{\"page\": null, \"유형\": \"객관식\", \"question_data\": {{\"question\": \"주제는?\", \"choices\": [\"A. 옵션1\", \"B. 옵션2\", \"C. 옵션3\", \"D. 옵션4\"], \"E. 옵션5\"],\"answer\": \"A\", \"explanation\": \"설명.\"}}}}]\n"
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

def generate_questions_from_pdf(pdf_path: str,
                                total_questions: int,
                                difficulty: str,
                                selected_types: list) -> list:
    """
    PDF 파일 경로(pdf_path)에서 페이지별 텍스트를 뽑은 뒤,
    선택된 문제 유형(selected_types)을 순환하며 total_questions만큼
    한 문제씩 생성합니다.
    """
    # 1) PDF 텍스트 통합 (페이지 정보는 q["page"]를 따로 채워도 되고)
    pdf_text = extract_text_from_pdf_parallel(pdf_path)

    # 2) 순환할 타입 리스트 만들기
    types_cycle = []
    n_types = len(selected_types)
    for i in range(total_questions):
        types_cycle.append(selected_types[i % n_types])

    all_questions = []
    for ptype in types_cycle:
        template = PROMPT_TEMPLATES[ptype]
        # 한 번에 한 문제씩만 생성
        qs = generate_questions(template,
                                pdf_text,
                                count=1,
                                difficulty=difficulty,
                                data_scope="PDF",
                                qtype=ptype)
        if qs:
            q = qs[0]
            all_questions.append(q)

    # 3) 중복 제거 (혹시 같은 질문이 들어왔을 때)
    unique = remove_duplicate_questions(all_questions)
    # 4) 총합이 모자라면(rare), 빈 유형으로 채워 넣거나 그대로 리턴
    return unique[:total_questions]




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
    """
    question: 원본 문제 dict, '유형'과 'question_data' 키가 포함됨
    user_answer: 사용자가 제출한 답안
    반환값: {
      "question_data": {...},
      "user_ans": 사용자 답안,
      "result": "정답" 또는 "오답",
      "feedback": 피드백 문자열,
      "score": 정수 점수 (정답이면 AI 점수, 오답이어도 AI 점수),
      "is_correct": True or False
    }
    """
    q_type = question.get("유형", "")
    q_data = question.get("question_data", {})
    reference_answer = q_data.get("answer", "").strip()
    explanation = (q_data.get("explanation") or "").strip()

    # 객관식, OX, 빈칸 채우기
    if q_type in ["객관식", "빈칸 채우기", "OX문제"]:
        ua = user_answer.strip().lower()
        ra = reference_answer.lower()

        # OX 문제: 한글 표현 허용
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
            "is_correct": is_correct
        }

    # 주관식, 서술형: AI 채점 결과 점수 그대로 반영
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
            "feedback": ai_result.get("feedback", "").strip(),
            "score": ai_score,  # ✅ 항상 AI 점수 그대로 사용
            "is_correct": is_correct
        }

    # AI 실패 시
    return {
        "question_data": q_data,
        "user_ans": user_answer,
        "result": "오답",
        "feedback": f"정답: {reference_answer}" if reference_answer else "오답",
        "score": 0,
        "is_correct": False
    }

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
    """
    Handles file upload (PDF/image), generates questions via OpenAI, and persists Quiz + Questions + Choices to DB.
    """
    file = request.files.get('file')
    if not file or file.filename == "":
        flash("파일이 선택되지 않았습니다.")
        return redirect(request.url)

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        flash("지원되지 않는 파일 형식입니다.")
        return redirect(request.url)

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Parse form inputs
    total_q   = int(request.form.get('total_questions', 9))
    difficulty = request.form.get('difficulty', '보통')
    selected   = request.form.getlist('question_types') or available_types.copy()

    # Generate questions from PDF or image
    try:
        if ext == '.pdf':
            questions = generate_questions_from_pdf(filepath, total_q, difficulty, selected)
        else:
            questions = generate_questions_from_image(filepath, total_q, difficulty, selected)
    except Exception as e:
        flash(str(e))
        return redirect(url_for('index'))

    # Persist Quiz record
    quiz = Quiz(user_id=session.get('user_id'), name=filename)
    db.session.add(quiz)
    db.session.flush()  # Assign quiz.quiz_id

    # Persist Questions and Choices
    question_ids = []
    for q in questions:
        data = q['question_data']
        q_rec = Question(
            quiz_id=quiz.quiz_id,
            type=q['유형'],
            page=q.get('page'),
            question_text=data['question'],
            answer=data['answer'],
            explanation=data.get('explanation')
        )
        db.session.add(q_rec)
        db.session.flush()  # Assign question_id
        question_ids.append(q_rec.question_id)
        # Save each choice for 객관식/OX
        for choice_text in data.get('choices', []):
            letter = choice_text.split('.')[0].strip()
            choice = Choice(
                question_id=q_rec.question_id,
                letter=letter,
                text=choice_text
            )
            db.session.add(choice)

    # Commit all DB changes
    db.session.commit()

    # Store quiz and question IDs in session for solving flow
    session['quiz_id'] = quiz.quiz_id
    session['question_ids'] = question_ids

    return redirect(url_for('solve', q=0))


@app.route("/youtube_generate", methods=["POST"])
def youtube_generate():
    youtube_url = request.form.get("youtube_url", "").strip()
    if not youtube_url:
        flash("유튜브 링크를 입력해주세요.")
        return redirect(url_for("create"))

    try:
        # 1) PDF 생성
        pdf_path, pdf_filename = youtube_to_pdf(youtube_url)
        # 2) 텍스트 추출 & 질문 생성
        questions = generate_questions_from_pdf(
            pdf_path,
            int(request.form.get("total_questions", 9)),
            request.form.get("difficulty", "보통"),
            request.form.getlist("question_types") or AVAILABLE_TYPES.copy()
        )
        # 3) 세션·파일 저장 (기존 generate 로직과 동일)
        session["questions"]      = questions
        base = os.path.splitext(pdf_filename)[0]
        gen_file = get_unique_filename(base)
        session["generated_file"] = gen_file
        session["answers"]        = {}
        out_path = os.path.join(app.config["UPLOAD_FOLDER"], gen_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        return redirect(url_for("solve", q=0))
    except Exception as e:
        flash(f"YouTube 기반 질문 생성 실패: {e}")
        return redirect(url_for("create"))

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
        # 저장된 JSON 파일 목록 표시
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith("_문제.json")]
        return render_template(
            "recreate.html",
            files=files,
            available_types=AVAILABLE_TYPES
        )
    else:
        # 클라이언트가 선택한 파일들 로드
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

        # 텍스트 결합 및 프롬프트 추가
        combined_text = "\n".join(combined_texts)
        combined_text = (
            "다음 문제들을 참고하여, 동일한 내용이 중복되지 않고 새로운 관점에서 문제를 생성해 주세요.\n"
            + combined_text
        )

        # 사용자 옵션 파싱
        try:
            total_questions = int(request.form.get("total_questions", "9"))
        except ValueError:
            total_questions = 9
        difficulty = request.form.get("difficulty", "보통")
        selected_types = request.form.getlist("question_types")
        if not selected_types:
            selected_types = AVAILABLE_TYPES.copy()

        # PDF vs. 문자열 자동 분기 처리
        source_text = extract_text_from_pdf_parallel(combined_text)

        # 질문 생성
        new_questions = generate_questions_from_pdf(
            source_text,
            total_questions,
            difficulty,
            selected_types
        )
        new_questions = clean_question_text(new_questions)
        # 생성된 질문/답안 세션에 저장
        session["questions"] = new_questions
        session["answers"]   = {}
        
        # 결과 JSON 파일 저장
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
    # 새롭게 들어올 때마다 이전 채점 결과 삭제
    session.pop("grading_results", None)
    questions = session.get("questions", [])
    if not questions:
        flash("문제가 존재하지 않습니다.")
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
        # 마지막 문제까지 다 풀었으면 grade로 이동
        if q_index >= total:
            return redirect(url_for("grade"))
        else:
            return redirect(url_for("solve", q=q_index))

    # GET: 현재 문제 보여주기
    current_question = questions[q_index]
    progress = f"문제 {q_index+1}/{total}"
    return render_template(
        "solve.html",
        question=current_question,
        progress=progress,
        q_index=q_index,
        total=total,
        hide_sidebar=True
    )

# ------------------------------
# 단계별 채점 결과 라우트
# ------------------------------
@app.route('/grade_step/<int:q_idx>', methods=['GET', 'POST'])
def grade_step(q_idx):
    """
    한 문제씩 풀이하고 채점한 뒤, 다음 문제로 이동시키는 흐름.
    q_idx: 0부터 시작하는 문항 인덱스
    """
    questions = session.get('questions', [])
    if q_idx < 0 or q_idx >= len(questions):
        return redirect(url_for('grade'))

    # POST: 사용자가 답안을 제출했을 때
    if request.method == 'POST':
        user_ans = request.form.get('answer', '').strip()
        # 채점
        result = grade_problem(questions[q_idx], user_ans)

        # 세션에 저장
        grading = session.get('grading_results', [])
        grading.append(result)
        session['grading_results'] = grading

        answers = session.get('answers', {})
        answers[q_idx] = user_ans
        session['answers'] = answers

        # 다음 문제로
        return redirect(url_for('grade_step', q_idx=q_idx+1))

    # GET: 현재 문제 보여주기
    q = questions[q_idx]['question_data']
    return render_template(
        'grade_step.html',
        question=q,
        page=q.get('page'),
        idx=q_idx,
        total=len(questions),
        prev_idx=q_idx-1 if q_idx>0 else None
    )

#-----------------------------
# 결과, 다운로드, 전체 채점 라우트
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

@app.route('/grade', methods=['GET'])
def grade():
    """
    Grades all answered questions, persists UserAnswer records, and shows results.
    """
    q_ids   = session.get('question_ids', [])
    answers = session.get('answers', {})
    results = []
    # Loop through each question and compute result
    for idx, qid in enumerate(q_ids):
        q = Question.query.get(qid)
        user_ans = answers.get(str(idx), '')
        # Build a dict for grade_problem function
        question_data = {
            '유형': q.type,
            'question_data': {
                'question': q.question_text,
                'answer': q.answer,
                'explanation': q.explanation,
                'choices': [c.text for c in q.choices]
            }
        }
        res = grade_problem(question_data, user_ans)
        results.append(res)
        # Save grading result to DB
        ua = UserAnswer(
            user_id=session.get('user_id'),
            question_id=qid,
            user_answer=user_ans,
            is_correct=res['is_correct'],
            feedback=res['feedback'],
            score=res['score']
        )
        db.session.add(ua)
    db.session.commit()

    return render_template(
        'grade.html',
        grading_results=results,
        total=len(q_ids),
        hide_sidebar=True
    )
# ---------------




@app.route('/make_short', methods=['POST'])
def make_short():
    """
    Generates shorts videos from quiz JSON and saves metadata to Video table.
    """
    quiz_id   = session.get('quiz_id')
    json_fname = f"uploads/{session.get('generated_file')}"
    try:
        video_paths = create_shorts_from_json(json_fname)
    except Exception as e:
        flash(str(e))
        return redirect(url_for('results'))

    # Persist each video record
    for path in video_paths:
        vid = Video(
            quiz_id=quiz_id,
            file_path=os.path.basename(path)
        )
        db.session.add(vid)
    db.session.commit()

    session['shorts'] = [os.path.basename(p) for p in video_paths]
    flash(f"{len(video_paths)}개의 쇼츠 영상이 생성되었습니다.")
    return redirect(url_for('results'))


@app.route('/shorts', methods=['GET', 'POST'])
def shorts():
    # 업로드된 문제 JSON 파일 목록
    json_files = sorted([
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.endswith('_문제.json')
    ])

    if request.method == 'POST':
        selected = request.form.getlist('selected_files')
        if not selected:
            flash('선택된 JSON 파일이 없습니다.')
            return redirect(url_for('shorts'))

        video_paths = []
        for fname in selected:
            json_path = os.path.join(UPLOAD_FOLDER, fname)
            try:
                # JSON 별로 쇼츠 생성
                paths = create_shorts_from_json(json_path)
                video_paths.extend(paths)
            except Exception as e:
                flash(f'{fname} 처리 중 오류: {e}')

        # OUTPUT_DIR(static/output) 안의 실제 파일명만 읽기
        shorts_list = sorted([
            os.path.basename(p)
            for p in video_paths
            if os.path.isfile(p)
        ])
        session['shorts'] = shorts_list
        flash(f'{len(shorts_list)}개의 쇼츠가 생성되었습니다.')
        return redirect(url_for('shorts'))

    # GET: 세션에 남은 쇼츠 모음 혹은 static/output 전체
    existing = session.get('shorts')
    if existing is None:
        # 세션이 비어 있으면 디스크상의 전체 목록
        existing = sorted([
            f for f in os.listdir(os.path.join(app.static_folder, 'output'))
            if f.lower().endswith('.mp4')
        ])

    return render_template(
        'shorts.html',
        files=json_files,
        shorts=existing
    )

@app.route('/output')
def output_list():
    """
    static/output 폴더 안의 모든 파일을 읽어와
    템플릿에 넘깁니다.
    """


    # mp4 등 미디어 파일만 걸러내고 싶으면 확장자 필터 추가 가능
    files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    ])
    return render_template('output_list.html', files=files)

@app.route('/json_list')
def json_list():
    """
    uploads 폴더 안의 모든 .json 파일 목록을 읽어서
    json_list.html 템플릿에 넘겨줍니다.
    """
    files = sorted(
        f for f in os.listdir(UPLOAD_FOLDER)
        if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)) and f.endswith('.json')
    )
    return render_template('json_list.html', files=files)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """
    /uploads/<filename> 으로 호출하면
    uploads/ 디렉터리에서 해당 파일을 inline 서빙합니다.
    JSON, MP4, 뭐든 다 지원.
    """
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)    



if __name__ == "__main__":
    app.run(host="0.0.0.0" , port = 5000, debug=True)
