import os
import json
import logging
import re
import concurrent.futures
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import openai

# 업로드된 JSON 파일들
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_unique_filename(base_name, tag="_문제", extension=".json", directory=UPLOAD_FOLDER):
    candidate = f"{base_name}{tag}{extension}"
    counter = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base_name}{tag}({counter}){extension}"
        counter += 1
    return candidate

def extract_text_from_pdf_parallel(pdf_path_or_text: str, page_range: tuple = None) -> str:
    if not os.path.isfile(pdf_path_or_text):
        return pdf_path_or_text

    pdf_path = pdf_path_or_text
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        start_page, end_page = (0, total_pages)
        if page_range:
            start_page = max(0, page_range[0] - 1)
            end_page   = min(total_pages, page_range[1])

        if start_page < 0 or end_page > total_pages or start_page >= end_page:
            logging.warning("페이지 범위가 전체 페이지 수를 벗어났습니다. 전체 페이지 사용합니다.")
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

def _extract_text_from_page(pdf_path: str, page_index: int) -> str:
    doc = fitz.open(pdf_path)
    text = doc[page_index].get_text()
    doc.close()
    return text or ""

def generate_questions_from_pdf(pdf_path: str,
                              total_questions: int,
                              difficulty: str,
                              selected_types: list,
                              page_range: str = "",
                              requirements: str = "") -> list:
    try:
        page_range_tuple = None
        if page_range:
            try:
                start, end = map(int, page_range.split("-"))
                page_range_tuple = (start, end)
            except ValueError:
                logging.warning("페이지 범위 입력 형식 오류. 전체 페이지 사용.")

        text = extract_text_from_pdf_parallel(pdf_path, page_range_tuple)
        if not text:
            raise Exception("PDF에서 텍스트를 추출할 수 없습니다.")

        all_questions = []
        for qtype in selected_types:
            template = PROMPT_TEMPLATES[qtype]
            questions = generate_questions(template, text, total_questions, difficulty, "PDF", qtype, requirements)
            all_questions.extend(questions)

        return all_questions

    except Exception as e:
        logging.error(f"PDF 문제 생성 실패: {e}")
        raise

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

def generate_questions(prompt_template: str, source_text: str, count: int, difficulty: str, data_scope: str, qtype: str, requirements: str = "") -> list:
    if requirements:
        prompt_template += f"\n\n추가 요구사항:\n{requirements}"
    
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

def truncate_text(text: str, max_word_count: int = 1500) -> str:
    words = text.split()
    return " ".join(words[:max_word_count]) if len(words) > max_word_count else text

def generate_question_with_prompt(prompt: str) -> dict:
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    content = response.choices[0].message.content.strip()
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