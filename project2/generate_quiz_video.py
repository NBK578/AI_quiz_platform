import os
import json
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, VideoClip




# 경로 설정
FONT_PATH = os.path.join(os.getcwd(), "BMJUA_TTF.TTF")
BG_PATH = os.path.join(os.getcwd(), "background1.jpg")
BGM_PATH = os.path.join(os.getcwd(), "bgm.mp3")
IMAGE_DIR = "./images"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# generate_quiz_video.py 에서
OUTPUT_DIR = os.path.join(MODULE_DIR, 'static', 'output')


os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def wrap_text(text, font, max_width):
    """주어진 텍스트를 최대 너비에 맞춰 줄바꿈을 합니다."""
    lines = []
    words = text.split()
    current_line = ""
    # PIL 객체 생성
    bg_img = Image.open(BG_PATH).convert("RGB")
    draw = ImageDraw.Draw(bg_img)

    for word in words:
        # 현재 줄에 해당 단어를 추가했을 때 최대 너비를 초과하는지 여부 체크
        test_line = f"{current_line} {word}".strip()
        if draw.textbbox((0, 0), test_line, font=font)[2] <= max_width:
            current_line = test_line
        else:
            if current_line:  # 현재 줄이 있을 경우 추가
                lines.append(current_line)
            current_line = word  # 새 줄 시작

    if current_line:  # 마지막 줄 추가
        lines.append(current_line)

    return lines


def create_shorts_from_json(json_path: str, max_items: int = None):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    quiz_data = normalize_quiz_data(raw_data)
    total = min(len(quiz_data), max_items) if max_items else len(quiz_data)

    video_paths = []
    for i, item in enumerate(quiz_data[:total]):
        path = process_quiz_item(item, i)
        video_paths.append(path)

    return video_paths


def create_image(text_lines, filename, size=(1080, 1920), fontsize=60):
    bg_img = Image.open(BG_PATH).convert("RGB").resize(size)
    draw = ImageDraw.Draw(bg_img)
    font = ImageFont.truetype(FONT_PATH, fontsize)

    # 줄바꿈을 처리한 텍스트 리스트
    wrapped_lines = []
    for line in text_lines:
        wrapped_lines.extend(wrap_text(line, font, size[0] - 40))  # 여백을 고려한 최대 너비

    total_height = sum([draw.textbbox((0, 0), line, font=font)[3] + 50 for line in wrapped_lines])
    y = (size[1] - total_height) // 2

    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2]
        draw.text(((size[0] - text_width) / 2, y), line, fill="black", font=font)
        y += bbox[3] + 50

    filepath = os.path.join(IMAGE_DIR, filename)
    bg_img.save(filepath)
    return filepath

def create_typing_clip(text, duration, size=(1080, 1920), fontsize=60):
    font = ImageFont.truetype(FONT_PATH, fontsize)
    bg_img = Image.open(BG_PATH).convert("RGB").resize(size)
    base = np.array(bg_img)
    draw = ImageDraw.Draw(bg_img)

    wrapped_text = wrap_text(text, font, size[0] - 40)  # 여백을 고려한 최대 너비
    full_text = "\n".join(wrapped_text)  # 줄바꿈 처리된 텍스트

    def make_frame(t):
        idx = int(len(full_text) * (t / duration))  # 전체 텍스트의 길이에 비례하여 보여줄 텍스트 결정
        display_text = full_text[:idx]  # 보여줄 텍스트
        img = Image.fromarray(base.copy())
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), display_text, font=font)
        x = (size[0] - bbox[2]) // 2
        y = (size[1] - bbox[3]) // 2
        draw.text((x, y), display_text, font=font, fill="black")
        return np.array(img)

    return VideoClip(make_frame, duration=duration)

# 타이핑 지속 시간 자동 계산
def dynamic_typing_duration(text, min_duration=3, speed=15):
    return max(min_duration, len(text) / speed)

# 영상 클립 결합 및 BGM 추가
def make_video_from_clips(image_audio_data, output_path):
    clips = []
    for data in image_audio_data:
        if isinstance(data['clip'], str):
            clip = ImageClip(data['clip']).with_duration(data['duration'])
        else:
            clip = data['clip'].with_duration(data['duration'])
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")
    bgm = AudioFileClip(BGM_PATH).subclipped(0, final.duration)
    final = final.with_audio(bgm)
    final.write_videofile(output_path, fps=24)

# 문제 1개 처리
def process_quiz_item(item, index):
    image_audio_data = []

    if item["type"] == "multiple_choice":
        # 문제 타이핑
        question_text = f"퀴즈! {item['question']}"
        q_duration = dynamic_typing_duration(question_text)
        q_clip = create_typing_clip(question_text, duration=q_duration)
        image_audio_data.append({'clip': q_clip, 'duration': q_duration+2})

        # 선택지 이미지 (고정 4초)
        choices = item.get("choices", [])
        choice_lines = [f"{ch}" for i, ch in enumerate(choices)]
        c_img = create_image(["다음 중 무엇일까요?"] + choice_lines, f"choices_{index}.png")
        image_audio_data.append({'clip': c_img, 'duration': 4})

        # 정답 타이핑
        answer_text = f"정답은... {item['answer']}!"
        a_duration = dynamic_typing_duration(answer_text)
        a_clip = create_typing_clip(answer_text, duration=a_duration)
        image_audio_data.append({'clip': a_clip, 'duration': a_duration})

    else:  # 주관식/서술형
        # 문제 타이핑
        question_text = f"퀴즈! {item['question']}"
        q_duration = dynamic_typing_duration(question_text)
        q_clip = create_typing_clip(question_text, duration=q_duration)
        image_audio_data.append({'clip': q_clip, 'duration': q_duration})

        # 정지 이미지 (4초)
        q_img = create_image([question_text], f"question_{index}.png")
        image_audio_data.append({'clip': q_img, 'duration': 4})

        # 정답 타이핑
        answer_text = f"정답 공개! {item['answer']}"
        a_duration = dynamic_typing_duration(answer_text)
        a_clip = create_typing_clip(answer_text, duration=a_duration)
        image_audio_data.append({'clip': a_clip, 'duration': a_duration})

    out_path = os.path.join(OUTPUT_DIR, f"quiz_{index:03}.mp4")
    # 변수명 맞춰서 호출
    make_video_from_clips(image_audio_data, out_path)

    # 생성된 경로를 호출자에게 반환
    return out_path


# JSON 데이터 구조 정규화
def normalize_quiz_data(raw_data):
    type_map = {
        "객관식": "multiple_choice",
        "주관식": "short_answer",
        "OX문제": "multiple_choice",
        "빈칸 채우기": "short_answer",
        "서술형": "short_answer"
    }

    normalized = []
    for item in raw_data:
        q = item.get("question_data", {})
        if not q:
            continue
        normalized.append({
            "type": type_map.get(item.get("유형", "").strip(), "short_answer"),
            "question": q.get("question", ""),
            "choices": q.get("choices", []),
            "answer": q.get("answer", ""),
            "explanation": q.get("explanation", "")
        })
    return normalized

# 메인 실행 함수
def main():
    input_path = "quiz.json"  # 테스트용 파일 이름
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    quiz_data = normalize_quiz_data(raw_data)

    for i, item in enumerate(quiz_data[:3]):  # 앞 3개만 생성
        process_quiz_item(item, i)

if __name__ == "__main__":
    main()