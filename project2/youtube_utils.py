import os
import re
import time
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
import whisper
from transformers import pipeline
from fpdf import FPDF

# 파일 및 폴더 경로 설정
BASE_DIR = os.path.dirname(__file__)
FONT_DIR = os.path.join(BASE_DIR, "fonts")
FONT_PATH = os.path.join(FONT_DIR, "NotoSansKR-Regular.ttf")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")  # PDF 저장 폴더

# 필수 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FONT_DIR, exist_ok=True)

# 폰트 파일 확인
if not os.path.isfile(FONT_PATH):
    raise FileNotFoundError(
        f"Unicode 폰트 파일을 찾을 수 없습니다: {FONT_PATH}\n"
        """프로젝트 내 'fonts' 폴더에 NotoSansKR-Regular.ttf 파일을 다운로드하여 넣어주세요.
        예시 경로: project/fonts/NotoSansKR-Regular.ttf"""
    )

def get_video_id(url):
    """
    유튜브 URL로부터 동영상 ID를 추출합니다.
    """
    match = re.search(r'(?:v=|youtu\.be/)([^&\n]+)', url)
    if match:
        return match.group(1)
    raise ValueError("유효한 유튜브 URL이 아닙니다.")


def download_audio(video_id):
    """
    pytube를 사용해 유튜브 동영상의 오디오 스트림을 다운로드합니다.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    output_path = os.path.join(tempfile.gettempdir(), f"{video_id}.mp3")
    audio_stream.download(filename=output_path)
    return output_path


def transcribe_audio(audio_path):
    """
    Whisper 모델을 사용해 오디오 파일을 텍스트로 변환합니다.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result.get("text", "")


def summarize_text(text):
    """
    transformers의 summarization pipeline을 사용해 텍스트를 요약합니다.
    """
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']


def youtube_to_pdf(youtube_url):
    """
    1. 자막 우선 시도 (YouTubeTranscriptApi)
    2. 자막 없을 시 오디오 다운로드 -> Whisper 음성 인식 -> 텍스트 요약
    3. 결과를 PDF로 생성하여 저장
    4. (pdf_path, filename) 반환
    """
    video_id = get_video_id(youtube_url)

    # 1) 자막 추출 시도
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko", "en"])
        text = "\n".join([entry['text'] for entry in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        # 2) 자막 없으면 음성 인식 및 요약
        try:
            audio_path = download_audio(video_id)
            full_text = transcribe_audio(audio_path)
            summary = summarize_text(full_text)
            text = (
                "음성 인식으로 추출한 전체 텍스트:\n" + full_text + "\n\n" +
                "추가 요약 결과:\n" + summary
            )
        except Exception as e:
            raise Exception(f"자막 및 음성 인식 모두 실패했습니다: {e}")
    except Exception as e:
        raise Exception(f"자막을 불러오는 중 오류 발생: {e}")

    # 3) PDF 생성
    filename = f"yt_{int(time.time())}.pdf"
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        pdf = FPDF()
        pdf.add_page()
        # Unicode 지원 폰트 등록, uni=True 필수
        pdf.add_font("Noto", "", FONT_PATH, uni=True)
        pdf.set_font("Noto", "", 12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(pdf_path)
    except Exception as e:
        raise Exception(f"PDF 생성 중 오류 발생: {e}")

    return pdf_path, filename


if __name__ == "__main__":
    test_url = "https://youtu.be/your_video_id_here"
    try:
        path, fname = youtube_to_pdf(test_url)
        print(f"PDF 생성 완료: {path}")
    except Exception as e:
        print(f"오류: {e}")
