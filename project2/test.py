import os
from gtts import gTTS
import kss  # Korean Sentence Splitter
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips
)

# 🗣️ TTS 생성
def text_to_speech(sentence, output_path):
    tts = gTTS(sentence, lang='ko')
    tts.save(output_path)

# 📐 화면 정중앙 계산 함수 (가로·세로 모두 센터)
def center_position(clip, frame_size):
    W, H = frame_size
    return ((W - clip.w) / 2, (H - clip.h) / 2)

# 🎞️ 타자기 스타일 자막 (예시 3 수정: 동적 중앙 배치)
def create_typewriter_subtitle_centered(sentence, duration, video_size, font_size):
    clips = []
    total_chars  = len(sentence)
    time_per_char = duration / total_chars

    for i in range(1, total_chars + 1):
        txt_clip = TextClip(
            text=sentence[:i],
            font_size=font_size,
            font=r"C:\Windows\Fonts\malgun.ttf",
            color="black",
            method="caption",     # caption 방식 사용
            size=(int(video_size[0] * 0.8), None),
            align="center"        # 텍스트 내부 중앙 정렬
        ).with_duration(time_per_char) \
         .with_position(center_position)  # 화면 정가운데
        clips.append(txt_clip)

    return concatenate_videoclips(clips)

# 🎞️ 개별 문장 클립 생성
def create_sentence_clip(sentence, index, bg_img_path, video_size, font_size):
    # 1) TTS 오디오 생성
    audio_path = f"temp_audio_{index}.mp3"
    text_to_speech(sentence, audio_path)
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    # 2) 배경 이미지 클립 준비
    bg_clip = (
        ImageClip(bg_img_path)
        .resize(video_size)
        .set_duration(duration)
        .set_audio(audio_clip)
        .set_fps(30)
    )

    # 3) 자막 클립 합성 (가운데)
    subtitle = create_typewriter_subtitle_centered(
        sentence, duration, video_size, font_size
    )
    final_clip = CompositeVideoClip([bg_clip, subtitle])
    return final_clip

# 🎬 전체 쇼츠 영상 생성 (가운데 자막)
def create_educational_shorts_centered(
    script,
    bg_img_path,
    output_path="shorts_centered.mp4",
    video_size=(1080, 1920),
    font_size=60
):
    print("📌 문장 분리 중...")
    sentences = kss.split_sentences(script)

    clips = []
    for i, sent in enumerate(sentences):
        print(f"🎙️ 문장 {i+1}/{len(sentences)} 처리 중: {sent}")
        clips.append(
            create_sentence_clip(
                sent, i, bg_img_path, video_size, font_size
            )
        )

    print("📌 클립 합치는 중...")
    final_clip = concatenate_videoclips(clips, method="compose")

    print("📌 렌더링 중...")
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac"
    )

    # 임시 오디오 파일 삭제
    for i in range(len(sentences)):
        os.remove(f"temp_audio_{i}.mp3")

# ✅ 실행 예시
if __name__ == "__main__":
    script = (
        "지진파에는 두 가지 주요한 종류가 있습니다. "
        "이 중 가장 먼저 도달하는 지진파의 이름은 무엇인가요? "
        "정답: P파 (Primary Wave)"
    )
    background_image_path = r"C:/Users/USER/Desktop/project/Shorts/background.jpg"

    # kss 설치: pip install kss
    create_educational_shorts_centered(
        script,
        background_image_path,
        video_size=(1080, 1920),
        font_size=64
    )