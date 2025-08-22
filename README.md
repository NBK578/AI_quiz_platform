📚 AI 기반 학습 지원 플랫폼 (AI Learning Support Platform)
🎯 프로젝트 개요

이 프로젝트는 PDF 학습 자료를 기반으로 자동 퀴즈를 생성하고, 사용자의 정답을 기반으로 난이도 조절 및 채점, 나아가 GPT를 활용한 토론 시뮬레이션 기능까지 포함하는 통합 학습 지원 웹 플랫폼입니다.

💡 주요 기능

📄 PDF 기반 문제 생성: 사용자가 업로드한 PDF 문서를 분석하여 객관식, OX, 주관식, 빈칸 문제를 자동 생성

🧠 AI 기반 채점 및 피드백: 사용자의 답안을 GPT가 채점하고 결과를 분석

💬 오답 토론 시뮬레이션: 틀린 문제에 대해 GPT와 함께 이유를 묻고 답하는 토론 기능

🎥 YouTube 자막 퀴즈 생성: 유튜브 영상의 자막을 분석하여 퀴즈 출제

📝 문제 다운로드 기능: 생성된 문제들을 JSON 형식으로 저장 및 다운로드

👤 구글 OAuth 로그인 지원: 구글 계정을 통한 간편 로그인

🛠️ 사용 기술
영역	기술 스택
백엔드	Python, Flask

AI	OpenAI GPT API (Function calling)

프론트엔드	HTML, CSS, Bootstrap, Jinja2

문서 분석	PyMuPDF (fitz), pytube, youtube_transcript_api

음성 변환	faster-whisper

환경 관리	pip, requirements.txt, venv
