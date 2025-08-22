# AI Learning Support Platform

통합적인 학습환경 제공을 목표로 한 웹 기반 AI 학습 지원 시스템입니다.  
사용자가 업로드한 학습 자료(PDF, 이미지, YouTube)를 바탕으로 자동으로 문제를 생성하고,  
답안에 따라 AI가 채점한 후, 오답에 대한 AI 토론 기능까지 제공하며 학습의 몰입성과 깊이를 향상시킵니다.

---

## 기능 요약

- **문제 자동 생성**  
  PDF, 이미지, YouTube 자료에서 다양한 유형(객관식, 주관식, OX, 빈칸, 서술형)의 퀴즈 생성

- **AI 채점 및 상세 피드백**  
  사용자의 답안에 대해 GPT 기반 채점 및 해설 제공으로 학습 효과 극대화

- **오답 토론 시뮬레이션**  
  오답에 대해 AI 간 토론 형식으로 "왜 정답인가"에 대한 논리적 이해 유도

- **토론 모드 확장**  
  - GPT vs GPT 토론  
  - 사용자 vs AI 토론  
  직관적인 인터페이스로 다양한 사고 방식 체험 가능

- **문제 콘텐츠 다운로드**  
  생성된 문제를 JSON 형식으로 저장, 편리한 활용과 공유 지원

- **쉬운 학습 접근성**  
  Google OAuth로 간편 로그인, 직관적인 UI/UX 구현

---

## Tech Stack

| 영역         | 기술 |
|--------------|------|
| **백엔드**    | Python, Flask |
| **AI 엔진**   | OpenAI GPT-4 (ChatCompletion API) |
| **문서 분석**  | PyMuPDF (fitz), pytesseract, pytube |
| **추가 기능**  | YouTube transcript API, faster-whisper |
| **프론트엔드** | HTML, Jinja2, Bootstrap, CSS, JS |
| **환경 관리**  | pip, virtualenv, requirements.txt |

---

## 설치 및 실행

```bash
# 복제 및 이동
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 가상환경 (선택)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 의존성 설치
pip install -r requirements.txt

# 실행
python app.py

---

## 디렉토리 구조

/project-root
├── app.py                  # 메인 서버 파일
├── debate_routes.py        # 토론 관련 라우트
├── generate_quiz_video.py  # YouTube 기반 문제 생성
├── templates/              # HTML 템플릿
│   ├── index.html
│   ├── solve.html
│   ├── grade.html
│   └── debate_result.html
├── static/                 # CSS, JS, 이미지 등 정적 리소스
├── uploads/                # 업로드 파일 저장 폴더
├── requirements.txt        # 패키지 의존성 리스트
└── README.md               # 프로젝트 설명 (이 파일)

