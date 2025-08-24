from flask import Flask, request, render_template, flash, redirect, session, send_file, url_for
import os
from youtube_utils import youtube_to_pdf

app = Flask(__name__)
app.secret_key = "myexperimentsecretkey"  # 실험용 비밀 키(운영 시엔 환경변수나 config로 관리)

# uploads 폴더는 youtube_utils.py와 동일하게 사용합니다.
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), "uploads")

@app.route("/")
def index():
    # 기본 접속 시 바로 변환 폼으로 이동
    return redirect(url_for("youtube_to_pdf_route"))

@app.route("/youtube_to_pdf", methods=["GET", "POST"])
def youtube_to_pdf_route():
    if request.method == "GET":
        return render_template("youtube_form.html")
    # POST: 사용자 입력 처리
    youtube_url = request.form.get("youtube_url", "").strip()
    if not youtube_url:
        flash("유튜브 링크를 입력해주세요.")
        return redirect(url_for("youtube_to_pdf_route"))

    try:
        # 자막 혹은 음성→요약 → PDF 생성
        pdf_path, filename = youtube_to_pdf(youtube_url)
        session["youtube_pdf"] = filename
        return render_template("youtube_result.html", filename=filename)
    except Exception as e:
        flash(f"처리 중 오류 발생: {e}")
        return redirect(url_for("youtube_to_pdf_route"))

@app.route("/download_youtube_pdf/<filename>")
def download_youtube_pdf(filename):
    # 생성된 PDF를 다운로드
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
