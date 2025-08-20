# youtube_routes.py
import os
from flask import Blueprint, render_template, request, redirect, url_for, session, send_file, flash, current_app, jsonify
from youtube_utils import youtube_to_pdf
from utils import generate_questions_from_pdf, get_unique_filename
import json

youtube_bp = Blueprint('youtube', __name__)

@youtube_bp.route('/youtube_to_pdf', methods=['GET', 'POST'])
def youtube_to_pdf_route():
    if request.method == 'GET':
        return render_template('youtube_form.html')
    url = request.form.get('youtube_url', '').strip()
    if not url:
        flash('유튜브 링크를 입력해주세요.')
        return redirect(url_for('youtube.youtube_to_pdf_route'))
    try:
        pdf_path, filename = youtube_to_pdf(url)
        session['youtube_pdf'] = filename
        return render_template('youtube_result.html', filename=filename)
    except Exception as e:
        flash(f'처리 중 오류 발생: {e}')
        return redirect(url_for('youtube.youtube_to_pdf_route'))

@youtube_bp.route('/download_youtube_pdf/<filename>')
def download_youtube_pdf(filename):
    path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    return send_file(path, as_attachment=True)

@youtube_bp.route('/youtube_create')
def youtube_create():
    return render_template('youtube_create.html')

@youtube_bp.route('/youtube_generate', methods=['POST'])
def youtube_generate():
    youtube_url = request.form.get('youtube_url')
    if not youtube_url:
        return jsonify({'error': 'YouTube URL이 필요합니다.'}), 400
    
    # TODO: YouTube URL에서 PDF를 다운로드하는 로직 구현
    pdf_path = "path/to/downloaded/pdf"  # 임시 경로
    
    custom_prompt = request.form.get("custom_prompt", "")
    # requirements와 custom_prompt를 합쳐서 프롬프트에 전달
    full_requirements = ""
    if custom_prompt:
        full_requirements += "\n" + custom_prompt
    
    try:
        questions = generate_questions_from_pdf(
            pdf_path=pdf_path,
            total_questions=5,  # 기본값
            difficulty="중급",  # 기본값
            selected_types=["객관식", "주관식"],  # 기본값
            requirements=full_requirements
        )
        
        # 결과를 JSON 파일로 저장
        filename = get_unique_filename("youtube_questions")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
            
        return jsonify({'success': True, 'questions': questions})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
