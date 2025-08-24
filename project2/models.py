from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# Flask 앱에서 init_app(app)으로 초기화할 "빈" SQLAlchemy 인스턴스
db = SQLAlchemy()

# 필요시: public 외 스키마라면 아래처럼 스키마 지정
# SCHEMA = "public"
# __table_args__ = {"schema": SCHEMA} 를 각 모델에 추가

class User(db.Model):
    __tablename__ = 'users'
    user_id    = db.Column(db.Integer, primary_key=True)
    google_id  = db.Column(db.String(255), unique=True, nullable=False)
    email      = db.Column(db.String(255), unique=True, nullable=False)
    name       = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    quizzes = db.relationship('Quiz', back_populates='owner', cascade='all, delete-orphan')
    answers = db.relationship('UserAnswer', back_populates='user', cascade='all, delete-orphan')

class Quiz(db.Model):
    __tablename__ = 'quizzes'
    quiz_id    = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)
    name       = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    owner     = db.relationship('User', back_populates='quizzes')
    questions = db.relationship('Question', back_populates='quiz', cascade='all, delete-orphan')
    videos    = db.relationship('Video', back_populates='quiz', cascade='all, delete-orphan')

class Question(db.Model):
    __tablename__ = 'questions'
    question_id   = db.Column(db.Integer, primary_key=True)
    quiz_id       = db.Column(db.Integer, db.ForeignKey('quizzes.quiz_id', ondelete='CASCADE'), nullable=False)
    type          = db.Column(db.String(50), nullable=False)  # 객관식/빈칸/OX/주관식/서술형
    page          = db.Column(db.Integer)
    question_text = db.Column(db.Text, nullable=False)
    answer        = db.Column(db.Text, nullable=False)
    explanation   = db.Column(db.Text)

    quiz         = db.relationship('Quiz', back_populates='questions')
    choices      = db.relationship('Choice', back_populates='question', cascade='all, delete-orphan')
    user_answers = db.relationship('UserAnswer', back_populates='question', cascade='all, delete-orphan')

class Choice(db.Model):
    __tablename__ = 'choices'
    choice_id   = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.question_id', ondelete='CASCADE'), nullable=False)
    letter      = db.Column(db.String(1), nullable=False)   # 'A'/'B'… 또는 'O'/'X'
    text        = db.Column(db.Text, nullable=False)

    question = db.relationship('Question', back_populates='choices')

class UserAnswer(db.Model):
    __tablename__ = 'user_answers'
    answer_id   = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete='SET NULL'))
    question_id = db.Column(db.Integer, db.ForeignKey('questions.question_id', ondelete='CASCADE'), nullable=False)
    user_answer = db.Column(db.Text)
    is_correct  = db.Column(db.Boolean)
    feedback    = db.Column(db.Text)
    score       = db.Column(db.Integer)
    answered_at = db.Column(db.DateTime, default=datetime.utcnow)

    user     = db.relationship('User', back_populates='answers')
    question = db.relationship('Question', back_populates='user_answers')

class Video(db.Model):
    __tablename__ = 'videos'
    video_id   = db.Column(db.Integer, primary_key=True)
    quiz_id    = db.Column(db.Integer, db.ForeignKey('quizzes.quiz_id', ondelete='CASCADE'), nullable=False)
    file_path  = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    quiz = db.relationship('Quiz', back_populates='videos')