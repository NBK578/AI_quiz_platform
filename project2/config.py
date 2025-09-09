# config.py
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
load_dotenv()


def get_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).lower()
    return v in ("1", "true", "yes", "y")

class BaseConfig:
    # .env에서 로드된 값 사용

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
    
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD_RAW = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "quizdb")

    SQLALCHEMY_DATABASE_URI = (
    f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASSWORD_RAW)}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
)

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = get_env_bool("SQL_ECHO", False)

    
    # 커넥션 풀(선택)
    POOL_SIZE = int(os.getenv("POOL_SIZE", "5"))
    POOL_TIMEOUT = int(os.getenv("POOL_TIMEOUT", "30"))
    POOL_RECYCLE = int(os.getenv("POOL_RECYCLE", "1800"))

class DevConfig(BaseConfig):
    DEBUG = True

class ProdConfig(BaseConfig):
    DEBUG = False
