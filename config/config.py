import os
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv("DB_HOST", "localhost")
USER = os.getenv("DB_USER")
PW   = os.getenv("DB_PASSWORD")
DB   = os.getenv("DB_NAME", "oulad_db")
PORT = os.getenv("DB_PORT", "5432")
