from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PDF_PATH = "./data/AWS Customer Agreement.pdf"

CHROMA_PATH = "chroma_db"

DB_PATH = "logs.db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4

SIMILARITY_THRESHOLD = 0.7
