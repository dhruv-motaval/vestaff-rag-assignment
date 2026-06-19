import sqlite3

from app.config import DB_PATH


def get_connection():
    """Returns a SQLite connection to the logs database."""

    return sqlite3.connect(DB_PATH)


def init_db():
    """Creates the query_logs table if it doesnt already exists."""

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer_found BOOLEAN,
        latency_ms REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


def log_query(question: str, answer_found: bool, latency_ms: float):
    """Inserts a single query record into query_logs with timestamp auto-set by SQLite."""

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO query_logs (question, answer_found, latency_ms)
        VALUES (?, ?, ?)
        """,
        (question, answer_found, latency_ms),
    )
    conn.commit()
    conn.close()
