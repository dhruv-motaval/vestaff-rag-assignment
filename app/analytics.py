import sqlite3
import numpy as np

from app.config import DB_PATH


def get_analytics():
    """Runs three SQL queries againts query_logs:

    1. Top 10 most frequent questions asked ( GROUP BY + COUNT)
    2. All questions where the answer was not found
    3. Latencies for each query (avg and 95th percentile)

    Returns results as a single dict.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT question, COUNT(*)
        FROM query_logs
        GROUP BY question
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """
    )

    top_questions = [{"question": row[0], "count": row[1]} for row in cursor.fetchall()]

    cursor.execute("""
    SELECT question
    FROM query_logs
    WHERE answer_found = 0
    """)

    no_answer = [row[0] for row in cursor.fetchall()]

    cursor.execute("""
    SELECT latency_ms
    FROM query_logs
    """)

    latencies = [x[0] for x in cursor.fetchall()]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    p95 = float(np.percentile(latencies, 95)) if latencies else 0

    conn.close()

    return {
        "most_frequent_question": top_questions,
        "no_answer_questions": no_answer,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95,
    }
