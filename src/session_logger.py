import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("outputs/prediction_log.db")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence_aml REAL NOT NULL,
            confidence_non_aml REAL NOT NULL,
            raw_probs TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_prediction(filename, prediction, probs):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT INTO predictions (timestamp, filename, prediction, confidence_aml, confidence_non_aml, raw_probs) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            filename,
            prediction,
            round(float(probs[1]), 6),
            round(float(probs[0]), 6),
            json.dumps([round(float(p), 6) for p in probs]),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_logs(limit=50):
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT timestamp, filename, prediction, confidence_aml, confidence_non_aml "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def get_stats():
    conn = sqlite3.connect(str(DB_PATH))
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    aml = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='AML'").fetchone()[0]
    non_aml = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Non-AML'").fetchone()[0]
    conn.close()
    return {"total": total, "aml": aml, "non_aml": non_aml}
