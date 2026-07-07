"""
database/db.py
----------------
Lightweight SQLite persistence layer for the banking support-ticket system.

We use plain SQLite (via Python's built-in sqlite3 module) rather than a
heavier ORM because the schema is tiny and the whole point of this module is
to give the LangChain tools something concrete and inspectable to read from
and write to. Everything the agents do to "the database" happens here and
only here, so behaviour is easy to trace and to unit test.
"""

import sqlite3
import random
import string
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "support_tickets.db"


@contextmanager
def get_connection():
    """Yield a SQLite connection with foreign keys / row factory configured."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables if they do not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id      TEXT PRIMARY KEY,
                customer_name  TEXT,
                message        TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'Open',
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_logs (
                log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message   TEXT NOT NULL,
                classification TEXT NOT NULL,
                agent_path     TEXT NOT NULL,
                response       TEXT NOT NULL,
                prompt_trace   TEXT,
                ticket_id      TEXT,
                success        INTEGER NOT NULL DEFAULT 1,
                timestamp      TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_log (
                feedback_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id         INTEGER NOT NULL,
                rating         TEXT NOT NULL,      -- 'up' or 'down'
                note           TEXT,
                timestamp      TEXT NOT NULL,
                FOREIGN KEY (log_id) REFERENCES interaction_logs(log_id)
            )
            """
        )


def _generate_ticket_id(conn) -> str:
    """Generate a unique 6-digit ticket number, retrying on collision."""
    while True:
        candidate = "".join(random.choices(string.digits, k=6))
        exists = conn.execute(
            "SELECT 1 FROM support_tickets WHERE ticket_id = ?", (candidate,)
        ).fetchone()
        if not exists:
            return candidate


def create_ticket(message: str, customer_name: str = "Customer") -> str:
    """Insert a new unresolved ticket and return its 6-digit id."""
    with get_connection() as conn:
        ticket_id = _generate_ticket_id(conn)
        now = datetime.utcnow().isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT INTO support_tickets
                (ticket_id, customer_name, message, status, created_at, updated_at)
            VALUES (?, ?, ?, 'Open', ?, ?)
            """,
            (ticket_id, customer_name, message, now, now),
        )
        return ticket_id


def get_ticket(ticket_id: str):
    """Return a ticket row (as a dict) or None if it does not exist."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM support_tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        return dict(row) if row else None


def update_ticket_status(ticket_id: str, status: str) -> bool:
    with get_connection() as conn:
        now = datetime.utcnow().isoformat(timespec="seconds")
        cur = conn.execute(
            "UPDATE support_tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
            (status, now, ticket_id),
        )
        return cur.rowcount > 0


def list_tickets(limit: int = 50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM support_tickets ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def log_interaction(
    user_message: str,
    classification: str,
    agent_path: str,
    response: str,
    prompt_trace: str = "",
    ticket_id: str = None,
    success: bool = True,
) -> int:
    """Insert an interaction log row and return its new log_id."""
    with get_connection() as conn:
        now = datetime.utcnow().isoformat(timespec="seconds")
        cur = conn.execute(
            """
            INSERT INTO interaction_logs
                (user_message, classification, agent_path, response, prompt_trace, ticket_id, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_message, classification, agent_path, response, prompt_trace, ticket_id, int(success), now),
        )
        return cur.lastrowid


def list_logs(limit: int = 100):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM interaction_logs ORDER BY log_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def routing_success_rate() -> dict:
    """Return {'total': n, 'successful': n, 'rate': 0-1} across all logged interactions."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, SUM(success) AS successful FROM interaction_logs"
        ).fetchone()
        total = row["total"] or 0
        successful = row["successful"] or 0
        rate = (successful / total) if total else 1.0
        return {"total": total, "successful": successful, "rate": rate}


def log_feedback(log_id: int, rating: str, note: str = None):
    """Record a thumbs-up/thumbs-down on a past interaction (the improvement-loop signal)."""
    with get_connection() as conn:
        now = datetime.utcnow().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO feedback_log (log_id, rating, note, timestamp) VALUES (?, ?, ?, ?)",
            (log_id, rating, note, now),
        )


def feedback_stats() -> dict:
    """Aggregate thumbs-up/down counts, overall and broken down by classification."""
    with get_connection() as conn:
        overall = conn.execute(
            """
            SELECT rating, COUNT(*) AS n FROM feedback_log GROUP BY rating
            """
        ).fetchall()
        by_class = conn.execute(
            """
            SELECT il.classification, fl.rating, COUNT(*) AS n
            FROM feedback_log fl
            JOIN interaction_logs il ON il.log_id = fl.log_id
            GROUP BY il.classification, fl.rating
            """
        ).fetchall()
        flagged = conn.execute(
            """
            SELECT il.log_id, il.user_message, il.classification, il.response, fl.note
            FROM feedback_log fl
            JOIN interaction_logs il ON il.log_id = fl.log_id
            WHERE fl.rating = 'down'
            ORDER BY fl.feedback_id DESC
            """
        ).fetchall()
        return {
            "overall": {r["rating"]: r["n"] for r in overall},
            "by_class": [dict(r) for r in by_class],
            "flagged_for_review": [dict(r) for r in flagged],
        }


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
