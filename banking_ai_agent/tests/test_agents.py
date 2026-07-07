"""
tests/test_agents.py
----------------------
Basic unit + integration tests, run in OFFLINE mode (no API key needed) so
they work in any environment, including CI.

Run with:
    LLM_PROVIDER=offline python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
os.environ["LLM_PROVIDER"] = "offline"

import pytest
from database import db
from agents.classifier_agent import ClassifierAgent
from agents.query_agent import QueryAgent
from orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Point db.DB_PATH at a throwaway file for every test."""
    test_db = tmp_path / "test_tickets.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)
    db.init_db()
    yield


def test_classifier_positive():
    clf = ClassifierAgent()
    label, _ = clf.classify("Thanks for sorting out my net banking login issue.")
    assert label == "POSITIVE_FEEDBACK"


def test_classifier_negative():
    clf = ClassifierAgent()
    label, _ = clf.classify("My debit card replacement still hasn't arrived.")
    assert label == "NEGATIVE_FEEDBACK"


def test_classifier_query():
    clf = ClassifierAgent()
    label, _ = clf.classify("Could you check the status of ticket 650932?")
    assert label == "QUERY"


def test_query_agent_extracts_ticket_id():
    qa = QueryAgent()
    assert qa.extract_ticket_id("status of ticket 650932 please") == "650932"
    assert qa.extract_ticket_id("no ticket number here") is None


def test_orchestrator_negative_feedback_creates_ticket():
    orch = Orchestrator()
    result = orch.handle_message("My debit card replacement still hasn't arrived.", "Rohit")
    assert result.classification == "NEGATIVE_FEEDBACK"
    assert result.ticket_id is not None

    ticket = db.get_ticket(result.ticket_id)
    assert ticket is not None
    assert ticket["status"] == "Open"


def test_orchestrator_query_reads_created_ticket():
    orch = Orchestrator()
    negative_result = orch.handle_message("My net banking keeps failing.", "Meera")
    ticket_id = negative_result.ticket_id

    query_result = orch.handle_message(f"Could you check the status of ticket {ticket_id}?", "Meera")
    assert query_result.classification == "QUERY"
    assert ticket_id in query_result.response
    assert "Open" in query_result.response


def test_orchestrator_positive_feedback_message_format():
    orch = Orchestrator()
    result = orch.handle_message("Thanks for resolving my credit card issue.", "Aditi")
    assert result.classification == "POSITIVE_FEEDBACK"
    assert "Aditi" in result.response


def test_logging_records_every_interaction():
    orch = Orchestrator()
    orch.handle_message("Thanks a lot!", "Sam")
    logs = db.list_logs()
    assert len(logs) >= 1
    assert logs[0]["classification"] == "POSITIVE_FEEDBACK"
    assert logs[0]["prompt_trace"]  # prompt trace was captured
    assert logs[0]["success"] == 1


def test_feedback_loop_records_rating():
    orch = Orchestrator()
    result = orch.handle_message("Thanks a lot!", "Sam")
    db.log_feedback(result.log_id, "down", note="tone felt generic")

    stats = db.feedback_stats()
    assert stats["overall"].get("down", 0) == 1
    assert len(stats["flagged_for_review"]) == 1
    assert stats["flagged_for_review"][0]["log_id"] == result.log_id


def test_routing_success_rate_reports_full_success_when_no_fallback():
    orch = Orchestrator()
    orch.handle_message("Thanks for resolving my issue.", "Aditi")
    orch.handle_message("My card is still not replaced.", "Rohit")
    stats = db.routing_success_rate()
    assert stats["total"] >= 2
    assert stats["rate"] == 1.0


def test_classifier_evaluation_runs_end_to_end():
    from evaluation import run_classifier_evaluation
    report = run_classifier_evaluation()
    assert report["total_cases"] > 0
    assert 0 <= report["overall_accuracy"] <= 1


def test_response_quality_evaluation_runs_end_to_end():
    from evaluation import run_response_quality_evaluation
    report = run_response_quality_evaluation()
    assert 0 <= report["overall_quality_score"] <= 1
    assert len(report["cases"]) > 0
