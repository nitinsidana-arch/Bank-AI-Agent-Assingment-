"""
evaluation.py
--------------
Implements the "Model Evaluation" requirements from Part 2 of the brief:

  - Assess the quality of generated responses across the agents (feedback
    accuracy, empathy level, clarity of status updates)
  - Use QA-based scoring and test case coverage for classification logic
  - Evaluate agent routing success rate

Three independent pieces:

1. classifier evaluation  - runs a labeled test set through ClassifierAgent
   and reports per-label and overall accuracy (test case coverage).
2. response quality (QA) scoring - runs the same test set through the full
   Orchestrator and scores each response against simple, explainable rubric
   checks (does a thank-you include the customer's name? does an apology
   include the ticket number and read as empathetic? does a status reply
   follow the exact required format?). Each check is a yes/no rubric item,
   which is what "QA-based scoring" means here - not a hidden ML score.
3. routing success rate - read straight from the interaction_logs table
   (see database/db.routing_success_rate), reflecting real traffic rather
   than only the curated test set.

Run standalone with:
    LLM_PROVIDER=offline python evaluation.py
"""

import json
import os
import re
from pathlib import Path

os.environ.setdefault("LLM_PROVIDER", "offline")

from agents.classifier_agent import ClassifierAgent
from orchestrator import Orchestrator
from database import db

EVAL_DATASET_PATH = Path(__file__).parent / "tests" / "eval_dataset.json"


def load_eval_dataset():
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 1. Classifier evaluation (test case coverage + accuracy)
# ----------------------------------------------------------------------
def run_classifier_evaluation(dataset=None, llm=None):
    dataset = dataset or load_eval_dataset()
    classifier = ClassifierAgent(llm=llm)

    results = []
    correct = 0
    per_label_totals = {}
    per_label_correct = {}

    for case in dataset:
        predicted, used_fallback = classifier.classify(case["message"])
        expected = case["expected_label"]
        is_correct = predicted == expected

        per_label_totals[expected] = per_label_totals.get(expected, 0) + 1
        if is_correct:
            correct += 1
            per_label_correct[expected] = per_label_correct.get(expected, 0) + 1

        results.append({
            "message": case["message"],
            "expected_label": expected,
            "predicted_label": predicted,
            "correct": is_correct,
            "used_fallback": used_fallback,
        })

    per_label_accuracy = {
        label: round(per_label_correct.get(label, 0) / total, 3)
        for label, total in per_label_totals.items()
    }

    return {
        "total_cases": len(dataset),
        "correct": correct,
        "overall_accuracy": round(correct / len(dataset), 3) if dataset else 0,
        "per_label_accuracy": per_label_accuracy,
        "cases": results,
    }


# ----------------------------------------------------------------------
# 2. Response quality (QA rubric) scoring
# ----------------------------------------------------------------------
EMPATHY_WORDS = ("apologize", "sorry", "understand", "inconvenience")
GRATITUDE_WORDS = ("thank", "delighted", "appreciate", "glad", "happy")


def _score_positive_response(response: str, customer_name: str) -> dict:
    checks = {
        "includes_customer_name": customer_name in response,
        "expresses_gratitude": any(w in response.lower() for w in GRATITUDE_WORDS),
        "reasonable_length": 0 < len(response) <= 400,
    }
    return checks


def _score_negative_response(response: str, ticket_id: str) -> dict:
    checks = {
        "includes_ticket_number": bool(ticket_id) and ticket_id in response,
        "reads_empathetic": any(w in response.lower() for w in EMPATHY_WORDS),
        "reasonable_length": 0 < len(response) <= 400,
    }
    return checks


def _score_query_response(response: str, message: str) -> dict:
    ticket_match = re.search(r"\b(\d{6})\b", message)
    lowered = response.lower()
    checks = {
        "echoes_ticket_number": (ticket_match.group(1) in response) if ticket_match else True,
        "gives_clear_status_or_not_found": (
            bool(re.search(r"is currently marked as", lowered)) or "couldn't find" in lowered
        ),
    }
    return checks


def run_response_quality_evaluation(dataset=None, llm=None):
    dataset = dataset or load_eval_dataset()
    orchestrator = Orchestrator(llm=llm)

    scored = []
    for case in dataset:
        customer_name = case.get("customer_name", "Customer")
        result = orchestrator.handle_message(case["message"], customer_name)

        if result.classification == "POSITIVE_FEEDBACK":
            checks = _score_positive_response(result.response, customer_name)
        elif result.classification == "NEGATIVE_FEEDBACK":
            checks = _score_negative_response(result.response, result.ticket_id)
        else:
            checks = _score_query_response(result.response, case["message"])

        pass_rate = sum(checks.values()) / len(checks) if checks else 0
        scored.append({
            "message": case["message"],
            "classification": result.classification,
            "response": result.response,
            "checks": checks,
            "pass_rate": round(pass_rate, 3),
        })

    overall_quality = round(sum(s["pass_rate"] for s in scored) / len(scored), 3) if scored else 0
    return {
        "overall_quality_score": overall_quality,
        "cases": scored,
    }


# ----------------------------------------------------------------------
# 3. Routing success rate (from real logged traffic)
# ----------------------------------------------------------------------
def run_routing_success_rate():
    return db.routing_success_rate()


# ----------------------------------------------------------------------
# Combined report
# ----------------------------------------------------------------------
def generate_evaluation_report(llm=None):
    db.init_db()
    classifier_eval = run_classifier_evaluation(llm=llm)
    quality_eval = run_response_quality_evaluation(llm=llm)
    routing_eval = run_routing_success_rate()

    return {
        "classifier_evaluation": classifier_eval,
        "response_quality_evaluation": quality_eval,
        "routing_success_rate": routing_eval,
    }


if __name__ == "__main__":
    report = generate_evaluation_report()

    print("=" * 70)
    print("CLASSIFIER EVALUATION (test case coverage)")
    print("=" * 70)
    print(f"Overall accuracy : {report['classifier_evaluation']['overall_accuracy'] * 100:.1f}% "
          f"({report['classifier_evaluation']['correct']}/{report['classifier_evaluation']['total_cases']})")
    for label, acc in report["classifier_evaluation"]["per_label_accuracy"].items():
        print(f"  {label:<18}: {acc * 100:.1f}%")

    print("\n" + "=" * 70)
    print("RESPONSE QUALITY (QA rubric scoring)")
    print("=" * 70)
    print(f"Overall quality score: {report['response_quality_evaluation']['overall_quality_score'] * 100:.1f}%")
    for case in report["response_quality_evaluation"]["cases"]:
        failed = [k for k, v in case["checks"].items() if not v]
        status = "PASS" if not failed else f"FAIL ({', '.join(failed)})"
        print(f"  [{case['classification']:<17}] {status}")

    print("\n" + "=" * 70)
    print("ROUTING SUCCESS RATE (from logged interactions)")
    print("=" * 70)
    r = report["routing_success_rate"]
    print(f"  {r['successful']}/{r['total']} successful ({r['rate'] * 100:.1f}%)")

    out_path = Path(__file__).parent / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")
