"""
demo_cli.py
------------
Command-line walkthrough of the three sample flows from the capstone brief.
Useful for a quick sanity check, and for capturing terminal-output evidence
of the pipeline working end-to-end without needing the Streamlit UI open.

Run with:
    LLM_PROVIDER=offline python demo_cli.py
"""

import os
os.environ.setdefault("LLM_PROVIDER", "offline")

from database import db
from orchestrator import Orchestrator


def run_flow(orchestrator, label, message, customer="Aditi"):
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")
    print(f"User Input     : \"{message}\"")
    result = orchestrator.handle_message(message, customer)
    print(f"Classification : {result.classification}")
    print(f"Agent Path     : {result.agent_path}")
    if result.ticket_id:
        print(f"Ticket Created : #{result.ticket_id}")
    print(f"Response       : {result.response}")
    return result


if __name__ == "__main__":
    db.init_db()
    orch = Orchestrator()

    run_flow(orch, "Example 1: Positive Feedback",
              "Thanks for sorting out my net banking login issue.")

    neg_result = run_flow(orch, "Example 2: Negative Feedback",
                           "My debit card replacement still hasn't arrived.")

    run_flow(orch, "Example 3: Query (using the ticket just created)",
              f"Could you check the status of ticket {neg_result.ticket_id}?")

    print(f"\n{'=' * 70}")
    print("All tickets currently in the database:")
    print(f"{'=' * 70}")
    for t in db.list_tickets():
        print(f"  #{t['ticket_id']}  [{t['status']:^11}]  {t['message']}")
