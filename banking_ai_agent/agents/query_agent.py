"""
agents/query_agent.py
----------------------
Handles QUERY-classified messages: extract a ticket number from free text
and report its current status from the support database.

Ticket-number extraction is done with a regex rather than an LLM call - it
is a narrow, well-defined pattern (exactly 6 digits) and a regex is both
faster and more reliable than asking a model to "find the number".
"""

import re
from database import db

TICKET_PATTERN = re.compile(r"\b(\d{6})\b")


class QueryAgent:
    def extract_ticket_id(self, user_message: str) -> str | None:
        match = TICKET_PATTERN.search(user_message)
        return match.group(1) if match else None

    def handle(self, user_message: str) -> str:
        ticket_id = self.extract_ticket_id(user_message)

        if not ticket_id:
            return (
                "I couldn't find a ticket number in your message. Could you "
                "share the 6-digit ticket number so I can look up its status?"
            )

        ticket = db.get_ticket(ticket_id)
        if not ticket:
            return f"I couldn't find any ticket matching #{ticket_id}. Please double-check the number."

        return f"Your ticket #{ticket_id} is currently marked as: {ticket['status']}."
