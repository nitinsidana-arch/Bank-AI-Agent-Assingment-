"""
agents/feedback_agent.py
-------------------------
Handles both branches of "feedback": positive and negative.

Positive branch: purely generative - ask the LLM for a warm, personalised
thank-you message.

Negative branch: this is the one branch with a real side effect - it opens
a new support ticket in the database (via database.db) and returns an
empathetic acknowledgement that includes the new ticket number. The LLM
is used for the wording only; the ticket creation itself is deterministic
code, not something we trust the model to "decide" to do correctly.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from llm_provider import get_llm
from database import db

THANK_YOU_SYSTEM_PROMPT = """You are a warm, professional banking support \
assistant. The customer just left positive feedback. Write a short (1-2 \
sentence) personalized thank-you message. Do not invent details about their \
issue that they did not mention."""

APOLOGY_SYSTEM_PROMPT = """You are an empathetic banking support assistant. \
The customer just left negative feedback about an unresolved issue, and a \
new support ticket has already been created for them. Write a short (1-2 \
sentence) empathetic message. You MUST include the exact placeholder \
{ticket_id} in your reply so the ticket number can be inserted."""


class FeedbackAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def handle_positive(self, user_message: str, customer_name: str = "Customer") -> str:
        messages = [
            SystemMessage(content=THANK_YOU_SYSTEM_PROMPT),
            HumanMessage(content=f"Customer name: {customer_name}\nMessage: {user_message}"),
        ]
        result = self.llm.invoke(messages)
        text = result.content.strip()
        # Guarantee the required format is always present even if the model
        # phrases things differently - the assignment spec requires this
        # exact acknowledgement pattern.
        if customer_name not in text:
            text = f"Thank you for your kind words, {customer_name}! {text}"
        return text

    def handle_negative(self, user_message: str, customer_name: str = "Customer") -> str:
        ticket_id = db.create_ticket(message=user_message, customer_name=customer_name)

        messages = [
            SystemMessage(content=APOLOGY_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        result = self.llm.invoke(messages)
        text = result.content.strip()

        if "{ticket_id}" in text:
            text = text.format(ticket_id=ticket_id)
        if ticket_id not in text:
            text = (
                f"We apologize for the inconvenience. A new ticket "
                f"#{ticket_id} has been generated, and our team will follow "
                f"up shortly."
            )
        return text, ticket_id
