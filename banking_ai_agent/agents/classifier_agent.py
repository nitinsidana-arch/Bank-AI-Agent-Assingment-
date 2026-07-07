"""
agents/classifier_agent.py
---------------------------
The Classifier Agent is the front door of the system. Every incoming
message passes through here first. Its only job is to decide which of the
three lanes the message belongs in:

    POSITIVE_FEEDBACK | NEGATIVE_FEEDBACK | QUERY

It is intentionally a *thin* agent - one focused LLM call constrained by a
strict system prompt, wrapped in output-normalisation so a slightly messy
model response ("Query.", "query", "QUERY\n") still routes correctly.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from llm_provider import get_llm

CLASSIFIER_SYSTEM_PROMPT = """You are a strict message classifier for a bank's \
customer support system. Classify the user's message into exactly one label:

- POSITIVE_FEEDBACK: the user is thanking the bank or praising a resolved issue.
- NEGATIVE_FEEDBACK: the user is complaining about an unresolved problem, \
  with no existing ticket number referenced.
- QUERY: the user is asking about the status of an existing ticket (often \
  includes a ticket number) or asking a general question.

Respond with ONLY the label, nothing else. Classify this message."""

VALID_LABELS = {"POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK", "QUERY"}


class ClassifierAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def classify(self, user_message: str) -> tuple[str, bool]:
        """
        Returns (label, used_fallback).
        used_fallback=True means the model's raw output didn't match a valid
        label and we had to default to QUERY - this is logged as a routing
        failure so the success-rate metric in the Evaluation tab reflects it.
        """
        messages = [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        result = self.llm.invoke(messages)
        label = result.content.strip().upper().replace(".", "")
        label = label.replace(" ", "_")

        used_fallback = label not in VALID_LABELS
        if used_fallback:
            label = "QUERY"
        return label, used_fallback
