"""
orchestrator.py
-----------------
This is the "multi-agent orchestration" layer described in the capstone
brief: Classifier -> {Positive Feedback Handler | Negative Feedback Handler
| Query Handler}.

Design choice: a supervisor/router pattern.
Each agent (ClassifierAgent, FeedbackAgent, QueryAgent) is built on top of
LangChain's chat-model abstraction (langchain_core.language_models,
SystemMessage/HumanMessage) so swapping the underlying model - Anthropic,
OpenAI, or the offline demo model - never touches this file. The
Orchestrator itself plays the role of a LangChain "supervisor agent": it
inspects the Classifier's output and dispatches to exactly one downstream
agent, mirroring the hand-off pattern used in LangGraph's supervisor
architectures, without requiring a graph runtime for a workflow this
linear and small.

Every request that passes through here is logged to interaction_logs,
including a compact prompt trace (which system prompt was used) and a
success flag (False only if the classifier had to fall back to a default
label), so the Streamlit "Logs & Debugging" and "Evaluation" tabs have
real, meaningful data to show - not just a raw echo of inputs/outputs.
"""

from dataclasses import dataclass
from typing import Optional

from agents.classifier_agent import ClassifierAgent, CLASSIFIER_SYSTEM_PROMPT
from agents.feedback_agent import FeedbackAgent, THANK_YOU_SYSTEM_PROMPT, APOLOGY_SYSTEM_PROMPT
from agents.query_agent import QueryAgent
from database import db


@dataclass
class OrchestratorResult:
    classification: str
    agent_path: str
    response: str
    ticket_id: Optional[str] = None
    used_fallback: bool = False
    log_id: Optional[int] = None


class Orchestrator:
    def __init__(self, llm=None):
        # All three agents can share one underlying LLM instance.
        self.classifier = ClassifierAgent(llm=llm)
        self.feedback_agent = FeedbackAgent(llm=llm)
        self.query_agent = QueryAgent()

    def handle_message(self, user_message: str, customer_name: str = "Customer") -> OrchestratorResult:
        classification, used_fallback = self.classifier.classify(user_message)
        ticket_id = None
        prompt_trace = f"[Classifier system prompt]\n{CLASSIFIER_SYSTEM_PROMPT}"

        if classification == "POSITIVE_FEEDBACK":
            agent_path = "Classifier -> Positive Feedback Handler"
            response = self.feedback_agent.handle_positive(user_message, customer_name)
            prompt_trace += f"\n\n[Positive Feedback Handler system prompt]\n{THANK_YOU_SYSTEM_PROMPT}"

        elif classification == "NEGATIVE_FEEDBACK":
            agent_path = "Classifier -> Negative Feedback Handler"
            response, ticket_id = self.feedback_agent.handle_negative(user_message, customer_name)
            prompt_trace += f"\n\n[Negative Feedback Handler system prompt]\n{APOLOGY_SYSTEM_PROMPT}"

        else:  # QUERY
            agent_path = "Classifier -> Query Handler"
            response = self.query_agent.handle(user_message)
            prompt_trace += "\n\n[Query Handler] regex ticket-number extraction, no LLM call."

        log_id = db.log_interaction(
            user_message=user_message,
            classification=classification,
            agent_path=agent_path,
            response=response,
            prompt_trace=prompt_trace,
            ticket_id=ticket_id,
            success=not used_fallback,
        )

        return OrchestratorResult(
            classification=classification,
            agent_path=agent_path,
            response=response,
            ticket_id=ticket_id,
            used_fallback=used_fallback,
            log_id=log_id,
        )
