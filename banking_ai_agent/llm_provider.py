"""
llm_provider.py
----------------
Central place that decides which language model backs the agents.

Two modes:

1. LIVE MODE - if ANTHROPIC_API_KEY (or OPENAI_API_KEY) is set in the
   environment, we build a real LangChain chat model (ChatAnthropic or
   ChatOpenAI) and every agent uses it for classification and generation.

2. OFFLINE / DEMO MODE - if no key is present, we fall back to a small
   deterministic "FakeChatModel" that mimics the LangChain chat interface
   (so the rest of the codebase doesn't need to know the difference) but
   uses keyword rules instead of an API call. This keeps the project
   runnable, testable, and demoable without any network access or paid
   API calls - useful for CI, grading, and offline demos.

Swap providers by editing get_llm() below, or by setting the
LLM_PROVIDER environment variable to "anthropic" / "openai" / "offline".
"""

import os
import re
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class OfflineRuleBasedChatModel(BaseChatModel):
    """
    A drop-in stand-in for a real chat model.

    It looks at the *last* human message content and returns a plausible
    response using simple keyword heuristics. This is intentionally simple:
    its only job is to let the multi-agent workflow run end-to-end without
    an API key so the orchestration logic can be developed, tested and
    demoed offline. Swap in ChatAnthropic/ChatOpenAI for production use.
    """

    NEGATIVE_WORDS = (
        "not", "hasn't", "hasnt", "still", "broken", "issue", "problem",
        "delay", "delayed", "wrong", "failed", "fail", "worst", "bad",
        "disappointed", "unhappy", "angry", "frustrat", "never arrived",
        "complaint", "poor", "slow", "error",
    )
    POSITIVE_WORDS = (
        "thanks", "thank you", "great", "awesome", "excellent",
        "happy", "appreciate", "good job", "well done", "love", "perfect",
        "satisfied", "helpful",
    )
    QUERY_WORDS = (
        "status", "check", "ticket", "track", "update on", "what is",
        "could you check", "any update", "?",
    )

    @property
    def _llm_type(self) -> str:
        return "offline-rule-based"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        last_human = ""
        system_hint = ""
        for m in messages:
            if m.type == "system":
                system_hint = m.content
            if m.type == "human":
                last_human = m.content

        text = self._route(system_hint, last_human)
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _route(self, system_hint: str, text: str) -> str:
        lowered = text.lower()

        if "classify" in system_hint.lower():
            has_ticket_number = bool(re.search(r"\b\d{6}\b", lowered))
            if has_ticket_number and any(w in lowered for w in self.QUERY_WORDS):
                return "QUERY"
            # Positive words are checked first: short thank-you messages often
            # mention the resolved problem ("sorted out my login issue"),
            # which would otherwise trip a generic negative keyword.
            if any(w in lowered for w in self.POSITIVE_WORDS):
                return "POSITIVE_FEEDBACK"
            if any(w in lowered for w in self.NEGATIVE_WORDS):
                return "NEGATIVE_FEEDBACK"
            if has_ticket_number or "?" in lowered:
                return "QUERY"
            return "POSITIVE_FEEDBACK"

        if "thank-you" in system_hint.lower() or "thank you" in system_hint.lower():
            return (
                "We're delighted to have helped, and we'll keep working "
                "hard to make every interaction this smooth."
            )

        if "empathetic" in system_hint.lower():
            return (
                "We're really sorry for the trouble this has caused you. "
                "Your case has been escalated to our support team and "
                "we'll make sure it's resolved as quickly as possible."
            )

        return "Thank you for reaching out. Let us know if there's anything else we can help with."

    def bind_tools(self, tools, **kwargs):
        # Not used in offline mode - agents call the model directly.
        return self


def get_llm() -> BaseChatModel:
    """
    Return a LangChain chat model.

    Priority:
      1. ANTHROPIC_API_KEY present  -> ChatAnthropic (claude-sonnet-4-6)
      2. OPENAI_API_KEY present     -> ChatOpenAI (gpt-4o-mini)
      3. Otherwise                 -> OfflineRuleBasedChatModel (demo mode)
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "offline":
        return OfflineRuleBasedChatModel()

    if os.getenv("ANTHROPIC_API_KEY") and provider in ("", "anthropic"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    if os.getenv("OPENAI_API_KEY") and provider in ("", "openai"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    return OfflineRuleBasedChatModel()
