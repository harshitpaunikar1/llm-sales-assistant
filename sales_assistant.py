"""
LLM-powered sales assistant for B2B lead qualification and objection handling.
Uses an LLM (Ollama or Gemini) to generate context-aware sales responses.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class SalesContext:
    product_name: str
    product_description: str
    target_persona: str
    key_value_props: List[str]
    common_objections: Dict[str, str]  # objection -> response
    pricing_tiers: Optional[List[Dict]] = None
    competitor_notes: Optional[Dict[str, str]] = None


@dataclass
class ConversationTurn:
    role: str   # user, assistant
    content: str
    timestamp: float = field(default_factory=time.time)


class LLMBackend:
    """Abstracts over Ollama (local) and Gemini (cloud) LLM backends."""

    def __init__(self, provider: str = "ollama", model: str = "llama3",
                 gemini_api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model
        self.gemini_api_key = gemini_api_key
        self._gemini_client = None
        self._setup()

    def _setup(self) -> None:
        if self.provider == "gemini" and GEMINI_AVAILABLE and self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_client = genai.GenerativeModel(self.model or "gemini-1.5-flash")
        elif self.provider == "ollama" and not OLLAMA_AVAILABLE:
            logger.warning("ollama package not installed. Using stub responses.")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response given a list of {'role': ..., 'content': ...} messages."""
        if self.provider == "ollama" and OLLAMA_AVAILABLE:
            return self._ollama_generate(messages)
        elif self.provider == "gemini" and GEMINI_AVAILABLE and self._gemini_client:
            return self._gemini_generate(messages)
        else:
            return self._stub_generate(messages)

    def _ollama_generate(self, messages: List[Dict]) -> str:
        try:
            response = ollama.chat(model=self.model, messages=messages)
            return response["message"]["content"]
        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            return "I'm having trouble connecting to the model right now."

    def _gemini_generate(self, messages: List[Dict]) -> str:
        try:
            formatted = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            response = self._gemini_client.generate_content(formatted)
            return response.text
        except Exception as exc:
            logger.error("Gemini error: %s", exc)
            return "I'm having trouble connecting to the model right now."

    def _stub_generate(self, messages: List[Dict]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "price" in last_user.lower() or "cost" in last_user.lower():
            return "Our pricing is competitive and scales with your team size. Can I share our pricing guide?"
        elif "competitor" in last_user.lower():
            return "Great question. Let me walk you through what sets us apart from the competition."
        elif "demo" in last_user.lower():
            return "I'd love to set up a demo for you. What time works best this week?"
        return "Thank you for your question. Let me get back to you with the details."


class SalesAssistant:
    """
    Stateful conversational sales assistant.
    Maintains conversation history, injects product context into the system prompt,
    and handles objection recognition with explicit overrides.
    """

    SYSTEM_TEMPLATE = """You are a professional B2B sales assistant for {product_name}.

Product: {product_description}

Target persona: {target_persona}

Key value propositions:
{value_props}

Common objections and suggested responses:
{objections}

Guidelines:
- Be consultative, not pushy.
- Ask qualifying questions to understand the prospect's pain points.
- Tie product features to the prospect's specific use case.
- If asked about pricing, share the value before the price.
- Keep responses concise (2-4 sentences unless elaboration is requested).
"""

    def __init__(self, context: SalesContext, llm_backend: LLMBackend):
        self.context = context
        self.llm = llm_backend
        self.history: List[ConversationTurn] = []
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        value_props = "\n".join(f"- {v}" for v in self.context.key_value_props)
        objections = "\n".join(
            f"- Objection: '{k}' -> Response: '{v}'"
            for k, v in self.context.common_objections.items()
        )
        return self.SYSTEM_TEMPLATE.format(
            product_name=self.context.product_name,
            product_description=self.context.product_description,
            target_persona=self.context.target_persona,
            value_props=value_props,
            objections=objections,
        )

    def _detect_objection(self, user_input: str) -> Optional[str]:
        user_lower = user_input.lower()
        for objection in self.context.common_objections:
            if any(word in user_lower for word in objection.lower().split()):
                return self.context.common_objections[objection]
        return None

    def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant reply."""
        self.history.append(ConversationTurn(role="user", content=user_message))
        objection_response = self._detect_objection(user_message)
        if objection_response and len(self.history) < 3:
            reply = objection_response
        else:
            messages = [{"role": "system", "content": self._system_prompt}]
            for turn in self.history:
                messages.append({"role": turn.role, "content": turn.content})
            reply = self.llm.generate(messages)
        self.history.append(ConversationTurn(role="assistant", content=reply))
        return reply

    def get_conversation_summary(self) -> Dict:
        return {
            "turns": len(self.history),
            "user_turns": sum(1 for t in self.history if t.role == "user"),
            "last_user_message": next(
                (t.content for t in reversed(self.history) if t.role == "user"), None
            ),
        }

    def reset(self) -> None:
        self.history = []


if __name__ == "__main__":
    context = SalesContext(
        product_name="LogiTrack Pro",
        product_description="AI-powered yard and logistics management platform for enterprises.",
        target_persona="VP of Logistics or Supply Chain Director at companies with 500+ vehicles.",
        key_value_props=[
            "Real-time dock assignment optimization reducing dwell time by 35%",
            "Integrates with SAP, Oracle, and legacy WMS in under 2 weeks",
            "AI-driven predictive maintenance alerts to prevent trailer breakdown",
            "Mobile-first inspection app for yard jockeys with offline capability",
        ],
        common_objections={
            "expensive": "Our ROI calculator shows typical payback in under 6 months through fuel savings and dock efficiency.",
            "integration": "We have pre-built connectors for all major WMS and ERP systems with a 2-week onboarding guarantee.",
            "competitor": "Unlike legacy YMS vendors, we offer real-time AI optimization without the multi-year implementation timeline.",
        },
    )

    backend = LLMBackend(provider="stub")
    assistant = SalesAssistant(context=context, llm_backend=backend)

    conversations = [
        "Tell me more about your product.",
        "That sounds interesting. What about pricing, it seems expensive.",
        "How does it integrate with our existing SAP system?",
        "Can I get a demo?",
    ]

    print(f"Sales Assistant for: {context.product_name}\n{'='*50}")
    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        reply = assistant.chat(user_msg)
        print(f"Assistant: {reply}")

    print("\nConversation summary:", assistant.get_conversation_summary())
