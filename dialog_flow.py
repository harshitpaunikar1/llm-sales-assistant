"""
Dialog flow manager for the LLM sales assistant.
Tracks qualification stage, intent detection, and CRM update triggers.
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SalesStage(str, Enum):
    DISCOVERY = "discovery"
    QUALIFICATION = "qualification"
    DEMO_REQUESTED = "demo_requested"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class Intent(str, Enum):
    PRICING_INQUIRY = "pricing_inquiry"
    DEMO_REQUEST = "demo_request"
    COMPETITOR_MENTION = "competitor_mention"
    OBJECTION = "objection"
    POSITIVE_SIGNAL = "positive_signal"
    CHURN_RISK = "churn_risk"
    GENERAL = "general"


INTENT_PATTERNS = {
    Intent.PRICING_INQUIRY: [r"\bprice\b", r"\bcost\b", r"\bbudget\b", r"\bhow much\b", r"\bexpensive\b"],
    Intent.DEMO_REQUEST: [r"\bdemo\b", r"\btrial\b", r"\bpilot\b", r"\btest drive\b", r"\bsee it\b"],
    Intent.COMPETITOR_MENTION: [r"\bcompetitor\b", r"\balternative\b", r"\bvs\b", r"\bcompare\b",
                                  r"\bsap\b", r"\boracle\b", r"\bjda\b", r"\bmanhattan\b"],
    Intent.OBJECTION: [r"\bnot interested\b", r"\bno budget\b", r"\bnot now\b",
                        r"\btoo expensive\b", r"\bnot a fit\b"],
    Intent.POSITIVE_SIGNAL: [r"\bimpressed\b", r"\binterested\b", r"\bsounds good\b",
                               r"\bwant to move forward\b", r"\blet.s do it\b"],
    Intent.CHURN_RISK: [r"\bcancel\b", r"\bdisappointed\b", r"\bnot happy\b",
                         r"\bswitching\b", r"\bmoving on\b"],
}


@dataclass
class QualificationScore:
    budget: Optional[bool] = None    # Does the prospect have budget?
    authority: Optional[bool] = None  # Are they a decision maker?
    need: Optional[bool] = None       # Is there a clear need?
    timeline: Optional[str] = None    # When do they want to decide?

    def bant_score(self) -> int:
        """BANT score out of 4 criteria confirmed."""
        return sum([
            1 if self.budget is True else 0,
            1 if self.authority is True else 0,
            1 if self.need is True else 0,
            1 if self.timeline is not None else 0,
        ])

    def is_qualified(self) -> bool:
        return self.bant_score() >= 3


@dataclass
class DialogState:
    session_id: str
    lead_id: str
    stage: SalesStage = SalesStage.DISCOVERY
    qualification: QualificationScore = field(default_factory=QualificationScore)
    intents_seen: List[Intent] = field(default_factory=list)
    topics_covered: List[str] = field(default_factory=list)
    turns: int = 0
    last_intent: Optional[Intent] = None
    crm_updates: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class IntentDetector:
    """Detects sales intents from user message text using regex patterns."""

    def detect(self, text: str) -> List[Intent]:
        text_lower = text.lower()
        detected = []
        for intent, patterns in INTENT_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                detected.append(intent)
        return detected if detected else [Intent.GENERAL]

    def primary_intent(self, text: str) -> Intent:
        detected = self.detect(text)
        priority = [Intent.DEMO_REQUEST, Intent.OBJECTION, Intent.PRICING_INQUIRY,
                    Intent.COMPETITOR_MENTION, Intent.POSITIVE_SIGNAL, Intent.CHURN_RISK,
                    Intent.GENERAL]
        for intent in priority:
            if intent in detected:
                return intent
        return Intent.GENERAL


class QualificationExtractor:
    """Updates qualification scores based on conversation signals."""

    BUDGET_SIGNALS = [r"\bwe have budget\b", r"\bapproved\b", r"\bfunded\b", r"\bROI\b"]
    NO_BUDGET_SIGNALS = [r"\bno budget\b", r"\bfreeze\b", r"\bcuts\b"]
    AUTHORITY_SIGNALS = [r"\bI decide\b", r"\bmy team\b", r"\bI approve\b", r"\bdirector\b",
                          r"\bVP\b", r"\bhead of\b"]
    NEED_SIGNALS = [r"\bneed\b", r"\bproblem\b", r"\bchallenge\b", r"\bpain\b", r"\bstruggle\b"]
    TIMELINE_PATTERNS = [
        (r"\bthis quarter\b", "this_quarter"),
        (r"\bq[1-4]\b", "next_quarter"),
        (r"\bthis year\b", "this_year"),
        (r"\basap\b", "immediate"),
        (r"\bnext month\b", "next_month"),
    ]

    def update(self, text: str, state: DialogState) -> DialogState:
        text_lower = text.lower()
        if any(re.search(p, text_lower) for p in self.BUDGET_SIGNALS):
            state.qualification.budget = True
        elif any(re.search(p, text_lower) for p in self.NO_BUDGET_SIGNALS):
            state.qualification.budget = False
        if any(re.search(p, text_lower) for p in self.AUTHORITY_SIGNALS):
            state.qualification.authority = True
        if any(re.search(p, text_lower) for p in self.NEED_SIGNALS):
            state.qualification.need = True
        for pattern, label in self.TIMELINE_PATTERNS:
            if re.search(pattern, text_lower):
                state.qualification.timeline = label
                break
        return state


class DialogFlowManager:
    """
    Orchestrates sales conversation flow:
    intent detection, qualification tracking, stage transitions, and CRM event generation.
    """

    def __init__(self, crm_callback: Optional[Callable[[Dict], None]] = None):
        self.intent_detector = IntentDetector()
        self.qual_extractor = QualificationExtractor()
        self.crm_callback = crm_callback
        self._sessions: Dict[str, DialogState] = {}

    def get_or_create_state(self, session_id: str, lead_id: str) -> DialogState:
        if session_id not in self._sessions:
            self._sessions[session_id] = DialogState(session_id=session_id, lead_id=lead_id)
        return self._sessions[session_id]

    def _advance_stage(self, state: DialogState, intent: Intent) -> SalesStage:
        if intent == Intent.DEMO_REQUEST:
            return SalesStage.DEMO_REQUESTED
        if intent == Intent.POSITIVE_SIGNAL and state.qualification.is_qualified():
            return SalesStage.PROPOSAL
        if intent == Intent.OBJECTION:
            return state.stage
        return state.stage

    def process_turn(self, session_id: str, lead_id: str,
                     user_message: str) -> Tuple[DialogState, Intent]:
        """Process a user turn and return updated state and detected intent."""
        state = self.get_or_create_state(session_id, lead_id)
        state.turns += 1
        intent = self.intent_detector.primary_intent(user_message)
        state.last_intent = intent
        state.intents_seen.append(intent)
        state = self.qual_extractor.update(user_message, state)
        new_stage = self._advance_stage(state, intent)
        if new_stage != state.stage:
            crm_event = {
                "lead_id": lead_id,
                "event": "stage_change",
                "from_stage": state.stage.value,
                "to_stage": new_stage.value,
                "timestamp": time.time(),
                "trigger_intent": intent.value,
            }
            state.crm_updates.append(crm_event)
            if self.crm_callback:
                self.crm_callback(crm_event)
            state.stage = new_stage

        if state.qualification.is_qualified() and Intent.POSITIVE_SIGNAL not in state.intents_seen[:-1]:
            qual_event = {
                "lead_id": lead_id,
                "event": "qualified",
                "bant_score": state.qualification.bant_score(),
                "timestamp": time.time(),
            }
            state.crm_updates.append(qual_event)
            if self.crm_callback:
                self.crm_callback(qual_event)

        return state, intent

    def session_report(self, session_id: str) -> Optional[Dict]:
        state = self._sessions.get(session_id)
        if not state:
            return None
        return {
            "session_id": state.session_id,
            "lead_id": state.lead_id,
            "stage": state.stage.value,
            "turns": state.turns,
            "bant_score": state.qualification.bant_score(),
            "is_qualified": state.qualification.is_qualified(),
            "intents": [i.value for i in state.intents_seen],
            "crm_updates": len(state.crm_updates),
        }


if __name__ == "__main__":
    def crm_printer(event: Dict):
        print(f"  [CRM] {event['event']}: {json.dumps({k: v for k, v in event.items() if k != 'timestamp'})}")

    manager = DialogFlowManager(crm_callback=crm_printer)

    messages = [
        "Hi, we have budget approved for this year and I'm the VP of Logistics.",
        "We have a big problem with dock scheduling - we need to solve it ASAP.",
        "Can you set up a demo for us next week?",
        "The solution sounds good, let's move forward.",
    ]

    print("Dialog flow simulation:\n")
    for msg in messages:
        state, intent = manager.process_turn("session_001", "lead_xyz", msg)
        print(f"User: {msg}")
        print(f"  Intent: {intent.value} | Stage: {state.stage.value} | "
              f"BANT: {state.qualification.bant_score()}/4 | Qualified: {state.qualification.is_qualified()}")

    print("\nSession report:")
    print(json.dumps(manager.session_report("session_001"), indent=2))
