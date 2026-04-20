"""
agent.py — Daily Reflection AI Agent (Core Engine)
====================================================
Rule-based, deterministic, hallucination-safe agent that evaluates
daily productivity through structured Q&A and transparent reasoning.

Architecture
------------
  ReflectionAgent
    ├── GuardrailEngine   — validates every input before it touches logic
    ├── DecisionEngine    — pure rule-based tree (no ML, no randomness)
    ├── ReasoningEngine   — builds human-readable explanation chains
    └── AuditLogger       — appends every decision to a structured log file

Design guarantees
-----------------
  * Deterministic  : identical inputs → identical outputs, always
  * Transparent    : every output carries a full reasoning trace
  * Safe           : no input reaches decision logic until it clears guardrails
  * Auditable      : every session is written to reflection_log.jsonl
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class AgentQuestion:
    """A single question the agent can ask."""
    key: str           # internal identifier
    text: str          # text shown to the user
    allowed: tuple     # exhaustive set of valid answers


@dataclass
class ValidatedAnswer:
    """The result of passing a raw string through the guardrail engine."""
    raw: str
    normalised: str
    value: bool
    is_valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class ReasoningStep:
    """One step in the agent's reasoning chain."""
    condition: str
    result: str
    triggered: bool


@dataclass
class AgentDecision:
    """The final output produced by the decision engine."""
    outcome_code: str
    outcome_label: str
    advice: str
    score: int                          # 0-100 productivity score
    reasoning_chain: list[ReasoningStep]
    inputs: dict[str, bool]


@dataclass
class SessionRecord:
    """Everything written to the audit log for one agent run."""
    session_id: str
    timestamp: str
    raw_inputs: dict[str, str]
    validated_inputs: dict[str, bool]
    guardrail_events: list[dict]
    decision: dict
    agent_version: str = "1.0.0"


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — GUARDRAIL ENGINE
# Hallucination prevention starts here.
# No answer reaches the decision engine until it passes all checks.
# ══════════════════════════════════════════════════════════════════

class GuardrailEngine:
    """
    Validates every user answer before it enters decision logic.

    Guardrails implemented
    ----------------------
    G1  Type check    — input must be a string
    G2  Length check  — must be 1–50 characters (no empty, no paragraphs)
    G3  Encoding      — strips control characters that could corrupt logs
    G4  Allow-list    — only pre-approved tokens are accepted
    G5  No inference  — agent never guesses; invalid input = rejection

    These guardrails collectively prevent:
      • Prompt injection via long or structured inputs
      • Ambiguous answers that could be interpreted multiple ways
      • Downstream hallucination caused by unvalidated state
    """

    ALLOWED_YES = frozenset({"yes", "y"})
    ALLOWED_NO  = frozenset({"no", "n"})
    ALLOWED_ALL = ALLOWED_YES | ALLOWED_NO

    MAX_LENGTH = 50
    MIN_LENGTH = 1

    # Pattern that strips anything outside printable ASCII
    _SAFE_PATTERN = re.compile(r"[^\x20-\x7E]")

    def validate(self, raw: str, question_key: str) -> ValidatedAnswer:
        """
        Run all guardrails on *raw* and return a ValidatedAnswer.
        On any failure the answer is marked invalid — never coerced.
        """

        # G1 — type check
        if not isinstance(raw, str):
            return ValidatedAnswer(
                raw=str(raw), normalised="", value=False,
                is_valid=False,
                rejection_reason=f"G1: expected str, got {type(raw).__name__}"
            )

        # G3 — encoding sanitisation (done before length check so we measure clean text)
        sanitised = self._SAFE_PATTERN.sub("", raw).strip()

        # G2 — length check
        if len(sanitised) < self.MIN_LENGTH:
            return ValidatedAnswer(
                raw=raw, normalised="", value=False,
                is_valid=False,
                rejection_reason="G2: answer is empty after sanitisation"
            )
        if len(sanitised) > self.MAX_LENGTH:
            return ValidatedAnswer(
                raw=raw, normalised=sanitised[:self.MAX_LENGTH], value=False,
                is_valid=False,
                rejection_reason=f"G2: answer too long ({len(sanitised)} chars; max {self.MAX_LENGTH})"
            )

        normalised = sanitised.lower()

        # G4 — allow-list check (this is the primary hallucination guardrail)
        if normalised not in self.ALLOWED_ALL:
            return ValidatedAnswer(
                raw=raw, normalised=normalised, value=False,
                is_valid=False,
                rejection_reason=(
                    f"G4: '{normalised}' not in allowed set "
                    f"{sorted(self.ALLOWED_ALL)}. "
                    "Agent refuses to infer meaning from ambiguous input."
                )
            )

        # G5 — deterministic boolean mapping (no inference, no ML)
        boolean_value = normalised in self.ALLOWED_YES

        return ValidatedAnswer(
            raw=raw, normalised=normalised,
            value=boolean_value, is_valid=True
        )


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — DECISION ENGINE
# Pure rule-based tree. No randomness. No ML. No external calls.
# ══════════════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Maps validated boolean inputs to outcomes via an explicit rule table.

    Score formula (additive, each component is independent):
      task_completed   → +50 pts
      felt_satisfied   → +30 pts  (only meaningful when task done)
      not_distracted   → +20 pts

    Outcome table (exhaustive — all 8 boolean triples are covered):
    ┌──────────────┬────────────┬──────────────┬──────────────────────────────┐
    │ task_done    │ satisfied  │ distracted   │ outcome                      │
    ├──────────────┼────────────┼──────────────┼──────────────────────────────┤
    │ True         │ True       │ False        │ EXCELLENT   (score: 100)     │
    │ True         │ True       │ True         │ REDUCE_DIST (score:  80)     │
    │ True         │ False      │ False        │ IMPROVE_QUL (score:  70)     │
    │ True         │ False      │ True         │ IMPROVE_QUL (score:  50)     │
    │ False        │ *          │ True         │ REDUCE_DIST (score:  20)     │
    │ False        │ *          │ False        │ IMPROVE_PLN (score:  20)     │
    └──────────────┴────────────┴──────────────┴──────────────────────────────┘
    (* = irrelevant when task not completed)
    """

    # Rule table: (task_completed, felt_satisfied, was_distracted) → outcome
    # Each entry: code, label, advice, base_score
    _RULES: dict[tuple, tuple] = {
        (True,  True,  False): (
            "EXCELLENT",
            "Excellent productivity",
            "You completed your task, felt satisfied, and stayed focused. "
            "Keep this routine — it is working. Consider documenting what "
            "made today effective so you can replicate it.",
            100
        ),
        (True,  True,  True): (
            "REDUCE_DIST",
            "Reduce distractions",
            "Great output today — but distractions cost you focus time. "
            "Tomorrow: silence notifications for your deep-work block and "
            "use a time-boxing technique (e.g. Pomodoro) to stay on track.",
            80
        ),
        (True,  False, False): (
            "IMPROVE_QUL",
            "Improve work quality",
            "Task done and no distractions — but you weren't satisfied with "
            "the result. Identify the specific gap (speed? depth? accuracy?) "
            "and set a concrete quality target for tomorrow.",
            70
        ),
        (True,  False, True): (
            "IMPROVE_QUL",
            "Improve work quality",
            "You finished the task despite distractions, but quality suffered. "
            "Tackle both issues: schedule a focused block early in the day "
            "and define a clear quality bar before you start.",
            50
        ),
        (False, True,  True): (
            "REDUCE_DIST",
            "Reduce distractions",
            "Distractions blocked your main task today. Tomorrow: identify "
            "your top two distraction sources the night before and remove "
            "them proactively. Protect at least 90 minutes of uninterrupted time.",
            20
        ),
        (False, True,  False): (
            "IMPROVE_PLN",
            "Improve planning",
            "No major distractions, yet the task wasn't completed — which "
            "points to a planning or scoping issue. Break your main task into "
            "sub-tasks tonight, estimate each one, and set a realistic daily goal.",
            20
        ),
        (False, False, True): (
            "REDUCE_DIST",
            "Reduce distractions",
            "Both task completion and satisfaction missed today — distractions "
            "played a key role. Reset tomorrow with a written plan, a distraction-"
            "free environment, and a single achievable goal.",
            20
        ),
        (False, False, False): (
            "IMPROVE_PLN",
            "Improve planning",
            "No distractions but the task still wasn't completed and satisfaction "
            "is low — this is a planning signal. Tonight, scope tomorrow's main "
            "task so it fits in 60% of your available time, leaving room for surprises.",
            20
        ),
    }

    def evaluate(self, inputs: dict[str, bool]) -> AgentDecision:
        """
        Evaluate validated inputs and return a fully-reasoned AgentDecision.
        Raises ValueError if inputs are incomplete (defensive coding).
        """
        required = {"task_completed", "felt_satisfied", "was_distracted"}
        missing = required - inputs.keys()
        if missing:
            raise ValueError(f"DecisionEngine: missing inputs: {missing}")

        task = inputs["task_completed"]
        satisfied = inputs["felt_satisfied"]
        distracted = inputs["was_distracted"]

        lookup_key = (task, satisfied, distracted)
        code, label, advice, score = self._RULES[lookup_key]

        # Build transparent reasoning chain — every condition is explicit
        reasoning = [
            ReasoningStep(
                condition="task_completed == True",
                result="Task branch selected (+50 pts base)",
                triggered=task
            ),
            ReasoningStep(
                condition="felt_satisfied == True  [only scored if task done]",
                result="+30 pts for satisfaction" if (task and satisfied) else "0 pts",
                triggered=(task and satisfied)
            ),
            ReasoningStep(
                condition="was_distracted == False",
                result="+20 pts for focus" if not distracted else "0 pts (distracted)",
                triggered=not distracted
            ),
            ReasoningStep(
                condition=f"Rule lookup: {lookup_key}",
                result=f"→ {code}: '{label}'",
                triggered=True
            ),
        ]

        return AgentDecision(
            outcome_code=code,
            outcome_label=label,
            advice=advice,
            score=score,
            reasoning_chain=reasoning,
            inputs=inputs,
        )


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — AUDIT LOGGER
# Every session is appended to a JSON Lines file for full traceability.
# ══════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Appends one JSON Lines record per session to *log_path*.

    Format: one JSON object per line (JSONL / NDJ), so the file is both
    human-readable and machine-parseable without loading the whole file.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        # Ensure the log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: SessionRecord) -> None:
        """Append *record* to the log file as a single JSON line."""
        line = json.dumps(asdict(record), ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_all(self) -> list[dict]:
        """Return all past session records as a list of dicts."""
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip corrupted lines silently
        return records


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — REFLECTION AGENT (orchestrator)
# Wires all components together; owns the conversation loop.
# ══════════════════════════════════════════════════════════════════

class ReflectionAgent:
    """
    Top-level orchestrator for the Daily Reflection AI Agent.

    Conversation flow
    -----------------
      1. Greet user
      2. For each question:
           a. Display question
           b. Collect raw input
           c. Run GuardrailEngine.validate()
           d. If invalid → log guardrail event, re-prompt (max 3 attempts)
           e. If still invalid after 3 attempts → abort session safely
      3. Pass validated inputs to DecisionEngine.evaluate()
      4. Print decision + full reasoning chain
      5. Write SessionRecord to AuditLogger

    The agent never modifies, infers, or extrapolates beyond the
    validated answer. If validation fails, the session aborts — it
    never falls back to a "best guess".
    """

    VERSION = "1.0.0"
    MAX_RETRIES = 3

    QUESTIONS: list[AgentQuestion] = [
        AgentQuestion(
            key="task_completed",
            text="Did you complete your main task today?",
            allowed=("yes", "no", "y", "n"),
        ),
        AgentQuestion(
            key="felt_satisfied",
            text="Did you feel satisfied with the quality of your work?",
            allowed=("yes", "no", "y", "n"),
        ),
        AgentQuestion(
            key="was_distracted",
            text="Were you significantly distracted during the day?",
            allowed=("yes", "no", "y", "n"),
        ),
    ]

    def __init__(self, log_path: Path = Path("logs/reflection_log.jsonl")):
        self.guardrail = GuardrailEngine()
        self.decision_engine = DecisionEngine()
        self.logger = AuditLogger(log_path)

    # ── public entry point ────────────────────────────────────────

    def run(self) -> Optional[AgentDecision]:
        """
        Run one full agent session interactively.
        Returns the AgentDecision on success, or None if aborted.
        """
        session_id = str(uuid.uuid4())[:8]
        timestamp  = datetime.now().isoformat(timespec="seconds")

        self._print_header(session_id)

        raw_inputs:       dict[str, str]  = {}
        validated_inputs: dict[str, bool] = {}
        guardrail_events: list[dict]      = []

        # ── question loop ─────────────────────────────────────────
        for question in self.QUESTIONS:
            answer, events = self._ask_with_retries(question, session_id)
            guardrail_events.extend(events)

            if answer is None:
                # Max retries exhausted — abort safely
                self._print_abort(question.key)
                self._log_aborted_session(
                    session_id, timestamp,
                    raw_inputs, validated_inputs, guardrail_events
                )
                return None

            raw_inputs[question.key]       = answer.raw
            validated_inputs[question.key] = answer.value

        # ── decision ──────────────────────────────────────────────
        decision = self.decision_engine.evaluate(validated_inputs)
        self._print_decision(decision)

        # ── audit log ─────────────────────────────────────────────
        record = SessionRecord(
            session_id=session_id,
            timestamp=timestamp,
            raw_inputs=raw_inputs,
            validated_inputs=validated_inputs,
            guardrail_events=guardrail_events,
            decision={
                "outcome_code":  decision.outcome_code,
                "outcome_label": decision.outcome_label,
                "score":         decision.score,
                "advice":        decision.advice,
                "reasoning": [
                    {
                        "condition": s.condition,
                        "result":    s.result,
                        "triggered": s.triggered,
                    }
                    for s in decision.reasoning_chain
                ],
            },
            agent_version=self.VERSION,
        )
        self.logger.write(record)
        print(f"\n  Session {session_id} saved to audit log.")

        return decision

    # ── internal helpers ──────────────────────────────────────────

    def _ask_with_retries(
        self,
        question: AgentQuestion,
        session_id: str,
    ) -> tuple[Optional[ValidatedAnswer], list[dict]]:
        """
        Prompt the user up to MAX_RETRIES times.
        Returns (ValidatedAnswer, guardrail_events) or (None, events).
        """
        events: list[dict] = []

        for attempt in range(1, self.MAX_RETRIES + 1):
            raw = input(f"\n  Q: {question.text}\n     [yes / no] > ").strip()
            result = self.guardrail.validate(raw, question.key)

            if result.is_valid:
                return result, events

            # Record the rejection
            event = {
                "session_id":   session_id,
                "question_key": question.key,
                "attempt":      attempt,
                "raw_input":    raw,
                "rejection":    result.rejection_reason,
            }
            events.append(event)
            print(
                f"\n  [GUARDRAIL] Input rejected (attempt {attempt}/{self.MAX_RETRIES}).\n"
                f"  Reason: {result.rejection_reason}\n"
                f"  Please answer exactly 'yes' or 'no'."
            )

        return None, events

    # ── display helpers ───────────────────────────────────────────

    @staticmethod
    def _print_header(session_id: str) -> None:
        width = 56
        print("\n" + "═" * width)
        print("  Daily Reflection AI Agent  v1.0.0")
        print(f"  Session: {session_id}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("═" * width)
        print("  Answer each question with 'yes' or 'no'.")
        print("  The agent is rule-based and will never guess.\n")

    @staticmethod
    def _print_decision(d: AgentDecision) -> None:
        width = 56
        print("\n" + "─" * width)
        print("  AGENT DECISION")
        print("─" * width)
        print(f"  Outcome  : {d.outcome_label}")
        print(f"  Score    : {d.score}/100")
        print(f"\n  Advice\n  {'─'*6}")

        # Word-wrap advice at 50 chars
        words = d.advice.split()
        line, lines = [], []
        for w in words:
            if sum(len(x)+1 for x in line) + len(w) > 50:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for ln in lines:
            print(f"  {ln}")

        print(f"\n  Reasoning chain")
        print(f"  {'─'*15}")
        for i, step in enumerate(d.reasoning_chain, 1):
            triggered = "✓" if step.triggered else "✗"
            print(f"  {i}. [{triggered}] {step.condition}")
            print(f"       → {step.result}")
        print("─" * width)

    @staticmethod
    def _print_abort(question_key: str) -> None:
        print(
            f"\n  [AGENT ABORT] Max retries exceeded for '{question_key}'.\n"
            "  The agent cannot proceed without a valid answer.\n"
            "  Session has been logged and terminated safely."
        )

    def _log_aborted_session(
        self,
        session_id, timestamp,
        raw_inputs, validated_inputs, guardrail_events
    ) -> None:
        record = SessionRecord(
            session_id=session_id,
            timestamp=timestamp,
            raw_inputs=raw_inputs,
            validated_inputs=validated_inputs,
            guardrail_events=guardrail_events,
            decision={"outcome_code": "ABORTED", "reason": "max retries exceeded"},
            agent_version=self.VERSION,
        )
        self.logger.write(record)
