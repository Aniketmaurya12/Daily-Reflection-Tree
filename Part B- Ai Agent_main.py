"""
main.py — Entry point for the Daily Reflection AI Agent
=========================================================
Run interactively   : python main.py
Run automated tests : python main.py --test
View past sessions  : python main.py --history [N]
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

from agent import ReflectionAgent


LOG_PATH = Path("logs/reflection_log.jsonl")


def run_interactive() -> None:
    """Start one live agent session."""
    agent = ReflectionAgent(log_path=LOG_PATH)
    agent.run()


def run_tests() -> None:
    """
    Automated test suite — exercises every rule-table path plus guardrails.
    Uses a headless version of the agent that bypasses stdin.
    """
    from agent import GuardrailEngine, DecisionEngine, ReasoningStep

    guardrail = GuardrailEngine()
    engine    = DecisionEngine()

    print("═" * 56)
    print("  AUTOMATED TEST SUITE")
    print("═" * 56)

    passed = failed = 0

    def check(label: str, actual, expected) -> None:
        nonlocal passed, failed
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {label}")
        if not ok:
            print(f"         expected : {expected!r}")
            print(f"         got      : {actual!r}")

    # ── Guardrail tests ──────────────────────────────────────────
    print("\n  Guardrail engine")
    print("  " + "─" * 40)

    cases_guardrail = [
        ("yes → valid True",          "yes",         True,  True),
        ("y → valid True",            "y",           True,  True),
        ("no → valid False",          "no",          False, True),
        ("n → valid False",           "n",           False, True),
        ("YES → case-insensitive",    "YES",         True,  True),
        ("maybe → rejected",          "maybe",       False, False),
        ("empty → rejected",          "",            False, False),
        ("101-char → rejected",       "y" * 101,     False, False),
        ("integer → rejected",        "1",           False, False),
        ("injection attempt",         "yes; rm -rf", False, False),
    ]

    for label, raw, exp_value, exp_valid in cases_guardrail:
        result = guardrail.validate(raw, "test")
        check(f"Guardrail: {label}", (result.value, result.is_valid), (exp_value, exp_valid))

    # ── Decision engine tests ─────────────────────────────────────
    print("\n  Decision engine (all 8 paths)")
    print("  " + "─" * 40)

    cases_decision = [
        ((True,  True,  False), "EXCELLENT",   100),
        ((True,  True,  True),  "REDUCE_DIST",  80),
        ((True,  False, False), "IMPROVE_QUL",  70),
        ((True,  False, True),  "IMPROVE_QUL",  50),
        ((False, True,  True),  "REDUCE_DIST",  20),
        ((False, True,  False), "IMPROVE_PLN",  20),
        ((False, False, True),  "REDUCE_DIST",  20),
        ((False, False, False), "IMPROVE_PLN",  20),
    ]

    for (task, sat, dist), exp_code, exp_score in cases_decision:
        inputs = {
            "task_completed": task,
            "felt_satisfied": sat,
            "was_distracted": dist,
        }
        decision = engine.evaluate(inputs)
        check(
            f"Decision ({int(task)},{int(sat)},{int(dist)}) → {exp_code} score={exp_score}",
            (decision.outcome_code, decision.score),
            (exp_code, exp_score)
        )

    # ── Determinism check ─────────────────────────────────────────
    print("\n  Determinism (same input × 3 runs)")
    print("  " + "─" * 40)

    inputs_det = {"task_completed": True, "felt_satisfied": True, "was_distracted": False}
    results = [engine.evaluate(inputs_det).outcome_code for _ in range(3)]
    check("All three runs identical", len(set(results)) == 1, True)

    # ── Summary ───────────────────────────────────────────────────
    total = passed + failed
    print(f"\n  {'─' * 40}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — all green")
    print("═" * 56)


def show_history(n: int = 5) -> None:
    """Print the last *n* sessions from the audit log."""
    agent = ReflectionAgent(log_path=LOG_PATH)
    records = agent.logger.read_all()

    if not records:
        print("  No sessions logged yet.")
        return

    recent = records[-n:]
    print(f"\n  Last {len(recent)} session(s) from audit log\n  {'─'*40}")

    for r in recent:
        d = r.get("decision", {})
        events = r.get("guardrail_events", [])
        print(
            f"  {r['timestamp']}  [{r['session_id']}]  "
            f"{d.get('outcome_code','?')}  score={d.get('score','?')}  "
            f"guardrail_hits={len(events)}"
        )
    print()


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--test" in args:
        run_tests()
    elif "--history" in args:
        idx = args.index("--history")
        n = int(args[idx + 1]) if idx + 1 < len(args) else 5
        show_history(n)
    else:
        run_interactive()
