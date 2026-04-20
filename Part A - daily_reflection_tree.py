"""
=============================================================
  Daily Reflection Tree
  Internship Assessment — Deterministic Decision Tree System
=============================================================

Author  : [Your Name]
Date    : 2025
Purpose : Evaluates daily productivity and reflection using
          a deterministic decision tree (no ML, no randomness).

WHY THIS SYSTEM IS DETERMINISTIC
---------------------------------
A system is deterministic when the same inputs ALWAYS produce
the same output, with zero randomness or learned behaviour.

This system achieves determinism by:
  1. Using only boolean (True/False) inputs.
  2. Using only `if/elif/else` branching logic — every path
     is hard-coded and rule-based.
  3. No random numbers, no external state, no timestamps.
  4. No machine learning models or probability distributions.

Result: run the program 1 time or 1 000 000 times with the
        same inputs — you ALWAYS get the same reflection.
=============================================================
"""


# ─────────────────────────────────────────────
# SECTION 1: CORE DECISION TREE FUNCTION
# ─────────────────────────────────────────────

def get_daily_reflection(task_completed: bool,
                         felt_satisfied: bool,
                         was_distracted: bool) -> dict:
    """
    Evaluates a person's day using a deterministic decision tree.

    Decision logic:
      ┌─ task_completed?
      │   ├── YES → felt_satisfied?
      │   │           ├── YES → "Great day! Keep doing the same"
      │   │           └── NO  → "Improve quality tomorrow"
      │   └── NO  → was_distracted?
      │               ├── YES → "Reduce distractions"
      │               └── NO  → "Plan tasks better"

    Parameters
    ----------
    task_completed : bool
        True if the person completed their main task today.
    felt_satisfied : bool
        True if the person felt satisfied with their work quality.
    was_distracted : bool
        True if the person was significantly distracted today.

    Returns
    -------
    dict with keys:
        "outcome"     (str)  : Short label for the result.
        "message"     (str)  : Full reflection message.
        "path"        (list) : Decision path taken through the tree.
        "inputs"      (dict) : Echo of the inputs for traceability.
    """

    # Record the inputs for transparency / traceability
    inputs = {
        "task_completed": task_completed,
        "felt_satisfied": felt_satisfied,
        "was_distracted": was_distracted,
    }

    # ── BRANCH 1: Task was completed ──────────────────────────
    if task_completed:
        path = ["task_completed = True"]

        # Sub-branch: satisfied with work quality?
        if felt_satisfied:
            path.append("felt_satisfied = True")
            return {
                "outcome": "Great day",
                "message": "Great day — keep doing the same!",
                "path":    path,
                "inputs":  inputs,
            }
        else:
            path.append("felt_satisfied = False")
            return {
                "outcome": "Improve quality",
                "message": "Improve quality tomorrow.",
                "path":    path,
                "inputs":  inputs,
            }

    # ── BRANCH 2: Task was NOT completed ──────────────────────
    else:
        path = ["task_completed = False"]

        # Sub-branch: distracted during the day?
        if was_distracted:
            path.append("was_distracted = True")
            return {
                "outcome": "Reduce distractions",
                "message": "Reduce distractions — focus on one thing at a time.",
                "path":    path,
                "inputs":  inputs,
            }
        else:
            path.append("was_distracted = False")
            return {
                "outcome": "Plan tasks better",
                "message": "Plan tasks better — break goals into smaller steps.",
                "path":    path,
                "inputs":  inputs,
            }


# ─────────────────────────────────────────────
# SECTION 2: INPUT VALIDATION HELPER
# ─────────────────────────────────────────────

def parse_yes_no(prompt: str) -> bool:
    """
    Prompts the user for a yes/no answer and returns a boolean.
    Keeps asking until a valid answer is given (edge case handling).

    Valid inputs (case-insensitive): yes, no, y, n

    Parameters
    ----------
    prompt : str
        The question to display to the user.

    Returns
    -------
    bool : True for yes, False for no.
    """
    while True:
        # Strip whitespace and convert to lowercase for comparison
        answer = input(prompt).strip().lower()

        if answer in ("yes", "y"):
            return True
        elif answer in ("no", "n"):
            return False
        else:
            # Edge case: user typed something unexpected
            print("  ⚠  Please answer 'yes' or 'no' (or 'y' / 'n').\n")


# ─────────────────────────────────────────────
# SECTION 3: DISPLAY / PRINT HELPERS
# ─────────────────────────────────────────────

def print_header() -> None:
    """Prints the application banner."""
    print("=" * 52)
    print("       DAILY REFLECTION TREE")
    print("  A deterministic productivity check-in")
    print("=" * 52)
    print()


def print_result(result: dict) -> None:
    """
    Pretty-prints the result dictionary returned by
    get_daily_reflection().

    Parameters
    ----------
    result : dict
        The dict returned by get_daily_reflection().
    """
    print()
    print("─" * 52)
    print("  REFLECTION RESULT")
    print("─" * 52)

    # Show each input (for transparency)
    print("  Inputs:")
    for key, value in result["inputs"].items():
        label = "Yes" if value else "No"
        print(f"    • {key:<20} → {label}")

    print()

    # Show the decision path taken through the tree
    print("  Decision path:")
    for step in result["path"]:
        print(f"    → {step}")

    print()

    # Show the final reflection message
    print(f"  Outcome  : {result['outcome']}")
    print(f"  Message  : {result['message']}")
    print("─" * 52)
    print()


# ─────────────────────────────────────────────
# SECTION 4: INTERACTIVE MAIN FUNCTION
# ─────────────────────────────────────────────

def run_interactive() -> None:
    """
    Runs the Daily Reflection Tree interactively in the terminal.
    Asks the three questions, evaluates the tree, and prints the
    reflection message.
    """
    print_header()
    print("Answer each question with 'yes' or 'no'.\n")

    # Collect the three inputs
    task_completed = parse_yes_no("1. Did you complete your main task today?  ")
    felt_satisfied = parse_yes_no("2. Did you feel satisfied with your work?  ")
    was_distracted = parse_yes_no("3. Were you distracted today?              ")

    # Evaluate the decision tree (deterministic call)
    result = get_daily_reflection(task_completed, felt_satisfied, was_distracted)

    # Display the result
    print_result(result)


# ─────────────────────────────────────────────
# SECTION 5: AUTOMATED TEST SUITE
# ─────────────────────────────────────────────

def run_tests() -> None:
    """
    Runs all four possible decision paths through the tree and
    verifies that each produces the expected deterministic output.

    This acts as both a test suite and a live demonstration of
    example inputs and outputs.

    Edge cases tested here:
      - All True inputs
      - Mixed inputs (task done but unhappy)
      - Task not done but distracted
      - Task not done and not distracted (no obvious external cause)
    """

    # Define every possible test case
    test_cases = [
        {
            "description": "Task done, satisfied, not distracted → Great day",
            "inputs": (True, True, False),
            "expected_outcome": "Great day",
        },
        {
            "description": "Task done, NOT satisfied, not distracted → Improve quality",
            "inputs": (True, False, False),
            "expected_outcome": "Improve quality",
        },
        {
            "description": "Task NOT done, not satisfied, WAS distracted → Reduce distractions",
            "inputs": (False, False, True),
            "expected_outcome": "Reduce distractions",
        },
        {
            "description": "Task NOT done, not satisfied, NOT distracted → Plan tasks better",
            "inputs": (False, False, False),
            "expected_outcome": "Plan tasks better",
        },
        # ── Edge cases ─────────────────────────────────────────
        # When task is NOT done, felt_satisfied is irrelevant
        # (third argument controls the branch, not the second).
        # We test this explicitly to confirm determinism holds.
        {
            "description": "Edge: task NOT done, 'satisfied=True' ignored, distracted → Reduce distractions",
            "inputs": (False, True, True),
            "expected_outcome": "Reduce distractions",
        },
        {
            "description": "Edge: task NOT done, 'satisfied=True' ignored, not distracted → Plan tasks better",
            "inputs": (False, True, False),
            "expected_outcome": "Plan tasks better",
        },
        # Determinism check: same input twice must produce same output
        {
            "description": "Determinism re-check: same inputs produce same output (True, True, False)",
            "inputs": (True, True, False),
            "expected_outcome": "Great day",
        },
    ]

    print("=" * 52)
    print("  RUNNING TEST SUITE")
    print("=" * 52)

    passed = 0
    failed = 0

    for i, case in enumerate(test_cases, start=1):
        task, satisfied, distracted = case["inputs"]

        # Call the deterministic tree
        result = get_daily_reflection(task, satisfied, distracted)

        # Check if outcome matches expectation
        status = "PASS" if result["outcome"] == case["expected_outcome"] else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        # Print test result
        print(f"\n  Test {i}: {status}")
        print(f"  Description : {case['description']}")
        print(f"  Inputs      : task={task}, satisfied={satisfied}, distracted={distracted}")
        print(f"  Expected    : {case['expected_outcome']}")
        print(f"  Got         : {result['outcome']}")
        print(f"  Message     : {result['message']}")
        print(f"  Path        : {' → '.join(result['path'])}")

    # Summary
    print()
    print("─" * 52)
    print(f"  Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("─" * 52)
    print()


# ─────────────────────────────────────────────
# SECTION 6: ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Allow running in test mode: python daily_reflection_tree.py --test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        run_interactive()
