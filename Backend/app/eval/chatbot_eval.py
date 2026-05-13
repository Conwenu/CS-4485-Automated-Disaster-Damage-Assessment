"""Eval harness for the parser.

Usage:
    python -m app.eval.chatbot_eval
    python -m app.eval.chatbot_eval --verbose
    python -m app.eval.chatbot_eval --intent GET_DAMAGE_FOR_LOCATION
    python -m app.eval.chatbot_eval --limit 10
"""

import argparse
import sys
import time
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

from app.services.query_parser import QueryParser
from app.eval.eval_cases import EVAL_CASES
from app.config import settings

FIELDS = [
    "intent",
    "city",
    "cities",
    "id",
    "ids",
    "scene_id",
    "top_k",
    "damage_level",
    "confidence_threshold",
    "direction",
    "status",
    "needs_external_knowledge",
    "needs_clarification",
    "is_follow_up",
    "is_answer_to_pending",
    "missing_param",
]


def _norm(value):
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return sorted(_norm(v) for v in value)
    return value


def _compare(expected, actual) -> bool:
    return _norm(expected) == _norm(actual)


def run_case(
    parser: QueryParser, case: Dict[str, Any]
) -> Tuple[bool, List[str], Dict[str, Any]]:
    history_turns = case.get("history") or []
    formatted = (
        "\n".join(f"{t['role'].upper()}: {t['content']}" for t in history_turns)
        or "(no prior turns)"
    )
    pending = case.get("pending")

    parsed = parser.parse(
        query=case["query"],
        history=formatted,
        pending_clarification=pending,
    )
    actual = parsed.model_dump()
    actual["intent"] = parsed.intent.value

    failures: List[str] = []
    for field in FIELDS:
        if field not in case["expected"]:
            continue
        exp = case["expected"][field]
        act = actual.get(field)
        if not _compare(exp, act):
            failures.append(f"  {field}: expected={exp!r} actual={act!r}")

    return (len(failures) == 0), failures, actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="Show passing cases too")
    ap.add_argument("--intent", help="Only run cases whose expected intent matches")
    ap.add_argument("--limit", type=int, help="Run only the first N matching cases")
    args = ap.parse_args()

    cases = EVAL_CASES
    if args.intent:
        cases = [c for c in cases if c["expected"].get("intent") == args.intent]
    if args.limit:
        cases = cases[: args.limit]

    print("Tier distribution:", Counter(c.get("tier") for c in cases))

    parser = QueryParser()

    start_time = time.time()

    total = 0
    passed = 0
    per_intent_total: Dict[str, int] = defaultdict(int)
    per_intent_pass: Dict[str, int] = defaultdict(int)
    per_tier_total = defaultdict(int)
    per_tier_pass = defaultdict(int)

    for i, case in enumerate(cases, start=1):
        intent = case["expected"].get("intent", "UNKNOWN")
        per_intent_total[intent] += 1
        tier = case.get("tier", "1")
        per_tier_total[tier] += 1
        total += 1

        try:
            ok, failures, actual = run_case(parser, case)
        except Exception as e:
            ok = False
            failures = [f"  exception: {type(e).__name__}: {e}"]
            actual = {}

        if ok:
            passed += 1
            per_intent_pass[intent] += 1
            per_tier_pass[tier] += 1
            if args.verbose:
                print(f"[{i:3d}] PASS  [{intent}]  {case['query'][:80]}")
        else:
            print(f"[{i:3d}] FAIL  [{intent}]  {case['query'][:80]}")
            for f in failures:
                print(f)
            if actual.get("reasoning"):
                print(f"  reasoning: {actual['reasoning']}")
            print()

        # Throttle for Gemini free tier (5 RPM). Skip on last case.
        if i < len(cases):
            time.sleep(settings.EVAL_REQUEST_DELAY_SECONDS)

    print()
    print("=" * 70)
    pct_overall = (passed / total * 100) if total else 0
    print(f"OVERALL: {passed}/{total} = {pct_overall:.1f}%")
    print("=" * 70)
    for intent in sorted(per_intent_total):
        p = per_intent_pass[intent]
        t = per_intent_total[intent]
        pct = (p / t * 100) if t else 0
        bar = "#" * int(pct // 5)
        print(f"  {intent:<30} {p:>3}/{t:<3} {pct:>5.1f}%  {bar}")
    print()

    # inside the loop:
    tier = case.get("tier", 1)
    per_tier_total[tier] += 1
    if ok:
        per_tier_pass[tier] += 1

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
