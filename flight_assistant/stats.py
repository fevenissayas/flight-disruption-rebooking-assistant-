"""Counts from finished turns: success, escalation, and intent."""


def summarize(results: list[dict]) -> dict:
    total = len(results)
    by_intent: dict[str, int] = {}
    escalated = 0
    rebook_total = 0
    rebook_success = 0
    for result in results:
        state = result.get("state") or {}
        intent = state.get("intent") or "unknown"
        by_intent[intent] = by_intent.get(intent, 0) + 1
        if state.get("escalated"):
            escalated += 1
        if intent == "rebook":
            rebook_total += 1
            if state.get("policy_passed") and not state.get("escalated"):
                rebook_success += 1
    return {
        "requests": total,
        "by_intent": by_intent,
        "escalation_rate": (escalated / total) if total else 0.0,
        "rebooking_success_rate": (rebook_success / rebook_total) if rebook_total else 0.0,
        "rebook_success": rebook_success,
        "rebook_total": rebook_total,
        "escalated": escalated,
    }


def format_summary(summary: dict) -> str:
    lines = [
        "## Statistics",
        "",
        f"Requests: {summary['requests']}",
        f"Rebooking success rate: {summary['rebooking_success_rate']:.0%} "
        f"({summary['rebook_success']} of {summary['rebook_total']})",
        f"Escalation rate: {summary['escalation_rate']:.0%} ({summary['escalated']} of {summary['requests']})",
        "Requests by intent:",
    ]
    for intent, count in sorted(summary["by_intent"].items()):
        lines.append(f"- {intent}: {count}")
    return "\n".join(lines)
