from typing import Any


def render_outcome(payload: dict[str, Any]) -> str:
    lines = ["## Retraining outcome", ""]
    lines.append(f"- Triggered: **{payload['triggered']}** ({payload['reason']})")
    lines.append(f"- Champion: `{payload.get('champion_version') or 'none'}`")
    gate = payload.get("gate")
    if gate:
        champion_value = gate["champion_value"]
        champion_text = f"{champion_value:.4f}" if champion_value is not None else "n/a"
        improvement = gate["improvement"]
        improvement_text = f"{improvement:+.2%}" if improvement is not None else "n/a"
        lines += [
            f"- Candidate: `{gate['candidate_version']}`",
            "",
            f"| {gate['metric']} | candidate | champion | improvement | promoted |",
            "|---|---|---|---|---|",
            f"| test | {gate['candidate_value']:.4f} | {champion_text} | "
            f"{improvement_text} | **{gate['promoted']}** |",
            "",
            f"{gate['reason']}",
        ]
    return "\n".join(lines) + "\n"
