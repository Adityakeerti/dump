"""Explainability for live signal decisions: named features, Q-value margins and attributions.

Attribution is gradient x input on the margin Q(switch) - Q(keep) of the online network, so a positive
value means that feature pushed the policy towards switching the phase.
"""

from typing import Any

import numpy as np
import torch

from citysense.config.schemas import RLConfig

FEATURE_NAMES: list[str] = [
    "arterial_queue",
    "cross_queue",
    "phase",
    "time_in_phase",
    "disruption_flag",
    "upstream_queue",
    "downstream_queue",
]

PHASE_LABELS: dict[int, str] = {0: "arterial green", 1: "cross-street green"}


def q_values(net: torch.nn.Module, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net(torch.tensor(obs, dtype=torch.float32)).numpy()


def attribute(net: torch.nn.Module, obs_row: np.ndarray, rl_cfg: RLConfig) -> list[dict[str, float]]:
    x = torch.tensor(obs_row, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    q = net(x)[0]
    margin = q[rl_cfg.action_switch] - q[rl_cfg.action_keep]
    margin.backward()
    contrib = (x.grad[0] * x[0]).detach().numpy()
    rows = [
        {"feature": FEATURE_NAMES[i], "value": float(obs_row[i]), "contribution": float(contrib[i])}
        for i in range(min(len(FEATURE_NAMES), len(obs_row)))
    ]
    return sorted(rows, key=lambda r: abs(r["contribution"]), reverse=True)


def describe(rec: dict[str, Any], rl_cfg: RLConfig) -> str:
    """Deterministic one-line explanation built only from the numbers in the decision record."""
    j = rec["junction"] + 1
    verb = "switched to " + PHASE_LABELS.get(rec["executed_phase"], "?") if rec["switched"] else (
        "kept " + PHASE_LABELS.get(rec["executed_phase"], "?")
    )
    head = f"J{j} {verb}"
    shield = rec.get("shield")
    if shield == "min_green":
        return f"{head}: policy asked to switch but the safety shield held min green ({rl_cfg.shield_min_green_sec:g}s)."
    if shield == "max_cycle":
        return f"{head}: safety shield forced a switch at max phase time ({rl_cfg.shield_max_cycle_sec:g}s)."
    mode = rec.get("mode")
    if mode == "explore":
        return (
            f"{head}: exploration (epsilon {rec['epsilon']:.2f}) picked a random action; "
            f"the greedy choice would have been {'switch' if rec['greedy'] == rl_cfg.action_switch else 'keep'}."
        )
    if mode == "exploit" and rec.get("q") is not None:
        q_keep, q_switch = rec["q"][rl_cfg.action_keep], rec["q"][rl_cfg.action_switch]
        drivers = ", ".join(
            f"{a['feature']} {'+' if a['contribution'] >= 0 else '-'}{abs(a['contribution']):.2f}"
            for a in (rec.get("attribution") or [])[:2]
        )
        return f"{head}: Q(switch) {q_switch:.2f} vs Q(keep) {q_keep:.2f}; main drivers: {drivers}."
    return f"{head}: {rec.get('rule', 'rule-based controller')}."
