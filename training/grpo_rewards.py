"""
IntelliCredit v2 — GRPO Reward Functions (Phase 6, Step 6.3)
==============================================================
Four reward functions for GRPO training.

GRPO requires: reward_funcs that take (prompts, completions, **metadata)
and return list[float].

Functions:
  1. reward_correctness       — PD-based decision quality     [-2.0, +1.0]
  2. reward_hard_rule_compliance — RBI hard rule adherence     [-2.0, +0.5]
  3. reward_format_compliance — Output format quality          [-0.3, +0.3]
  4. reward_portfolio_awareness — Portfolio-state sensitivity  [-0.8, +0.3]

Total reward per completion = sum of all 4 functions.
GRPO compares completions within a group of N to compute relative advantage.
"""

import re
import sys
import os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.action_parser import parse_llm_output, ACTION_MAP


# ═══════════════════════════════════════════════════════════════
# HELPER: Extract action from completion text
# ═══════════════════════════════════════════════════════════════

def _extract_action(completion: str) -> tuple:
    """
    Parse the completion text and extract the action + parse metadata.

    Returns:
        (action_int, parse_type, parse_confidence)
    """
    parsed = parse_llm_output(completion)
    return parsed["action"], parsed["parse_type"], parsed["parse_confidence"]


def _extract_from_prompt(prompt: str, field: str, default=None):
    """Extract a value from the prompt text using regex."""
    if field == "npa_rate":
        m = re.search(r"NPA Rate:\s*([\d.]+)%", prompt)
        return float(m.group(1)) / 100 if m else (default or 0.0)
    elif field == "crar":
        m = re.search(r"CRAR:\s*([\d.]+)%", prompt)
        return float(m.group(1)) / 100 if m else (default or 0.18)
    elif field == "dscr":
        m = re.search(r"DSCR:\s*([\d.]+)x", prompt)
        return float(m.group(1)) if m else (default or 1.5)
    return default


def _has_hard_rule_in_prompt(prompt: str) -> List[str]:
    """Check if prompt contains hard rule triggers."""
    rules = []
    # Check for RED alerts
    if "🔴" in prompt or "[RED]" in prompt:
        rules.append("HR-03")
    # Check for low DSCR
    dscr = _extract_from_prompt(prompt, "dscr", 1.5)
    if dscr < 1.0:
        rules.append("HR-01")
    # Check for high cheque bounce
    m = re.search(r"Cheque Bounce Rate:\s*([\d.]+)%", prompt)
    if m and float(m.group(1)) > 25:
        rules.append("HR-04")
    return rules


# ═══════════════════════════════════════════════════════════════
# REWARD FUNCTION 1: CORRECTNESS
# ═══════════════════════════════════════════════════════════════

def reward_correctness(
    prompts: List[str],
    completions: List[str],
    ground_truth_pd: List[float] = None,
    optimal_action: List[int] = None,
    **kwargs,
) -> List[float]:
    """
    PD-based decision quality reward.

    Logic:
      - PD < 0.25 (good borrower):
          APPROVE → +1.0, CONDITIONAL → +0.3, REJECT → -0.3 (opportunity cost)
      - PD 0.25-0.45 (borderline):
          CONDITIONAL → +0.8, APPROVE → -0.5, REJECT → +0.2
      - PD >= 0.45 (risky):
          REJECT → +1.0, CONDITIONAL → +0.2, APPROVE → -2.0 (dangerous)

    Range: [-2.0, +1.0]
    """
    rewards = []
    for i, completion in enumerate(completions):
        action, _, _ = _extract_action(completion)
        pd = ground_truth_pd[i] if ground_truth_pd else 0.5

        if pd < 0.25:       # Good borrower
            if action == 0:   reward = 1.0
            elif action == 1: reward = 0.3
            else:             reward = -0.3   # Missed opportunity
        elif pd < 0.45:      # Borderline
            if action == 1:   reward = 0.8
            elif action == 2: reward = 0.2
            else:             reward = -0.5   # Risky approve
        else:                # High risk
            if action == 2:   reward = 1.0
            elif action == 1: reward = 0.2
            else:             reward = -2.0   # Dangerous approve

        rewards.append(reward)
    return rewards


# ═══════════════════════════════════════════════════════════════
# REWARD FUNCTION 2: HARD RULE COMPLIANCE
# ═══════════════════════════════════════════════════════════════

def reward_hard_rule_compliance(
    prompts: List[str],
    completions: List[str],
    hard_rules: List[list] = None,
    has_red_alerts: List[bool] = None,
    **kwargs,
) -> List[float]:
    """
    RBI hard rule adherence reward.

    Logic:
      - If hard rules triggered and agent REJECTS → +0.5
      - If hard rules triggered and agent APPROVES → -2.0
      - If hard rules triggered and agent CONDITIONAL → -1.0
      - If no hard rules → 0.0 (neutral)

    Range: [-2.0, +0.5]
    """
    rewards = []
    for i, completion in enumerate(completions):
        action, _, _ = _extract_action(completion)

        # Get hard rules from metadata (may arrive as JSON string from dataset)
        hr_raw = hard_rules[i] if hard_rules else []
        if isinstance(hr_raw, str):
            import json as _json
            try:
                hr = _json.loads(hr_raw)
            except Exception:
                hr = []
        else:
            hr = list(hr_raw) if hr_raw else []
        red = has_red_alerts[i] if has_red_alerts else False

        # Also check prompt text for hard rules
        prompt_hr = _has_hard_rule_in_prompt(prompts[i]) if not hr else []
        all_hr = list(set(hr + prompt_hr))

        if red and "HR-03" not in all_hr:
            all_hr.append("HR-03")

        if all_hr:
            if action == 2:   reward = 0.5    # Correctly rejected
            elif action == 1: reward = -1.0   # Should have fully rejected
            else:             reward = -2.0   # Dangerous violation
        else:
            reward = 0.0  # No hard rules — neutral

        rewards.append(reward)
    return rewards


# ═══════════════════════════════════════════════════════════════
# REWARD FUNCTION 3: FORMAT COMPLIANCE
# ═══════════════════════════════════════════════════════════════

def reward_format_compliance(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Output format quality reward.

    Logic:
      - submit_decision() with reasoning → +0.3
      - submit_decision() without reasoning → +0.1
      - Standalone keyword (APPROVE/REJECT) → 0.0
      - No recognizable decision → -0.3

    Range: [-0.3, +0.3]
    """
    rewards = []
    for completion in completions:
        _, parse_type, confidence = _extract_action(completion)

        if parse_type == "final_decision":
            reward = 0.3 if confidence > 0.8 else 0.1
        elif parse_type == "fallback_keyword":
            reward = 0.0
        elif parse_type == "tool_call":
            # Tool call without decision — slight penalty
            reward = -0.1
        else:  # default_reject
            reward = -0.3

        rewards.append(reward)
    return rewards


# ═══════════════════════════════════════════════════════════════
# REWARD FUNCTION 4: PORTFOLIO AWARENESS
# ═══════════════════════════════════════════════════════════════

def reward_portfolio_awareness(
    prompts: List[str],
    completions: List[str],
    npa_rate: List[float] = None,
    crar: List[float] = None,
    ground_truth_pd: List[float] = None,
    **kwargs,
) -> List[float]:
    """
    Portfolio-state sensitivity reward.

    Logic:
      - High NPA (>8%) + APPROVE risky loan → -0.5
      - High NPA (>8%) + REJECT → +0.3 (conservative, good)
      - Low CRAR (<14%) + APPROVE → -0.3
      - Good portfolio + approve good loan → +0.2

    Range: [-0.8, +0.3]
    """
    rewards = []
    for i, completion in enumerate(completions):
        action, _, _ = _extract_action(completion)
        reward = 0.0

        npa = npa_rate[i] if npa_rate else _extract_from_prompt(prompts[i], "npa_rate", 0.02)
        cr = crar[i] if crar else _extract_from_prompt(prompts[i], "crar", 0.18)
        pd = ground_truth_pd[i] if ground_truth_pd else 0.5

        # High NPA environment
        if npa > 0.08:
            if action == 0 and pd > 0.30:
                reward = -0.5  # Approving risky in stressed portfolio
            elif action == 2:
                reward = 0.3   # Conservative in stressed portfolio

        # Low CRAR buffer
        if cr < 0.14:
            if action == 0:
                reward -= 0.3  # Approving depletes thin capital buffer

        # Good portfolio + good loan
        if npa < 0.03 and cr > 0.16 and action == 0 and pd < 0.20:
            reward = 0.2  # Healthy approve in healthy portfolio

        reward = max(-0.8, min(0.3, reward))
        rewards.append(reward)
    return rewards


# ═══════════════════════════════════════════════════════════════
# COMBINED REWARD (convenience for GRPO)
# ═══════════════════════════════════════════════════════════════

def combined_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Sum of all 4 reward functions.
    This is the function passed to GRPOTrainer's reward_funcs.
    """
    r1 = reward_correctness(prompts, completions, **kwargs)
    r2 = reward_hard_rule_compliance(prompts, completions, **kwargs)
    r3 = reward_format_compliance(prompts, completions, **kwargs)
    r4 = reward_portfolio_awareness(prompts, completions, **kwargs)

    return [
        round(r1[i] + r2[i] + r3[i] + r4[i], 4)
        for i in range(len(completions))
    ]


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  GRPO Reward Functions — Self-Test")
    print("=" * 65)

    prompts = ["... NPA Rate: 2.0% ... CRAR: 18.0% ..."] * 5
    completions = [
        "submit_decision('APPROVE', 'Strong financials, DSCR 1.85x, clean record')",
        "submit_decision('REJECT', 'HR-01 triggered: DSCR below 1.0')",
        "submit_decision('CONDITIONAL', 'Approve with quarterly review covenants')",
        "REJECT",
        "I am not sure what to decide here...",
    ]
    metadata = {
        "ground_truth_pd": [0.10, 0.60, 0.30, 0.70, 0.50],
        "hard_rules": [[], ["HR-01"], [], ["HR-03"], []],
        "has_red_alerts": [False, False, False, True, False],
        "npa_rate": [0.02, 0.02, 0.02, 0.10, 0.02],
        "crar": [0.18, 0.18, 0.18, 0.13, 0.18],
    }

    print("\n  Testing individual reward functions:")
    r1 = reward_correctness(prompts, completions, **metadata)
    print(f"  R1 Correctness:    {r1}")
    r2 = reward_hard_rule_compliance(prompts, completions, **metadata)
    print(f"  R2 Hard Rules:     {r2}")
    r3 = reward_format_compliance(prompts, completions, **metadata)
    print(f"  R3 Format:         {r3}")
    r4 = reward_portfolio_awareness(prompts, completions, **metadata)
    print(f"  R4 Portfolio:      {r4}")

    total = combined_reward(prompts, completions, **metadata)
    print(f"\n  Combined:          {total}")

    # Verify ranges
    assert all(-2.5 <= t <= 2.5 for t in total), f"Out of range: {total}"
    assert r1[0] == 1.0, f"Good approve should be +1.0, got {r1[0]}"
    assert r1[3] == 1.0, f"High-PD reject should be +1.0, got {r1[3]}"
    assert r2[1] == 0.5, f"HR reject should be +0.5, got {r2[1]}"
    assert r3[0] == 0.3, f"Good format should be +0.3, got {r3[0]}"
    assert r3[4] == -0.3, f"Parse failure should be -0.3, got {r3[4]}"

    print("\n  All assertions passed ✓")
    print(f"\n{'='*65}")
    print(f"  GRPO Reward Functions Self-Test Complete ✓")
    print(f"{'='*65}")
