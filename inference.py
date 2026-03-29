"""
IntelliCredit-CreditAppraisal-v1 — Inference Script
===================================================
MANDATORY VARIABLES:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier for inference
    HF_TOKEN       Your Hugging Face / API key

This script runs an LLM agent against all tasks and reports scores.
"""

import os
import re
import json
import textwrap
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env if it exists

from openai import OpenAI

# ─── Environment imports ──────────────────────────────────────
from server.intellicredit_env import IntelliCreditEnvironment, TASK_CONFIGS
from models import IntelliCreditAction

# ─── Configuration ────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

TEMPERATURE = 0.2
MAX_TOKENS = 300

SYSTEM_PROMPT = textwrap.dedent("""
You are an experienced Senior Credit Officer at an Indian MSME lending institution.
You are evaluating corporate credit applications one at a time.

For each application, review the financial profile, banking behavior, GST compliance,
management quality, and any forensic alerts carefully.

You must respond with EXACTLY ONE of these decisions:
- APPROVE (action 0): Strong financials, low risk, no red flags
- CONDITIONAL (action 1): Acceptable but with concerns requiring monitoring
- REJECT (action 2): High risk, hard rule violations, or severe red flags

IMPORTANT RULES:
- If DSCR < 1.0 → ALWAYS REJECT (regulatory requirement)
- If any RED forensic alert → ALWAYS REJECT
- If Cheque Bounce Rate > 25% → ALWAYS REJECT
- If CIBIL < 550 equivalent → Strong lean toward REJECT
- Consider portfolio diversification and capital constraints
- When financial data shows 'DATA UNAVAILABLE', treat this as increased uncertainty.
  Missing critical metrics (DSCR, Current Ratio, Collateral) should bias toward
  CONDITIONAL or REJECT unless other indicators are exceptionally strong.
- Consider portfolio alerts (CC Spike, Bounce Surge, etc.) as systemic risk signals.

Respond with ONLY the decision word: APPROVE, CONDITIONAL, or REJECT.
Do NOT include explanations.
""").strip()


ACTION_MAP = {
    "approve": 0,
    "conditional": 1,
    "reject": 2,
    "conditional_approve": 1,
    "conditional approve": 1,
}


def parse_decision(response_text: str) -> int:
    """Parse LLM response into action integer."""
    if not response_text:
        return 2  # Default to REJECT for safety

    text = response_text.strip().lower()

    # Direct match
    for key, val in ACTION_MAP.items():
        if key in text:
            return val

    # Number match
    numbers = re.findall(r'\b[012]\b', text)
    if numbers:
        return int(numbers[0])

    return 2  # Default conservative


def run_inference(task_id: str, client: OpenAI) -> Dict:
    """Run a single episode and return the score."""
    env = IntelliCreditEnvironment(task_id=task_id)
    obs = env.reset()

    print(f"\n{'='*60}")
    print(f"  Task: {task_id} — {TASK_CONFIGS[task_id]['description']}")
    print(f"  Steps: {TASK_CONFIGS[task_id]['num_steps']}")
    print(f"{'='*60}")

    actions = []
    total_reward = 0.0

    while not obs.done:
        # Build prompt from observation
        summary = obs.application_summary.text_summary

        # Portfolio context (GAP 13: include full portfolio state)
        alert_labels = ["CC Spike", "Bounce Surge", "GST Filing Miss", "Adverse Media", "Credit Degradation"]
        active_alerts = [
            alert_labels[i]
            for i, v in enumerate(obs.alert_state)
            if v > 0.5
        ]
        alert_text = ", ".join(active_alerts) if active_alerts else "None"

        portfolio_info = (
            f"\n── PORTFOLIO STATUS ──\n"
            f"  Capital Remaining: {obs.portfolio_state[0]*100:.0f}%\n"
            f"  Capital Deployed: {obs.portfolio_state[1]*100:.0f}%\n"
            f"  NPA Rate: {obs.portfolio_state[2]*100:.1f}%\n"
            f"  CRAR: {obs.portfolio_state[9]*100:.1f}%\n"
            f"  Sector Diversification: {obs.portfolio_state[3]*100:.0f}%\n"
            f"  Top Sector Concentration: {obs.portfolio_state[4]*100:.0f}%\n"
            f"  Portfolio Alerts: {alert_text}\n"
            f"  Timestep: {obs.timestep + 1}/{TASK_CONFIGS[task_id]['num_steps']}\n"
        )

        user_prompt = f"{summary}\n{portfolio_info}"

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  ⚠ LLM call failed: {exc}. Defaulting to REJECT.")
            response_text = "REJECT"

        decision = parse_decision(response_text)
        decision_labels = {0: "APPROVE", 1: "CONDITIONAL", 2: "REJECT"}

        action = IntelliCreditAction(decision=decision)
        obs = env.step(action)

        actions.append(decision)
        total_reward += obs.reward

        print(
            f"  Step {obs.timestep}: {decision_labels[decision]} | "
            f"reward={obs.reward:+.2f} | cumulative={total_reward:+.2f}"
        )

        if obs.done:
            print(f"\n  ── EPISODE COMPLETE ──")
            print(f"  Final Score: {obs.episode_score:.4f}")
            if obs.score_breakdown:
                for k, v in obs.score_breakdown.items():
                    print(f"    {k}: {v}")
            break

    return {
        "task_id": task_id,
        "score": obs.episode_score or 0.0,
        "total_reward": round(total_reward, 4),
        "decisions": actions,
        "breakdown": obs.score_breakdown or {},
    }


def main():
    """Run inference across all tasks."""
    print("═" * 60)
    print("  IntelliCredit-CreditAppraisal-v1 — LLM Baseline Agent")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API: {API_BASE_URL}")
    print("═" * 60)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = {}
    tasks_to_run = ["task1", "task2", "task3"]  # Minimum 3 required

    for task_id in tasks_to_run:
        result = run_inference(task_id, client)
        results[task_id] = result

    # Summary
    print("\n" + "═" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("═" * 60)
    for task_id, result in results.items():
        print(f"  {task_id}: score={result['score']:.4f} | reward={result['total_reward']:+.2f}")

    avg_score = sum(r["score"] for r in results.values()) / len(results)
    print(f"\n  Average Score: {avg_score:.4f}")
    print("═" * 60)

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
