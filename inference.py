"""
IntelliCredit-CreditAppraisal-v1 — Inference Script
====================================================
Required environment variables:
    API_BASE_URL   LLM API endpoint  (default: HF Router)
    MODEL_NAME     Model identifier  (default: Llama-3.3-70B-Instruct)
    HF_TOKEN       Hugging Face / API key  (REQUIRED)
    ENV_URL        Environment URL   (default: http://localhost:7860)

Runtime < 20 minutes on 2 vCPU / 8 GB RAM.
"""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import uuid
import time

import requests
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)

ENV_URL      = os.getenv("ENV_URL", "https://vssksn-intellicredit-openenv.hf.space")
SEED         = 42

TEMPERATURE  = 0.2
MAX_TOKENS   = 300

TASK_CONFIGS = {
    "task1": {"num_steps": 5,  "description": "Easy — Clean profiles"},
    "task2": {"num_steps": 8,  "description": "Medium — Forensic alerts"},
    "task3": {"num_steps": 12, "description": "Medium-Hard — Macro shocks + missing data"},
}

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── Environment HTTP Client ──────────────────────────────────────

class EnvHTTPClient:
    """Plain HTTP client for the IntelliCredit environment server."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: str, seed: int = 42, episode_id: str | None = None) -> dict:
        """POST /reset → {observation, reward, done}"""
        body: dict = {"seed": seed, "task_id": task_id}
        if episode_id:
            body["episode_id"] = episode_id
        r = self.session.post(f"{self.base}/reset", json=body, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: dict, episode_id: str | None = None) -> dict:
        """POST /step → {observation, reward, done}"""
        body: dict = {"action": action}
        if episode_id:
            body["episode_id"] = episode_id
        r = self.session.post(f"{self.base}/step", json=body, timeout=30)
        r.raise_for_status()
        return r.json()


# ── LLM Prompt ────────────────────────────────────────────────────

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


# ── Episode Runner ────────────────────────────────────────────────

def run_episode(env: EnvHTTPClient, task_id: str, seed: int = 42) -> dict:
    """Run a single task episode against the remote environment."""
    config = TASK_CONFIGS[task_id]
    num_steps = config["num_steps"]

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Task: {task_id} — {config['description']}", file=sys.stderr)
    print(f"  Steps: {num_steps} | Seed: {seed}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Generate a unique episode_id for session persistence across HTTP calls
    episode_id = f"{task_id}-{uuid.uuid4().hex[:8]}"

    # Reset environment
    reset_resp = env.reset(task_id=task_id, seed=seed, episode_id=episode_id)
    obs_data = reset_resp.get("observation", reset_resp)

    # Structured stdout marker
    print(f"[START] task={task_id}", flush=True)

    actions = []
    total_reward = 0.0
    step = 0
    t_start = time.time()

    done = reset_resp.get("done", False) or obs_data.get("done", False)

    while not done and step < num_steps:
        # Per-task 6-minute safety limit
        if time.time() - t_start > 360:
            print("  Time limit for this task reached", file=sys.stderr)
            break

        step += 1

        # Build prompt from observation
        summary = ""
        app_summary = obs_data.get("application_summary", {})
        if isinstance(app_summary, dict):
            summary = app_summary.get("text_summary", "")
        elif isinstance(app_summary, str):
            summary = app_summary

        # Portfolio context
        portfolio_state = obs_data.get("portfolio_state", [0]*10)
        alert_state = obs_data.get("alert_state", [0]*5)

        alert_labels = ["CC Spike", "Bounce Surge", "GST Filing Miss", "Adverse Media", "Credit Degradation"]
        active_alerts = [
            alert_labels[i]
            for i, v in enumerate(alert_state)
            if i < len(alert_labels) and isinstance(v, (int, float)) and v > 0.5
        ]
        alert_text = ", ".join(active_alerts) if active_alerts else "None"

        # Safely extract portfolio metrics
        def safe_pct(lst, idx, default=0.0):
            try:
                return lst[idx] * 100
            except (IndexError, TypeError):
                return default

        portfolio_info = (
            f"\n── PORTFOLIO STATUS ──\n"
            f"  Capital Remaining: {safe_pct(portfolio_state, 0):.0f}%\n"
            f"  Capital Deployed: {safe_pct(portfolio_state, 1):.0f}%\n"
            f"  NPA Rate: {safe_pct(portfolio_state, 2):.1f}%\n"
            f"  CRAR: {safe_pct(portfolio_state, 9):.1f}%\n"
            f"  Sector Diversification: {safe_pct(portfolio_state, 3):.0f}%\n"
            f"  Top Sector Concentration: {safe_pct(portfolio_state, 4):.0f}%\n"
            f"  Portfolio Alerts: {alert_text}\n"
            f"  Timestep: {step}/{num_steps}\n"
        )

        user_prompt = f"{summary}\n{portfolio_info}"

        # LLM call with error handling
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  ⚠ LLM call failed: {exc}. Defaulting to REJECT.", file=sys.stderr)
            response_text = "REJECT"

        decision = parse_decision(response_text)
        decision_labels = {0: "APPROVE", 1: "CONDITIONAL", 2: "REJECT"}

        # Step the environment
        try:
            step_resp = env.step(
                action={"decision": decision},
                episode_id=episode_id,
            )
            obs_data = step_resp.get("observation", step_resp)
            reward = float(step_resp.get("reward", 0.0) or 0.0)
            done = bool(step_resp.get("done", False) or obs_data.get("done", False))
        except Exception as exc:
            print(f"  Step error: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            break

        actions.append(decision)
        total_reward += reward

        print(f"[STEP] step={step} reward={reward}", flush=True)

        print(
            f"  Step {step}: {decision_labels[decision]} | "
            f"reward={reward:+.2f} | cumulative={total_reward:+.2f}",
            file=sys.stderr,
        )

        if done:
            episode_score = obs_data.get("episode_score", 0.0) or 0.0
            score_breakdown = obs_data.get("score_breakdown", {}) or {}
            print(f"\n  ── EPISODE COMPLETE ──", file=sys.stderr)
            print(f"  Final Score: {episode_score:.4f}", file=sys.stderr)
            if score_breakdown:
                for k, v in score_breakdown.items():
                    print(f"    {k}: {v}", file=sys.stderr)

    # Extract final score
    episode_score = obs_data.get("episode_score", 0.0) or 0.0
    score_breakdown = obs_data.get("score_breakdown", {}) or {}
    elapsed = time.time() - t_start

    # Structured stdout marker
    print(f"[END] task={task_id} score={episode_score} steps={step}", flush=True)

    return {
        "task_id": task_id,
        "score": episode_score,
        "total_reward": round(total_reward, 4),
        "decisions": actions,
        "steps": step,
        "elapsed_s": round(elapsed, 1),
        "breakdown": score_breakdown,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60, file=sys.stderr)
    print("  IntelliCredit-CreditAppraisal-v1 — LLM Baseline Agent", file=sys.stderr)
    print(f"  Model:   {MODEL_NAME}", file=sys.stderr)
    print(f"  API:     {API_BASE_URL}", file=sys.stderr)
    print(f"  Env URL: {ENV_URL}", file=sys.stderr)
    print("═" * 60, file=sys.stderr)

    env = EnvHTTPClient(ENV_URL)

    # Wait for environment to be ready
    print("\n  Waiting for environment...", file=sys.stderr)
    for i in range(30):
        if env.health():
            print("  Environment ready ✓", file=sys.stderr)
            break
        print(f"  Attempt {i+1}/30...", file=sys.stderr)
        time.sleep(2)
    else:
        print("  ERROR: Environment not reachable after 60s", file=sys.stderr)
        print(f"  Tried: {ENV_URL}/health", file=sys.stderr)
        sys.exit(1)

    results = []
    tasks_to_run = ["task1", "task2", "task3"]  # Minimum 3 required
    t_global = time.time()

    for task_id in tasks_to_run:
        # 18 minute global safety limit
        if time.time() - t_global > 1080:
            print("Global time limit reached", file=sys.stderr)
            break

        try:
            result = run_episode(env, task_id=task_id, seed=SEED)
            results.append(result)
        except Exception as exc:
            print(f"  Task {task_id} failed: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            results.append({
                "task_id": task_id,
                "score": 0.0,
                "total_reward": 0.0,
                "decisions": [],
                "steps": 0,
                "elapsed_s": 0.0,
                "breakdown": {},
            })

    # Summary (stderr only)
    print("\n" + "═" * 60, file=sys.stderr)
    print("  BASELINE RESULTS SUMMARY", file=sys.stderr)
    print("═" * 60, file=sys.stderr)
    for result in results:
        print(
            f"  {result['task_id']}: score={result['score']:.4f} | "
            f"reward={result['total_reward']:+.2f} | "
            f"steps={result['steps']} | "
            f"time={result['elapsed_s']:.1f}s",
            file=sys.stderr,
        )

    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"\n  Average Score: {avg_score:.4f}", file=sys.stderr)

    elapsed = time.time() - t_global
    status = "OK" if elapsed < 1200 else "OVER TIME LIMIT"
    print(f"\n  Total elapsed: {elapsed:.1f}s | Status: {status}", file=sys.stderr)
    print("═" * 60, file=sys.stderr)

    # Save results
    output = {
        "model": MODEL_NAME,
        "env_url": ENV_URL,
        "seed": SEED,
        "results": results,
        "average_score": round(sum(r["score"] for r in results) / max(len(results), 1), 4),
        "total_elapsed": round(elapsed, 1),
        "status": status,
    }

    try:
        with open("baseline_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print("  Saved: baseline_results.json", file=sys.stderr)
    except Exception as exc:
        print(f"  Warning: Could not save results: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
