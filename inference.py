"""
IntelliCredit-CreditAppraisal-v1 — Inference Script
====================================================
Required environment variables:
    API_BASE_URL   LLM API endpoint  (default: HF Router)
    MODEL_NAME     Model identifier  (default: Llama-3.3-70B-Instruct)
    HF_TOKEN       Hugging Face / API key  (REQUIRED)
    ENV_URL        Environment server URL  (default: HF Space URL)

stdout format (parsed by validator):
    [START] task=<task_id> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Runtime < 20 minutes on 2 vCPU / 8 GB RAM.
"""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
import uuid

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

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "intellicredit_credit_appraisal"
SEED         = 42
TEMPERATURE  = 0.2
MAX_TOKENS   = 300
SUCCESS_THRESHOLD = 0.1  # minimum score to count as success

TASK_CONFIGS = {
    "task1": {"num_steps": 5,  "description": "Easy — Clean profiles"},
    "task2": {"num_steps": 8,  "description": "Medium — Forensic alerts"},
    "task3": {"num_steps": 12, "description": "Medium-Hard — Macro shocks + missing data"},
}

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── stdout helpers (validator parses these lines) ────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment HTTP Client ──────────────────────────────────────

class EnvHTTPClient:
    """Plain HTTP client for the IntelliCredit environment server."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/health", timeout=15)
            return r.status_code == 200
        except Exception as e:
            print(f"  Health check error: {e}", file=sys.stderr)
            return False

    def reset(self, task_id: str, seed: int = 42, episode_id: str | None = None) -> dict:
        """POST /reset → {observation, reward, done}"""
        body: dict = {"seed": seed, "task_id": task_id}
        if episode_id:
            body["episode_id"] = episode_id
        r = self.session.post(f"{self.base}/reset", json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def step(self, action: dict, episode_id: str | None = None) -> dict:
        """POST /step → {observation, reward, done}"""
        body: dict = {"action": action}
        if episode_id:
            body["episode_id"] = episode_id
        r = self.session.post(f"{self.base}/step", json=body, timeout=60)
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
    "approve":            0,
    "conditional":        1,
    "reject":             2,
    "conditional_approve":1,
    "conditional approve":1,
}
DECISION_LABELS = {0: "APPROVE", 1: "CONDITIONAL", 2: "REJECT"}


def parse_decision(response_text: str) -> int:
    """Parse LLM response into action integer (defaults to REJECT)."""
    if not response_text:
        return 2
    text = response_text.strip().lower()
    for key, val in ACTION_MAP.items():
        if key in text:
            return val
    numbers = re.findall(r'\b[012]\b', text)
    if numbers:
        return int(numbers[0])
    return 2


# ── Episode Runner ────────────────────────────────────────────────

def run_episode(env: EnvHTTPClient, task_id: str, seed: int = 42) -> dict:
    """Run a single task episode. [END] is always emitted via finally."""
    config   = TASK_CONFIGS[task_id]
    num_steps = config["num_steps"]

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Task: {task_id} — {config['description']}", file=sys.stderr)
    print(f"  Steps: {num_steps} | Seed: {seed}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Tracking state (must be accessible in finally)
    step          = 0
    rewards: list[float] = []
    episode_score = 0.0
    done          = False
    success       = False
    obs_data: dict = {}

    # Emit [START] immediately
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Generate unique episode_id for HTTP session persistence
        episode_id = f"{task_id}-{uuid.uuid4().hex[:8]}"

        reset_resp = env.reset(task_id=task_id, seed=seed, episode_id=episode_id)
        obs_data   = reset_resp.get("observation", reset_resp)
        done       = bool(reset_resp.get("done", False) or obs_data.get("done", False))

        t_start = time.time()

        while not done and step < num_steps:
            # Per-task 6-minute safety limit
            if time.time() - t_start > 360:
                print("  Time limit for this task reached", file=sys.stderr)
                break

            step += 1

            # ── Build prompt ──────────────────────────────────────
            app_summary = obs_data.get("application_summary", {})
            if isinstance(app_summary, dict):
                summary = app_summary.get("text_summary", "")
            else:
                summary = str(app_summary)

            portfolio_state = obs_data.get("portfolio_state", [0] * 10)
            alert_state     = obs_data.get("alert_state",     [0] * 5)

            alert_labels  = ["CC Spike", "Bounce Surge", "GST Filing Miss", "Adverse Media", "Credit Degradation"]
            active_alerts = [
                alert_labels[i]
                for i, v in enumerate(alert_state)
                if i < len(alert_labels) and isinstance(v, (int, float)) and v > 0.5
            ]
            alert_text = ", ".join(active_alerts) if active_alerts else "None"

            def safe_pct(lst: list, idx: int, default: float = 0.0) -> float:
                try:
                    return float(lst[idx]) * 100
                except (IndexError, TypeError, ValueError):
                    return default

            portfolio_info = (
                f"\n── PORTFOLIO STATUS ──\n"
                f"  Capital Remaining: {safe_pct(portfolio_state, 0):.0f}%\n"
                f"  Capital Deployed:  {safe_pct(portfolio_state, 1):.0f}%\n"
                f"  NPA Rate:          {safe_pct(portfolio_state, 2):.1f}%\n"
                f"  CRAR:              {safe_pct(portfolio_state, 9):.1f}%\n"
                f"  Sector Conc.:      {safe_pct(portfolio_state, 4):.0f}%\n"
                f"  Portfolio Alerts:  {alert_text}\n"
                f"  Timestep:          {step}/{num_steps}\n"
            )
            user_prompt = f"{summary}\n{portfolio_info}"

            # ── LLM call ─────────────────────────────────────────
            last_error: str | None = None
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
                last_error    = str(exc)
                response_text = "REJECT"
                print(f"  ⚠ LLM call failed: {exc}. Defaulting to REJECT.", file=sys.stderr)

            decision = parse_decision(response_text)
            action_str = DECISION_LABELS[decision]

            # ── Step the environment ──────────────────────────────
            try:
                step_resp = env.step(
                    action={"decision": decision},
                    episode_id=episode_id,
                )
                obs_data = step_resp.get("observation", step_resp)
                reward   = float(step_resp.get("reward", 0.0) or 0.0)
                done     = bool(step_resp.get("done", False) or obs_data.get("done", False))
            except Exception as exc:
                last_error = str(exc)
                reward     = 0.0
                done       = True   # treat step failure as terminal
                print(f"  Step error: {exc}", file=sys.stderr)
                import traceback; traceback.print_exc(file=sys.stderr)

            rewards.append(reward)

            # ── Emit [STEP] ───────────────────────────────────────
            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

            print(
                f"  Step {step}: {action_str} | "
                f"reward={reward:+.2f} | cumulative={sum(rewards):+.2f}",
                file=sys.stderr,
            )

            if done:
                episode_score = float(obs_data.get("episode_score") or 0.0)
                score_bd      = obs_data.get("score_breakdown") or {}
                print(f"\n  ── EPISODE COMPLETE ──", file=sys.stderr)
                print(f"  Final Score: {episode_score:.4f}", file=sys.stderr)
                for k, v in score_bd.items():
                    print(f"    {k}: {v}", file=sys.stderr)

        # If loop ended without natural done, try to grab score from last obs
        if not done or episode_score == 0.0:
            episode_score = float(obs_data.get("episode_score") or 0.0)

        success = episode_score >= SUCCESS_THRESHOLD

    except Exception as exc:
        # Catch-all: unexpected errors still reach finally → [END] always prints
        print(f"  Episode error: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)

    finally:
        # ── Always emit [END] ─────────────────────────────────────
        log_end(success=success, steps=step, score=episode_score, rewards=rewards)

    return {
        "task_id":      task_id,
        "score":        episode_score,
        "total_reward": round(sum(rewards), 4),
        "decisions":    [DECISION_LABELS.get(a, str(a)) for a in rewards],
        "steps":        step,
        "success":      success,
        "breakdown":    obs_data.get("score_breakdown") or {},
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

    # Wait for environment to be ready (20 × 3s = 60s max)
    print("\n  Waiting for environment...", file=sys.stderr)
    ready = False
    for i in range(20):
        if env.health():
            print("  Environment ready ✓", file=sys.stderr)
            ready = True
            break
        print(f"  Attempt {i+1}/20...", file=sys.stderr)
        time.sleep(3)

    if not ready:
        raise RuntimeError(f"Environment not reachable after 60s. Tried: {ENV_URL}/health")

    results      = []
    tasks_to_run = ["task1", "task2", "task3"]
    t_global     = time.time()

    for task_id in tasks_to_run:
        # 18-minute global safety limit
        if time.time() - t_global > 1080:
            print("Global time limit reached", file=sys.stderr)
            break

        try:
            result = run_episode(env, task_id=task_id, seed=SEED)
            results.append(result)
        except Exception as exc:
            # run_episode's finally ensures [END] was already printed;
            # we still need to record a zero-score entry for the summary.
            print(f"  Task {task_id} outer error: {exc}", file=sys.stderr)
            results.append({
                "task_id":      task_id,
                "score":        0.0,
                "total_reward": 0.0,
                "decisions":    [],
                "steps":        0,
                "success":      False,
                "breakdown":    {},
            })

    # ── Summary (stderr only) ─────────────────────────────────────
    print("\n" + "═" * 60, file=sys.stderr)
    print("  BASELINE RESULTS SUMMARY", file=sys.stderr)
    print("═" * 60, file=sys.stderr)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(
            f"  {status} {r['task_id']}: score={r['score']:.4f} | "
            f"reward={r['total_reward']:+.2f} | steps={r['steps']}",
            file=sys.stderr,
        )

    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"\n  Average Score: {avg:.4f}", file=sys.stderr)

    elapsed = time.time() - t_global
    status  = "OK" if elapsed < 1200 else "OVER TIME LIMIT"
    print(f"\n  Total elapsed: {elapsed:.1f}s | Status: {status}", file=sys.stderr)
    print("═" * 60, file=sys.stderr)

    # ── Save results ──────────────────────────────────────────────
    output = {
        "model":         MODEL_NAME,
        "env_url":       ENV_URL,
        "benchmark":     BENCHMARK,
        "seed":          SEED,
        "results":       results,
        "average_score": round(sum(r["score"] for r in results) / max(len(results), 1), 4),
        "total_elapsed": round(elapsed, 1),
        "status":        status,
    }

    try:
        with open("baseline_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print("  Saved: baseline_results.json", file=sys.stderr)
    except Exception as exc:
        print(f"  Warning: Could not save results: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
