"""
eval_llm.py
===========
Evaluate a local HuggingFace model (base or GRPO-trained) against the
IntelliCredit environment server and save results in the SAME format as
baseline_results.json — so compare_results.py works directly.

Usage:
    # Evaluate the GRPO-trained model (default):
    python eval_llm.py

    # Evaluate the base model:
    python eval_llm.py --model mistralai/Mistral-7B-Instruct-v0.3 --out base_results.json

    # Evaluate the trained model explicitly:
    python eval_llm.py --model vssksn/intellicredit-mistral-7b-grpo --out grpo_results.json

    # Then compare:
    python compare_results.py --baseline baseline_results.json --after grpo_results.json

Requirements:
    pip install transformers torch accelerate
    Environment server must be running at ENV_URL (default: http://localhost:7860)
"""

from __future__ import annotations

import argparse
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ENV_URL         = os.getenv("ENV_URL", "http://localhost:7860")
DEFAULT_MODEL   = "vssksn/intellicredit-mistral-7b-grpo"
BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.3"

TEMPERATURE     = 0.2
MAX_NEW_TOKENS  = 128    # We only need APPROVE/CONDITIONAL/REJECT (+ brief reasoning)

TASK_CONFIGS = {
    "task1": {"num_steps": 5,  "description": "Easy — Clean profiles"},
    "task2": {"num_steps": 8,  "description": "Medium — Forensic alerts"},
    "task3": {"num_steps": 12, "description": "Medium-Hard — Macro shocks + missing data"},
}

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
Do NOT include explanations or any other text.
""").strip()

ACTION_MAP = {
    "approve":             0,
    "conditional":         1,
    "reject":              2,
    "conditional_approve": 1,
    "conditional approve": 1,
}
DECISION_LABELS = {0: "APPROVE", 1: "CONDITIONAL", 2: "REJECT"}


# ─────────────────────────────────────────────────────────────────────────────
# Environment HTTP Client  (same as inference.py)
# ─────────────────────────────────────────────────────────────────────────────

class EnvHTTPClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health(self) -> bool:
        try:
            return self.session.get(f"{self.base}/health", timeout=15).status_code == 200
        except Exception:
            return False

    def reset(self, task_id: str, seed: int = 42, episode_id: str | None = None) -> dict:
        body: dict = {"seed": seed, "task_id": task_id}
        if episode_id:
            body["episode_id"] = episode_id
        r = self.session.post(f"{self.base}/reset", json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def step(self, action: dict, episode_id: str | None = None) -> dict:
        body: dict = {"action": action}
        if episode_id:
            body["episode_id"] = episode_id
        r = self.session.post(f"{self.base}/step", json=body, timeout=60)
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Local LLM Agent
# ─────────────────────────────────────────────────────────────────────────────

class LocalLLMAgent:
    """
    Wraps a local HuggingFace model (loaded via transformers pipeline)
    and provides a decision() method compatible with the environment loop.
    """

    def __init__(self, model_name: str):
        print(f"\n{'─'*60}")
        print(f"  Loading model: {model_name}")
        print(f"{'─'*60}")

        from transformers import pipeline, AutoTokenizer
        import torch

        self.model_name = model_name
        device = 0 if __import__("torch").cuda.is_available() else -1
        device_label = "GPU (CUDA)" if device == 0 else "CPU"
        print(f"  Device: {device_label}")

        # Load tokenizer separately to apply chat template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device=device,
            dtype=__import__("torch").bfloat16 if device == 0 else None,
            trust_remote_code=True,
        )
        # Override any saved generation_config that might conflict with our
        # max_new_tokens (e.g. a saved max_length=20 from GRPO training).
        from transformers import GenerationConfig
        self.pipe.model.generation_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        print(f"  ✅ Model ready\n")

    def decide(self, obs_data: dict) -> int:
        """
        Format the observation as a prompt, call the model, parse the decision.

        Returns: 0 (APPROVE), 1 (CONDITIONAL), 2 (REJECT)
        """
        user_prompt = self._build_prompt(obs_data)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            output = self.pipe(
                messages,
                # All generation params already set in generation_config above;
                # passing them again here causes the deprecation warning.
            )
            # pipeline returns list of dicts; extract the generated assistant turn
            generated = output[0]["generated_text"]
            if isinstance(generated, list):
                # Chat format: list of role/content dicts
                response_text = generated[-1].get("content", "")
            else:
                response_text = str(generated)
        except Exception as exc:
            print(f"  ⚠ LLM call failed: {exc} — defaulting to REJECT", file=sys.stderr)
            response_text = "REJECT"

        return self._parse(response_text)

    def _build_prompt(self, obs_data: dict) -> str:
        """Build user message from environment observation (same format as inference.py)."""
        app_summary = obs_data.get("application_summary", {})
        summary = app_summary.get("text_summary", "") if isinstance(app_summary, dict) else str(app_summary)

        portfolio_state = obs_data.get("portfolio_state", [0] * 10)
        alert_state     = obs_data.get("alert_state",     [0] * 5)

        alert_labels  = ["CC Spike", "Bounce Surge", "GST Filing Miss", "Adverse Media", "Credit Degradation"]
        active_alerts = [
            alert_labels[i]
            for i, v in enumerate(alert_state)
            if i < len(alert_labels) and isinstance(v, (int, float)) and v > 0.5
        ]
        alert_text = ", ".join(active_alerts) if active_alerts else "None"

        def safe_pct(lst, idx, default=0.0):
            try:    return float(lst[idx]) * 100
            except: return default

        portfolio_info = (
            f"\n── PORTFOLIO STATUS ──\n"
            f"  Capital Remaining: {safe_pct(portfolio_state, 0):.0f}%\n"
            f"  Capital Deployed:  {safe_pct(portfolio_state, 1):.0f}%\n"
            f"  NPA Rate:          {safe_pct(portfolio_state, 2):.1f}%\n"
            f"  CRAR:              {safe_pct(portfolio_state, 9):.1f}%\n"
            f"  Sector Conc.:      {safe_pct(portfolio_state, 4):.0f}%\n"
            f"  Portfolio Alerts:  {alert_text}\n"
        )
        return f"{summary}\n{portfolio_info}"

    @staticmethod
    def _parse(text: str) -> int:
        """Parse APPROVE / CONDITIONAL / REJECT from model output."""
        if not text:
            return 2
        lower = text.strip().lower()
        for key, val in ACTION_MAP.items():
            if key in lower:
                return val
        numbers = re.findall(r'\b[012]\b', lower)
        if numbers:
            return int(numbers[0])
        return 2   # default safe: REJECT


# ─────────────────────────────────────────────────────────────────────────────
# Episode Runner  (mirrors inference.py logic but uses local model)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env: EnvHTTPClient, agent: LocalLLMAgent,
                task_id: str, seed: int = 42) -> dict:
    config    = TASK_CONFIGS[task_id]
    num_steps = config["num_steps"]
    episode_id = f"{task_id}-{uuid.uuid4().hex[:8]}"

    print(f"\n  ── {task_id}: {config['description']} (seed={seed}) ──")

    reset_resp = env.reset(task_id=task_id, seed=seed, episode_id=episode_id)
    obs_data   = reset_resp.get("observation", reset_resp)
    done       = bool(reset_resp.get("done", False) or obs_data.get("done", False))

    rewards:     list[float] = []
    decisions:   list[str]  = []
    episode_score = 0.0
    breakdown:   dict = {}
    step = 0
    last_resp:   dict = {}

    while not done and step < num_steps:
        step += 1

        decision   = agent.decide(obs_data)
        action_str = DECISION_LABELS[decision]

        step_resp = env.step(action={"decision": decision}, episode_id=episode_id)
        last_resp = step_resp                              # keep reference for post-loop
        obs_data  = step_resp.get("observation", step_resp)
        reward    = float(step_resp.get("reward", 0.0) or 0.0)
        done      = bool(step_resp.get("done", False) or obs_data.get("done", False))

        rewards.append(reward)
        decisions.append(action_str)

        print(f"    Step {step:2d}: {action_str:<12} | reward={reward:+.2f} | "
              f"cumulative={sum(rewards):+.2f}")

        # Debug: print raw response keys on first step so we know the structure
        if step == 1:
            print(f"    [debug] resp keys: {list(step_resp.keys())}")

        if done:
            # episode_score may be at top level OR inside observation
            episode_score = (
                float(step_resp.get("episode_score") or 0.0)
                or float(obs_data.get("episode_score") or 0.0)
            )
            breakdown = (
                step_resp.get("score_breakdown")
                or obs_data.get("score_breakdown")
                or {}
            )

    # Post-loop fallback: try both locations
    if episode_score == 0.0:
        episode_score = (
            float(last_resp.get("episode_score") or 0.0)
            or float(obs_data.get("episode_score") or 0.0)
        )
    if not breakdown:
        breakdown = (
            last_resp.get("score_breakdown")
            or obs_data.get("score_breakdown")
            or {}
        )

    print(f"  Score: {episode_score:.4f} | Steps: {step}")

    return {
        "task_id":      task_id,
        "score":        round(episode_score, 4),
        "total_reward": round(sum(rewards), 4),
        "decisions":    decisions,
        "breakdown":    breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IntelliCredit LLM Evaluation")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"HF model name or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--seed",   type=int, default=42, help="Environment seed")
    parser.add_argument("--tasks",  nargs="+", default=["task1", "task2", "task3"],
                        choices=list(TASK_CONFIGS.keys()),
                        help="Tasks to evaluate (default: task1 task2 task3)")
    parser.add_argument("--env-url", default=ENV_URL,
                        help=f"Environment server URL (default: {ENV_URL})")
    parser.add_argument("--out",    default=None,
                        help="Output JSON path (default: auto-named from model)")
    args = parser.parse_args()

    # Auto-name output file
    if args.out is None:
        tag = "grpo" if "grpo" in args.model.lower() else "base"
        args.out = f"{tag}_results.json"

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         IntelliCredit — Local LLM Evaluation                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Model   : {args.model}")
    print(f"  Tasks   : {', '.join(args.tasks)}")
    print(f"  Seed    : {args.seed}")
    print(f"  Env URL : {args.env_url}")
    print(f"  Output  : {args.out}")

    # ── Check environment server ──────────────────────────────────────────
    env = EnvHTTPClient(args.env_url)
    print("\n  Waiting for environment server...", end="", flush=True)
    for attempt in range(20):
        if env.health():
            print(" ✅ Ready")
            break
        time.sleep(3)
        print(".", end="", flush=True)
    else:
        print(f"\n  ❌ Environment server not reachable at {args.env_url}/health")
        print("     Start it first: python -m server.app  (or check ENV_URL in .env)")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────
    agent = LocalLLMAgent(args.model)

    # ── Run episodes ──────────────────────────────────────────────────────
    results: dict = {}
    t_start = time.time()

    for task_id in args.tasks:
        try:
            result = run_episode(env, agent, task_id=task_id, seed=args.seed)
            results[task_id] = result
        except Exception as exc:
            print(f"\n  ❌ Task {task_id} failed: {exc}", file=sys.stderr)
            results[task_id] = {
                "task_id":      task_id,
                "score":        0.0,
                "total_reward": 0.0,
                "decisions":    [],
                "breakdown":    {},
            }

    elapsed = time.time() - t_start

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  RESULTS SUMMARY — {args.model.split('/')[-1]}")
    print(f"{'═'*60}")
    print(f"  {'Task':<8} {'Score':>7} {'Reward':>8} {'Steps':>6} {'Decision Counts'}")
    print(f"  {'─'*52}")
    for task_id, r in results.items():
        decs  = r["decisions"]
        n_app = decs.count("APPROVE")
        n_con = decs.count("CONDITIONAL")
        n_rej = decs.count("REJECT")
        print(f"  {task_id:<8} {r['score']:>7.4f} {r['total_reward']:>8.2f} "
              f"{len(decs):>6}   A:{n_app} C:{n_con} R:{n_rej}")

    avg_score  = sum(r["score"]        for r in results.values()) / max(len(results), 1)
    avg_reward = sum(r["total_reward"] for r in results.values()) / max(len(results), 1)
    print(f"  {'─'*52}")
    print(f"  {'AVG':<8} {avg_score:>7.4f} {avg_reward:>8.2f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'═'*60}")

    # ── Save JSON (matches baseline_results.json format exactly) ──────────
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ Saved → {args.out}")
    print(f"  Run comparison with:")
    print(f"     python compare_results.py --baseline baseline_results.json --after {args.out}")


if __name__ == "__main__":
    main()
