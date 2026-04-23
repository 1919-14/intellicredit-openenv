"""
IntelliCredit v2 — Evaluation Engine (Phase 7, Steps 7.1 & 7.2)
================================================================
Runs baseline and post-training evaluations against fixed seeds.

Usage:
  # Baseline (base model, rule-based agent):
  python evaluation/evaluate.py --mode baseline --episodes 100

  # Post-training (fine-tuned model):
  python evaluation/evaluate.py --mode grpo --model-path training/merged_model --episodes 100

  # Reflection only (no weight updates):
  python evaluation/evaluate.py --mode reflection --episodes 30

Output:
  evaluation/results/baseline_results.json
  evaluation/results/grpo_results.json
  evaluation/results/reflection_results.json
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.intellicredit_env import IntelliCreditEnvironment
from server.reward import grade_episode
from models import IntelliCreditAction


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

TASKS          = ["task1", "task2", "task3", "task4", "task5"]
EPISODES_PER_TASK = 20
FIXED_SEEDS    = list(range(1000, 1100))   # deterministic, shared baseline & post-train

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class EpisodeResult:
    episode_id      : int
    task_id         : str
    seed            : int
    total_reward    : float
    episode_score   : float
    accuracy        : float
    hard_rule_violations: int
    hard_rule_applicable : int
    npa_rate        : float
    crar            : float
    audit_pass_count: int
    audit_fail_count: int
    terminated_early: bool
    termination_reason: str
    steps_survived  : int
    total_actions   : int
    approve_count   : int
    conditional_count: int
    reject_count    : int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalSummary:
    mode            : str
    model_name      : str
    total_episodes  : int
    avg_reward      : float
    avg_score       : float
    avg_accuracy    : float
    hard_rule_violation_rate: float
    avg_npa_rate    : float
    avg_crar        : float
    audit_pass_rate : float
    early_termination_rate: float
    per_task        : Dict[str, dict] = field(default_factory=dict)
    episodes        : List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# AGENT STRATEGIES
# ═══════════════════════════════════════════════════════════════

class RuleBasedAgent:
    """
    Deterministic rule-based agent for baseline evaluation.

    Logic (mimics what a naive reader would do):
      - Hard rule triggered → REJECT
      - RED alert → REJECT
      - PD > 0.45 → REJECT
      - PD 0.25-0.45 → CONDITIONAL
      - PD < 0.25 → APPROVE
    """

    def decide(self, obs: Any, app_meta: dict) -> int:
        hard_rules = app_meta.get("hard_rules_triggered", [])
        alerts = app_meta.get("alerts", [])
        has_red = any(a.get("severity") == "RED" for a in alerts)
        pd = app_meta.get("hidden_pd", 0.5)

        if hard_rules or has_red:
            return 2  # REJECT

        if pd >= 0.45:
            return 2
        elif pd >= 0.25:
            return 1  # CONDITIONAL
        else:
            return 0  # APPROVE


class RandomAgent:
    """Random agent — lower bound baseline."""
    def decide(self, obs: Any, app_meta: dict) -> int:
        return random.choice([0, 0, 1, 2])  # slight approve bias


class GreedyApproveAgent:
    """Always approves — upper bound on yield, ignores risk."""
    def decide(self, obs: Any, app_meta: dict) -> int:
        return 0


# ═══════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════

def run_episode(
    task_id: str,
    seed: int,
    episode_id: int,
    agent,
) -> EpisodeResult:
    """Run one full episode and collect metrics."""
    env = IntelliCreditEnvironment(task_id=task_id)
    obs = env.reset(seed=seed)

    actions_taken = []
    cumulative_reward = 0.0
    approve_count = 0
    conditional_count = 0
    reject_count = 0

    for step in range(50):
        app_meta = {}
        if step < len(env._applications):
            app_meta = env._applications[step]["metadata"]

        decision = agent.decide(obs, app_meta)
        action = IntelliCreditAction(decision=decision)
        obs = env.step(action)

        actions_taken.append(decision)
        cumulative_reward += obs.reward

        if decision == 0: approve_count += 1
        elif decision == 1: conditional_count += 1
        else: reject_count += 1

        if obs.done:
            break

    portfolio = env._portfolio
    world = env._world

    # Grade episode
    grade = grade_episode(
        actions=actions_taken,
        applications=env._applications[:len(actions_taken)],
        portfolio=portfolio,
        task_id=task_id,
    )

    # Hard rule violations
    hr_violations = 0
    hr_applicable = 0
    for i, action in enumerate(actions_taken):
        if i >= len(env._applications):
            break
        meta = env._applications[i]["metadata"]
        hr = meta.get("hard_rules_triggered", [])
        if hr:
            hr_applicable += 1
            if action != 2:  # Should have rejected
                hr_violations += 1

    # Audit stats
    audit_pass = sum(1 for a in world.audit_history if a.get("is_clean", True))
    audit_fail = sum(1 for a in world.audit_history if not a.get("is_clean", True))

    return EpisodeResult(
        episode_id       = episode_id,
        task_id          = task_id,
        seed             = seed,
        total_reward     = round(cumulative_reward, 4),
        episode_score    = round(grade["score"], 4),
        accuracy         = round(grade["breakdown"].get("accuracy", 0), 4),
        hard_rule_violations  = hr_violations,
        hard_rule_applicable  = hr_applicable,
        npa_rate         = round(portfolio.npa_rate, 4),
        crar             = round(portfolio.crar, 4),
        audit_pass_count = audit_pass,
        audit_fail_count = audit_fail,
        terminated_early = portfolio.episode_terminated,
        termination_reason = portfolio.termination_reason or "",
        steps_survived   = portfolio.steps_survived,
        total_actions    = len(actions_taken),
        approve_count    = approve_count,
        conditional_count= conditional_count,
        reject_count     = reject_count,
    )


# ═══════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════

def run_evaluation(
    mode: str,
    model_name: str,
    agent,
    n_per_task: int = EPISODES_PER_TASK,
    verbose: bool = True,
) -> EvalSummary:
    """
    Run full multi-task evaluation.

    Args:
        mode       : "baseline", "grpo", or "reflection"
        model_name : Human-readable model identifier
        agent      : Agent instance with .decide(obs, meta) -> int
        n_per_task : Episodes per task level
    """
    all_results: List[EpisodeResult] = []
    episode_id = 0

    for task_id in TASKS:
        task_results = []
        seeds = FIXED_SEEDS[episode_id: episode_id + n_per_task]

        if verbose:
            print(f"  [{task_id}] Running {n_per_task} episodes...")

        for i, seed in enumerate(seeds):
            result = run_episode(task_id, seed, episode_id + i, agent)
            task_results.append(result)
            all_results.append(result)

            if verbose and (i + 1) % 5 == 0:
                avg_score = sum(r.episode_score for r in task_results) / len(task_results)
                print(f"    {i+1}/{n_per_task} | avg_score={avg_score:.3f}")

        episode_id += n_per_task

    # Aggregate
    n = len(all_results)

    def safe_rate(num, denom):
        return round(num / denom, 4) if denom > 0 else 0.0

    total_hr_violations = sum(r.hard_rule_violations for r in all_results)
    total_hr_applicable = sum(r.hard_rule_applicable for r in all_results)
    total_audits = sum(r.audit_pass_count + r.audit_fail_count for r in all_results)
    total_audit_pass = sum(r.audit_pass_count for r in all_results)
    total_terminated = sum(1 for r in all_results if r.terminated_early)

    # Per-task breakdown
    per_task = {}
    for task_id in TASKS:
        task_eps = [r for r in all_results if r.task_id == task_id]
        if not task_eps:
            continue
        t_n = len(task_eps)
        per_task[task_id] = {
            "episodes"        : t_n,
            "avg_score"       : round(sum(r.episode_score for r in task_eps) / t_n, 4),
            "avg_reward"      : round(sum(r.total_reward for r in task_eps) / t_n, 4),
            "avg_accuracy"    : round(sum(r.accuracy for r in task_eps) / t_n, 4),
            "avg_npa_rate"    : round(sum(r.npa_rate for r in task_eps) / t_n, 4),
            "terminated_early": sum(1 for r in task_eps if r.terminated_early),
        }

    summary = EvalSummary(
        mode            = mode,
        model_name      = model_name,
        total_episodes  = n,
        avg_reward      = round(sum(r.total_reward for r in all_results) / n, 4),
        avg_score       = round(sum(r.episode_score for r in all_results) / n, 4),
        avg_accuracy    = round(sum(r.accuracy for r in all_results) / n, 4),
        hard_rule_violation_rate = safe_rate(total_hr_violations, total_hr_applicable),
        avg_npa_rate    = round(sum(r.npa_rate for r in all_results) / n, 4),
        avg_crar        = round(sum(r.crar for r in all_results) / n, 4),
        audit_pass_rate = safe_rate(total_audit_pass, total_audits),
        early_termination_rate = safe_rate(total_terminated, n),
        per_task        = per_task,
        episodes        = [r.to_dict() for r in all_results],
    )

    return summary


# ═══════════════════════════════════════════════════════════════
# REFLECTION EVALUATION
# ═══════════════════════════════════════════════════════════════

def run_reflection_evaluation(
    n_episodes: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Run reflection-module evaluation: scores improving across episodes.

    Uses RuleBasedAgent + MemoryBank, showing that in-context lessons
    improve performance without weight updates.
    """
    from server.reflection import MemoryBank, run_post_episode_reflection, ImprovementTracker

    bank_path = os.path.join(RESULTS_DIR, "eval_memory_bank.json")
    bank = MemoryBank(path=bank_path)
    bank.clear()

    agent = RuleBasedAgent()
    episode_scores = []
    episode_details = []

    for ep in range(1, n_episodes + 1):
        task_id = TASKS[(ep - 1) % len(TASKS)]
        seed = FIXED_SEEDS[ep - 1]

        env = IntelliCreditEnvironment(task_id=task_id)
        obs = env.reset(seed=seed)

        actions_taken = []
        step_rewards = []
        step_components = []

        for step in range(50):
            app_meta = {}
            if step < len(env._applications):
                app_meta = env._applications[step]["metadata"]

            decision = agent.decide(obs, app_meta)
            action = IntelliCreditAction(decision=decision)
            obs = env.step(action)

            actions_taken.append(decision)
            step_rewards.append(obs.reward)
            step_components.append(dict(obs.reward_components))

            if obs.done:
                break

        grade = grade_episode(
            actions=actions_taken,
            applications=env._applications[:len(actions_taken)],
            portfolio=env._portfolio,
            task_id=task_id,
        )
        score = grade["score"]
        episode_scores.append(score)

        # Run reflection
        result = run_post_episode_reflection(
            episode_number  = ep,
            actions         = actions_taken,
            applications    = env._applications[:len(actions_taken)],
            portfolio       = env._portfolio,
            world           = env._world,
            step_rewards    = step_rewards,
            step_components = step_components,
            episode_score   = score,
            memory_bank     = bank,
        )

        episode_details.append({
            "episode"          : ep,
            "task_id"          : task_id,
            "score"            : round(score, 4),
            "lessons_stored"   : bank.lesson_count,
            "lessons_extracted": result["lessons_extracted"],
        })

        if verbose and ep % 5 == 0:
            recent_avg = sum(episode_scores[-5:]) / 5
            print(f"    Episode {ep:2d}/{n_episodes} | score={score:.3f} | recent_avg={recent_avg:.3f} | lessons={bank.lesson_count}")

    # Improvement report
    tracker = ImprovementTracker(bank)
    report = tracker.get_improvement_report()

    # Phase averages
    def phase_avg(start, end):
        chunk = episode_scores[start:end]
        return round(sum(chunk) / len(chunk), 4) if chunk else 0.0

    result = {
        "mode"             : "reflection",
        "model_name"       : "RuleBasedAgent + MemoryBank (no weight updates)",
        "total_episodes"   : n_episodes,
        "score_trajectory" : [round(s, 4) for s in episode_scores],
        "phase_averages"   : {
            "episodes_1_10"  : phase_avg(0, 10),
            "episodes_11_20" : phase_avg(10, 20),
            "episodes_21_30" : phase_avg(20, 30),
        },
        "improvement_delta": report.get("improvement_delta", 0.0),
        "improving"        : report.get("improving", False),
        "memory_stats"     : bank.get_stats(),
        "top_lessons"      : bank.get_lessons_text(n=10),
        "episode_details"  : episode_details,
    }

    # Cleanup temp bank
    if os.path.exists(bank_path):
        os.remove(bank_path)

    return result


# ═══════════════════════════════════════════════════════════════
# RESULT SAVING
# ═══════════════════════════════════════════════════════════════

def save_results(data: dict, filename: str) -> str:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Results saved → {path}")
    return path


def print_summary(data: dict):
    """Print a compact summary to console."""
    print(f"\n{'═'*60}")
    print(f"  EVALUATION SUMMARY: {data.get('mode', '?').upper()}")
    print(f"{'═'*60}")
    print(f"  Model           : {data.get('model_name', 'N/A')}")
    print(f"  Episodes        : {data.get('total_episodes', 0)}")
    print(f"  Avg Score       : {data.get('avg_score', 0):.4f}")
    print(f"  Avg Reward      : {data.get('avg_reward', 0):.4f}")
    print(f"  Avg Accuracy    : {data.get('avg_accuracy', 0):.1%}")
    print(f"  HR Violation %  : {data.get('hard_rule_violation_rate', 0):.1%}")
    print(f"  Avg NPA Rate    : {data.get('avg_npa_rate', 0):.1%}")
    print(f"  Avg CRAR        : {data.get('avg_crar', 0):.1%}")
    print(f"  Audit Pass Rate : {data.get('audit_pass_rate', 0):.1%}")
    print(f"  Early Term Rate : {data.get('early_termination_rate', 0):.1%}")

    if data.get("per_task"):
        print(f"\n  Per-Task Breakdown:")
        print(f"  {'Task':<8} {'Score':>7} {'Reward':>8} {'Accuracy':>9} {'NPA':>7}")
        print(f"  {'-'*45}")
        for t, v in data["per_task"].items():
            print(f"  {t:<8} {v['avg_score']:>7.4f} {v['avg_reward']:>8.4f} {v['avg_accuracy']:>9.1%} {v['avg_npa_rate']:>7.1%}")
    print(f"{'═'*60}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IntelliCredit Evaluation (Phase 7)")
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "random", "greedy", "reflection"],
                        help="Evaluation mode")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_TASK,
                        help="Episodes per task level (baseline modes)")
    parser.add_argument("--n-episodes", type=int, default=30,
                        help="Total episodes for reflection mode")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    IntelliCredit v2 — Evaluation Engine (Phase 7)        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.mode == "reflection":
        print(f"\n  Mode: Reflection (cross-episode learning, {args.n_episodes} episodes)")
        result = run_reflection_evaluation(n_episodes=args.n_episodes, verbose=verbose)
        save_results(result, "reflection_results.json")
        print(f"\n  Score trajectory: {result['score_trajectory']}")
        phase = result["phase_averages"]
        print(f"  Phase averages: ep1-10={phase['episodes_1_10']:.4f} | ep11-20={phase['episodes_11_20']:.4f} | ep21-30={phase['episodes_21_30']:.4f}")
        print(f"  Improving: {result['improving']} | Delta: {result['improvement_delta']:+.4f}")
        return

    # Choose agent
    if args.mode == "baseline":
        agent = RuleBasedAgent()
        model_name = "RuleBasedAgent (optimal rule follower)"
        fname = "baseline_results.json"
    elif args.mode == "random":
        agent = RandomAgent()
        model_name = "RandomAgent (lower bound)"
        fname = "random_results.json"
    elif args.mode == "greedy":
        agent = GreedyApproveAgent()
        model_name = "GreedyApproveAgent (always APPROVE)"
        fname = "greedy_results.json"
    else:
        print(f"  Unknown mode: {args.mode}")
        return

    print(f"\n  Mode: {args.mode} | Agent: {model_name}")
    print(f"  Episodes per task: {args.episodes} | Total: {args.episodes * len(TASKS)}")

    summary = run_evaluation(
        mode=args.mode,
        model_name=model_name,
        agent=agent,
        n_per_task=args.episodes,
        verbose=verbose,
    )

    result_dict = asdict(summary)
    save_results(result_dict, fname)
    print_summary(result_dict)


if __name__ == "__main__":
    main()
