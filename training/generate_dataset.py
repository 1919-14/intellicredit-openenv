"""
IntelliCredit v2 — GRPO Training Dataset Generator (Phase 6, Step 6.1)
=======================================================================
Generates 2000 prompts across all 5 task levels for GRPO training.

Each prompt contains:
  - Role definition
  - Tool descriptions
  - RBI regulatory rules (6 hard rules)
  - Application data (from dataset.py)
  - Forensic alerts
  - Portfolio snapshot
  - Macro environment
  - Instruction to call tools then submit_decision()

Output:
  - JSONL file: training/grpo_dataset.jsonl
  - Optional HF Hub push: vssksn/intellicredit-grpo-dataset
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.dataset import generate_application, application_to_text
from server.intellicredit_env import IntelliCreditEnvironment
from server.agent_loop import build_system_prompt


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

PROMPTS_PER_TASK = 400
TASKS = ["task1", "task2", "task3", "task4", "task5"]
TOTAL_PROMPTS = PROMPTS_PER_TASK * len(TASKS)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "grpo_dataset.jsonl")


# ═══════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_grpo_prompt(env: IntelliCreditEnvironment, step_idx: int = 0) -> dict:
    """
    Build a single GRPO training prompt from the current env state.

    Returns dict with:
        prompt    : str  — the full system+application prompt
        metadata  : dict — ground truth for reward computation
    """
    if step_idx >= len(env._applications):
        return None

    app = env._applications[step_idx]
    meta = app["metadata"]
    features = app["features"]

    # Build the application text
    app_text = application_to_text(app)

    # Portfolio snapshot (from env state)
    portfolio = env._portfolio
    npa_rate = portfolio.npa_rate if portfolio else 0.0
    crar = portfolio.crar if portfolio else 0.18
    cap_remaining = portfolio.capital_remaining if portfolio else 100.0
    cap_deployed_pct = (
        (portfolio.capital_deployed / portfolio.total_capital * 100)
        if portfolio and portfolio.total_capital > 0 else 0.0
    )

    # Macro state
    macro = env._world.get_macro_obs() if env._world else [0.2, 0.0, 0.5, 0.5, 0.5]
    macro_label = "HIGH STRESS" if macro[0] > 0.6 else ("MODERATE" if macro[0] > 0.35 else "STABLE")

    # Build the full prompt
    prompt = f"""You are a Senior Credit Officer at an Indian NBFC reviewing MSME loan applications.
Your decisions must balance yield (approving good loans), risk (avoiding defaults), and regulatory compliance (RBI rules).

═══ AVAILABLE TOOLS ═══
You may call up to 4 tools before submitting your decision. Each tool returns information.

TOOL 1 — get_financial_report("company_name")
  Returns: 3-year revenue, EBITDA margins, debt schedule, auditor remarks.

TOOL 2 — check_compliance_status("company_name")
  Returns: MCA filings, GST returns, DIN status, NCLT cases, CIBIL score, prior defaults.

TOOL 3 — get_market_intelligence("sector_name")
  Returns: Sector stress, RBI advisory, peer NPA rate, headwinds/tailwinds, portfolio exposure.

═══ RBI REGULATORY RULES ═══
HARD RULES (ANY of these = MANDATORY REJECT):
  HR-01: DSCR < 1.0           HR-04: Cheque bounce rate > 25%
  HR-02: Director disqualified HR-05: GST compliance < 40%
  HR-03: RED forensic alert   HR-06: Severe adverse media (>0.80)

Regulatory Limits:
  • CRAR must stay above 12.5% (current: {crar:.1%})
  • NPA rate must stay below 5% (current: {npa_rate:.1%})
  • Sector concentration must stay below 30%
  • Single borrower exposure must stay below 15%

═══ PORTFOLIO STATUS ═══
  CRAR: {crar:.1%} | NPA Rate: {npa_rate:.1%}
  Capital Deployed: {cap_deployed_pct:.1f}% | Remaining: ₹{cap_remaining:.1f} Cr
  Macro Environment: {macro_label} (stress={macro[0]:.2f})

═══ CURRENT APPLICATION ═══
{app_text}

═══ YOUR TASK ═══
Analyze this application. Use tools if you need more information.
Then submit your final decision using: submit_decision("ACTION", "your detailed reasoning")
where ACTION is one of: APPROVE, CONDITIONAL, REJECT.

Think step by step. If any hard rule is triggered, you MUST reject."""

    # Ground truth metadata for reward computation
    metadata = {
        "task_id"            : env._task_id,
        "ground_truth_pd"    : meta.get("hidden_pd", 0.5),
        "optimal_action"     : meta.get("optimal_action", 2),
        "hard_rules"         : meta.get("hard_rules_triggered", []),
        "alerts"             : meta.get("alerts", []),
        "sector"             : meta.get("sector", "Unknown"),
        "tier"               : meta.get("tier", "C"),
        "company_name"       : meta.get("company_name", "Unknown"),
        "loan_amount_cr"     : meta.get("loan_amount_cr", 5.0),
        "is_repeat"          : meta.get("is_repeat_applicant", False),
        "has_missing_data"   : meta.get("has_missing_features", False),
        "npa_rate"           : npa_rate,
        "crar"               : crar,
        "dscr"               : app["raw_values"].get("dscr", 1.5),
        "has_red_alerts"     : any(a.get("severity") == "RED" for a in meta.get("alerts", [])),
    }

    return {
        "prompt": prompt.strip(),
        "metadata": metadata,
    }


# ═══════════════════════════════════════════════════════════════
# DATASET GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_dataset(
    prompts_per_task: int = PROMPTS_PER_TASK,
    output_path: str = OUTPUT_JSONL,
    seed_base: int = 42,
) -> str:
    """
    Generate the full GRPO training dataset.

    Creates 5 × prompts_per_task prompts across all task levels.
    Each row contains the prompt text and ground truth metadata.

    Returns:
        Path to the generated JSONL file.
    """
    all_samples = []
    sample_id = 0

    for task_id in TASKS:
        print(f"  Generating {prompts_per_task} prompts for {task_id}...")
        task_count = 0

        # Generate multiple episodes to get enough unique applications
        episodes_needed = (prompts_per_task // 40) + 1  # ~40-50 apps per episode

        for ep in range(episodes_needed):
            if task_count >= prompts_per_task:
                break

            seed = seed_base + hash(task_id) % 10000 + ep * 100
            env = IntelliCreditEnvironment(task_id=task_id)
            env.reset(seed=seed)

            # Extract prompts from each application in the episode
            for step_idx in range(min(50, len(env._applications))):
                if task_count >= prompts_per_task:
                    break

                result = build_grpo_prompt(env, step_idx)
                if result is None:
                    break

                row = {
                    "id"       : f"{task_id}_{sample_id:05d}",
                    "prompt"   : result["prompt"],
                    "metadata" : result["metadata"],
                }
                all_samples.append(row)
                sample_id += 1
                task_count += 1

    # Shuffle for training variety
    random.seed(seed_base)
    random.shuffle(all_samples)

    # Write JSONL
    with open(output_path, "w") as f:
        for row in all_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n  Dataset written: {output_path}")
    print(f"  Total samples: {len(all_samples)}")

    # Distribution report
    task_dist = {}
    for s in all_samples:
        tid = s["metadata"]["task_id"]
        task_dist[tid] = task_dist.get(tid, 0) + 1
    print(f"  Distribution: {task_dist}")

    # Hard rule stats
    hr_count = sum(1 for s in all_samples if s["metadata"]["hard_rules"])
    red_count = sum(1 for s in all_samples if s["metadata"]["has_red_alerts"])
    repeat_count = sum(1 for s in all_samples if s["metadata"]["is_repeat"])
    print(f"  Hard rules triggered: {hr_count}/{len(all_samples)} ({hr_count/len(all_samples)*100:.1f}%)")
    print(f"  RED alerts: {red_count}/{len(all_samples)} ({red_count/len(all_samples)*100:.1f}%)")
    print(f"  Repeat applicants: {repeat_count}/{len(all_samples)} ({repeat_count/len(all_samples)*100:.1f}%)")

    return output_path


# ═══════════════════════════════════════════════════════════════
# HF DATASET PUSH (optional)
# ═══════════════════════════════════════════════════════════════

def push_to_hub(jsonl_path: str, repo_id: str = "vssksn/intellicredit-grpo-dataset"):
    """
    Push generated dataset to Hugging Face Hub.
    Requires: pip install datasets huggingface_hub
    """
    try:
        from datasets import Dataset

        rows = []
        with open(jsonl_path, "r") as f:
            for line in f:
                row = json.loads(line)
                # Flatten metadata for HF dataset
                flat = {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "task_id": row["metadata"]["task_id"],
                    "ground_truth_pd": row["metadata"]["ground_truth_pd"],
                    "optimal_action": row["metadata"]["optimal_action"],
                    "hard_rules": json.dumps(row["metadata"]["hard_rules"]),
                    "has_red_alerts": row["metadata"]["has_red_alerts"],
                    "sector": row["metadata"]["sector"],
                    "tier": row["metadata"]["tier"],
                }
                rows.append(flat)

        ds = Dataset.from_list(rows)
        ds.push_to_hub(repo_id, private=False)
        print(f"\n  ✅ Dataset pushed to https://huggingface.co/datasets/{repo_id}")
    except ImportError:
        print("\n  ⚠️  Install 'datasets' and 'huggingface_hub' to push to HF Hub.")
    except Exception as e:
        print(f"\n  ❌ Push failed: {e}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  IntelliCredit GRPO Dataset Generator (Phase 6, Step 6.1)")
    print("=" * 65)

    path = generate_dataset()

    # Sanity check: read back and verify
    print(f"\n  Verifying...")
    with open(path, "r") as f:
        lines = f.readlines()
    first = json.loads(lines[0])
    assert "prompt" in first, "Missing prompt field"
    assert "metadata" in first, "Missing metadata field"
    assert "ground_truth_pd" in first["metadata"], "Missing ground_truth_pd"
    print(f"  ✓ First sample: {first['id']} | pd={first['metadata']['ground_truth_pd']:.3f}")
    print(f"  ✓ Prompt length: {len(first['prompt'])} chars")

    print(f"\n{'='*65}")
    print(f"  Dataset Generation Complete ✓")
    print(f"{'='*65}")

    # Optional: push to HF Hub
    if "--push" in sys.argv:
        push_to_hub(path)
