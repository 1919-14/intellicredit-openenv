"""
IntelliCredit v2 — GRPO Training Script (Phase 6, Steps 6.2-6.6)
==================================================================
Complete GRPO fine-tuning pipeline using Unsloth + TRL.

Features:
  - 3-stage curriculum training (easy → medium → full)
  - 4 reward functions (correctness, hard rules, format, portfolio)
  - QLoRA 4-bit quantization for single-GPU training
  - Checkpoint saving after each stage
  - Training monitoring and red flag detection
  - Merged model export and HF Hub push

Usage:
  # Stage 1 only (easy tasks):
  python training/train_grpo.py --stage 1

  # All 3 stages sequentially:
  python training/train_grpo.py --stage all

  # Resume from stage 2 checkpoint:
  python training/train_grpo.py --stage 2 --resume

  # Export merged model:
  python training/train_grpo.py --export
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════
# PATH SETUP
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

TRAINING_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(TRAINING_DIR, "grpo_dataset.jsonl")
CHECKPOINTS   = os.path.join(TRAINING_DIR, "checkpoints")
LOGS_DIR      = os.path.join(TRAINING_DIR, "logs")
MERGED_DIR    = os.path.join(TRAINING_DIR, "merged_model")


# ═══════════════════════════════════════════════════════════════
# STEP 6.2: MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Model selection (order of preference)
MODEL_CONFIGS = {
    "llama3_8b": {
        "model_name"    : "unsloth/Llama-3.1-8B-Instruct",
        "max_seq_length": 1024,
        "load_in_4bit"  : True,
        "lora_r"        : 16,
        "lora_alpha"    : 16,
        "lora_dropout"  : 0.0,
        "target_modules" : ["q_proj", "v_proj", "k_proj", "o_proj"],
        "description"   : "Primary: Llama-3.1-8B-Instruct (best quality)",
    },
    "qwen_7b": {
        "model_name"    : "unsloth/Qwen2.5-7B-Instruct",
        "max_seq_length": 1024,
        "load_in_4bit"  : True,
        "lora_r"        : 16,
        "lora_alpha"    : 16,
        "lora_dropout"  : 0.0,
        "target_modules" : ["q_proj", "v_proj", "k_proj", "o_proj"],
        "description"   : "Alternative: Qwen-2.5-7B-Instruct",
    },
    "gemma_1b": {
        "model_name"    : "unsloth/gemma-3-1b-it",
        "max_seq_length": 1024,
        "load_in_4bit"  : True,
        "lora_r"        : 8,
        "lora_alpha"    : 8,
        "lora_dropout"  : 0.0,
        "target_modules" : ["q_proj", "v_proj"],
        "description"   : "Debug/fastest: Gemma-3-1B-Instruct",
    },
}

DEFAULT_MODEL = "llama3_8b"


# ═══════════════════════════════════════════════════════════════
# STEP 6.4: TRAINING HYPERPARAMETERS (3-Stage Curriculum)
# ═══════════════════════════════════════════════════════════════

STAGE_CONFIGS = {
    1: {
        "name"                       : "Stage 1: Easy (task1 only)",
        "task_filter"                : ["task1"],
        "num_train_epochs"           : 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate"              : 5e-6,
        "max_completion_length"      : 300,
        "num_generations"            : 8,
        "temperature"                : 0.9,
        "beta"                       : 0.001,
        "warmup_ratio"               : 0.1,
        "logging_steps"              : 10,
        "save_steps"                 : 100,
    },
    2: {
        "name"                       : "Stage 2: Medium (task1 + task2)",
        "task_filter"                : ["task1", "task2"],
        "num_train_epochs"           : 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate"              : 5e-6,
        "max_completion_length"      : 300,
        "num_generations"            : 8,
        "temperature"                : 0.9,
        "beta"                       : 0.001,
        "warmup_ratio"               : 0.1,
        "logging_steps"              : 10,
        "save_steps"                 : 100,
    },
    3: {
        "name"                       : "Stage 3: Full (all tasks)",
        "task_filter"                : ["task1", "task2", "task3", "task4", "task5"],
        "num_train_epochs"           : 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate"              : 2e-6,
        "max_completion_length"      : 300,
        "num_generations"            : 8,
        "temperature"                : 0.8,
        "beta"                       : 0.001,
        "warmup_ratio"               : 0.1,
        "logging_steps"              : 10,
        "save_steps"                 : 100,
    },
}


# ═══════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════

def load_dataset(task_filter: List[str] = None) -> List[dict]:
    """Load the GRPO dataset, optionally filtering by task levels."""
    if not os.path.exists(DATASET_PATH):
        print(f"  ❌ Dataset not found at {DATASET_PATH}")
        print(f"  Run: python training/generate_dataset.py first")
        sys.exit(1)

    samples = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            row = json.loads(line)
            if task_filter:
                if row["metadata"]["task_id"] not in task_filter:
                    continue
            samples.append(row)

    print(f"  Loaded {len(samples)} samples (filter={task_filter})")
    return samples


def prepare_hf_dataset(samples: List[dict]):
    """Convert to HuggingFace Dataset format for GRPOTrainer."""
    try:
        from datasets import Dataset
    except ImportError:
        print("  ❌ Install: pip install datasets")
        sys.exit(1)

    rows = []
    for s in samples:
        rows.append({
            "prompt": s["prompt"],
            # Metadata fields for reward functions
            "ground_truth_pd": s["metadata"]["ground_truth_pd"],
            "optimal_action" : s["metadata"]["optimal_action"],
            "hard_rules"     : json.dumps(s["metadata"].get("hard_rules", [])),
            "has_red_alerts" : s["metadata"].get("has_red_alerts", False),
            "npa_rate"       : s["metadata"].get("npa_rate", 0.02),
            "crar"           : s["metadata"].get("crar", 0.18),
            "sector"         : s["metadata"].get("sector", "Unknown"),
        })

    return Dataset.from_list(rows)


# ═══════════════════════════════════════════════════════════════
# REWARD FUNCTIONS WRAPPER (for GRPOTrainer)
# ═══════════════════════════════════════════════════════════════

def build_reward_funcs():
    """
    Build the 4 reward functions in the format GRPOTrainer expects.

    GRPOTrainer calls: reward_func(prompts, completions, **metadata_columns)
    """
    from training.grpo_rewards import (
        reward_correctness,
        reward_hard_rule_compliance,
        reward_format_compliance,
        reward_portfolio_awareness,
    )

    return [
        reward_correctness,
        reward_hard_rule_compliance,
        reward_format_compliance,
        reward_portfolio_awareness,
    ]


# ═══════════════════════════════════════════════════════════════
# STEP 6.5: TRAINING MONITOR
# ═══════════════════════════════════════════════════════════════

class TrainingMonitor:
    """
    Monitors training for red flags and logs metrics.

    Red flags:
      - Model always outputs REJECT (safe but misses yield)
      - Model always outputs APPROVE (maximizes short-term)
      - Format reward up but correctness flat
      - Action distribution collapse
    """

    def __init__(self, log_path: str):
        self._log_path = log_path
        self._history = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_step(self, step: int, metrics: dict):
        """Log training step metrics."""
        entry = {"step": step, "timestamp": time.time(), **metrics}
        self._history.append(entry)

        # Write to log file
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def check_red_flags(self, recent_n: int = 50):
        """
        Check for red flags in recent training history.
        Returns list of warning strings.
        """
        if len(self._history) < recent_n:
            return []

        recent = self._history[-recent_n:]
        flags = []

        # Check action distribution if tracked
        if "action_dist" in recent[-1]:
            dist = recent[-1]["action_dist"]
            if dist.get("reject_pct", 0) > 0.85:
                flags.append("🚨 RED FLAG: Model is reject-biased (>85% REJECT)")
            if dist.get("approve_pct", 0) > 0.85:
                flags.append("🚨 RED FLAG: Model is approve-biased (>85% APPROVE)")

        # Check if reward is flat
        if len(recent) >= 20:
            first_10_avg = sum(r.get("reward", 0) for r in recent[:10]) / 10
            last_10_avg = sum(r.get("reward", 0) for r in recent[-10:]) / 10
            if abs(last_10_avg - first_10_avg) < 0.05:
                flags.append("⚠️ WARNING: Reward appears flat — learning may have stalled")

        return flags

    def get_summary(self) -> dict:
        """Get training summary statistics."""
        if not self._history:
            return {"status": "no_data"}

        rewards = [h.get("reward", 0) for h in self._history]
        return {
            "total_steps"  : len(self._history),
            "avg_reward"   : round(sum(rewards) / len(rewards), 4),
            "max_reward"   : round(max(rewards), 4),
            "min_reward"   : round(min(rewards), 4),
            "last_reward"  : round(rewards[-1], 4),
            "red_flags"    : self.check_red_flags(),
        }


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_stage(
    stage: int,
    model_key: str = DEFAULT_MODEL,
    resume: bool = False,
    dry_run: bool = False,
):
    """
    Execute one training stage.

    Args:
        stage   : 1, 2, or 3
        model_key: Key from MODEL_CONFIGS
        resume  : Load from previous stage checkpoint
        dry_run : Just validate config without training
    """
    config = STAGE_CONFIGS[stage]
    model_config = MODEL_CONFIGS[model_key]

    print(f"\n{'='*65}")
    print(f"  {config['name']}")
    print(f"  Model: {model_config['description']}")
    print(f"{'='*65}")

    # Load dataset
    samples = load_dataset(task_filter=config["task_filter"])
    if not samples:
        print("  ❌ No samples found for this task filter!")
        return

    if dry_run:
        print(f"  [DRY RUN] Would train on {len(samples)} samples")
        print(f"  Config: {json.dumps(config, indent=2, default=str)}")
        return

    # ── Import training dependencies (fail fast if not installed) ──
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        import torch
    except ImportError as e:
        print(f"\n  ❌ Missing dependency: {e}")
        print(f"  Install with:")
        print(f"    pip install unsloth trl torch")
        print(f"  Or run with --dry-run to validate config")
        return

    # ── Step 6.2: Load model ──────────────────────────────────────
    print(f"\n  Loading model: {model_config['model_name']}...")

    checkpoint_path = None
    if resume and stage > 1:
        prev_checkpoint = os.path.join(CHECKPOINTS, f"stage_{stage-1}")
        if os.path.exists(prev_checkpoint):
            checkpoint_path = prev_checkpoint
            print(f"  Resuming from: {checkpoint_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = checkpoint_path or model_config["model_name"],
        max_seq_length = model_config["max_seq_length"],
        load_in_4bit   = model_config["load_in_4bit"],
        fast_inference  = True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r              = model_config["lora_r"],
        lora_alpha     = model_config["lora_alpha"],
        lora_dropout   = model_config["lora_dropout"],
        target_modules = model_config["target_modules"],
    )

    print(f"  ✓ Model loaded ({model_config['model_name']})")

    # ── Prepare dataset ──────────────────────────────────────────
    hf_dataset = prepare_hf_dataset(samples)
    print(f"  ✓ Dataset prepared ({len(hf_dataset)} samples)")

    # ── Build reward functions ───────────────────────────────────
    reward_funcs = build_reward_funcs()
    print(f"  ✓ Reward functions loaded (4 functions)")

    # ── Step 6.4: GRPO training config ───────────────────────────
    stage_output_dir = os.path.join(CHECKPOINTS, f"stage_{stage}")
    os.makedirs(stage_output_dir, exist_ok=True)

    training_args = GRPOConfig(
        output_dir                 = stage_output_dir,
        num_train_epochs           = config["num_train_epochs"],
        per_device_train_batch_size= config["per_device_train_batch_size"],
        gradient_accumulation_steps= config["gradient_accumulation_steps"],
        learning_rate              = config["learning_rate"],
        max_completion_length      = config["max_completion_length"],
        num_generations            = config["num_generations"],
        temperature                = config["temperature"],
        beta                       = config["beta"],
        warmup_ratio               = config["warmup_ratio"],
        logging_steps              = config["logging_steps"],
        save_steps                 = config["save_steps"],
        fp16                       = True,
        report_to                  = "none",
        log_level                  = "info",
    )

    # ── Training monitor ─────────────────────────────────────────
    log_path = os.path.join(LOGS_DIR, f"stage_{stage}_training.jsonl")
    monitor = TrainingMonitor(log_path)

    # ── Step 6.4: Run GRPO training ──────────────────────────────
    print(f"\n  Starting GRPO training...")
    print(f"  Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Num generations: {config['num_generations']}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Beta (KL): {config['beta']}")

    trainer = GRPOTrainer(
        model         = model,
        args          = training_args,
        train_dataset = hf_dataset,
        reward_funcs  = reward_funcs,
        tokenizer     = tokenizer,
    )

    # Train
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    print(f"\n  ✓ Training completed in {elapsed/60:.1f} minutes")

    # ── Step 6.6: Save checkpoint ─────────────────────────────────
    trainer.save_model(stage_output_dir)
    tokenizer.save_pretrained(stage_output_dir)
    print(f"  ✓ Checkpoint saved to: {stage_output_dir}")

    # ── Quick inference test ──────────────────────────────────────
    print(f"\n  Running quick inference test...")
    test_prompts = samples[:5]
    for tp in test_prompts:
        inputs = tokenizer(tp["prompt"][:500], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"    [{tp['metadata']['task_id']}] PD={tp['metadata']['ground_truth_pd']:.2f} → {response[:80]}...")

    # Summary
    summary = monitor.get_summary()
    print(f"\n  Stage {stage} Summary:")
    print(f"    Total steps: {summary.get('total_steps', 'N/A')}")
    print(f"    Red flags: {summary.get('red_flags', [])}")

    return stage_output_dir


# ═══════════════════════════════════════════════════════════════
# STEP 6.6: MODEL EXPORT
# ═══════════════════════════════════════════════════════════════

def export_merged_model(
    model_key: str = DEFAULT_MODEL,
    stage: int = 3,
    push_to_hub: bool = False,
    hub_repo: str = "vssksn/intellicredit-grpo-llama3",
):
    """
    Merge LoRA adapters and export full model.

    Uses Unsloth's save_pretrained_merged() to avoid
    manual 4-bit → 16-bit upcasting issues.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("  ❌ Install unsloth first")
        return

    checkpoint = os.path.join(CHECKPOINTS, f"stage_{stage}")
    if not os.path.exists(checkpoint):
        print(f"  ❌ Checkpoint not found: {checkpoint}")
        return

    print(f"\n  Loading stage {stage} checkpoint for merging...")
    model_config = MODEL_CONFIGS[model_key]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = checkpoint,
        max_seq_length = model_config["max_seq_length"],
        load_in_4bit   = True,
    )

    os.makedirs(MERGED_DIR, exist_ok=True)

    print(f"  Merging LoRA adapters → 16-bit model...")
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"  ✓ Merged model saved to: {MERGED_DIR}")

    if push_to_hub:
        print(f"  Pushing to HF Hub: {hub_repo}...")
        model.push_to_hub_merged(hub_repo, tokenizer, save_method="merged_16bit")
        print(f"  ✓ Pushed to https://huggingface.co/{hub_repo}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IntelliCredit GRPO Training (Phase 6)")
    parser.add_argument("--stage", type=str, default="1",
                        help="Stage to train: 1, 2, 3, or 'all'")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous stage checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without training")
    parser.add_argument("--export", action="store_true",
                        help="Export merged model from stage 3")
    parser.add_argument("--push", action="store_true",
                        help="Push merged model to HF Hub")

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       IntelliCredit v2 — GRPO Training Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.export:
        export_merged_model(
            model_key=args.model,
            push_to_hub=args.push,
        )
        return

    if args.stage == "all":
        for s in [1, 2, 3]:
            train_stage(s, model_key=args.model,
                       resume=(s > 1), dry_run=args.dry_run)
    else:
        stage = int(args.stage)
        train_stage(stage, model_key=args.model,
                   resume=args.resume, dry_run=args.dry_run)

    print(f"\n{'='*65}")
    print(f"  GRPO Training Pipeline Complete ✓")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
