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
import warnings
from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════
# ALLOCATOR CONFIG — must be set BEFORE torch is imported.
# expandable_segments:True lets the CUDA allocator grow/shrink segments
# instead of holding monolithic blocks, eliminating the "reserved but
# unallocated" fragmentation that triggers false OOMs on 8 GB GPUs.
# ═══════════════════════════════════════════════════════════════
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

# Suppress verbose deprecation noise from transformers/trl internals
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ═══════════════════════════════════════════════════════════════
# PATH SETUP + .env LOADING
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

TRAINING_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(TRAINING_DIR, "grpo_dataset.jsonl")
CHECKPOINTS   = os.path.join(TRAINING_DIR, "checkpoints")
LOGS_DIR      = os.path.join(TRAINING_DIR, "logs")
MERGED_DIR    = os.path.join(TRAINING_DIR, "merged_model")

# Load .env so HF_TOKEN is available for authenticated Hub downloads
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(PROJECT_ROOT, ".env")
    load_dotenv(_env_path, override=False)
    _hf_token = os.environ.get("HF_TOKEN")
    if _hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)
        print(f"  [env] HF_TOKEN loaded from {_env_path}")
except ImportError:
    pass  # python-dotenv optional; export HF_TOKEN manually if needed


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
    "gemma_2b": {
        "model_name"    : "unsloth/gemma-2-2b-it",
        # 1024 with num_gen=2 is the proven-stable config on RTX 5050 8GB:
        # Peak VRAM = model(1.5) + KV(2seqs×1024tok×~108MB) + acts(~0.6GB)
        #           ≈ 2.4 GB total. Safe headroom of ~5.6 GB.
        # DO NOT raise to 2048: that triples activation memory and OOMs.
        "max_seq_length": 1024,
        "load_in_4bit"  : True,
        "lora_r"        : 8,
        "lora_alpha"    : 16,
        "lora_dropout"  : 0.0,
        "target_modules" : ["q_proj", "v_proj", "k_proj", "o_proj"],
        "description"   : "Gemma-2-2B-Instruct",
    },
}

DEFAULT_MODEL = "gemma_1b"


# ═══════════════════════════════════════════════════════════════
# STEP 6.4: TRAINING HYPERPARAMETERS (3-Stage Curriculum)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# VRAM Budget for Gemma-2-2B-4bit on RTX 5050 (7.96 GB usable)
# ----------------------------------------------------------------
# Component              |  Memory
# Model weights (4-bit)  |  ~1.5 GB
# LoRA adapters          |  ~0.05 GB
# Optimizer (adamw_8bit) |  ~0.3 GB
# GRPO generation peak:  |
#   num_gen × batch_size forward passes simultaneously
#   Gemma-2-2B: 26 layers, 4 KV-heads, head_dim=256, bf16
#   KV per seq @ 512 tok:  26×2×4×256×512×2  = ~54 MB
#   2 seqs × 54 MB   = ~108 MB KV cache
#   Activation (checkpointing on): ~150 MB × 2 = ~300 MB
# Total safe estimate:   |  ~2.3 GB
# Headroom:              |  ~5.7 GB available ✔
# NOTE: num_generations ≥2 is required for GRPO advantage estimation.
# ═══════════════════════════════════════════════════════════════
STAGE_CONFIGS = {
    1: {
        "name"                       : "Stage 1: Easy (task1 only)",
        "task_filter"                : ["task1"],
        "num_train_epochs"           : 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate"              : 5e-6,
        # prompt+completion budget: 800+128=928 < 1024 max_seq_length.
        # max_prompt_length here is used for BOTH dataset truncation
        # (at source, via tokenizer) AND GRPOConfig (collator guard).
        "max_prompt_length"          : 800,
        "max_completion_length"      : 128,
        "num_generations"            : 2,
        "temperature"                : 0.9,
        "beta"                       : 0.001,
        "warmup_steps"               : 20,
        "logging_steps"              : 10,
        "save_steps"                 : 50,
    },
    2: {
        "name"                       : "Stage 2: Medium (task1 + task2)",
        "task_filter"                : ["task1", "task2"],
        "num_train_epochs"           : 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate"              : 3e-6,
        "max_prompt_length"          : 800,
        "max_completion_length"      : 128,
        "num_generations"            : 2,
        "temperature"                : 0.9,
        "beta"                       : 0.001,
        "warmup_steps"               : 20,
        "logging_steps"              : 10,
        "save_steps"                 : 50,
    },
    3: {
        "name"                       : "Stage 3: Full (all tasks)",
        "task_filter"                : ["task1", "task2", "task3", "task4", "task5"],
        "num_train_epochs"           : 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate"              : 2e-6,
        "max_prompt_length"          : 800,
        "max_completion_length"      : 128,
        "num_generations"            : 2,
        "temperature"                : 0.8,
        "beta"                       : 0.001,
        "warmup_steps"               : 30,
        "logging_steps"              : 10,
        "save_steps"                 : 50,
    },
}


# ═══════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════

def load_dataset(task_filter: List[str] = None, hf_dataset: str = None) -> List[dict]:
    """Load the GRPO dataset, optionally filtering by task levels."""
    samples = []
    
    if hf_dataset:
        print(f"  Downloading dataset from Hugging Face Hub: {hf_dataset}...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("  ❌ Install: pip install huggingface_hub")
            sys.exit(1)
            
        try:
            file_path = hf_hub_download(repo_id=hf_dataset, repo_type="dataset", filename="grpo_dataset.jsonl")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    # If it's the flattened format from push_to_hub:
                    if "metadata" not in row:
                        metadata = {
                            "task_id": row.get("task_id", ""),
                            "ground_truth_pd": row.get("ground_truth_pd", 0.0),
                            "optimal_action": row.get("optimal_action", 0),
                            "hard_rules": json.loads(row.get("hard_rules", "[]")) if isinstance(row.get("hard_rules"), str) else row.get("hard_rules", []),
                            "has_red_alerts": row.get("has_red_alerts", False),
                            "npa_rate": row.get("npa_rate", 0.02),
                            "crar": row.get("crar", 0.18),
                            "sector": row.get("sector", "Unknown"),
                        }
                        row = {"prompt": row["prompt"], "metadata": metadata}
                        
                    if task_filter and row["metadata"]["task_id"] not in task_filter:
                        continue
                    samples.append(row)
        except Exception as e:
            print(f"  ❌ Failed to download or read dataset: {e}")
            sys.exit(1)
    else:
        if not os.path.exists(DATASET_PATH):
            print(f"  ❌ Dataset not found at {DATASET_PATH}")
            print(f"  Run: python training/generate_dataset.py first")
            sys.exit(1)

        with open(DATASET_PATH, "r") as f:
            for line in f:
                row = json.loads(line)
                if task_filter:
                    if row["metadata"]["task_id"] not in task_filter:
                        continue
                samples.append(row)

    print(f"  Loaded {len(samples)} samples (filter={task_filter})")
    return samples


def prepare_hf_dataset(samples: List[dict], tokenizer=None, max_prompt_length: int = 800):
    """
    Convert raw samples to HuggingFace Dataset for GRPOTrainer.

    CRITICAL: Truncate prompts at SOURCE using the real tokenizer.
    GRPOConfig.max_prompt_length only applies inside the trainer collator,
    AFTER Unsloth shapes attention masks to max_seq_length. If a prompt
    exceeds max_seq_length at that point, the tensor shape mismatch crashes:
      RuntimeError: size of tensor a (512) must match tensor b (795)
    Truncating here guarantees the dataset never contains over-length prompts.
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("  ❌ Install: pip install datasets")
        sys.exit(1)

    truncated = 0
    rows = []
    for s in samples:
        prompt = s["prompt"]

        # Token-level truncation: decode back to text so GRPOTrainer's
        # internal tokenizer call sees a correctly-sized string.
        if tokenizer is not None and len(prompt) > 0:
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_length,
                add_special_tokens=True,
                return_tensors=None,
            )
            if len(enc["input_ids"]) < len(tokenizer(prompt, add_special_tokens=True)["input_ids"]):
                truncated += 1
            prompt = tokenizer.decode(enc["input_ids"], skip_special_tokens=False)

        rows.append({
            "prompt"         : prompt,
            "ground_truth_pd": s["metadata"]["ground_truth_pd"],
            "optimal_action" : s["metadata"]["optimal_action"],
            "hard_rules"     : json.dumps(s["metadata"].get("hard_rules", [])),
            "has_red_alerts" : s["metadata"].get("has_red_alerts", False),
            "npa_rate"       : s["metadata"].get("npa_rate", 0.02),
            "crar"           : s["metadata"].get("crar", 0.18),
            "sector"         : s["metadata"].get("sector", "Unknown"),
        })

    if truncated:
        print(f"  [dataset] Truncated {truncated}/{len(samples)} prompts to ≤{max_prompt_length} tokens")

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
    hf_repo: str = None,
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
    samples = load_dataset(task_filter=config["task_filter"], hf_dataset=hf_repo)
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
        fast_inference = False,
        token          = os.environ.get("HF_TOKEN"),
    )

    # ── CRITICAL: Override generation_config to prevent max_length=8192 OOM ──
    # Gemma-2's generation_config.json sets max_length=8192 by default.
    # Unsloth's fast_forward_inference pre-allocates a KV cache sized to
    # generation_config.max_length BEFORE our max_completion_length is applied.
    # This causes 43× larger KV allocation than needed → CUDA OOM.
    # Fix: pin max_new_tokens and clear max_length from the model config.
    _max_new = config["max_completion_length"]
    model.generation_config.max_new_tokens = _max_new
    if hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
    # use_cache=False during training prevents the KV cache from growing
    # unboundedly during the backward pass (training != inference).
    model.config.use_cache = False
    print(f"  [mem] generation_config patched: max_new_tokens={_max_new}, max_length=None, use_cache=False")

    if torch.cuda.is_available():
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated()) / 1024**3
        print(f"  [mem] VRAM after model load: {free_gb:.2f} GB free")

    # Apply LoRA — use_gradient_checkpointing="unsloth" halves VRAM for
    # activations at ~10% throughput cost; worthwhile on 8GB.
    model = FastLanguageModel.get_peft_model(
        model,
        r                        = model_config["lora_r"],
        lora_alpha               = model_config["lora_alpha"],
        lora_dropout             = model_config["lora_dropout"],
        target_modules           = model_config["target_modules"],
        use_gradient_checkpointing= "unsloth",
        random_state             = 42,
    )

    print(f"  ✓ Model loaded ({model_config['model_name']})")

    # ── Prepare dataset (AFTER tokenizer is loaded so we can truncate) ──────
    hf_dataset = prepare_hf_dataset(
        samples,
        tokenizer       = tokenizer,
        max_prompt_length = config["max_prompt_length"],
    )
    print(f"  ✓ Dataset prepared ({len(hf_dataset)} samples, prompts ≤{config['max_prompt_length']} tokens)")

    # ── Build reward functions ───────────────────────────────────
    reward_funcs = build_reward_funcs()
    print(f"  ✓ Reward functions loaded (4 functions)")

    # ── Step 6.4: GRPO training config ───────────────────────────
    stage_output_dir = os.path.join(CHECKPOINTS, f"stage_{stage}")
    os.makedirs(stage_output_dir, exist_ok=True)

    training_args = GRPOConfig(
        output_dir                   = stage_output_dir,
        num_train_epochs             = config["num_train_epochs"],
        per_device_train_batch_size  = config["per_device_train_batch_size"],
        gradient_accumulation_steps  = config["gradient_accumulation_steps"],
        learning_rate                = config["learning_rate"],
        # Sequence budget: prompt + completion must fit in max_seq_length.
        # Capping prompt prevents truncation warnings from Unsloth.
        max_prompt_length            = config["max_prompt_length"],
        max_completion_length        = config["max_completion_length"],
        num_generations              = config["num_generations"],
        temperature                  = config["temperature"],
        beta                         = config["beta"],
        warmup_steps                 = config["warmup_steps"],  # warmup_ratio deprecated
        logging_steps                = config["logging_steps"],
        save_steps                   = config["save_steps"],
        fp16                         = False,
        bf16                         = True,
        # Disable compile + cache to avoid generation_config conflict warnings
        optim                        = "adamw_8bit",
        dataloader_num_workers       = 0,   # WSL has no forked dataloader
        report_to                    = "none",
        log_level                    = "warning",   # suppress info spam
    )

    # ── Training monitor ─────────────────────────────────────────
    log_path = os.path.join(LOGS_DIR, f"stage_{stage}_training.jsonl")
    monitor = TrainingMonitor(log_path)

    # ── Step 6.4: Run GRPO training ──────────────────────────────
    eff_batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    print(f"\n  Starting GRPO training...")
    print(f"  Effective batch size : {eff_batch}")
    print(f"  Epochs               : {config['num_train_epochs']}")
    print(f"  Learning rate        : {config['learning_rate']}")
    print(f"  Num generations      : {config['num_generations']}")
    print(f"  Max prompt length    : {config['max_prompt_length']}")
    print(f"  Max completion length: {config['max_completion_length']}")
    print(f"  Temperature          : {config['temperature']}")
    print(f"  Beta (KL)            : {config['beta']}")

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

    # Report VRAM after training
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM after training : {used:.2f} GB / {total:.2f} GB")

    # ── Step 6.6: Save checkpoint ─────────────────────────────────
    trainer.save_model(stage_output_dir)
    tokenizer.save_pretrained(stage_output_dir)
    print(f"  ✓ Checkpoint saved to: {stage_output_dir}")

    # ── Release VRAM before next stage ───────────────────────────
    # Explicitly delete trainer + model so Python GC + CUDA allocator
    # can reclaim all memory before the next stage re-loads the model.
    del trainer, model, tokenizer
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1024**3
        print(f"  VRAM freed. Available for next stage: {free:.2f} GB")

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

    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="Hugging Face dataset repository to use (e.g. vssksn/intellicredit-grpo-v2)")

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
                       resume=(s > 1), dry_run=args.dry_run, hf_repo=args.hf_dataset)
            # Flush VRAM between stages — train_stage already does this
            # internally, but a second empty_cache ensures the allocator
            # has fully reset before the next stage's model load.
            try:
                import torch, gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        stage = int(args.stage)
        train_stage(stage, model_key=args.model,
                   resume=args.resume, dry_run=args.dry_run, hf_repo=args.hf_dataset)

    print(f"\n{'='*65}")
    print(f"  GRPO Training Pipeline Complete ✓")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
