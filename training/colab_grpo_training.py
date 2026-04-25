"""
╔══════════════════════════════════════════════════════════════════╗
║  IntelliCredit v2 — GRPO Fine-Tuning on Google Colab           ║
║  Model: Qwen2.5-1.5B-Instruct (4-bit QLoRA via HF+PEFT)       ║
║  NO UNSLOTH — pure transformers + trl + peft                   ║
║  v3: Fixed zero-loss issue with gradient-aware rewards          ║
╚══════════════════════════════════════════════════════════════════╝

Copy each section below into separate Colab cells.
"""

# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 1: INSTALL DEPENDENCIES ═══
# ═══════════════════════════════════════════════════════════════════
# !pip uninstall unsloth unsloth_zoo -y
# !pip install --upgrade pip
# !pip install "trl>=0.15.2" peft accelerate bitsandbytes
# !pip install "transformers>=4.45.0" datasets matplotlib huggingface_hub


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 2: IMPORTS & CONFIG ═══
# ═══════════════════════════════════════════════════════════════════

import os
import re
import json
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# ── Configuration ────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN  = 1024
HF_DATASET   = "vssksn/intellicredit-grpo-v2"
OUTPUT_BASE  = "intellicredit-grpo-results"
FINAL_MODEL  = "qwen-intellicredit-grpo-final"
PUSH_TO_HUB  = False
HUB_REPO     = "YOUR_USERNAME/intellicredit-grpo-qwen1.5b"

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/charts", exist_ok=True)

print("✅ Imports complete")
print(f"   Model   : {MODEL_NAME}")
print(f"   Dataset : {HF_DATASET}")
print(f"   GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"   VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 3: ACTION PARSER (INLINED) ═══
# ═══════════════════════════════════════════════════════════════════

ACTION_MAP = {
    "APPROVE": 0, "APPROVED": 0,
    "CONDITIONAL": 1, "CONDITIONAL_APPROVE": 1, "CONDITIONAL APPROVE": 1,
    "REJECT": 2, "REJECTED": 2, "DECLINE": 2, "DENY": 2,
}

_RE_SUBMIT  = re.compile(
    r"submit_decision\s*\(\s*['\"]?([A-Z_]+)['\"]?\s*,\s*['\"]?(.*?)['\"]?\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_RE_KEYWORD = re.compile(
    r"\b(APPROVE(?:D)?|CONDITIONAL(?:_APPROVE)?|REJECT(?:ED)?|DECLINE|DENY)\b",
    re.IGNORECASE,
)
_RE_TOOL    = re.compile(
    r"\b(get_financial_report|check_compliance_status|get_market_intelligence)\s*\([^)]*\)",
    re.IGNORECASE,
)

# Credit-domain keywords the model should learn to use
CREDIT_KEYWORDS = [
    "risk", "credit", "loan", "dscr", "npa", "crar", "approve",
    "reject", "conditional", "compliance", "rbi", "capital",
    "borrower", "default", "probability", "portfolio", "decision",
    "financial", "assessment", "collateral", "ratio", "debt",
]


def _unwrap(text):
    """Convert GRPOTrainer list-of-dicts completion to plain string."""
    if isinstance(text, list):
        return " ".join(m.get("content", "") for m in text if isinstance(m, dict))
    return str(text) if not isinstance(text, str) else text


def parse_llm_output(text) -> Dict[str, Any]:
    text = _unwrap(text)
    if not text.strip():
        return {"action": 2, "parse_type": "default_reject", "parse_confidence": 0.0}
    t = text.strip()
    if _RE_TOOL.search(t):
        return {"action": 2, "parse_type": "tool_call", "parse_confidence": 0.95}
    ms = list(_RE_SUBMIT.finditer(t))
    if ms:
        m   = ms[-1]
        raw = m.group(1).upper().strip()
        rsn = (m.group(2) or "").strip()
        act = ACTION_MAP.get(raw, ACTION_MAP.get(raw.replace("_", " "), 2))
        return {"action": act, "parse_type": "final_decision",
                "parse_confidence": 0.90 if rsn else 0.65}
    kms = list(_RE_KEYWORD.finditer(t))
    if kms:
        kw  = kms[-1].group(1).upper()
        act = ACTION_MAP.get(kw, 2)
        return {"action": act, "parse_type": "fallback_keyword", "parse_confidence": 0.55}
    return {"action": 2, "parse_type": "default_reject", "parse_confidence": 0.0}


def _extract_action(completion):
    p = parse_llm_output(_unwrap(completion))
    return p["action"], p["parse_type"], p["parse_confidence"]


print("✅ Action parser loaded")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 4: REWARD FUNCTIONS (v3 — gradient-aware) ═══
# ═══════════════════════════════════════════════════════════════════
#
# KEY INSIGHT: GRPO loss = advantage * log_prob_ratio
# advantage = (reward - mean_reward) / std_reward
# If all N generations get the SAME reward → std=0 → advantage=0 → loss=0
#
# So rewards MUST vary across the N generations for the same prompt.
# When the model is untrained, ALL outputs are garbage → same parse →
# same reward → zero loss → death spiral.
#
# FIX: Add continuous, content-based reward signals that naturally
# differ between completions even when all are "garbage":
#   - Length bonus (different garbage = different length)
#   - English coherence bonus (some garbage has more English words)
#   - Credit keyword density (some garbage accidentally uses domain terms)
#   - Repetition penalty (detect degenerate repetition loops)
# ═══════════════════════════════════════════════════════════════════

def _count_english_ratio(text: str) -> float:
    """Fraction of text that is ASCII/English vs non-English."""
    if not text: return 0.0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text)


def _count_credit_keywords(text: str) -> int:
    """Count credit-domain keywords in text."""
    t = text.lower()
    return sum(1 for kw in CREDIT_KEYWORDS if kw in t)


def _repetition_penalty(text: str) -> float:
    """Detect degenerate repetition. Returns penalty in [-1.0, 0.0]."""
    if len(text) < 20: return 0.0
    # Check for repeating n-grams (sign of degenerate output)
    words = text.split()
    if len(words) < 6: return 0.0
    # Check 3-gram repetition
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams: return 0.0
    unique_ratio = len(set(trigrams)) / len(trigrams)
    if unique_ratio < 0.3: return -1.0   # heavily repetitive
    if unique_ratio < 0.5: return -0.5
    return 0.0


def reward_correctness(prompts, completions, ground_truth_pd=None, **kwargs):
    """R1: PD-based decision quality + language quality signal [-2.0, +1.5]"""
    rewards = []
    for i, c in enumerate(completions):
        text = _unwrap(c)
        action, ptype, conf = _extract_action(c)
        pd = float(ground_truth_pd[i]) if ground_truth_pd else 0.5

        # Base decision reward
        if ptype == "default_reject":
            # No parseable decision — give continuous signal based on text quality
            eng_ratio = _count_english_ratio(text)
            kw_count  = _count_credit_keywords(text)
            rep_pen   = _repetition_penalty(text)
            # Score: English text with credit keywords > random garbage
            reward = -1.0 + eng_ratio * 0.5 + min(kw_count * 0.1, 0.3) + rep_pen
        else:
            # Has a parseable decision — reward based on correctness
            if   pd < 0.25: reward = {0: 1.0, 1: 0.3}.get(action, -0.3)
            elif pd < 0.45: reward = {1: 0.8, 2: 0.2}.get(action, -0.5)
            else:           reward = {2: 1.0, 1: 0.2}.get(action, -2.0)
            # Bonus for proper format
            if ptype == "final_decision": reward += 0.3
            elif ptype == "fallback_keyword": reward += 0.1

        rewards.append(float(np.clip(reward, -2.0, 1.5)))
    return rewards


def reward_hard_rule_compliance(prompts, completions, hard_rules=None,
                                has_red_alerts=None, **kwargs):
    """R2: RBI hard rule adherence [-2.0, +0.5]"""
    rewards = []
    for i, c in enumerate(completions):
        text = _unwrap(c)
        action, ptype, _ = _extract_action(c)
        prompt_str = _unwrap(prompts[i]) if i < len(prompts) else ""

        hr_raw = (hard_rules or [])[i] if hard_rules else []
        try:
            hr = json.loads(hr_raw) if isinstance(hr_raw, str) else list(hr_raw or [])
        except Exception:
            hr = []
        red = bool((has_red_alerts or [])[i]) if has_red_alerts else False

        if "🔴" in prompt_str or "[RED]" in prompt_str:
            if "HR-03" not in hr: hr.append("HR-03")
        m = re.search(r"DSCR:\s*([\d.]+)x", prompt_str)
        if m and float(m.group(1)) < 1.0:
            if "HR-01" not in hr: hr.append("HR-01")
        if red and "HR-03" not in hr: hr.append("HR-03")

        if ptype == "default_reject":
            # No parseable output — small continuous signal
            eng_ratio = _count_english_ratio(text)
            reward = -0.5 + eng_ratio * 0.3
        elif hr:
            reward = {2: 0.5, 1: -1.0}.get(action, -2.0)
        else:
            reward = 0.0

        rewards.append(float(np.clip(reward, -2.0, 0.5)))
    return rewards


def reward_format_compliance(prompts, completions, **kwargs):
    """R3: Output format quality — continuous signal [-1.0, +0.5]"""
    rewards = []
    for c in completions:
        text = _unwrap(c)
        _, pt, conf = _extract_action(c)

        if pt == "final_decision":
            reward = 0.5 if conf > 0.8 else 0.3
        elif pt == "fallback_keyword":
            reward = 0.1
        elif pt == "tool_call":
            reward = 0.0
        else:
            # No decision found — continuous signal
            eng_ratio = _count_english_ratio(text)
            kw_count  = _count_credit_keywords(text)
            rep_pen   = _repetition_penalty(text)
            # Length bonus (prefer non-trivial outputs, cap at 200 chars)
            len_bonus = min(len(text.strip()), 200) / 200.0 * 0.2
            reward = -0.8 + eng_ratio * 0.3 + min(kw_count * 0.05, 0.2) + len_bonus + rep_pen

        rewards.append(float(np.clip(reward, -1.0, 0.5)))
    return rewards


def reward_portfolio_awareness(prompts, completions, npa_rate=None,
                               crar=None, ground_truth_pd=None, **kwargs):
    """R4: Portfolio-state sensitivity [-0.8, +0.3]"""
    rewards = []
    for i, c in enumerate(completions):
        text = _unwrap(c)
        action, ptype, _ = _extract_action(c)
        npa = float(npa_rate[i])        if npa_rate        else 0.02
        cr  = float(crar[i])            if crar            else 0.18
        pd  = float(ground_truth_pd[i]) if ground_truth_pd else 0.5

        if ptype == "default_reject":
            # Continuous signal for garbage outputs
            eng = _count_english_ratio(text)
            reward = -0.3 + eng * 0.1
        else:
            reward = 0.0
            if npa > 0.08:
                if action == 0 and pd > 0.30: reward = -0.5
                elif action == 2: reward = 0.3
            if cr < 0.14 and action == 0: reward -= 0.3
            if npa < 0.03 and cr > 0.16 and action == 0 and pd < 0.20: reward = 0.2

        rewards.append(float(np.clip(reward, -0.8, 0.3)))
    return rewards


REWARD_FUNCS = [reward_correctness, reward_hard_rule_compliance,
                reward_format_compliance, reward_portfolio_awareness]

print("✅ 4 reward functions loaded (v3 — gradient-aware)")
print("   R1 correctness [-2.0,+1.5]  R2 hard_rules [-2.0,+0.5]")
print("   R3 format      [-1.0,+0.5]  R4 portfolio  [-0.8,+0.3]")
print("   ℹ️  All rewards now give continuous signal even for garbage outputs")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 5: VERIFY REWARD DIVERSITY (debug check) ═══
# ═══════════════════════════════════════════════════════════════════

print("\n🧪 Verifying reward diversity with fake completions...")

# Simulate different quality completions
test_completions = [
    [{"role": "assistant", "content": "submit_decision('APPROVE', 'Low risk borrower with strong DSCR')"}],
    [{"role": "assistant", "content": "I think we should REJECT this loan application"}],
    [{"role": "assistant", "content": "The risk assessment indicates potential default concerns"}],
    [{"role": "assistant", "content": "下面是小下面是小下面是小下面是小"}],  # Chinese gibberish
    [{"role": "assistant", "content": "aaaa bbbb cccc dddd"}],  # random garbage
    [{"role": "assistant", "content": ""}],  # empty
]
test_prompts = [test_completions[0]] * len(test_completions)  # dummy
test_pd      = [0.1] * len(test_completions)

for rf in REWARD_FUNCS:
    rewards = rf(test_prompts, test_completions, ground_truth_pd=test_pd,
                 hard_rules=["[]"]*len(test_completions),
                 has_red_alerts=[False]*len(test_completions),
                 npa_rate=[0.02]*len(test_completions),
                 crar=[0.18]*len(test_completions))
    unique = len(set(rewards))
    print(f"   {rf.__name__:35s} rewards={[f'{r:.2f}' for r in rewards]}  unique={unique}")
    if unique == 1:
        print(f"   ⚠️  WARNING: {rf.__name__} returns identical rewards for all!")

print("✅ Reward diversity check done\n")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 6: LOAD MODEL (HF + PEFT, no unsloth) ═══
# ═══════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading model: {MODEL_NAME}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float32,
    bnb_4bit_use_double_quant = True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map          = "auto",
    trust_remote_code   = True,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA config — higher rank for better capacity
lora_cfg = LoraConfig(
    r              = 32,
    lora_alpha     = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,     # small dropout for regularization
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)

print("✅ Model loaded + LoRA applied")
model.print_trainable_parameters()


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 7: TEST MODEL BEFORE TRAINING ═══
# ═══════════════════════════════════════════════════════════════════

print("\n🔍 Testing base model before training...")
test_prompt = "You are a credit analyst. Based on the following: PD=0.85, DSCR=0.7x, NPA=3%. What is your decision? Use submit_decision('DECISION', 'reason')."
messages = [{"role": "user", "content": test_prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"   Base model response: {resp[:200]}")
print(f"   Parse: {parse_llm_output(resp)}")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 8: LOAD & PREPARE DATASET ═══
# ═══════════════════════════════════════════════════════════════════

from huggingface_hub import hf_hub_download

print(f"\n🔄 Downloading dataset: {HF_DATASET}...")
jsonl_path = hf_hub_download(
    repo_id   = HF_DATASET,
    filename  = "grpo_dataset.jsonl",
    repo_type = "dataset",
)
print(f"   Downloaded: {jsonl_path}")

rows = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        row  = json.loads(line)
        meta = row.get("metadata", row)
        hr   = meta.get("hard_rules", [])
        rows.append({
            "prompt"          : row["prompt"],
            "task_id"         : meta.get("task_id", "task1"),
            "ground_truth_pd" : float(meta.get("ground_truth_pd", 0.5)),
            "optimal_action"  : int(meta.get("optimal_action", 2)),
            "hard_rules"      : json.dumps(hr) if isinstance(hr, list) else str(hr),
            "has_red_alerts"  : bool(meta.get("has_red_alerts", False)),
            "npa_rate"        : float(meta.get("npa_rate", 0.02)),
            "crar"            : float(meta.get("crar", 0.18)),
            "sector"          : str(meta.get("sector", "Unknown")),
        })

full_dataset = Dataset.from_list(rows)
print(f"   Total: {len(full_dataset)} samples")

task_counts = defaultdict(int)
for r in full_dataset:
    task_counts[r["task_id"]] += 1
print(f"   Tasks: {dict(sorted(task_counts.items()))}")


def format_prompt(example):
    """Wrap raw prompt text in chat template list."""
    return {"prompt": [{"role": "user", "content": example["prompt"]}]}


def filter_tasks(ds, tasks):
    return ds.filter(lambda x: x["task_id"] in tasks).map(format_prompt)


stage_datasets = {
    1: filter_tasks(full_dataset, ["task1"]),
    2: filter_tasks(full_dataset, ["task1", "task2"]),
    3: filter_tasks(full_dataset, list(task_counts.keys())),
}
for s, ds in stage_datasets.items():
    print(f"   Stage {s}: {len(ds)} samples")

print("✅ Dataset ready")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 9: STAGE CONFIGS (v3 — higher LR, more generations) ═══
# ═══════════════════════════════════════════════════════════════════

STAGE_CONFIGS = {
    1: {"name": "Stage 1: Easy (task1)",         "lr": 1e-4,  "steps": 100, "temp": 1.2, "epochs": 2},
    2: {"name": "Stage 2: Medium (task1+task2)", "lr": 5e-5,  "steps": 100, "temp": 1.0, "epochs": 2},
    3: {"name": "Stage 3: Full (all tasks)",     "lr": 2e-5,  "steps": 150, "temp": 0.9, "epochs": 3},
}

COMMON_ARGS = dict(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations             = 8,
    generation_batch_size       = 8,       # must be divisible by num_generations
    beta                        = 0.04,
    warmup_steps                = 10,
    logging_steps               = 1,
    save_steps                  = 50,
    report_to                   = "none",
    use_vllm                    = False,
    fp16                        = False,
    bf16                        = False,
)

print("✅ Stage configs ready (v3)")
for s, c in STAGE_CONFIGS.items():
    print(f"   {c['name']} — LR={c['lr']}, steps={c['steps']}, temp={c['temp']}")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 10: 3-STAGE TRAINING LOOP ═══
# ═══════════════════════════════════════════════════════════════════

from transformers import TrainerCallback

all_logs    = {1: [], 2: [], 3: []}
stage_times = {}


class RewardLogger(TrainerCallback):
    def __init__(self, stage, store):
        self.stage = stage
        self.store = store
        self._reward_printed = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {
                "step"          : state.global_step,
                "loss"          : logs.get("loss"),
                "reward"        : logs.get("reward"),
                "reward_std"    : logs.get("reward_std"),
                "kl"            : logs.get("kl"),
                "learning_rate" : logs.get("learning_rate"),
            }
            self.store[self.stage].append(entry)

            # Print detailed reward info for first 5 steps to debug
            if self._reward_printed < 5:
                self._reward_printed += 1
                r   = logs.get("reward", "?")
                std = logs.get("reward_std", "?")
                kl  = logs.get("kl", "?")
                print(f"   📊 Step {state.global_step}: reward={r}, reward_std={std}, kl={kl}")


print("\n" + "="*70)
print("  STARTING 3-STAGE GRPO CURRICULUM TRAINING (v3)")
print("="*70)

for stage in [1, 2, 3]:
    cfg = STAGE_CONFIGS[stage]
    ds  = stage_datasets[stage]

    print(f"\n{'─'*70}")
    print(f"  {cfg['name']}")
    print(f"  Samples: {len(ds)} | LR: {cfg['lr']} | Steps: {cfg['steps']} | Temp: {cfg['temp']}")
    print(f"{'─'*70}")

    stage_out = f"{OUTPUT_BASE}/stage_{stage}"
    os.makedirs(stage_out, exist_ok=True)

    training_args = GRPOConfig(
        output_dir          = stage_out,
        num_train_epochs    = cfg["epochs"],
        learning_rate       = cfg["lr"],
        temperature         = cfg["temp"],
        max_steps           = cfg["steps"],
        **COMMON_ARGS,
    )

    trainer = GRPOTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = ds,
        reward_funcs     = REWARD_FUNCS,
        processing_class = tokenizer,
        callbacks        = [RewardLogger(stage, all_logs)],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    stage_times[stage] = elapsed

    print(f"\n  ✅ Stage {stage} done in {elapsed/60:.1f} min")
    trainer.save_model(stage_out)
    tokenizer.save_pretrained(stage_out)
    print(f"  💾 Checkpoint: {stage_out}")

    # Quick sanity inference with chat template
    print(f"\n  🔍 Sample inference (3 prompts):")
    model.eval()
    for idx in range(min(3, len(ds))):
        sample     = ds[idx]
        prompt_txt = sample["prompt"][0]["content"] if isinstance(sample["prompt"], list) else sample["prompt"]

        # Use proper chat template for inference
        messages = [{"role": "user", "content": prompt_txt[:500]}]
        text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs   = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_SEQ_LEN).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100,
                                 do_sample=True, temperature=0.7, top_p=0.9)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_llm_output(resp)
        print(f"     [{idx+1}] PD={sample.get('ground_truth_pd','?'):.2f} "
              f"→ [{parsed['parse_type']}] {resp[:100]}...")
    model.train()

print(f"\n{'='*70}")
print(f"  ALL 3 STAGES COMPLETE ✅")
print(f"  Total: {sum(stage_times.values())/60:.1f} min")
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 11: LEARNING CURVES ═══
# ═══════════════════════════════════════════════════════════════════

print("\n📊 Generating charts...")

all_steps, all_losses, all_rewards, all_kl, all_lr = [], [], [], [], []
stage_boundaries = []
offset = 0

for stage in [1, 2, 3]:
    logs = all_logs[stage]
    if not logs: continue
    stage_boundaries.append(offset)
    for e in logs:
        s = e["step"] + offset
        all_steps.append(s)
        if e.get("loss")          is not None: all_losses.append((s, e["loss"]))
        if e.get("reward")        is not None: all_rewards.append((s, e["reward"]))
        if e.get("kl")            is not None: all_kl.append((s, e["kl"]))
        if e.get("learning_rate") is not None: all_lr.append((s, e["learning_rate"]))
    offset += max(e["step"] for e in logs) + 1

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("IntelliCredit GRPO — Qwen2.5-1.5B (3-Stage Curriculum v3)", fontsize=15, fontweight="bold")

def _plot(ax, data, color, title, ylabel):
    if not data: return
    xs, ys = zip(*data)
    ax.plot(xs, ys, color=color, alpha=0.3, linewidth=0.5)
    if len(ys) > 5:
        w = min(10, len(ys)//3)
        sm = np.convolve(ys, np.ones(w)/w, mode="valid")
        ax.plot(xs[:len(sm)], sm, color=color, linewidth=2, label="Smoothed")
    for b in stage_boundaries[1:]:
        ax.axvline(x=b, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel)
    ax.set_xlabel("Step"); ax.grid(True, alpha=0.3); ax.legend()

_plot(axes[0,0], all_losses,  "#E53935", "Training Loss",           "Loss")
_plot(axes[0,1], all_rewards, "#1976D2", "Mean Reward ↑",           "Reward")
_plot(axes[1,0], all_kl,      "#7B1FA2", "KL Divergence",           "KL")
_plot(axes[1,1], all_lr,      "#00796B", "Learning Rate Schedule",  "LR")

plt.tight_layout()
p1 = f"{OUTPUT_BASE}/charts/training_overview.png"
plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.show()
print(f"  💾 {p1}")

# Per-stage reward progression
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("Per-Stage Reward Progression", fontsize=14, fontweight="bold")
for i, stage in enumerate([1, 2, 3]):
    logs = all_logs[stage]
    if not logs: continue
    rewards = [(e["step"], e["reward"]) for e in logs if e.get("reward") is not None]
    if rewards:
        xs, ys = zip(*rewards)
        axes2[i].plot(xs, ys, color="#1976D2", alpha=0.4)
        if len(ys) > 3:
            w = min(5, len(ys)//2)
            sm = np.convolve(ys, np.ones(w)/w, mode="valid")
            axes2[i].plot(xs[:len(sm)], sm, color="#E53935", linewidth=2)
        axes2[i].set_title(STAGE_CONFIGS[stage]["name"])
        axes2[i].set_xlabel("Step"); axes2[i].set_ylabel("Reward")
        axes2[i].grid(True, alpha=0.3)
plt.tight_layout()
p2 = f"{OUTPUT_BASE}/charts/per_stage_rewards.png"
plt.savefig(p2, dpi=150); plt.show()
print(f"  💾 {p2}")

# Raw logs
with open(f"{OUTPUT_BASE}/training_logs.json", "w") as f:
    json.dump({"stage_1": all_logs[1], "stage_2": all_logs[2],
               "stage_3": all_logs[3], "stage_times": stage_times,
               "model": MODEL_NAME, "dataset": HF_DATASET}, f, indent=2, default=str)
print("✅ Charts done!")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 12: SAVE FINAL MODEL ═══
# ═══════════════════════════════════════════════════════════════════

print(f"\n💾 Saving final model → {FINAL_MODEL}")
model.save_pretrained(FINAL_MODEL)
tokenizer.save_pretrained(FINAL_MODEL)
print("✅ Done!")

if PUSH_TO_HUB:
    model.push_to_hub(HUB_REPO)
    tokenizer.push_to_hub(HUB_REPO)
    print(f"☁️  Pushed to https://huggingface.co/{HUB_REPO}")


# ═══════════════════════════════════════════════════════════════════
# ═══ CELL 13: TRAINING SUMMARY ═══
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  📊 FINAL TRAINING SUMMARY")
print("="*70)
for stage in [1, 2, 3]:
    logs = all_logs[stage]
    if not logs: continue
    rewards = [e["reward"] for e in logs if e.get("reward") is not None]
    losses  = [e["loss"]   for e in logs if e.get("loss")   is not None]
    print(f"\n  Stage {stage}: {STAGE_CONFIGS[stage]['name']}")
    print(f"    Steps : {len(logs)}  |  Time: {stage_times.get(stage,0)/60:.1f} min")
    if rewards:
        q = max(1, len(rewards)//4)
        print(f"    Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        print(f"    First→Last quartile: {np.mean(rewards[:q]):.4f} → {np.mean(rewards[-q:]):.4f}")
    if losses:
        print(f"    Loss  : {np.mean(losses):.4f} (final: {losses[-1]:.4f})")

print(f"\n  Total training time: {sum(stage_times.values())/60:.1f} min")
print(f"  Final model saved: {FINAL_MODEL}/")
