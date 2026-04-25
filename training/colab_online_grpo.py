"""
╔══════════════════════════════════════════════════════════════════╗
║  IntelliCredit v2 — ONLINE GRPO (Multi-Turn Tool Calling)        ║
║  Model  : Qwen/Qwen2.5-1.5B-Instruct  (4-bit QLoRA, no unsloth) ║
║  Dataset: vssksn/intellicredit-grpo-v2 (5000 samples)           ║
║  v4.0   : Fixed submit_decision() gap from offline training      ║
║                                                                  ║
║  Key fixes vs v3 offline:                                        ║
║  ✅ Multi-turn tool calling with synthesized results             ║
║  ✅ STRONG reward for submit_decision() format                   ║
║  ✅ Tool results synthesized from prompt data (no server needed) ║
║  ✅ Custom GRPO loss — no GRPOTrainer limitation                 ║
║  ✅ Memory bank for cross-step lesson injection                  ║
║  ✅ 3-stage curriculum (tasks 1→5)                               ║
╚══════════════════════════════════════════════════════════════════╝

Paste each ═══ CELL ═══ block into a separate Colab cell.
"""


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 1: INSTALL  (run once, then Runtime → Restart session) ═══
# ════════════════════════════════════════════════════════════════════

import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# Core training stack — version-pinned to avoid conflicts on Colab T4
_pip("--upgrade", "pip")
_pip("bitsandbytes>=0.46.1")                        # REQUIRED for 4-bit QLoRA
_pip("transformers>=4.45.0,<5.0.0")                 # stable API
_pip("trl>=0.15.2", "peft>=0.13.0", "accelerate>=1.0.0")
_pip("datasets>=2.20.0", "huggingface_hub>=0.24.0", "matplotlib")

# ── HF Token (optional but avoids rate-limit warnings) ──
# In Colab: Secrets (🔑 icon) → add HF_TOKEN with your token from
#   https://huggingface.co/settings/tokens
import os
try:
    from google.colab import userdata
    hf_tok = userdata.get("HF_TOKEN")
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok
        from huggingface_hub import login
        login(token=hf_tok, add_to_git_credential=False)
        print("✅ HF Token loaded from Colab secrets")
    else:
        print("⚠️  HF_TOKEN not set — using anonymous HF access (rate-limited but OK for public models)")
except Exception:
    print("⚠️  Not in Colab or no HF_TOKEN secret — anonymous HF access")

print("\n✅ All packages installed")
print("   ⚡ IMPORTANT: Runtime → Restart session → then run from Cell 2")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 2: IMPORTS & CONFIG ═══
# ════════════════════════════════════════════════════════════════════

import os, re, json, time, math, random, copy
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen2.5-1.5B-Instruct"
HF_DATASET    = "vssksn/intellicredit-grpo-v2"
OUTPUT_BASE   = "intellicredit-online-grpo"
FINAL_MODEL   = "qwen-intellicredit-online-final"

# Online GRPO hyper-params
NUM_GENERATIONS  = 6       # completions per prompt (must be ≥ 2 for contrast)
MAX_NEW_TOKENS   = 256     # longer = model can fit submit_decision
MAX_TOOL_TURNS   = 3       # max tool calls per episode before forcing decision
BATCH_SIZE       = 2       # prompts per update step (×NUM_GENERATIONS completions)
GRAD_ACCUM       = 4       # effective batch = BATCH_SIZE × GRAD_ACCUM
KL_BETA          = 0.04    # KL penalty coefficient
MAX_SEQ_LEN      = 1536

# ── Multi-Step GRPO flag ─────────────────────────────────────────────
# True  → sum log-probs over ALL assistant turns (tool calls + final)
#         This is TRUE multi-step GRPO: the gradient flows through every
#         turn in the episode, not just the final submit_decision().
#         Tool-calling behaviour is directly optimised.
# False → only the final turn contributes to the loss (simpler, faster)
#         Equivalent to outcome-supervised GRPO on single completions.
MULTI_STEP_GRPO  = True

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/charts", exist_ok=True)

print("✅ Imports complete")
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"   GPU  : {gpu}")
print(f"   VRAM : {vram:.1f} GB")
print(f"   Model: {MODEL_NAME}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 3: ACTION PARSER (inlined — no server import needed) ═══
# ════════════════════════════════════════════════════════════════════

ACTION_MAP = {
    "APPROVE": 0, "APPROVED": 0,
    "CONDITIONAL": 1, "CONDITIONAL_APPROVE": 1, "CONDITIONAL APPROVE": 1,
    "REJECT": 2, "REJECTED": 2, "DECLINE": 2, "DENY": 2,
}

_RE_SUBMIT  = re.compile(
    r"submit_decision\s*\(\s*['\"]?([A-Z_]+)['\"]?\s*,\s*['\"]?(.*?)['\"]?\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_RE_TOOL = re.compile(
    r"\b(get_financial_report|check_compliance_status|get_market_intelligence)"
    r"\s*\(\s*['\"]?([^)'\"]*)['\"]?\s*\)",
    re.IGNORECASE,
)
_RE_KEYWORD = re.compile(
    r"\b(APPROVE(?:D)?|CONDITIONAL(?:_APPROVE)?|REJECT(?:ED)?|DECLINE|DENY)\b",
    re.IGNORECASE,
)

CREDIT_KEYWORDS = [
    "risk", "credit", "loan", "dscr", "npa", "crar", "approve",
    "reject", "conditional", "compliance", "rbi", "capital", "borrower",
    "default", "probability", "portfolio", "decision", "financial",
    "assessment", "collateral", "ratio", "debt",
]


def _unwrap(text) -> str:
    if isinstance(text, list):
        return " ".join(m.get("content", "") for m in text if isinstance(m, dict))
    return str(text) if not isinstance(text, str) else text


def parse_llm_output(text) -> Dict[str, Any]:
    text = _unwrap(text)
    if not text.strip():
        return {"action": 2, "parse_type": "default_reject",
                "parse_confidence": 0.0, "reasoning": ""}
    t = text.strip()
    # Priority 1: tool call
    tm = _RE_TOOL.search(t)
    if tm:
        raw_arg   = tm.group(2).strip().strip("'\"")
        tool_name = tm.group(1).lower()
        # env-compatible: match actual parameter name in tool_executor.py
        #   get_financial_report(company_id)    → {"company_id": ...}
        #   check_compliance_status(company_id) → {"company_id": ...}
        #   get_market_intelligence(sector)     → {"sector": ...}
        tool_args_dict = (
            {"sector": raw_arg}
            if "market_intelligence" in tool_name
            else {"company_id": raw_arg}
        )
        return {
            "action": 2, "parse_type": "tool_call",
            "tool_name": tool_name,
            "tool_arg":  raw_arg,          # convenience: plain string
            "tool_args": tool_args_dict,   # env-compatible: correct key per tool
            "parse_confidence": 0.95,
            "reasoning":  f"calling {tool_name}",
            "parse_failure": False,
        }
    # Priority 2: submit_decision
    ms = list(_RE_SUBMIT.finditer(t))
    if ms:
        m   = ms[-1]
        raw = m.group(1).upper().strip()
        rsn = (m.group(2) or "").strip()
        act = ACTION_MAP.get(raw, ACTION_MAP.get(raw.replace("_", " "), 2))
        return {
            "action": act, "parse_type": "final_decision",
            "parse_confidence": 0.90 if rsn else 0.65,
            "reasoning": rsn, "tool_name": None, "tool_args": None,
            "parse_failure": False,
        }
    # Priority 3: keyword
    kms = list(_RE_KEYWORD.finditer(t))
    if kms:
        kw  = kms[-1].group(1).upper()
        act = ACTION_MAP.get(kw, 2)
        return {
            "action": act, "parse_type": "fallback_keyword",
            "parse_confidence": 0.55, "reasoning": t[-100:],
            "tool_name": None, "tool_args": None,
            "parse_failure": True,   # partial parse = soft failure
        }
    # Default
    return {
        "action": 2, "parse_type": "default_reject",
        "parse_confidence": 0.0, "reasoning": "",
        "tool_name": None, "tool_args": None,
        "parse_failure": True,
    }


print("✅ Action parser loaded")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 4: REWARD FUNCTIONS (v4 — submit_decision boosted) ═══
# ════════════════════════════════════════════════════════════════════
#
#  GRPO requires reward VARIANCE across the N completions/prompt.
#  Key fix: submit_decision gets a big bonus (+0.5) so the model
#  learns to prefer it over fallback_keyword or default_reject.
#
# ════════════════════════════════════════════════════════════════════

def _count_english_ratio(text: str) -> float:
    if not text: return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def _count_credit_kw(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in CREDIT_KEYWORDS if kw in t)


def _rep_penalty(text: str) -> float:
    words = text.split()
    if len(words) < 6: return 0.0
    tg = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    if not tg: return 0.0
    ratio = len(set(tg)) / len(tg)
    return -1.0 if ratio < 0.3 else (-0.5 if ratio < 0.5 else 0.0)


def _base_continuous(text: str) -> float:
    """Continuous signal for unparseable outputs."""
    return (-1.0 + _count_english_ratio(text) * 0.5
            + min(_count_credit_kw(text) * 0.1, 0.3)
            + _rep_penalty(text))


def reward_correctness(prompts, completions, ground_truth_pd=None, **kwargs):
    """R1: PD-based correctness + FORMAT bonus for submit_decision. [-2.0, +2.0]"""
    rewards = []
    for i, c in enumerate(completions):
        text   = _unwrap(c)
        parsed = parse_llm_output(text)
        action, ptype, conf = parsed["action"], parsed["parse_type"], parsed["parse_confidence"]
        pd = float(ground_truth_pd[i]) if ground_truth_pd else 0.5

        if ptype == "default_reject":
            reward = _base_continuous(text)
        else:
            # Correctness score based on PD
            if pd < 0.25:
                reward = {0: 1.0, 1: 0.3}.get(action, -0.3)
            elif pd < 0.45:
                reward = {1: 0.8, 2: 0.2}.get(action, -0.5)
            else:
                reward = {2: 1.0, 1: 0.2}.get(action, -2.0)

            # FORMAT BONUS — this is the key fix for the offline training gap
            if ptype == "final_decision":     reward += 0.5   # submit_decision used ✅
            elif ptype == "fallback_keyword": reward += 0.1   # partial credit
            # tool_call: no correctness bonus (haven't decided yet)

        rewards.append(float(np.clip(reward, -2.0, 2.0)))
    return rewards


def reward_hard_rule_compliance(prompts, completions, hard_rules=None,
                                has_red_alerts=None, **kwargs):
    """R2: RBI hard rule adherence. [-2.0, +0.5]"""
    rewards = []
    for i, c in enumerate(completions):
        text   = _unwrap(c)
        parsed = parse_llm_output(text)
        action, ptype = parsed["action"], parsed["parse_type"]
        prompt_str = _unwrap(prompts[i]) if i < len(prompts) else ""

        hr_raw = (hard_rules or [])[i] if hard_rules else []
        try:
            hr = json.loads(hr_raw) if isinstance(hr_raw, str) else list(hr_raw or [])
        except Exception:
            hr = []
        red = bool((has_red_alerts or [])[i]) if has_red_alerts else False

        # Detect hard rules from prompt text
        if "🔴" in prompt_str or "[RED]" in prompt_str:
            if "HR-03" not in hr: hr.append("HR-03")
        m = re.search(r"DSCR:\s*([\d.]+)x", prompt_str)
        if m and float(m.group(1)) < 1.0:
            if "HR-01" not in hr: hr.append("HR-01")
        m2 = re.search(r"[Cc]heque\s*[Bb]ounce\s*[Rr]ate:\s*([\d.]+)%", prompt_str)
        if m2 and float(m2.group(1)) > 25:
            if "HR-04" not in hr: hr.append("HR-04")
        if red and "HR-03" not in hr: hr.append("HR-03")

        if ptype == "default_reject":
            reward = -0.5 + _count_english_ratio(text) * 0.3
        elif hr:
            reward = {2: 0.5, 1: -1.0}.get(action, -2.0)
        else:
            reward = 0.0

        rewards.append(float(np.clip(reward, -2.0, 0.5)))
    return rewards


def reward_format_compliance(prompts, completions, **kwargs):
    """R3: Format quality — heavy bonus for submit_decision. [-1.0, +1.0]"""
    rewards = []
    for c in completions:
        text   = _unwrap(c)
        parsed = parse_llm_output(text)
        ptype, conf = parsed["parse_type"], parsed["parse_confidence"]
        rsn = parsed.get("reasoning", "")

        if ptype == "final_decision":
            # submit_decision with good reasoning = top reward
            if conf > 0.8 and len(rsn) > 20:  reward = 1.0
            elif conf > 0.8:                   reward = 0.7
            else:                              reward = 0.4  # no reasoning
        elif ptype == "fallback_keyword":
            reward = 0.1
        elif ptype == "tool_call":
            reward = 0.15   # tool calls are fine but not the goal
        else:
            eng = _count_english_ratio(text)
            kw  = _count_credit_kw(text)
            lb  = min(len(text.strip()), 200) / 200.0 * 0.2
            reward = -0.8 + eng * 0.3 + min(kw * 0.05, 0.2) + lb + _rep_penalty(text)

        rewards.append(float(np.clip(reward, -1.0, 1.0)))
    return rewards


def reward_portfolio_awareness(prompts, completions, npa_rate=None,
                               crar=None, ground_truth_pd=None, **kwargs):
    """R4: Portfolio sensitivity. [-0.8, +0.3]"""
    rewards = []
    for i, c in enumerate(completions):
        text   = _unwrap(c)
        parsed = parse_llm_output(text)
        action, ptype = parsed["action"], parsed["parse_type"]
        npa = float(npa_rate[i])        if npa_rate        else 0.02
        cr  = float(crar[i])            if crar            else 0.18
        pd  = float(ground_truth_pd[i]) if ground_truth_pd else 0.5

        if ptype == "default_reject":
            reward = -0.3 + _count_english_ratio(text) * 0.1
        else:
            reward = 0.0
            if npa > 0.08:
                if action == 0 and pd > 0.30: reward = -0.5
                elif action == 2:             reward = 0.3
            if cr < 0.14 and action == 0:    reward -= 0.3
            if npa < 0.03 and cr > 0.16 and action == 0 and pd < 0.20:
                reward = 0.2

        rewards.append(float(np.clip(reward, -0.8, 0.3)))
    return rewards


def combined_reward(prompts, completions, **kwargs) -> List[float]:
    r1 = reward_correctness(prompts, completions, **kwargs)
    r2 = reward_hard_rule_compliance(prompts, completions, **kwargs)
    r3 = reward_format_compliance(prompts, completions, **kwargs)
    r4 = reward_portfolio_awareness(prompts, completions, **kwargs)
    return [round(r1[i] + r2[i] + r3[i] + r4[i], 4) for i in range(len(completions))]


REWARD_FUNCS = [reward_correctness, reward_hard_rule_compliance,
                reward_format_compliance, reward_portfolio_awareness]

print("✅ Reward functions v4 loaded")
print("   R1 correctness  [-2.0,+2.0] — +0.5 bonus for submit_decision")
print("   R2 hard_rules   [-2.0,+0.5]")
print("   R3 format       [-1.0,+1.0] — +1.0 bonus for submit_decision+reasoning")
print("   R4 portfolio    [-0.8,+0.3]")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 5: TOOL RESULT SYNTHESIZER ═══
# ════════════════════════════════════════════════════════════════════
#
#  The dataset prompts already contain all financial data.
#  When the model calls a tool, we extract that data from the prompt
#  and re-format it as a structured tool result — making the tool
#  call genuinely useful (structured view of data already in prompt).
#
# ════════════════════════════════════════════════════════════════════

def _extract_float(text: str, pattern: str, default: float = 0.0) -> float:
    m = re.search(pattern, text)
    try:
        return float(m.group(1)) if m else default
    except Exception:
        return default


def synthesize_tool_result(tool_name: str, tool_arg: str, prompt: str) -> str:
    """
    Generate a structured tool result by extracting values from the
    prompt text.  No server call needed — data is already in the prompt.
    """
    tool_name = tool_name.lower()
    p = prompt  # shorthand

    if tool_name == "get_financial_report":
        com   = re.search(r"CREDIT APPRAISAL SUMMARY[^\n]*?—\s*(.+?)\s*═", p)
        cname = com.group(1).strip() if com else tool_arg or "Applicant"
        dscr  = _extract_float(p, r"DSCR:\s*([\d.]+)x", 1.5)
        cr    = _extract_float(p, r"[Cc]urrent\s*[Rr]atio:\s*([\d.]+)", 1.2)
        de    = _extract_float(p, r"[Dd]ebt.to.[Ee]quity:\s*([\d.]+)", 1.0)
        eb    = _extract_float(p, r"EBITDA\s*[Mm]argin:\s*([\d.]+)%", 15.0)
        rev   = re.search(r"[Rr]evenue:\s*₹([\d.]+)\s*Cr", p)
        nw    = re.search(r"[Nn]et\s*[Ww]orth:\s*₹([\d.]+)\s*Cr", p)
        loan  = re.search(r"[Ll]oan\s*[Rr]equested:\s*₹([\d.]+)\s*Cr", p)
        cibil = _extract_float(p, r"CIBIL\s*[Ss]core:\s*([\d.]+)", 650.0)
        ronw  = _extract_float(p, r"[Rr]eturn\s*on\s*[Nn]et\s*[Ww]orth:\s*([\d.]+)%", 10.0)
        coll  = _extract_float(p, r"[Cc]ollateral\s*[Cc]overage:\s*([\d.]+)x", 1.0)
        od    = _extract_float(p, r"OD\s*[Uu]tilisation:\s*([\d.]+)%", 50.0)
        ccvol = _extract_float(p, r"[Cc][Cc]\s*[Vv]olatility:\s*([\d.]+)%", 15.0)
        bbr   = _extract_float(p, r"[Cc]heque\s*[Bb]ounce\s*[Rr]ate:\s*([\d.]+)%", 5.0)
        wc    = _extract_float(p, r"[Ww]orking\s*[Cc]apital\s*[Cc]ycle:\s*([\d.]+)\s*days", 60.0)
        gst   = _extract_float(p, r"GST\s*[Tt]urnover\s*CAGR:\s*([-\d.]+)%", 5.0)

        dscr_flag = " ⚠️ BELOW 1.0 — HR-01 TRIGGERED" if dscr < 1.0 else (" ✅ Strong" if dscr >= 1.5 else " ✓ Adequate")
        bb_flag   = " ⚠️ ABOVE 25% — HR-04 TRIGGERED" if bbr > 25 else ""

        lines = [
            f"═══ FINANCIAL REPORT: {cname} ═══",
            "",
            "── Key Financial Ratios ──",
            f"  DSCR              : {dscr:.2f}x{dscr_flag}",
            f"  Current Ratio     : {cr:.2f}x  {'✅' if cr >= 1.2 else '⚠️'}",
            f"  Debt-to-Equity    : {de:.2f}x  {'✅' if de <= 2.0 else '⚠️ High leverage'}",
            f"  EBITDA Margin     : {eb:.1f}%  {'✅' if eb >= 15 else '⚠️ Thin margin'}",
            f"  CIBIL Score       : {int(cibil)}  {'✅' if cibil >= 700 else ('⚠️' if cibil >= 650 else '🚫 High risk')}",
            f"  RONW              : {ronw:.1f}%",
            f"  Collateral Cover  : {coll:.2f}x  {'✅' if coll >= 1.25 else '⚠️ Thin cover'}",
            "",
            "── Scale & Request ──",
            f"  Revenue           : ₹{rev.group(1) if rev else 'N/A'} Cr",
            f"  Net Worth         : ₹{nw.group(1)  if nw  else 'N/A'} Cr",
            f"  Loan Requested    : ₹{loan.group(1) if loan else 'N/A'} Cr",
            "",
            "── Banking Behaviour ──",
            f"  OD Utilisation    : {od:.1f}%  {'⚠️ High' if od > 75 else '✅'}",
            f"  CC Volatility     : {ccvol:.1f}%",
            f"  Cheque Bounce     : {bbr:.1f}%{bb_flag}",
            f"  WC Cycle          : {wc:.0f} days",
            "",
            "── GST & Turnover ──",
            f"  GST Turnover CAGR : {gst:+.1f}%  {'⚠️ Declining' if gst < 0 else '✅ Growing'}",
            "═══ END FINANCIAL REPORT ═══",
        ]
        return "\n".join(lines)

    elif tool_name == "check_compliance_status":
        com   = re.search(r"CREDIT APPRAISAL SUMMARY[^\n]*?—\s*(.+?)\s*═", p)
        cname = com.group(1).strip() if com else tool_arg or "Applicant"

        # Hard rules
        hr_lines = []
        dscr_val = _extract_float(p, r"DSCR:\s*([\d.]+)x", 1.5)
        if dscr_val < 1.0:
            hr_lines.append(f"   🚫 HR-01: DSCR = {dscr_val:.2f}x (below 1.0) — MANDATORY REJECT")
        bounce = _extract_float(p, r"[Cc]heque\s*[Bb]ounce\s*[Rr]ate:\s*([\d.]+)%", 0.0)
        if bounce > 25:
            hr_lines.append(f"   🚫 HR-04: Cheque bounce = {bounce:.1f}% (above 25%) — MANDATORY REJECT")
        gst_comp = _extract_float(p, r"GST.*?compliance.*?([\d.]+)%", 100.0)
        if gst_comp < 40:
            hr_lines.append(f"   🚫 HR-05: GST compliance = {gst_comp:.0f}% (below 40%) — MANDATORY REJECT")

        # Forensic alerts from prompt
        alert_lines = []
        for sev, icon in [("🔴", "🔴 RED"), ("🟡", "🟡 AMBER")]:
            for am in re.finditer(rf"\{sev}\s*\[(\w+)\]\s*(\w[\w_]+):\s*(.+?)(?=\n|$)", p):
                alert_lines.append(f"   {icon} | {am.group(2)}: {am.group(3).strip()}")

        is_repeat = bool(re.search(r"is_repeat.*?true|REPEAT\s*APPLICANT", p, re.IGNORECASE))
        lit = _extract_float(p, r"[Pp]romoter\s*[Ll]itigation\s*[Cc]ases:\s*([\d]+)", 0)
        rpt = re.search(r"[Rr]elated\s*[Pp]arty\s*[Tt]ransactions:\s*([\d.]+)%", p)
        circ = re.search(r"[Cc]ircular\s*[Tt]rading:\s*([\d.]+)%", p)

        lines = [
            f"═══ COMPLIANCE STATUS: {cname} ═══",
            "",
        ]
        if is_repeat:
            lines.append("  ⚠️  REPEAT APPLICANT — previously rejected")
        if hr_lines:
            lines.append("── HARD RULES TRIGGERED (MANDATORY REJECT) ──")
            lines.extend(hr_lines)
        else:
            lines.append("  ✅ No hard rules triggered")
        lines.append("")
        if alert_lines:
            lines.append("── Forensic Alerts ──")
            lines.extend(alert_lines)
        else:
            lines.append("  ✅ No forensic alerts")
        lines.append("")
        lines.append("── Governance ──")
        lines.append(f"  Promoter Litigation Cases : {int(lit)}")
        if rpt:
            rpt_pct = float(rpt.group(1))
            lines.append(f"  Related Party Transactions: {rpt_pct:.1f}%  {'⚠️ High' if rpt_pct > 20 else '✅'}")
        if circ:
            circ_pct = float(circ.group(1))
            lines.append(f"  Circular Trading          : {circ_pct:.1f}%  {'⚠️ Suspicious' if circ_pct > 15 else '✅'}")
        lines.append("═══ END COMPLIANCE ═══")
        return "\n".join(lines)

    elif tool_name == "get_market_intelligence":
        sector_arg = tool_arg or "Unknown"
        m_stress = re.search(r"[Mm]acro\s*[Ee]nvironment.*?stress=([\d.]+)", p)
        stress = float(m_stress.group(1)) if m_stress else 0.25
        npa = _extract_float(p, r"NPA\s*[Rr]ate:\s*([\d.]+)%", 2.0) / 100
        crar = _extract_float(p, r"CRAR:\s*([\d.]+)%", 18.0) / 100
        cap_dep = _extract_float(p, r"[Cc]apital\s*[Dd]eployed:\s*([\d.]+)%", 0.0)
        rem = re.search(r"[Rr]emaining:\s*₹([\d.]+)\s*Cr", p)

        stress_label = "HIGH STRESS ⚠️" if stress > 0.6 else ("MODERATE" if stress > 0.35 else "STABLE ✅")
        npa_flag     = "⚠️ Approaching limit" if npa > 0.04 else ("🚫 ABOVE 5% LIMIT" if npa >= 0.05 else "✅")
        crar_flag    = "🚫 BELOW 12.5% MINIMUM" if crar < 0.125 else ("⚠️ Thin buffer" if crar < 0.15 else "✅")

        lines = [
            f"═══ MARKET INTELLIGENCE: {sector_arg} ═══",
            "",
            "── Macro Environment ──",
            f"  Macro Stress     : {stress:.2f} ({stress_label})",
            f"  Macro Shock      : {'YES ⚠️' if stress > 0.5 else 'No'}",
            "",
            "── Portfolio Status ──",
            f"  NPA Rate         : {npa:.1%}  {npa_flag}",
            f"  CRAR             : {crar:.1%}  {crar_flag}",
            f"  Capital Deployed : {cap_dep:.1f}%",
            f"  Capital Remaining: ₹{rem.group(1) if rem else 'N/A'} Cr",
            "",
            "── Sector Intelligence ──",
        ]

        # Sector-specific commentary
        sector_comments = {
            "it": ["  • IT sector headwinds from global slowdown", "  • Peer NPA avg ~1.8%"],
            "manufacturing": ["  • Input cost pressures post-commodity spike", "  • Peer NPA avg ~3.2%"],
            "real estate": ["  • Inventory overhang in Tier-2 cities", "  • RBI advisory: heightened caution"],
            "textile": ["  • Export slowdown — USD/INR volatility", "  • Peer NPA avg ~4.1%"],
            "pharma": ["  • Stable domestic demand", "  • Export growth positive"],
            "food": ["  • Seasonal working capital spikes", "  • Stable demand fundamentals"],
            "construction": ["  • Infra spending boost from Govt capex", "  • Watch for receivable delays"],
        }
        matched = False
        for key, comments in sector_comments.items():
            if key in sector_arg.lower():
                lines.extend(comments)
                matched = True
                break
        if not matched:
            lines.append(f"  • No adverse RBI advisory for {sector_arg}")
            lines.append(f"  • Maintain standard underwriting criteria")

        lines.append("═══ END MARKET INTELLIGENCE ═══")
        return "\n".join(lines)

    return f"[TOOL ERROR] Unknown tool: {tool_name}. Supported: get_financial_report, check_compliance_status, get_market_intelligence"


print("✅ Tool result synthesizer loaded")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 6: LIGHTWEIGHT MEMORY BANK ═══
# ════════════════════════════════════════════════════════════════════

class MemoryBank:
    """
    Stores lessons from failed episodes and injects them into the
    system prompt.  This is the 'Phase 5 Reflection' feature —
    the agent becomes self-improving without weight updates.
    """
    def __init__(self, max_lessons: int = 8):
        self.lessons: List[Dict] = []
        self.max_lessons = max_lessons

    def add_lesson(self, lesson: str, reward: float, parse_type: str):
        if reward < 0.5:   # only learn from failures
            self.lessons.append({
                "lesson": lesson, "reward": reward,
                "parse_type": parse_type, "count": 1,
            })
        # Deduplicate and keep worst performers
        seen = {}
        for l in self.lessons:
            key = l["lesson"][:50]
            if key not in seen or l["reward"] < seen[key]["reward"]:
                seen[key] = l
        self.lessons = sorted(seen.values(), key=lambda x: x["reward"])[:self.max_lessons]

    def inject_prompt_suffix(self) -> str:
        if not self.lessons:
            return ""
        lines = ["", "── LESSONS FROM RECENT DECISIONS ──"]
        for i, l in enumerate(self.lessons[-3:], 1):
            lines.append(f"  Lesson {i}: {l['lesson']}")
        return "\n".join(lines)

    def learn_from_episode(self, prompt: str, final_text: str,
                           reward: float, parse_type: str):
        if parse_type == "default_reject":
            self.add_lesson("Always end with submit_decision('ACTION', 'reason')", reward, parse_type)
        elif parse_type == "fallback_keyword":
            self.add_lesson("Use submit_decision() format, not just a keyword", reward, parse_type)
        elif reward < 0.0 and parse_type == "final_decision":
            # Extract what went wrong
            dscr = _extract_float(prompt, r"DSCR:\s*([\d.]+)x", 9.9)
            if dscr < 1.0:
                self.add_lesson("DSCR < 1.0 → always REJECT (HR-01)", reward, parse_type)
            self.add_lesson(f"Negative reward ({reward:.2f}): review hard rules before deciding", reward, parse_type)


MEMORY_BANK = MemoryBank()
print("✅ Memory bank initialised")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 7: SYSTEM PROMPT BUILDER ═══
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_BASE = """\
You are a Senior Credit Officer at an Indian NBFC. Your job is to review MSME loan applications and make APPROVE / CONDITIONAL / REJECT decisions that balance yield, risk, and RBI regulatory compliance.

══ AVAILABLE TOOLS ══
Call up to {max_tool_turns} tools to gather information before your decision.

  get_financial_report("company_name")
    → Returns: 3-year revenue trends, DSCR, current ratio, D/E, EBITDA, CIBIL, collateral, OD utilisation.

  check_compliance_status("company_name")
    → Returns: Hard rules triggered, GST compliance, forensic alerts, litigation cases, repeat applicant flag.

  get_market_intelligence("sector_name")
    → Returns: Macro stress, sector headwinds, portfolio NPA, CRAR, capital utilisation.

══ RBI HARD RULES (MANDATORY REJECT if ANY triggered) ══
  HR-01: DSCR < 1.0               HR-04: Cheque bounce > 25%
  HR-02: Director disqualified     HR-05: GST compliance < 40%
  HR-03: RED forensic alert        HR-06: Severe adverse media > 0.80

══ REGULATORY LIMITS ══
  • CRAR must stay above 12.5%  • NPA rate must stay below 5%
  • Sector concentration < 30%  • Single borrower < 15%

══ OUTPUT FORMAT — YOU MUST END WITH THIS EXACT CALL ══
  submit_decision("APPROVE",      "your 20+ word reasoning here")
  submit_decision("CONDITIONAL",  "your 20+ word reasoning here")
  submit_decision("REJECT",       "your 20+ word reasoning here")

Think step by step. If a hard rule is triggered → REJECT immediately.
"""


def build_system_prompt(max_tool_turns: int = MAX_TOOL_TURNS,
                        memory_bank: Optional[MemoryBank] = None) -> str:
    base = SYSTEM_PROMPT_BASE.format(max_tool_turns=max_tool_turns)
    if memory_bank:
        suffix = memory_bank.inject_prompt_suffix()
        if suffix:
            base += suffix + "\n"
    return base


def build_messages(prompt_text: str, tool_transcript: List[Dict],
                   system_prompt: str) -> List[Dict]:
    """
    Build the message list for the current state of a multi-turn episode.
    The dataset prompt already has application data — we add tool results on top.
    """
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": prompt_text},
    ]
    for turn in tool_transcript:
        messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user",      "content": turn["tool_result"]})
    return messages


print("✅ System prompt builder loaded")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 8: LOAD MODEL ═══
# ════════════════════════════════════════════════════════════════════

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
tokenizer.padding_side = "left"   # for decoder-only left-pad

lora_cfg = LoraConfig(
    r              = 32,
    lora_alpha     = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
print("✅ Model + LoRA ready")


# Quick smoke test
print("\n🔍 Brief base-model test...")
_test_msgs = [
    {"role": "system", "content": "You are a credit officer. Use submit_decision() to decide."},
    {"role": "user",   "content": "DSCR=0.8x (below 1.0). Hard rule HR-01 triggered. Make your decision."},
]
_test_text = tokenizer.apply_chat_template(_test_msgs, tokenize=False, add_generation_prompt=True)
_inp = tokenizer(_test_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    _out = model.generate(**_inp, max_new_tokens=80, do_sample=True, temperature=0.7)
_resp = tokenizer.decode(_out[0][_inp.input_ids.shape[1]:], skip_special_tokens=True)
_parsed = parse_llm_output(_resp)
print(f"   Response  : {_resp[:120]}...")
print(f"   Parse type: {_parsed['parse_type']}  |  action: {_parsed['action']}")
print("✅ Smoke test done")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 9: LOAD DATASET ═══
# ════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading dataset: {HF_DATASET}...")

jsonl_path = hf_hub_download(
    repo_id   = HF_DATASET,
    filename  = "grpo_dataset.jsonl",
    repo_type = "dataset",
)

rows = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        row  = json.loads(line)
        meta = row.get("metadata", row)
        hr   = meta.get("hard_rules", [])
        rows.append({
            "prompt"          : row["prompt"],           # already rich text
            "task_id"         : meta.get("task_id", "task1"),
            "ground_truth_pd" : float(meta.get("ground_truth_pd", 0.5)),
            "optimal_action"  : int(meta.get("optimal_action", 2)),
            "hard_rules"      : json.dumps(hr) if isinstance(hr, list) else str(hr),
            "has_red_alerts"  : bool(meta.get("has_red_alerts", False)),
            "npa_rate"        : float(meta.get("npa_rate", 0.02)),
            "crar"            : float(meta.get("crar", 0.18)),
            "sector"          : str(meta.get("sector", "Unknown")),
            "company_name"    : str(meta.get("company_name", "Company")),
        })

print(f"   Total samples: {len(rows)}")
task_counts = defaultdict(int)
for r in rows:
    task_counts[r["task_id"]] += 1
print(f"   Tasks: {dict(sorted(task_counts.items()))}")

# Stage datasets
stage_rows = {
    1: [r for r in rows if r["task_id"] == "task1"],
    2: [r for r in rows if r["task_id"] in {"task1", "task2"}],
    3: rows,
}
for s, ds in stage_rows.items():
    print(f"   Stage {s}: {len(ds)} samples")

print("✅ Dataset ready")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 10: MULTI-TURN EPISODE RUNNER ═══
# ════════════════════════════════════════════════════════════════════

def run_episode(
    sample: Dict,
    model,
    tokenizer,
    temperature: float = 1.0,
    memory_bank: Optional[MemoryBank] = None,
) -> Dict:
    """
    Run one multi-turn episode for a single sample.

    Multi-Step GRPO support:
      Each assistant turn (tool call OR final decision) is stored as a
      (prompt_str, completion) pair in `all_turns`.  When MULTI_STEP_GRPO=True,
      the GRPO loss sums log-probs over every turn so that tool-calling
      behaviour is directly trained, not just the final decision.

    Env compatibility:
      parse_llm_output() now returns parse_failure + tool_args dict,
      matching the schema expected by IntelliCreditAction in models.py.

    Returns:
        {
          "prompt_text"    : str,    # original application text
          "completion"     : str,    # final assistant output (for reward)
          "all_turns"      : list,   # [(prompt_str, completion), ...] ALL turns
          "full_transcript": str,    # human-readable log
          "parse_type"     : str,
          "action"         : int,
          "tool_calls_made": int,
          "metadata"       : dict,   # for reward functions
        }
    """
    prompt_text   = sample["prompt"]
    sys_prompt    = build_system_prompt(MAX_TOOL_TURNS, memory_bank)
    tool_transcript: List[Dict] = []
    tool_calls_made = 0
    last_completion = ""
    last_parsed     = {}
    # ── Multi-step GRPO: collect every (prompt_str, completion) turn ──
    all_turns: List[Tuple[str, str]] = []

    for turn in range(MAX_TOOL_TURNS + 1):   # +1 for forced decision turn
        messages  = build_messages(prompt_text, tool_transcript, sys_prompt)
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # If at tool limit, append a forcing instruction
        if turn == MAX_TOOL_TURNS:
            chat_text += (
                "\n[SYSTEM: You have reached the tool call limit. "
                "You MUST now submit your final decision using submit_decision().]\n"
            )

        inputs = tokenizer(
            chat_text, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                do_sample      = True,
                temperature    = temperature,
                top_p          = 0.9,
                pad_token_id   = tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # ── Store this turn for multi-step GRPO log-prob computation ──
        # chat_text is the full prompt up to (not including) this completion
        all_turns.append((chat_text, completion))

        parsed          = parse_llm_output(completion)
        last_completion = completion
        last_parsed     = parsed

        if parsed["parse_type"] == "tool_call":
            # Execute tool → synthesize result → continue episode
            tool_result = synthesize_tool_result(
                parsed.get("tool_name", ""),
                parsed.get("tool_arg",  ""),
                prompt_text,
            )
            tool_transcript.append({
                "assistant":   completion,
                "tool_result": (
                    f"[TOOL RESULT — {parsed.get('tool_name','')}]\n{tool_result}"
                ),
            })
            tool_calls_made += 1
        else:
            # Final decision (submit_decision / keyword / default) — stop
            break

    # Assemble human-readable transcript for logging
    full_transcript = f"[USER PROMPT]\n{prompt_text[:300]}...\n"
    for i, t in enumerate(tool_transcript, 1):
        full_transcript += f"\n[TOOL CALL {i}]\n{t['assistant'][:120]}...\n"
        full_transcript += f"[RESULT {i}]\n{t['tool_result'][:200]}...\n"
    full_transcript += f"\n[FINAL]\n{last_completion[:200]}"

    return {
        "prompt_text"    : prompt_text,
        "completion"     : last_completion,
        "all_turns"      : all_turns,        # ← key addition for multi-step GRPO
        "full_transcript": full_transcript,
        "parse_type"     : last_parsed.get("parse_type", "default_reject"),
        "action"         : last_parsed.get("action", 2),
        "tool_calls_made": tool_calls_made,
        "metadata"       : {
            "ground_truth_pd": sample["ground_truth_pd"],
            "hard_rules"     : sample["hard_rules"],
            "has_red_alerts" : sample["has_red_alerts"],
            "npa_rate"       : sample["npa_rate"],
            "crar"           : sample["crar"],
        },
    }


print("✅ Episode runner ready  (multi-step GRPO: all_turns tracked)")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 11: CUSTOM GRPO LOSS ═══
# ════════════════════════════════════════════════════════════════════

def compute_log_probs(model, tokenizer, prompt_text: str,
                      completion: str) -> torch.Tensor:
    """
    Compute the sum of log-probs for the completion tokens given the prompt.
    Returns a scalar tensor (requires_grad=True via LoRA params).
    """
    full_text = prompt_text + completion
    full_ids   = tokenizer(full_text, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LEN
                           ).input_ids.to(model.device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LEN
                           ).input_ids.to(model.device)

    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    with torch.cuda.amp.autocast(enabled=False):
        logits = model(full_ids).logits  # [1, seq, vocab]

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[0, prompt_len - 1 : -1, :]   # [comp_len, vocab]
    shift_labels = full_ids[0, prompt_len:]              # [comp_len]

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_lps = log_probs[
        torch.arange(shift_labels.shape[0], device=model.device),
        shift_labels,
    ]
    return token_lps.sum()


def grpo_loss_step(
    model,
    tokenizer,
    samples: List[Dict],
    temperature: float,
    memory_bank: MemoryBank,
    ref_log_probs_cache: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict]:
    """
    One GRPO update step over BATCH_SIZE prompts × NUM_GENERATIONS completions.

    Returns (loss_tensor, metrics_dict).
    """
    all_losses  = []
    metrics     = defaultdict(list)

    for sample in samples:
        completions = []
        episodes    = []

        # Generate NUM_GENERATIONS rollouts
        for _ in range(NUM_GENERATIONS):
            ep = run_episode(sample, model, tokenizer, temperature, memory_bank)
            episodes.append(ep)
            completions.append(ep["completion"])

        # Compute rewards for all completions
        prompts_list = [ep["prompt_text"] for ep in episodes]
        meta_keys    = ["ground_truth_pd", "hard_rules", "has_red_alerts",
                        "npa_rate", "crar"]
        reward_kwargs = {k: [ep["metadata"][k] for ep in episodes] for k in meta_keys}
        reward_list  = combined_reward(prompts_list, completions, **reward_kwargs)

        metrics["reward"].extend(reward_list)
        metrics["tool_calls"].extend(ep["tool_calls_made"] for ep in episodes)
        metrics["parse_types"].extend(ep["parse_type"] for ep in episodes)

        # GRPO advantage = (r - mean) / (std + eps)
        r_arr   = np.array(reward_list, dtype=np.float32)
        r_mean  = r_arr.mean()
        r_std   = r_arr.std() + 1e-8
        advantages = (r_arr - r_mean) / r_std

        # Compute GRPO loss for each completion
        for ep, adv in zip(episodes, advantages):

            if MULTI_STEP_GRPO and len(ep.get("all_turns", [])) > 0:
                # ════════════════════════════════════════════════════
                # TRUE MULTI-STEP GRPO
                # Sum log-probs over EVERY assistant turn in the episode
                # (tool call turns + final decision turn).
                # The gradient flows through all turns → the model
                # directly learns WHEN to call tools and WHEN to decide.
                # ════════════════════════════════════════════════════
                turn_lps     = []
                turn_ref_lps = []
                for p_str, c_str in ep["all_turns"]:
                    # Current policy
                    lp_t = compute_log_probs(model, tokenizer, p_str, c_str)
                    turn_lps.append(lp_t)
                    # Reference policy (no grad, cached)
                    ck = hash(p_str[-40:] + c_str[:60])
                    if ck not in ref_log_probs_cache:
                        with torch.no_grad():
                            ref_log_probs_cache[ck] = compute_log_probs(
                                model, tokenizer, p_str, c_str
                            ).detach()
                    turn_ref_lps.append(ref_log_probs_cache[ck])

                lp     = sum(turn_lps)                  # total trajectory log-prob
                ref_lp = sum(turn_ref_lps)
                metrics["turns_per_ep"].append(len(ep["all_turns"]))
            else:
                # ════════════════════════════════════════════════════
                # SINGLE-TURN GRPO (fallback / MULTI_STEP_GRPO=False)
                # Only the final completion contributes to the loss.
                # ════════════════════════════════════════════════════
                sys_p    = build_system_prompt(MAX_TOOL_TURNS, memory_bank)
                messages = [
                    {"role": "system", "content": sys_p},
                    {"role": "user",   "content": ep["prompt_text"]},
                ]
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                completion_str = ep["completion"]
                lp     = compute_log_probs(model, tokenizer, prompt_str, completion_str)
                cache_key = hash(completion_str[:100])
                if cache_key not in ref_log_probs_cache:
                    with torch.no_grad():
                        ref_log_probs_cache[cache_key] = compute_log_probs(
                            model, tokenizer, prompt_str, completion_str
                        ).detach()
                ref_lp = ref_log_probs_cache[cache_key]
                metrics["turns_per_ep"].append(1)

            # KL penalty on trajectory log-prob difference
            kl = (lp - ref_lp).clamp(min=0)

            # GRPO objective: maximise advantage × trajectory log-prob, penalise KL
            advantage_tensor = torch.tensor(float(adv), device=model.device)
            step_loss = -(advantage_tensor * lp) + KL_BETA * kl
            all_losses.append(step_loss)
            metrics["kl"].append(kl.item())

        # Memory bank update — learn from worst episode
        worst_ep  = episodes[int(np.argmin(reward_list))]
        MEMORY_BANK.learn_from_episode(
            worst_ep["prompt_text"],
            worst_ep["completion"],
            min(reward_list),
            worst_ep["parse_type"],
        )

    if not all_losses:
        return torch.tensor(0.0, device=model.device, requires_grad=True), {}

    loss = torch.stack(all_losses).mean()
    return loss, metrics


print("✅ Custom GRPO loss ready")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 12: 3-STAGE TRAINING LOOP ═══
# ════════════════════════════════════════════════════════════════════

STAGE_CONFIGS = {
    1: {"name": "Stage 1: Easy (task1)",         "tasks": ["task1"],
        "lr": 1e-4,  "steps": 80,  "temp": 1.2},
    2: {"name": "Stage 2: Medium (task1+task2)", "tasks": ["task1","task2"],
        "lr": 5e-5,  "steps": 80,  "temp": 1.0},
    3: {"name": "Stage 3: Full (all tasks)",     "tasks": None,   # None = all
        "lr": 2e-5,  "steps": 120, "temp": 0.9},
}

all_logs    = {1: [], 2: [], 3: []}
stage_times = {}


def _get_optimizer(model, lr: float):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.999),
    )


print("=" * 70)
print("  STARTING 3-STAGE ONLINE GRPO CURRICULUM TRAINING (v4)")
print("=" * 70)

for stage_num in [1, 2, 3]:
    cfg = STAGE_CONFIGS[stage_num]

    # Filter dataset for this stage
    if cfg["tasks"] is None:
        stage_data = rows
    else:
        stage_data = [r for r in rows if r["task_id"] in cfg["tasks"]]

    random.shuffle(stage_data)

    print(f"\n{'─'*70}")
    print(f"  {cfg['name']}")
    print(f"  Samples: {len(stage_data)} | LR: {cfg['lr']} | Steps: {cfg['steps']} | Temp: {cfg['temp']}")
    print(f"{'─'*70}")

    optimizer = _get_optimizer(model, cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["steps"], eta_min=cfg["lr"] * 0.1
    )
    ref_log_probs_cache: Dict[str, torch.Tensor] = {}

    model.train()
    t_start = time.time()
    data_idx = 0

    for step in range(1, cfg["steps"] + 1):
        # Sample batch
        batch = []
        for _ in range(BATCH_SIZE):
            batch.append(stage_data[data_idx % len(stage_data)])
            data_idx += 1

        loss_accum = torch.tensor(0.0, device=model.device)

        for acc_step in range(GRAD_ACCUM):
            micro_batch = batch   # same batch, multiple accum steps
            loss, metrics = grpo_loss_step(
                model, tokenizer, micro_batch,
                cfg["temp"], MEMORY_BANK, ref_log_probs_cache,
            )
            (loss / GRAD_ACCUM).backward()
            loss_accum = loss_accum + loss.detach() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Logging
        avg_reward = float(np.mean(metrics.get("reward", [0])))
        std_reward = float(np.std(metrics.get("reward", [0])))
        avg_kl     = float(np.mean(metrics.get("kl", [0])))
        avg_tools  = float(np.mean(metrics.get("tool_calls", [0])))
        pt_counts  = defaultdict(int)
        for pt in metrics.get("parse_types", []):
            pt_counts[pt] += 1
        total_pt = max(sum(pt_counts.values()), 1)
        submit_pct = pt_counts.get("final_decision", 0) / total_pt * 100

        log_entry = {
            "step": step, "stage": stage_num,
            "loss": float(loss_accum), "reward": avg_reward,
            "reward_std": std_reward,  "kl": avg_kl,
            "tool_calls": avg_tools,   "submit_pct": submit_pct,
            "lr": scheduler.get_last_lr()[0],
        }
        all_logs[stage_num].append(log_entry)

        if step % 5 == 0 or step <= 3:
            print(
                f"  Step {step:4d}/{cfg['steps']} | "
                f"loss={float(loss_accum):+.4f} | "
                f"reward={avg_reward:+.3f}±{std_reward:.3f} | "
                f"kl={avg_kl:.5f} | "
                f"tools={avg_tools:.1f} | "
                f"submit={submit_pct:.0f}%"
            )

    elapsed = time.time() - t_start
    stage_times[stage_num] = elapsed

    # Save checkpoint
    ckpt_dir = f"{OUTPUT_BASE}/stage_{stage_num}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    print(f"\n  ✅ Stage {stage_num} done in {elapsed/60:.1f} min")
    print(f"  💾 Checkpoint: {ckpt_dir}")

    # Sample inference
    print(f"\n  🔍 Sample inference (3 prompts):")
    model.eval()
    for idx in range(min(3, len(stage_data))):
        s = stage_data[idx]
        ep = run_episode(s, model, tokenizer, cfg["temp"], MEMORY_BANK)
        print(
            f"     [{idx+1}] PD={s['ground_truth_pd']:.2f} | "
            f"[{ep['parse_type']}] tools={ep['tool_calls_made']} | "
            f"{ep['completion'][:80]}..."
        )
    model.train()

print(f"\n{'='*70}")
print(f"  ALL 3 STAGES COMPLETE ✅")
print(f"  Total: {sum(stage_times.values())/60:.1f} min")
print(f"{'='*70}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 13: LEARNING CURVES ═══
# ════════════════════════════════════════════════════════════════════

print("\n📊 Generating charts...")

# Gather all data
all_steps, all_losses, all_rewards, all_kl, all_tools, all_submit = [], [], [], [], [], []
stage_boundaries = []
offset = 0

for stage in [1, 2, 3]:
    logs = all_logs[stage]
    if not logs: continue
    stage_boundaries.append(offset)
    for e in logs:
        s = e["step"] + offset
        all_steps.append(s)
        if e.get("loss")       is not None: all_losses.append((s, e["loss"]))
        if e.get("reward")     is not None: all_rewards.append((s, e["reward"]))
        if e.get("kl")         is not None: all_kl.append((s, e["kl"]))
        if e.get("tool_calls") is not None: all_tools.append((s, e["tool_calls"]))
        if e.get("submit_pct") is not None: all_submit.append((s, e["submit_pct"]))
    offset += max(e["step"] for e in logs) + 1

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("IntelliCredit Online GRPO — Qwen2.5-1.5B (Multi-Turn Tool Calling, v4)",
             fontsize=14, fontweight="bold")


def _plot(ax, data, color, title, ylabel, ylim=None):
    if not data: return
    xs, ys = zip(*data)
    ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8)
    if len(ys) > 5:
        w  = min(10, len(ys) // 3)
        sm = np.convolve(ys, np.ones(w) / w, mode="valid")
        ax.plot(xs[:len(sm)], sm, color=color, linewidth=2.2, label="Smoothed")
    for b in stage_boundaries[1:]:
        ax.axvline(x=b, color="gray", linestyle="--", alpha=0.5, label="Stage")
    ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel)
    ax.set_xlabel("Step"); ax.grid(True, alpha=0.3)
    if ylim: ax.set_ylim(ylim)
    ax.legend(fontsize=8)


_plot(axes[0, 0], all_losses,  "#E53935", "GRPO Loss",              "Loss")
_plot(axes[0, 1], all_rewards, "#1976D2", "Mean Reward ↑",          "Reward")
_plot(axes[0, 2], all_kl,      "#7B1FA2", "KL Divergence",          "KL")
_plot(axes[1, 0], all_tools,   "#F57C00", "Avg Tool Calls / Step",  "Tool Calls", ylim=[0, MAX_TOOL_TURNS+0.5])
_plot(axes[1, 1], all_submit,  "#00796B", "submit_decision() Rate ↑", "% of Outputs", ylim=[0, 105])

# Action distribution last 20 steps (all stages combined)
all_pts = [e["parse_types"] for s in [1,2,3] for e in all_logs.get(s,[]) if "parse_types" in e]
if not all_pts:
    # Use what we logged as submit_pct
    labels = ["submit_decision", "fallback_keyword", "tool_call", "default_reject"]
    vals   = [
        np.mean([e.get("submit_pct", 0) for s in [1,2,3] for e in all_logs.get(s,[])]),
        0, 0, 0
    ]
    axes[1, 2].bar(labels, vals, color=["#1976D2", "#F57C00", "#43A047", "#E53935"])
else:
    flat_pts   = [pt for batch in all_pts[-20:] for pt in batch]
    pt_counter = defaultdict(int)
    for pt in flat_pts: pt_counter[pt] += 1
    axes[1, 2].bar(pt_counter.keys(), pt_counter.values(),
                   color=["#1976D2", "#F57C00", "#43A047", "#E53935"][:len(pt_counter)])
axes[1, 2].set_title("Parse Type Distribution (last 20 steps)", fontweight="bold")
axes[1, 2].set_ylabel("Count"); axes[1, 2].grid(True, alpha=0.3)
plt.setp(axes[1, 2].get_xticklabels(), rotation=20, ha="right")

plt.tight_layout()
chart_path = f"{OUTPUT_BASE}/charts/online_grpo_overview.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  💾 {chart_path}")

# Stage-by-stage reward chart
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("Per-Stage Reward Progression (Online GRPO v4)", fontsize=13, fontweight="bold")
colors = ["#E53935", "#1976D2", "#43A047"]
for i, stage in enumerate([1, 2, 3]):
    logs = all_logs[stage]
    if not logs: continue
    rewards = [(e["step"], e["reward"]) for e in logs if e.get("reward") is not None]
    submit  = [(e["step"], e.get("submit_pct", 0)) for e in logs]
    if rewards:
        xs, ys = zip(*rewards)
        axes2[i].plot(xs, ys, color=colors[i], alpha=0.35, linewidth=0.8, label="reward")
        if len(ys) > 3:
            w  = min(5, len(ys) // 2)
            sm = np.convolve(ys, np.ones(w) / w, mode="valid")
            axes2[i].plot(xs[:len(sm)], sm, color=colors[i], linewidth=2.5)
    if submit:
        xs2, ys2 = zip(*submit)
        ax_r = axes2[i].twinx()
        ax_r.plot(xs2, ys2, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="submit%")
        ax_r.set_ylabel("submit_decision %", fontsize=8)
        ax_r.set_ylim(0, 110)
    axes2[i].set_title(STAGE_CONFIGS[stage]["name"], fontsize=11)
    axes2[i].set_xlabel("Step"); axes2[i].set_ylabel("Reward")
    axes2[i].grid(True, alpha=0.3)
plt.tight_layout()
chart2_path = f"{OUTPUT_BASE}/charts/per_stage_rewards.png"
plt.savefig(chart2_path, dpi=150)
plt.show()
print(f"  💾 {chart2_path}")
print("✅ Charts done!")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 14: SAVE + FINAL SUMMARY ═══
# ════════════════════════════════════════════════════════════════════

print(f"\n💾 Saving final model → {FINAL_MODEL}")
model.save_pretrained(FINAL_MODEL)
tokenizer.save_pretrained(FINAL_MODEL)

# Save logs
log_path = f"{OUTPUT_BASE}/training_logs.json"
with open(log_path, "w") as f:
    json.dump({
        f"stage_{s}": all_logs[s] for s in [1, 2, 3]
    } | {"stage_times": stage_times, "model": MODEL_NAME,
         "dataset": HF_DATASET, "version": "v4-online-grpo"}, f, indent=2, default=str)
print(f"   Logs saved → {log_path}")

print("\n" + "=" * 70)
print("  📊 FINAL ONLINE GRPO TRAINING SUMMARY")
print("=" * 70)

for stage in [1, 2, 3]:
    logs = all_logs[stage]
    if not logs: continue
    rewards  = [e["reward"]      for e in logs if e.get("reward")      is not None]
    submits  = [e["submit_pct"]  for e in logs if e.get("submit_pct")  is not None]
    tools    = [e["tool_calls"]  for e in logs if e.get("tool_calls")  is not None]
    losses   = [e["loss"]        for e in logs if e.get("loss")        is not None]
    q = max(1, len(rewards) // 4) if rewards else 1
    print(f"\n  Stage {stage}: {STAGE_CONFIGS[stage]['name']}")
    print(f"    Steps      : {len(logs)}  |  Time: {stage_times.get(stage, 0)/60:.1f} min")
    if rewards:
        print(f"    Reward     : {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        print(f"    Trend      : {np.mean(rewards[:q]):.4f} → {np.mean(rewards[-q:]):.4f}")
    if submits:
        print(f"    submit()%  : {np.mean(submits[:q]):.1f}% → {np.mean(submits[-q:]):.1f}%  ← key metric")
    if tools:
        print(f"    Tool calls : {np.mean(tools):.2f} avg per step")
    if losses:
        print(f"    Loss       : {np.mean(losses):.6f} (final: {losses[-1]:.6f})")

print(f"\n  Total time  : {sum(stage_times.values())/60:.1f} min")
print(f"  Final model : {FINAL_MODEL}/")
print(f"  Memory bank : {len(MEMORY_BANK.lessons)} lessons accumulated")
if MEMORY_BANK.lessons:
    print("  Top lessons :")
    for l in MEMORY_BANK.lessons[:3]:
        print(f"    • {l['lesson']}")
print(f"\n  ✅ Online GRPO training complete!")
print("=" * 70)
