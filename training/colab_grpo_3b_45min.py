"""
╔══════════════════════════════════════════════════════════════════════╗
║  IntelliCredit v2 — SPEEDRUN GRPO (45-min · 3B · A100)              ║
║  Model  : Qwen/Qwen2.5-3B-Instruct  (4-bit QLoRA)                   ║
║  Target : HF JupyterLab Space — Nvidia A100 Large · 142 GB          ║
║  Budget : ~$6–8 out of your $30 HF credit                           ║
║                                                                      ║
║  Optimised tradeoffs vs full 7-hour run:                             ║
║  • 40 total steps (10+10+20) instead of 280                         ║
║  • NUM_GENERATIONS = 4  (min for GRPO contrast)                      ║
║  • GRAD_ACCUM = 1  (instant updates, noisier but fast)               ║
║  • MAX_TOOL_TURNS = 2  (2 tool calls max per episode)                ║
║  • MAX_NEW_TOKENS = 150  (enough for full submit_decision)           ║
║                                                                      ║
║  Expected outcome: model learns submit_decision() format + HR rules  ║
║  Perfect for hackathon demo — reward curve visible in under 1 hr     ║
╚══════════════════════════════════════════════════════════════════════╝

Paste each ═══ CELL ═══ block into a separate Jupyter cell.
Run Cell 1 → Restart Kernel → Run Cell 2 onward in order.
"""


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 1: INSTALL  (run once → Kernel → Restart → run Cell 2+) ═
# ════════════════════════════════════════════════════════════════════

import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# Force install PyTorch with CUDA 12.1 support first
_pip("torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121", "--upgrade")

_pip("--upgrade", "pip")
_pip("bitsandbytes>=0.46.1")
_pip("transformers>=4.45.0,<5.0.0")
_pip("trl>=0.15.2", "peft>=0.13.0", "accelerate>=1.0.0")
_pip("datasets>=2.20.0", "huggingface_hub>=0.24.0", "matplotlib")

# ── HF Token (load from JupyterLab env var or Colab secrets) ──────
import os
# On HF Spaces JupyterLab, secrets are available as env vars directly.
# Just set HF_TOKEN in your Space secrets — it's auto-mounted.
hf_tok = os.environ.get("HF_TOKEN", "")
if hf_tok:
    from huggingface_hub import login
    login(token=hf_tok, add_to_git_credential=False)
    print("✅ HF Token loaded from environment")
else:
    print("⚠️  HF_TOKEN not set — anonymous access (OK for public models)")

print("\n✅ All packages installed")
print("   ⚡ IMPORTANT: Kernel → Restart Kernel → then run from Cell 2")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 2: IMPORTS & CONFIG ═══════════════════════════════════════
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

# ── Core Config ──────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"          # 3B sweet-spot
HF_DATASET   = "vssksn/intellicredit-grpo-v2"
OUTPUT_BASE  = "intellicredit-grpo-3b-45min"
FINAL_MODEL  = "qwen3b-intellicredit-speedrun"

# ── Speedrun GRPO hyper-params (A100 optimised) ──────────────────────
NUM_GENERATIONS  = 4       # min for GRPO advantage contrast
MAX_NEW_TOKENS   = 150     # enough for submit_decision + reason
MAX_TOOL_TURNS   = 2       # 2 tool calls max per episode
BATCH_SIZE       = 2       # A100 handles 3B at batch 2 easily
GRAD_ACCUM       = 1       # instant weight updates (faster, noisier)
KL_BETA          = 0.04
MAX_SEQ_LEN      = 1400
MULTI_STEP_GRPO  = True    # gradient flows through tool call turns too

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/charts", exist_ok=True)

print("✅ Imports complete")
gpu  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"   GPU  : {gpu}")
print(f"   VRAM : {vram:.1f} GB")
print(f"   Model: {MODEL_NAME}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 3: ACTION PARSER ══════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

ACTION_MAP = {
    "APPROVE": 0, "APPROVED": 0,
    "CONDITIONAL": 1, "CONDITIONAL_APPROVE": 1, "CONDITIONAL APPROVE": 1,
    "REJECT": 2, "REJECTED": 2, "DECLINE": 2, "DENY": 2,
}

_RE_SUBMIT = re.compile(
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


def _unwrap(text) -> str:
    if isinstance(text, list):
        return " ".join(m.get("content", "") for m in text if isinstance(m, dict))
    return str(text) if not isinstance(text, str) else text


def parse_llm_output(text) -> Dict:
    t = _unwrap(text).strip()
    if not t:
        return {"action": 2, "parse_type": "default_reject",
                "parse_confidence": 0.0, "reasoning": "",
                "tool_name": None, "tool_args": None, "parse_failure": True}

    tm = _RE_TOOL.search(t)
    if tm:
        raw_arg   = tm.group(2).strip().strip("'\"")
        tool_name = tm.group(1).lower()
        tool_args_dict = (
            {"sector": raw_arg}
            if "market_intelligence" in tool_name
            else {"company_id": raw_arg}
        )
        return {
            "action": 2, "parse_type": "tool_call",
            "tool_name": tool_name,
            "tool_arg":  raw_arg,
            "tool_args": tool_args_dict,
            "parse_confidence": 0.95,
            "reasoning":  f"calling {tool_name}",
            "parse_failure": False,
        }

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

    kms = list(_RE_KEYWORD.finditer(t))
    if kms:
        kw  = kms[-1].group(1).upper()
        act = ACTION_MAP.get(kw, 2)
        return {
            "action": act, "parse_type": "fallback_keyword",
            "parse_confidence": 0.55, "reasoning": t[-100:],
            "tool_name": None, "tool_args": None, "parse_failure": True,
        }

    return {
        "action": 2, "parse_type": "default_reject",
        "parse_confidence": 0.0, "reasoning": "",
        "tool_name": None, "tool_args": None, "parse_failure": True,
    }


print("✅ Action parser loaded")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 4: REWARD FUNCTIONS v4 ════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

def _ef(text: str, pattern: str, default: float) -> float:
    m = re.search(pattern, text, re.IGNORECASE)
    try:
        return float(m.group(1)) if m else default
    except (ValueError, AttributeError):
        return default


def reward_correctness(prompts, completions, ground_truth_pd, **kw) -> List[float]:
    """R1: correctness vs PD ground truth + submit_decision bonus."""
    scores = []
    for prompt, comp, pd in zip(prompts, completions, ground_truth_pd):
        p   = parse_llm_output(comp)
        act = p["action"]
        s   = 0.0
        if pd > 0.45:                   # high-risk → REJECT correct
            s = 1.5 if act == 2 else (-1.5 if act == 0 else 0.0)
        elif pd < 0.15:                 # low-risk  → APPROVE correct
            s = 1.5 if act == 0 else (-0.5 if act == 2 else 0.5)
        else:                           # borderline → CONDITIONAL correct
            s = 1.0 if act == 1 else (0.2 if act == 0 else -0.5)
        if p["parse_type"] == "final_decision":
            s += 0.5                    # strong bonus for using submit_decision
        scores.append(max(-2.0, min(2.0, s)))
    return scores


def reward_hard_rules(prompts, completions, hard_rules, has_red_alerts, **kw) -> List[float]:
    """R2: hard-rule compliance — HR triggered → must REJECT."""
    scores = []
    for comp, hr_str, red in zip(completions, hard_rules, has_red_alerts):
        p   = parse_llm_output(comp)
        act = p["action"]
        try:
            hr_list = json.loads(hr_str) if isinstance(hr_str, str) else hr_str
        except Exception:
            hr_list = []
        has_hr = bool(hr_list) or red
        if has_hr:
            s = 0.5 if act == 2 else -2.0
        else:
            s = 0.3 if act != 2 else -0.5
        scores.append(s)
    return scores


def reward_format(prompts, completions, **kw) -> List[float]:
    """R3: output format quality — reward submit_decision with reasoning."""
    scores = []
    for comp in completions:
        p   = parse_llm_output(comp)
        rsn = p.get("reasoning", "")
        if p["parse_type"] == "final_decision" and len(rsn) > 20:
            s = 1.0
        elif p["parse_type"] == "final_decision":
            s = 0.3
        elif p["parse_type"] == "tool_call":
            s = 0.1
        elif p["parse_type"] == "fallback_keyword":
            s = -0.5
        else:
            s = -1.0
        scores.append(s)
    return scores


def reward_portfolio(prompts, completions, npa_rate, crar, **kw) -> List[float]:
    """R4: portfolio-aware penalty — penalise approvals when NPA/CRAR at limits."""
    scores = []
    for comp, npa, cr in zip(completions, npa_rate, crar):
        p   = parse_llm_output(comp)
        act = p["action"]
        s   = 0.0
        if npa > 0.048 and act == 0:   s -= 0.8
        if cr  < 0.13  and act == 0:   s -= 0.6
        if npa < 0.02  and act == 2:   s -= 0.3
        scores.append(max(-0.8, min(0.3, s)))
    return scores


def combined_reward(prompts, completions, ground_truth_pd, hard_rules,
                    has_red_alerts, npa_rate, crar) -> List[float]:
    r1 = reward_correctness(prompts, completions, ground_truth_pd)
    r2 = reward_hard_rules(prompts, completions, hard_rules, has_red_alerts)
    r3 = reward_format(prompts, completions)
    r4 = reward_portfolio(prompts, completions, npa_rate, crar)
    return [a + b + c + d for a, b, c, d in zip(r1, r2, r3, r4)]


print("✅ Reward functions v4 loaded")
print("   R1 correctness  [-2.0,+2.0] — +0.5 bonus for submit_decision")
print("   R2 hard_rules   [-2.0,+0.5]")
print("   R3 format       [-1.0,+1.0] — +1.0 bonus for submit_decision+reasoning")
print("   R4 portfolio    [-0.8,+0.3]")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 5: TOOL RESULT SYNTHESIZER ════════════════════════════════
# ════════════════════════════════════════════════════════════════════

def _extract_float(text: str, pattern: str, default: float) -> float:
    m = re.search(pattern, text, re.IGNORECASE)
    try:
        return float(m.group(1)) if m else default
    except (ValueError, AttributeError):
        return default


def synthesize_tool_result(tool_name: str, tool_arg: str, prompt_text: str) -> str:
    """Extract data from the prompt and return a formatted tool result."""
    p = prompt_text
    tool_name = (tool_name or "").lower().strip()

    if tool_name == "get_financial_report":
        cname = tool_arg or "Applicant"
        dscr  = _extract_float(p, r"DSCR:\s*([\d.]+)x", 1.5)
        cr    = _extract_float(p, r"Current\s*Ratio:\s*([\d.]+)x", 1.2)
        de    = _extract_float(p, r"[Dd]ebt.to.[Ee]quity:\s*([\d.]+)x", 1.5)
        eb    = _extract_float(p, r"EBITDA\s*Margin:\s*([\d.]+)%", 18.0)
        rev   = re.search(r"[Rr]evenue[^₹\n]*?₹([\d.]+)\s*Cr", p)
        nw    = re.search(r"[Nn]et\s*[Ww]orth[^₹\n]*?₹([\d.]+)\s*Cr", p)
        loan  = re.search(r"[Ll]oan\s*[Rr]equested:\s*₹([\d.]+)\s*Cr", p)
        cibil = _extract_float(p, r"CIBIL\s*[Ss]core:\s*([\d.]+)", 650.0)
        coll  = _extract_float(p, r"[Cc]ollateral\s*[Cc]overage:\s*([\d.]+)x", 1.0)
        bbr   = _extract_float(p, r"[Cc]heque\s*[Bb]ounce\s*[Rr]ate:\s*([\d.]+)%", 5.0)
        od    = _extract_float(p, r"OD\s*[Uu]tilisation:\s*([\d.]+)%", 50.0)
        gst   = _extract_float(p, r"GST\s*[Tt]urnover\s*CAGR:\s*([-\d.]+)%", 5.0)

        dscr_flag = " ⚠️ BELOW 1.0 — HR-01" if dscr < 1.0 else (" ✅ Strong" if dscr >= 1.5 else " ✓ Adequate")
        bb_flag   = " ⚠️ ABOVE 25% — HR-04" if bbr > 25 else ""

        return "\n".join([
            f"═══ FINANCIAL REPORT: {cname} ═══",
            f"  DSCR              : {dscr:.2f}x{dscr_flag}",
            f"  Current Ratio     : {cr:.2f}x  {'✅' if cr >= 1.2 else '⚠️'}",
            f"  Debt-to-Equity    : {de:.2f}x  {'✅' if de <= 2.0 else '⚠️ High leverage'}",
            f"  EBITDA Margin     : {eb:.1f}%  {'✅' if eb >= 15 else '⚠️ Thin margin'}",
            f"  CIBIL Score       : {int(cibil)}  {'✅' if cibil >= 700 else '⚠️'}",
            f"  Collateral Cover  : {coll:.2f}x  {'✅' if coll >= 1.25 else '⚠️ Thin cover'}",
            f"  Revenue           : ₹{rev.group(1) if rev else 'N/A'} Cr",
            f"  Net Worth         : ₹{nw.group(1) if nw else 'N/A'} Cr",
            f"  Loan Requested    : ₹{loan.group(1) if loan else 'N/A'} Cr",
            f"  OD Utilisation    : {od:.1f}%  {'⚠️ High' if od > 75 else '✅'}",
            f"  Cheque Bounce     : {bbr:.1f}%{bb_flag}",
            f"  GST Turnover CAGR : {gst:+.1f}%  {'⚠️ Declining' if gst < 0 else '✅ Growing'}",
            "═══ END FINANCIAL REPORT ═══",
        ])

    elif tool_name == "check_compliance_status":
        cname    = tool_arg or "Applicant"
        dscr_val = _extract_float(p, r"DSCR:\s*([\d.]+)x", 1.5)
        bounce   = _extract_float(p, r"[Cc]heque\s*[Bb]ounce\s*[Rr]ate:\s*([\d.]+)%", 0.0)
        gst_comp = _extract_float(p, r"GST.*?compliance.*?([\d.]+)%", 100.0)
        is_repeat = bool(re.search(r"is_repeat.*?true|REPEAT\s*APPLICANT", p, re.IGNORECASE))
        lit       = _extract_float(p, r"[Pp]romoter\s*[Ll]itigation\s*[Cc]ases:\s*([\d]+)", 0)

        hr_lines = []
        if dscr_val < 1.0:  hr_lines.append(f"   🚫 HR-01: DSCR = {dscr_val:.2f}x (below 1.0) — MANDATORY REJECT")
        if bounce   > 25:   hr_lines.append(f"   🚫 HR-04: Cheque bounce = {bounce:.1f}% (above 25%) — MANDATORY REJECT")
        if gst_comp < 40:   hr_lines.append(f"   🚫 HR-05: GST compliance = {gst_comp:.0f}% (below 40%) — MANDATORY REJECT")

        alert_lines = []
        for sev, icon in [("🔴", "🔴 RED"), ("🟡", "🟡 AMBER")]:
            for am in re.finditer(rf"\{sev}\s*\[(\w+)\]\s*(\w[\w_]+):\s*(.+?)(?=\n|$)", p):
                alert_lines.append(f"   {icon} | {am.group(2)}: {am.group(3).strip()}")

        lines = [f"═══ COMPLIANCE STATUS: {cname} ═══"]
        if is_repeat: lines.append("  ⚠️ REPEAT APPLICANT — previously rejected")
        if hr_lines:
            lines.append("── HARD RULES TRIGGERED (MANDATORY REJECT) ──")
            lines.extend(hr_lines)
        else:
            lines.append("  ✅ No hard rules triggered")
        if alert_lines:
            lines.append("── Forensic Alerts ──")
            lines.extend(alert_lines)
        else:
            lines.append("  ✅ No forensic alerts")
        lines.append(f"  Promoter Litigation Cases : {int(lit)}")
        lines.append("═══ END COMPLIANCE ═══")
        return "\n".join(lines)

    elif tool_name == "get_market_intelligence":
        sector_arg  = tool_arg or "General"
        m_stress    = re.search(r"[Mm]acro\s*[Ee]nvironment.*?stress=([\d.]+)", p)
        stress      = float(m_stress.group(1)) if m_stress else 0.25
        npa         = _extract_float(p, r"NPA\s*[Rr]ate:\s*([\d.]+)%", 2.0) / 100
        crar        = _extract_float(p, r"CRAR:\s*([\d.]+)%", 18.0) / 100
        cap_dep     = _extract_float(p, r"[Cc]apital\s*[Dd]eployed:\s*([\d.]+)%", 0.0)
        stress_lbl  = "HIGH STRESS ⚠️" if stress > 0.6 else ("MODERATE" if stress > 0.35 else "STABLE ✅")
        npa_flag    = "⚠️ Near limit" if npa > 0.04 else ("🚫 ABOVE 5%" if npa >= 0.05 else "✅")
        crar_flag   = "🚫 BELOW 12.5%" if crar < 0.125 else ("⚠️ Thin buffer" if crar < 0.15 else "✅")
        return "\n".join([
            f"═══ MARKET INTELLIGENCE: {sector_arg} ═══",
            f"  Macro Stress     : {stress:.2f} ({stress_lbl})",
            f"  NPA Rate         : {npa:.1%}  {npa_flag}",
            f"  CRAR             : {crar:.1%}  {crar_flag}",
            f"  Capital Deployed : {cap_dep:.1f}%",
            "═══ END MARKET INTELLIGENCE ═══",
        ])

    return f"[TOOL ERROR] Unknown tool: {tool_name}"


print("✅ Tool result synthesizer loaded")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 6: MEMORY BANK ════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

class MemoryBank:
    def __init__(self, max_lessons: int = 6):
        self.lessons: List[Dict] = []
        self.max_lessons = max_lessons

    def add_lesson(self, lesson: str, reward: float, parse_type: str):
        if reward < 0.5:
            self.lessons.append({"lesson": lesson, "reward": reward, "parse_type": parse_type})
        seen = {}
        for l in self.lessons:
            key = l["lesson"][:50]
            if key not in seen or l["reward"] < seen[key]["reward"]:
                seen[key] = l
        self.lessons = sorted(seen.values(), key=lambda x: x["reward"])[:self.max_lessons]

    def inject_prompt_suffix(self) -> str:
        if not self.lessons: return ""
        lines = ["", "── LESSONS FROM RECENT DECISIONS ──"]
        for i, l in enumerate(self.lessons[-3:], 1):
            lines.append(f"  Lesson {i}: {l['lesson']}")
        return "\n".join(lines)

    def learn_from_episode(self, prompt: str, final_text: str, reward: float, parse_type: str):
        if parse_type == "default_reject":
            self.add_lesson("Always end with submit_decision('ACTION', 'reason')", reward, parse_type)
        elif parse_type == "fallback_keyword":
            self.add_lesson("Use submit_decision() format, not just a keyword", reward, parse_type)
        elif reward < 0.0 and parse_type == "final_decision":
            dscr = _extract_float(prompt, r"DSCR:\s*([\d.]+)x", 9.9)
            if dscr < 1.0:
                self.add_lesson("DSCR < 1.0 → always REJECT (HR-01)", reward, parse_type)


MEMORY_BANK = MemoryBank()
print("✅ Memory bank initialised")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 7: SYSTEM PROMPT BUILDER ══════════════════════════════════
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_BASE = """\
You are a Senior Credit Officer at an Indian NBFC. Review MSME loan applications and make APPROVE / CONDITIONAL / REJECT decisions balancing yield, risk, and RBI compliance.

══ AVAILABLE TOOLS (call up to {max_tool_turns} times) ══
  get_financial_report("company_name")
    → DSCR, current ratio, D/E, EBITDA, CIBIL, collateral, OD utilisation.
  check_compliance_status("company_name")
    → Hard rules triggered, forensic alerts, repeat applicant flag.
  get_market_intelligence("sector_name")
    → Macro stress, NPA rate, CRAR, capital utilisation.

══ RBI HARD RULES — MANDATORY REJECT if triggered ══
  HR-01: DSCR < 1.0   HR-03: RED forensic alert   HR-04: Cheque bounce > 25%
  HR-05: GST < 40%    HR-02: Director disqualified

══ YOU MUST END WITH EXACTLY ONE OF THESE ══
  submit_decision("APPROVE",      "20+ word reasoning")
  submit_decision("CONDITIONAL",  "20+ word reasoning")
  submit_decision("REJECT",       "20+ word reasoning")

Think step by step. Check hard rules first. If any triggered → REJECT.
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt_text},
    ]
    for turn in tool_transcript:
        messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user",      "content": turn["tool_result"]})
    return messages


print("✅ System prompt builder loaded")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 8: LOAD MODEL + LoRA ══════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading model: {MODEL_NAME}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,  # bfloat16 is faster on A100
    bnb_4bit_use_double_quant = True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map          = "auto",
    trust_remote_code   = True,
    torch_dtype         = torch.bfloat16,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# LoRA — r=16 is enough for speedrun (smaller = faster)
lora_cfg = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
print("✅ Model + LoRA ready")

# Smoke test
print("\n🔍 Smoke test...")
_msgs = [
    {"role": "system", "content": "You are a credit officer. Use submit_decision()."},
    {"role": "user",   "content": "DSCR=0.8x (below 1.0). Hard rule HR-01 triggered."},
]
_txt = tokenizer.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
_inp = tokenizer(_txt, return_tensors="pt").to(model.device)
with torch.no_grad():
    _out = model.generate(**_inp, max_new_tokens=60, do_sample=True, temperature=0.7)
_resp = tokenizer.decode(_out[0][_inp.input_ids.shape[1]:], skip_special_tokens=True)
print(f"   Response  : {_resp[:120]}")
print(f"   Parse type: {parse_llm_output(_resp)['parse_type']}")
print("✅ Smoke test done")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 9: LOAD DATASET ═══════════════════════════════════════════
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
            "prompt"          : row["prompt"],
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
print("✅ Dataset ready")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 10: MULTI-TURN EPISODE RUNNER ═════════════════════════════
# ════════════════════════════════════════════════════════════════════

def run_episode(
    sample: Dict,
    model,
    tokenizer,
    temperature: float = 1.0,
    memory_bank: Optional[MemoryBank] = None,
) -> Dict:
    prompt_text      = sample["prompt"]
    sys_prompt       = build_system_prompt(MAX_TOOL_TURNS, memory_bank)
    tool_transcript: List[Dict] = []
    tool_calls_made  = 0
    last_completion  = ""
    last_parsed      = {}
    all_turns: List[Tuple[str, str]] = []   # for multi-step GRPO

    for turn in range(MAX_TOOL_TURNS + 1):
        messages  = build_messages(prompt_text, tool_transcript, sys_prompt)
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if turn == MAX_TOOL_TURNS:
            chat_text += (
                "\n[SYSTEM: Tool limit reached. "
                "You MUST submit your decision now using submit_decision().]"
            )

        inputs = tokenizer(
            chat_text, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                min_new_tokens = 10,           # prevent empty EOS collapse
                do_sample      = True,
                temperature    = temperature,
                top_p          = 0.9,
                pad_token_id   = tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Guard against empty completion
        if not completion:
            completion = 'submit_decision("REJECT", "Insufficient data to approve this application.")'

        all_turns.append((chat_text, completion))
        parsed          = parse_llm_output(completion)
        last_completion = completion
        last_parsed     = parsed

        if parsed["parse_type"] == "tool_call":
            tool_result = synthesize_tool_result(
                parsed.get("tool_name", ""),
                parsed.get("tool_arg",  ""),
                prompt_text,
            )
            tool_transcript.append({
                "assistant":   completion,
                "tool_result": f"[TOOL RESULT — {parsed.get('tool_name','')}]\n{tool_result}",
            })
            tool_calls_made += 1
        else:
            break

    return {
        "prompt_text"    : prompt_text,
        "completion"     : last_completion,
        "all_turns"      : all_turns,
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


print("✅ Episode runner ready")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 11: CUSTOM GRPO LOSS ══════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

def compute_log_probs(model, tokenizer, prompt_text: str,
                      completion: str) -> torch.Tensor:
    full_text  = prompt_text + completion
    full_ids   = tokenizer(full_text, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LEN
                           ).input_ids.to(model.device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LEN
                           ).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        logits = model(full_ids).logits

    shift_logits = logits[0, prompt_len - 1 : -1, :]
    shift_labels = full_ids[0, prompt_len:]
    log_probs    = F.log_softmax(shift_logits.float(), dim=-1)
    token_lps    = log_probs[
        torch.arange(shift_labels.shape[0], device=model.device), shift_labels
    ]
    return token_lps.sum()


def grpo_loss_step(
    model, tokenizer, samples: List[Dict],
    temperature: float, memory_bank: MemoryBank,
    ref_log_probs_cache: Dict,
) -> Tuple[torch.Tensor, Dict]:
    all_losses = []
    metrics    = defaultdict(list)

    for sample in samples:
        completions, episodes = [], []
        for _ in range(NUM_GENERATIONS):
            ep = run_episode(sample, model, tokenizer, temperature, memory_bank)
            episodes.append(ep)
            completions.append(ep["completion"])

        prompts_list  = [ep["prompt_text"] for ep in episodes]
        meta_keys     = ["ground_truth_pd", "hard_rules", "has_red_alerts", "npa_rate", "crar"]
        reward_kwargs = {k: [ep["metadata"][k] for ep in episodes] for k in meta_keys}
        reward_list   = combined_reward(prompts_list, completions, **reward_kwargs)

        metrics["reward"].extend(reward_list)
        metrics["tool_calls"].extend(ep["tool_calls_made"] for ep in episodes)
        metrics["parse_types"].extend(ep["parse_type"] for ep in episodes)

        r_arr      = np.array(reward_list, dtype=np.float32)
        advantages = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)

        for ep, adv in zip(episodes, advantages):
            if MULTI_STEP_GRPO and ep.get("all_turns"):
                turn_lps, turn_refs = [], []
                for p_str, c_str in ep["all_turns"]:
                    lp_t = compute_log_probs(model, tokenizer, p_str, c_str)
                    turn_lps.append(lp_t)
                    ck = hash(p_str[-40:] + c_str[:60])
                    if ck not in ref_log_probs_cache:
                        with torch.no_grad():
                            ref_log_probs_cache[ck] = compute_log_probs(
                                model, tokenizer, p_str, c_str
                            ).detach()
                    turn_refs.append(ref_log_probs_cache[ck])
                lp     = sum(turn_lps)
                ref_lp = sum(turn_refs)
                metrics["turns_per_ep"].append(len(ep["all_turns"]))
            else:
                sys_p = build_system_prompt(MAX_TOOL_TURNS, memory_bank)
                p_str = tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_p},
                     {"role": "user",   "content": ep["prompt_text"]}],
                    tokenize=False, add_generation_prompt=True
                )
                lp  = compute_log_probs(model, tokenizer, p_str, ep["completion"])
                ck  = hash(ep["completion"][:100])
                if ck not in ref_log_probs_cache:
                    with torch.no_grad():
                        ref_log_probs_cache[ck] = compute_log_probs(
                            model, tokenizer, p_str, ep["completion"]
                        ).detach()
                ref_lp = ref_log_probs_cache[ck]
                metrics["turns_per_ep"].append(1)

            kl           = (lp - ref_lp).clamp(min=0)
            adv_t        = torch.tensor(float(adv), device=model.device)
            step_loss    = -(adv_t * lp) + KL_BETA * kl
            all_losses.append(step_loss)
            metrics["kl"].append(kl.item())

        worst = episodes[int(np.argmin(reward_list))]
        MEMORY_BANK.learn_from_episode(
            worst["prompt_text"], worst["completion"],
            min(reward_list), worst["parse_type"],
        )

    if not all_losses:
        return torch.tensor(0.0, device=model.device, requires_grad=True), {}
    return torch.stack(all_losses).mean(), metrics


print("✅ Custom GRPO loss ready")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 12: 45-MINUTE SPEEDRUN TRAINING LOOP ══════════════════════
# ════════════════════════════════════════════════════════════════════
# Total: 40 steps (10+10+20) — designed to run in ~45 min on A100 3B

STAGE_CONFIGS = {
    1: {"name": "Stage 1 — Format (task1 only)",
        "tasks": ["task1"], "lr": 2e-4, "steps": 10, "temp": 1.2},
    2: {"name": "Stage 2 — Hard Rules (task1+task2)",
        "tasks": ["task1", "task2"], "lr": 1e-4, "steps": 10, "temp": 1.0},
    3: {"name": "Stage 3 — Portfolio (all tasks)",
        "tasks": None, "lr": 5e-5, "steps": 20, "temp": 0.9},
}
# What each stage specialises in:
#  Stage 1: learn submit_decision() format (very high LR, easy data)
#  Stage 2: learn hard-rule compliance (HR-01, HR-04 trigger = REJECT)
#  Stage 3: learn portfolio-aware decisions (NPA %, CRAR limits)

all_logs    = {1: [], 2: [], 3: []}
stage_times = {}


def _get_optimizer(model, lr: float):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )


print("=" * 68)
print("  STARTING 45-MIN SPEEDRUN GRPO — Qwen2.5-3B (A100)")
print(f"  Config: NUM_GENERATIONS={NUM_GENERATIONS} | BATCH_SIZE={BATCH_SIZE} "
      f"| GRAD_ACCUM={GRAD_ACCUM}")
print(f"  Total steps: {sum(c['steps'] for c in STAGE_CONFIGS.values())} "
      f"| Est. time: ~45 min on A100")
print("=" * 68)

for stage_num in [1, 2, 3]:
    cfg = STAGE_CONFIGS[stage_num]
    if cfg["tasks"] is None:
        stage_data = rows
    else:
        stage_data = [r for r in rows if r["task_id"] in cfg["tasks"]]
    random.shuffle(stage_data)

    print(f"\n{'─'*68}")
    print(f"  {cfg['name']}")
    print(f"  Samples: {len(stage_data)} | LR: {cfg['lr']} | "
          f"Steps: {cfg['steps']} | Temp: {cfg['temp']}")
    print(f"{'─'*68}")

    optimizer           = _get_optimizer(model, cfg["lr"])
    scheduler           = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["steps"], eta_min=cfg["lr"] * 0.1
    )
    ref_log_probs_cache: Dict = {}
    model.train()
    t_start  = time.time()
    data_idx = 0

    for step in range(1, cfg["steps"] + 1):
        # Sample batch
        batch = [stage_data[data_idx % len(stage_data)]
                 for _ in range(BATCH_SIZE)]
        data_idx += BATCH_SIZE

        # Single grad-accum step (GRAD_ACCUM=1 → no loop needed)
        loss, metrics = grpo_loss_step(
            model, tokenizer, batch,
            cfg["temp"], MEMORY_BANK, ref_log_probs_cache,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Metrics
        avg_reward = float(np.mean(metrics.get("reward", [0])))
        std_reward = float(np.std(metrics.get("reward", [0])))
        avg_kl     = float(np.mean(metrics.get("kl", [0])))
        avg_tools  = float(np.mean(metrics.get("tool_calls", [0])))
        avg_turns  = float(np.mean(metrics.get("turns_per_ep", [1])))
        pt_counts  = defaultdict(int)
        for pt in metrics.get("parse_types", []):
            pt_counts[pt] += 1
        total_pt   = max(sum(pt_counts.values()), 1)
        submit_pct = pt_counts.get("final_decision", 0) / total_pt * 100
        elapsed    = time.time() - t_start

        all_logs[stage_num].append({
            "step": step, "stage": stage_num,
            "loss": float(loss.detach()), "reward": avg_reward,
            "reward_std": std_reward, "kl": avg_kl,
            "tool_calls": avg_tools, "submit_pct": submit_pct,
            "turns": avg_turns, "lr": scheduler.get_last_lr()[0],
        })

        # Print every step (only 40 total — all important)
        eta = (elapsed / step) * (cfg["steps"] - step)
        print(
            f"  [{stage_num}] Step {step:2d}/{cfg['steps']} | "
            f"loss={float(loss.detach()):+.3f} | "
            f"reward={avg_reward:+.2f}±{std_reward:.2f} | "
            f"submit={submit_pct:.0f}% | "
            f"turns={avg_turns:.1f} | "
            f"ETA {eta/60:.1f}m"
        )

    elapsed = time.time() - t_start
    stage_times[stage_num] = elapsed

    # Save checkpoint
    ckpt_dir = f"{OUTPUT_BASE}/stage_{stage_num}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"\n  ✅ Stage {stage_num} done in {elapsed/60:.1f} min")
    print(f"  💾 Checkpoint saved: {ckpt_dir}")

    # 3 sample inferences
    print(f"\n  🔍 Sample inferences after Stage {stage_num}:")
    model.eval()
    for idx in range(min(3, len(stage_data))):
        s  = stage_data[idx]
        ep = run_episode(s, model, tokenizer, cfg["temp"], MEMORY_BANK)
        print(f"  [{idx+1}] PD={s['ground_truth_pd']:.2f} | "
              f"[{ep['parse_type']}] tools={ep['tool_calls_made']} | "
              f"{ep['completion'][:90]}...")
    model.train()

print(f"\n{'='*68}")
print(f"  ALL 3 STAGES COMPLETE ✅")
print(f"  Total training time: {sum(stage_times.values())/60:.1f} min")
print(f"  Total cost (@ $2.50/hr): "
      f"~${sum(stage_times.values())/3600*2.5:.2f}")
print(f"{'='*68}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 13: LEARNING CURVES ═══════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

print("\n📊 Generating training charts...")

all_steps, all_losses, all_rewards, all_kl = [], [], [], []
all_tools, all_submit, all_turns_data = [], [], []
stage_boundaries = []
offset = 0

for stage in [1, 2, 3]:
    logs = all_logs[stage]
    if not logs:
        continue
    stage_boundaries.append(offset)
    for e in logs:
        s = e["step"] + offset
        all_steps.append(s)
        all_losses.append((s, e["loss"]))
        all_rewards.append((s, e["reward"]))
        all_kl.append((s, e["kl"]))
        all_tools.append((s, e["tool_calls"]))
        all_submit.append((s, e["submit_pct"]))
        all_turns_data.append((s, e.get("turns", 1)))
    offset += max(e["step"] for e in logs) + 1

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("IntelliCredit 45-Min Speedrun — Qwen2.5-3B GRPO (A100)",
             fontsize=14, fontweight="bold")


def _plot(ax, data, color, title, ylabel, ylim=None):
    if not data:
        return
    xs, ys = zip(*data)
    ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.8)
    if len(ys) > 4:
        w  = min(5, len(ys) // 2)
        sm = np.convolve(ys, np.ones(w) / w, mode="valid")
        ax.plot(xs[:len(sm)], sm, color=color, linewidth=2.5, label="Smoothed")
    for b in stage_boundaries[1:]:
        ax.axvline(x=b, color="gray", linestyle="--", alpha=0.45, label="Stage →")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8)


_plot(axes[0, 0], all_losses,     "#E53935", "GRPO Loss",             "Loss")
_plot(axes[0, 1], all_rewards,    "#1976D2", "Mean Reward ↑",         "Reward")
_plot(axes[0, 2], all_kl,         "#7B1FA2", "KL Divergence",         "KL",     ylim=[0, None])
_plot(axes[1, 0], all_tools,      "#F57C00", "Avg Tool Calls",        "Calls",  ylim=[0, MAX_TOOL_TURNS + 0.5])
_plot(axes[1, 1], all_submit,     "#00796B", "submit_decision() Rate ↑", "%",   ylim=[0, 105])
_plot(axes[1, 2], all_turns_data, "#5D4037", "Avg Turns per Episode", "Turns",  ylim=[0.5, MAX_TOOL_TURNS + 1.5])

plt.tight_layout()
chart_path = f"{OUTPUT_BASE}/charts/speedrun_curves.png"
plt.savefig(chart_path, dpi=130)
plt.show()
print(f"✅ Charts saved → {chart_path}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 14: PUSH ADAPTER TO HF HUB ════════════════════════════════
# ════════════════════════════════════════════════════════════════════
# Pushes ONLY the LoRA adapter (~80MB) not the full 3B model (~6GB)

HF_USERNAME  = "vssksn"   # ← change to your HF username
PUSH_REPO    = f"{HF_USERNAME}/intellicredit-3b-grpo-speedrun"

print(f"\n🚀 Pushing final LoRA adapter to: {PUSH_REPO}")

# Save final merged adapter
final_dir = f"{OUTPUT_BASE}/final"
os.makedirs(final_dir, exist_ok=True)
model.eval()
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# Push to HF
try:
    model.push_to_hub(PUSH_REPO, private=True)
    tokenizer.push_to_hub(PUSH_REPO, private=True)
    print(f"✅ Adapter pushed → https://huggingface.co/{PUSH_REPO}")
    print(f"   Load it later with:")
    print(f'   from peft import PeftModel')
    print(f'   model = PeftModel.from_pretrained(base_model, "{PUSH_REPO}")')
except Exception as e:
    print(f"⚠️  Push failed ({e}) — adapter is saved locally at {final_dir}/")
    print(f"   Download it manually from the JupyterLab Files panel.")

print("\n🎉 DONE! Your IntelliCredit 3B agent is trained and ready.")
print(f"   Final checkpoint : {final_dir}")
print(f"   Training charts  : {OUTPUT_BASE}/charts/speedrun_curves.png")
