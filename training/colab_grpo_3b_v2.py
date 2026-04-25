"""
╔══════════════════════════════════════════════════════════════════════╗
║  IntelliCredit v2 — GRPO Mistral-7B (A100)                          ║
║  Model  : mistralai/Mistral-7B-Instruct-v0.3  (4-bit QLoRA)         ║
║  Target : HF JupyterLab Space — Nvidia A100 · 80 GB                 ║
║                                                                      ║
║  Why Mistral instead of Qwen2.5-3B:                                  ║
║  ✅ Native function-call training in v0.3                           ║
║  ✅ Dramatically better instruction following → submit_decision()   ║
║     naturally at cold start → std>0 from GRPO Step 1               ║
║  ✅ 7B params in 4-bit = ~6GB VRAM (trivial on A100 80GB)          ║
║  ✅ [INST] chat template handled automatically by tokenizer         ║
║  ✅ KL_BETA=0.15 + SFT warmup + robust parser v3                   ║
║  ✅ Stage 0 SFT warmup (9 gold examples) before GRPO               ║
╚══════════════════════════════════════════════════════════════════════╝

Paste each ═══ CELL ═══ block into a separate Jupyter cell.
Run Cell 1 → Restart Kernel → Run Cell 2 onward in order.
"""


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 1: INSTALL ════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

_pip("torch", "torchvision", "torchaudio",
     "--index-url", "https://download.pytorch.org/whl/cu121", "--upgrade")
_pip("--upgrade", "pip")
_pip("bitsandbytes>=0.46.1")
_pip("transformers>=4.45.0,<5.0.0")
_pip("trl>=0.15.2", "peft>=0.13.0", "accelerate>=1.0.0")
_pip("datasets>=2.20.0", "huggingface_hub>=0.24.0", "matplotlib", "langdetect")
_pip("sentencepiece", "protobuf")   # ← required for Mistral SentencePiece tokenizer

import os
hf_tok = os.environ.get("HF_TOKEN", "")
if hf_tok:
    from huggingface_hub import login
    login(token=hf_tok, add_to_git_credential=False)
    print("✅ HF Token loaded")
else:
    print("⚠️  HF_TOKEN not set — anonymous access (OK for public models)")

print("\n✅ All packages installed")
print("   ⚡ IMPORTANT: Kernel → Restart Kernel → then run from Cell 2")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 2: IMPORTS & CONFIG ═══════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

import os, re, json, time, random
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from huggingface_hub import hf_hub_download

# ── Core Config ──────────────────────────────────────────────────────
MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.3"   # ← switched from Qwen 3B
HF_DATASET   = "vssksn/intellicredit-grpo-v2"
OUTPUT_BASE  = "intellicredit-grpo-mistral-7b"

# ── GRPO hyper-params ────────────────────────────────────────────────
NUM_GENERATIONS  = 6       # good advantage contrast
MAX_NEW_TOKENS   = 350     # must be big enough for analysis + submit_decision()
MAX_TOOL_TURNS   = 2
BATCH_SIZE       = 2       # Mistral-7B 4-bit: ~6GB, easily fits 2 on A100
GRAD_ACCUM       = 1
KL_BETA          = 0.15   # tight leash — prevents drift
MAX_SEQ_LEN      = 1600   # Mistral supports 32K context, use a bit more
MULTI_STEP_GRPO  = False  # single-turn stable for cold-start

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/charts", exist_ok=True)

print("✅ Imports complete")
gpu  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"   GPU  : {gpu}")
print(f"   VRAM : {vram:.1f} GB")
print(f"   Model: {MODEL_NAME}")
print(f"   KL_BETA={KL_BETA} | NUM_GEN={NUM_GENERATIONS} | MULTI_STEP={MULTI_STEP_GRPO}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 3: ACTION PARSER v3 (robust — handles Qwen verbose style) ═
# ════════════════════════════════════════════════════════════════════
#
# Priority cascade (highest → lowest confidence):
#  L1: exact   submit_decision("ACTION", "reason")
#  L2: fuzzy   submit variants (missing quotes, typos, camelCase)
#  L3: tool    get_financial_report(...) / check_compliance_status(...)
#  L4: label   Final Decision: REJECT  /  Decision: APPROVE  / My recommendation: ...
#  L5: sentence I would REJECT / I recommend to APPROVE / Therefore, CONDITIONAL
#  L6: keyword last standalone APPROVE/REJECT/CONDITIONAL anywhere in text
#  L7: default silent REJECT (parse_failure=True, lowest reward)

ACTION_MAP = {
    "APPROVE": 0, "APPROVED": 0,
    "CONDITIONAL": 1, "CONDITIONAL_APPROVE": 1, "CONDITIONAL APPROVE": 1,
    "REJECT": 2, "REJECTED": 2, "DECLINE": 2, "DENY": 2,
    "CONDITIONALLY APPROVE": 1, "CONDITIONAL APPROVAL": 1,
}

# ── compiled regex patterns ────────────────────────────────────────
_ACT = r"(APPROVE[D]?|CONDITIONAL[_\s]?APPROVE[D]?|CONDITIONAL|REJECT(?:ED)?|DECLINE|DENY)"

# L1: exact submit_decision("ACTION", "reason")
_RE_L1_EXACT = re.compile(
    r"submit_decision\s*\(\s*['\"]?\s*" + _ACT + r"\s*['\"]?"
    r"(?:\s*,\s*['\"]?(.*?)['\"]?)?\s*\)",
    re.IGNORECASE | re.DOTALL,
)

# L2: fuzzy variants — submit decision / submitDecision / submit-decision / decision(...)
_RE_L2_FUZZY = re.compile(
    r"(?:submit[\s_-]?decision|submitDecision|final[\s_]?decision\s*\(|decision\s*\()"
    r"\s*\(\s*['\"]?\s*" + _ACT + r"\s*['\"]?"
    r"(?:\s*,\s*['\"]?(.*?)['\"]?)?\s*\)",
    re.IGNORECASE | re.DOTALL,
)

# L3: tool call
_RE_L3_TOOL = re.compile(
    r"\b(get_financial_report|check_compliance_status|get_market_intelligence)"
    r"\s*\(\s*['\"]?([^)'\"]*)['\"]?\s*\)",
    re.IGNORECASE,
)

# L4: structured label on its own line / after a colon
#     "Final Decision: REJECT"  |  "Decision: APPROVE"  |  "**REJECT**"
_RE_L4_LABEL = re.compile(
    r"(?:"
    r"(?:final\s+)?(?:decision|recommendation|answer|verdict|conclusion)"
    r"|my\s+(?:final\s+)?(?:recommendation|decision|answer)"
    r"|credit\s+(?:decision|recommendation)"
    r")"
    r"[\s:*─—]+\**\s*" + _ACT,
    re.IGNORECASE,
)

# Support bare bold like **REJECT** or __APPROVE__ at start of a line
_RE_L4_BOLD = re.compile(
    r"(?:^|\n)\s*[*_]{1,2}\s*" + _ACT + r"\s*[*_]{1,2}",
    re.IGNORECASE | re.MULTILINE,
)

# L5: decision embedded in a sentence
#     "I would reject"  |  "We should approve"  |  "Therefore, reject"
_RE_L5_SENTENCE = re.compile(
    r"(?:"
    r"(?:I|we)\s+(?:would|will|must|should|hereby|therefore|thus|hence)?\s*"
    r"(?:recommend(?:\s+(?:to|that\s+we))?\s+)?"
    r"|(?:therefore|thus|hence|consequently|accordingly|as\s+such),?\s+(?:(?:I|we)\s+)?"
    r"|(?:my\s+)?recommendation\s+is\s+(?:to\s+)?"
    r"|I\s+am\s+(?:going\s+to|recommending\s+(?:to\s+)?|submitting\s+(?:to\s+)?)?"
    r")"
    + _ACT,
    re.IGNORECASE,
)

# L6: bare keyword — last mention anywhere
_RE_L6_KEYWORD = re.compile(
    r"\b" + _ACT + r"\b",
    re.IGNORECASE,
)


def _unwrap(text) -> str:
    if isinstance(text, list):
        return " ".join(m.get("content", "") for m in text if isinstance(m, dict))
    return str(text) if not isinstance(text, str) else text


def _norm_action(raw: str) -> int:
    """Normalise a matched action string to 0/1/2."""
    k = raw.upper().strip()
    k = re.sub(r"[\s_]+", " ", k)   # collapse spaces/underscores
    return ACTION_MAP.get(k, ACTION_MAP.get(k.split()[0], 2))


def _extract_context(t: str, match_end: int, window: int = 200) -> str:
    """Extract surrounding text as reasoning context."""
    start = max(0, match_end - window)
    return t[start:match_end + window].strip()


def parse_llm_output(text) -> Dict:
    t = _unwrap(text).strip()
    if not t:
        return {"action": 2, "parse_type": "default_reject",
                "parse_confidence": 0.0, "reasoning": "",
                "tool_name": None, "tool_args": None, "parse_failure": True}

    # ── L3: tool call — check BEFORE decision patterns so tool text
    #         in the middle of a response doesn't get mis-parsed        ──────
    tm = _RE_L3_TOOL.search(t)
    if tm:
        raw_arg   = tm.group(2).strip().strip("'\"")
        tool_name = tm.group(1).lower()
        tool_args = ({"sector": raw_arg}
                     if "market_intelligence" in tool_name
                     else {"company_id": raw_arg})
        return {"action": 2, "parse_type": "tool_call",
                "tool_name": tool_name, "tool_arg": raw_arg,
                "tool_args": tool_args, "parse_confidence": 0.95,
                "reasoning": f"calling {tool_name}",
                "parse_failure": False}

    # ── L1: exact submit_decision ─────────────────────────────────────
    ms = list(_RE_L1_EXACT.finditer(t))
    if ms:
        m   = ms[-1]   # take last match (model may reason, then conclude)
        raw = m.group(1)
        rsn = (m.group(2) or "").strip()
        # fall back: grab text between previous sentence and match as context
        if not rsn:
            rsn = _extract_context(t, m.start(), 120)
        act = _norm_action(raw)
        return {"action": act, "parse_type": "final_decision",
                "parse_confidence": 0.95 if rsn else 0.70,
                "reasoning": rsn, "tool_name": None, "tool_args": None,
                "parse_failure": False}

    # ── L2: fuzzy submit variants ─────────────────────────────────────
    ms2 = list(_RE_L2_FUZZY.finditer(t))
    if ms2:
        m   = ms2[-1]
        raw = m.group(1)
        rsn = (m.group(2) or "").strip()
        act = _norm_action(raw)
        return {"action": act, "parse_type": "final_decision",
                "parse_confidence": 0.80,
                "reasoning": rsn or _extract_context(t, m.start(), 100),
                "tool_name": None, "tool_args": None, "parse_failure": False}

    # ── L4: structured label ──────────────────────────────────────────
    ml4a = list(_RE_L4_LABEL.finditer(t))
    ml4b = list(_RE_L4_BOLD.finditer(t))
    all_l4 = sorted(ml4a + ml4b, key=lambda m: m.end())
    if all_l4:
        m   = all_l4[-1]
        raw = m.group(1)
        act = _norm_action(raw)
        rsn = t[max(0, m.start() - 200): m.end()].strip()
        return {"action": act, "parse_type": "structured_label",
                "parse_confidence": 0.75, "reasoning": rsn,
                "tool_name": None, "tool_args": None, "parse_failure": False}

    # ── L5: sentence-level intent ─────────────────────────────────────
    ml5 = list(_RE_L5_SENTENCE.finditer(t))
    if ml5:
        m   = ml5[-1]
        raw = m.group(1)
        act = _norm_action(raw)
        rsn = t[max(0, m.start() - 150): m.end() + 50].strip()
        return {"action": act, "parse_type": "sentence_decision",
                "parse_confidence": 0.65, "reasoning": rsn,
                "tool_name": None, "tool_args": None, "parse_failure": False}

    # ── L6: bare keyword anywhere ─────────────────────────────────────
    ml6 = list(_RE_L6_KEYWORD.finditer(t))
    if ml6:
        m   = ml6[-1]
        raw = m.group(1)
        act = _norm_action(raw)
        return {"action": act, "parse_type": "fallback_keyword",
                "parse_confidence": 0.45, "reasoning": t[-150:],
                "tool_name": None, "tool_args": None, "parse_failure": True}

    # ── L7: silence / unparseable ─────────────────────────────────────
    return {"action": 2, "parse_type": "default_reject",
            "parse_confidence": 0.0, "reasoning": "",
            "tool_name": None, "tool_args": None, "parse_failure": True}


# ── Self-test ─────────────────────────────────────────────────────
_PARSER_TESTS = [
    ('submit_decision("REJECT", "DSCR below 1.0")',                          "final_decision",    2),
    ("submit_decision(APPROVE, strong financials DSCR 1.8x)",                "final_decision",    0),
    ("get_financial_report('Acme Corp')",                                    "tool_call",         2),
    ("Final Decision: CONDITIONAL",                                          "structured_label",  1),
    ("**REJECT**",                                                           "structured_label",  2),
    ("I would recommend to APPROVE this application given strong metrics.",  "sentence_decision", 0),
    ("Therefore, REJECT.",                                                   "sentence_decision", 2),
    ("Based on my analysis REJECT",                                          "fallback_keyword",  2),
    ("The applicant is a good risk.",                                        "default_reject",    2),
]
_pass = _fail = 0
for _txt, _expected_type, _expected_act in _PARSER_TESTS:
    _r = parse_llm_output(_txt)
    _ok = (_r["parse_type"] == _expected_type and _r["action"] == _expected_act)
    if _ok: _pass += 1
    else:
        _fail += 1
        print(f"  ⚠️  PARSER FAIL: '{_txt[:50]}' → got {_r['parse_type']}/{_r['action']}, "
              f"expected {_expected_type}/{_expected_act}")
print(f"✅ Action parser v3 loaded — {_pass}/{len(_PARSER_TESTS)} self-tests passed"
      + (f" ({_fail} failed)" if _fail else ""))


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 4: REWARD FUNCTIONS v5 (with language penalty) ════════════
# ════════════════════════════════════════════════════════════════════

# ── Language detection (penalise non-English drift) ──────────────────
try:
    from langdetect import detect as _lang_detect
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False

def _is_english(text: str) -> bool:
    """Returns True if text appears to be English, False otherwise."""
    if not _LANGDETECT_OK or len(text.strip()) < 20:
        return True
    try:
        return _lang_detect(text[:300]) == "en"
    except Exception:
        return True  # benefit of the doubt


def reward_correctness(prompts, completions, ground_truth_pd, **kw) -> List[float]:
    scores = []
    for comp, pd in zip(completions, ground_truth_pd):
        p   = parse_llm_output(comp)
        act = p["action"]
        if pd > 0.45:
            s = 1.5 if act == 2 else (-1.5 if act == 0 else 0.0)
        elif pd < 0.15:
            s = 1.5 if act == 0 else (-0.5 if act == 2 else 0.5)
        else:
            s = 1.0 if act == 1 else (0.2 if act == 0 else -0.5)
        if p["parse_type"] == "final_decision":
            s += 0.5   # strong bonus for using submit_decision()
        scores.append(max(-2.0, min(2.0, s)))
    return scores


def reward_hard_rules(prompts, completions, hard_rules, has_red_alerts, **kw) -> List[float]:
    scores = []
    for comp, hr_str, red in zip(completions, hard_rules, has_red_alerts):
        p   = parse_llm_output(comp)
        act = p["action"]
        try:
            hr_list = json.loads(hr_str) if isinstance(hr_str, str) else hr_str
        except Exception:
            hr_list = []
        has_hr = bool(hr_list) or red
        scores.append(0.5 if (has_hr and act == 2) else
                      (-2.0 if (has_hr and act != 2) else
                       (0.3 if act != 2 else -0.5)))
    return scores


def reward_format(prompts, completions, **kw) -> List[float]:
    """
    Smooth 7-tier reward gradient — guides model step by step toward
    full submit_decision() compliance:

    default_reject   : -1.0  (unparseable — worst)
    fallback_keyword : -0.3  (found a word, not a decision)
    sentence_decision: +0.2  (implied intent in text — partial credit)
    structured_label : +0.4  (e.g. "Final Decision: REJECT" — good progress)
    tool_call        : +0.1  (mid-episode tool use — neutral/slight positive)
    final_decision   : +0.6  (submit_decision() with short reasoning)
    final_decision   : +1.0  (submit_decision() with 20+ word reasoning — perfect)
    """
    scores = []
    for comp in completions:
        p   = parse_llm_output(comp)
        rsn = p.get("reasoning", "")
        pt  = p["parse_type"]

        # ── Language penalty (severe — prevents multilingual drift) ───
        lang_ok      = _is_english(comp)
        lang_penalty = 0.0 if lang_ok else -2.0

        # ── 7-tier format reward ──────────────────────────────────────
        if pt == "final_decision" and len(rsn) > 20:
            s = 1.0    # perfect: submit_decision + solid reasoning
        elif pt == "final_decision":
            s = 0.6    # good: submit_decision but short/missing reasoning
        elif pt == "structured_label":
            s = 0.4    # e.g. "Final Decision: REJECT" — model getting close
        elif pt == "sentence_decision":
            s = 0.2    # e.g. "I would REJECT..." — intent clear, wrong format
        elif pt == "tool_call":
            s = 0.1    # mid-episode tool use — correct behaviour
        elif pt == "fallback_keyword":
            s = -0.3   # bare keyword only — very weak signal
        else:          # default_reject: unparseable/multilingual garbage
            s = -1.0

        scores.append(s + lang_penalty)
    return scores


def reward_portfolio(prompts, completions, npa_rate, crar, **kw) -> List[float]:
    scores = []
    for comp, npa, cr in zip(completions, npa_rate, crar):
        p = parse_llm_output(comp); act = p["action"]; s = 0.0
        if npa > 0.048 and act == 0: s -= 0.8
        if cr  < 0.13  and act == 0: s -= 0.6
        if npa < 0.02  and act == 2: s -= 0.3
        scores.append(max(-0.8, min(0.3, s)))
    return scores


def combined_reward(prompts, completions, ground_truth_pd,
                    hard_rules, has_red_alerts, npa_rate, crar) -> List[float]:
    r1 = reward_correctness(prompts, completions, ground_truth_pd)
    r2 = reward_hard_rules(prompts, completions, hard_rules, has_red_alerts)
    r3 = reward_format(prompts, completions)
    r4 = reward_portfolio(prompts, completions, npa_rate, crar)
    return [a + b + c + d for a, b, c, d in zip(r1, r2, r3, r4)]

print("✅ Reward functions v5 loaded (with language penalty)")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 5: TOOL RESULT SYNTHESIZER ════════════════════════════════
# ════════════════════════════════════════════════════════════════════

def _ef(text: str, pattern: str, default: float) -> float:
    m = re.search(pattern, text, re.IGNORECASE)
    try:
        return float(m.group(1)) if m else default
    except Exception:
        return default


def synthesize_tool_result(tool_name: str, tool_arg: str, prompt_text: str) -> str:
    p = prompt_text
    tool_name = (tool_name or "").lower().strip()
    if tool_name == "get_financial_report":
        cname = tool_arg or "Applicant"
        dscr  = _ef(p, r"DSCR:\s*([\d.]+)x", 1.5)
        cr    = _ef(p, r"Current\s*Ratio:\s*([\d.]+)x", 1.2)
        de    = _ef(p, r"[Dd]ebt.to.[Ee]quity:\s*([\d.]+)x", 1.5)
        eb    = _ef(p, r"EBITDA\s*Margin:\s*([\d.]+)%", 18.0)
        cibil = _ef(p, r"CIBIL\s*[Ss]core:\s*([\d.]+)", 650.0)
        coll  = _ef(p, r"[Cc]ollateral\s*[Cc]overage:\s*([\d.]+)x", 1.0)
        bbr   = _ef(p, r"[Cc]heque\s*[Bb]ounce\s*[Rr]ate:\s*([\d.]+)%", 5.0)
        od    = _ef(p, r"OD\s*[Uu]tilisation:\s*([\d.]+)%", 50.0)
        dscr_flag = " ⚠️ BELOW 1.0 — HR-01 TRIGGERED" if dscr < 1.0 else (" ✅ Strong" if dscr >= 1.5 else " ✓ Adequate")
        return "\n".join([
            f"═══ FINANCIAL REPORT: {cname} ═══",
            f"  DSCR              : {dscr:.2f}x{dscr_flag}",
            f"  Current Ratio     : {cr:.2f}x  {'✅' if cr >= 1.2 else '⚠️'}",
            f"  Debt-to-Equity    : {de:.2f}x  {'✅' if de <= 2.0 else '⚠️ High leverage'}",
            f"  EBITDA Margin     : {eb:.1f}%  {'✅' if eb >= 15 else '⚠️ Thin margin'}",
            f"  CIBIL Score       : {int(cibil)}  {'✅' if cibil >= 700 else '⚠️'}",
            f"  Collateral Cover  : {coll:.2f}x  {'✅' if coll >= 1.25 else '⚠️ Thin cover'}",
            f"  OD Utilisation    : {od:.1f}%  {'⚠️ High' if od > 75 else '✅'}",
            f"  Cheque Bounce     : {bbr:.1f}%{'  ⚠️ HR-04 TRIGGERED' if bbr > 25 else ''}",
            "═══ END FINANCIAL REPORT ═══",
        ])
    elif tool_name == "check_compliance_status":
        cname = tool_arg or "Applicant"
        dscr_val = _ef(p, r"DSCR:\s*([\d.]+)x", 1.5)
        bounce   = _ef(p, r"[Cc]heque\s*[Bb]ounce\s*[Rr]ate:\s*([\d.]+)%", 0.0)
        hr_lines = []
        if dscr_val < 1.0: hr_lines.append(f"   🚫 HR-01: DSCR={dscr_val:.2f}x — MANDATORY REJECT")
        if bounce > 25:    hr_lines.append(f"   🚫 HR-04: Bounce={bounce:.1f}% — MANDATORY REJECT")
        lines = [f"═══ COMPLIANCE: {cname} ═══"]
        if hr_lines:
            lines.append("── HARD RULES TRIGGERED ──")
            lines.extend(hr_lines)
        else:
            lines.append("  ✅ No hard rules triggered")
        lines.append("═══ END COMPLIANCE ═══")
        return "\n".join(lines)
    elif tool_name == "get_market_intelligence":
        sector_arg = tool_arg or "General"
        npa  = _ef(p, r"NPA\s*[Rr]ate:\s*([\d.]+)%", 2.0) / 100
        crar = _ef(p, r"CRAR:\s*([\d.]+)%", 18.0) / 100
        return "\n".join([
            f"═══ MARKET INTELLIGENCE: {sector_arg} ═══",
            f"  NPA Rate : {npa:.1%}  {'⚠️ Near limit' if npa > 0.04 else '✅'}",
            f"  CRAR     : {crar:.1%}  {'🚫 BELOW 12.5%' if crar < 0.125 else '✅'}",
            "═══ END MARKET INTELLIGENCE ═══",
        ])
    return f"[TOOL ERROR] Unknown tool: {tool_name}"

print("✅ Tool result synthesizer loaded")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 6: MEMORY BANK ════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

class MemoryBank:
    def __init__(self, max_lessons: int = 5):
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
        for i, l in enumerate(self.lessons[-2:], 1):
            lines.append(f"  Lesson {i}: {l['lesson']}")
        return "\n".join(lines)

    def learn_from_episode(self, prompt: str, final_text: str, reward: float, parse_type: str):
        if parse_type == "default_reject":
            self.add_lesson("Always end with submit_decision('ACTION', 'reason')", reward, parse_type)
        elif parse_type == "fallback_keyword":
            self.add_lesson("Use submit_decision() format, not a plain keyword", reward, parse_type)

MEMORY_BANK = MemoryBank()
print("✅ Memory bank initialised")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 7: SYSTEM PROMPT WITH FEW-SHOT EXAMPLES ══════════════════
# ════════════════════════════════════════════════════════════════════
# ▶ KEY FIX: the model now sees concrete submit_decision() examples
#   so it knows the exact format and can generate it from turn 1.
#   This breaks the std=0 deadlock in GRPO.

SYSTEM_PROMPT_BASE = """\
You are a Senior Credit Officer at an Indian NBFC. Review MSME loan applications and make APPROVE / CONDITIONAL / REJECT decisions balancing yield, risk, and RBI compliance.

══ AVAILABLE TOOLS (call up to {max_tool_turns} times) ══
  get_financial_report("company_name")
    → Returns DSCR, current ratio, D/E, EBITDA, CIBIL, collateral, OD utilisation.
  check_compliance_status("company_name")
    → Returns hard rules triggered, forensic alerts, repeat applicant flag.
  get_market_intelligence("sector_name")
    → Returns macro stress, NPA rate, CRAR, capital utilisation.

══ RBI HARD RULES — MANDATORY REJECT if any triggered ══
  HR-01: DSCR < 1.0   HR-03: RED forensic alert   HR-04: Cheque bounce > 25%
  HR-05: GST < 40%    HR-02: Director disqualified

══ OUTPUT — YOU MUST END WITH EXACTLY THIS FORMAT ══
  submit_decision("APPROVE",      "your reasoning in at least 20 words here")
  submit_decision("CONDITIONAL",  "your reasoning in at least 20 words here")
  submit_decision("REJECT",       "your reasoning in at least 20 words here")

══ EXAMPLES OF CORRECT OUTPUT ══

Example 1 (REJECT due to hard rule):
  I reviewed the financials. The DSCR is 0.72x which is below the 1.0x minimum, triggering HR-01.
  submit_decision("REJECT", "Hard rule HR-01 triggered: DSCR is 0.72x, below mandatory 1.0x minimum. Application must be declined regardless of other factors.")

Example 2 (APPROVE with strong financials):
  The financials are strong with DSCR 1.85x, CIBIL 760, and no hard rules triggered.
  submit_decision("APPROVE", "Strong financials: DSCR 1.85x well above minimum, CIBIL 760, no compliance issues, stable revenue growth of 14% CAGR justifies approval.")

Example 3 (CONDITIONAL with thin collateral):
  Financials are acceptable but collateral coverage of 0.95x is below the 1.25x preferred level.
  submit_decision("CONDITIONAL", "Adequate DSCR 1.3x and CIBIL 720, however collateral coverage 0.95x is below preferred 1.25x. Approve subject to additional collateral or personal guarantee.")

Think step by step. Check hard rules first. If any triggered → REJECT immediately.
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

print("✅ System prompt with few-shot examples loaded")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 8: LOAD MODEL + LoRA ══════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading model: {MODEL_NAME}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map          = "auto",
    torch_dtype         = torch.bfloat16,   # Mistral: use torch_dtype not dtype
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Mistral tokenizer fix: no pad_token by default, use eos_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = "left"

# Mistral-7B LoRA — same target modules as Qwen (standard MistralForCausalLM)
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
print(f"✅ Model + LoRA ready (vocab size: {len(tokenizer)})")

# Smoke test — now with few-shot examples in prompt, should see submit_decision
print("\n🔍 Smoke test (should now show submit_decision in response)...")
_msgs = [
    {"role": "system", "content": build_system_prompt()},
    {"role": "user",   "content": "Application: DSCR=0.8x — this is below the 1.0x minimum (HR-01 triggered). What is your decision?"},
]
_txt = tokenizer.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
_inp = tokenizer(_txt, return_tensors="pt").to(model.device)
with torch.no_grad():
    _out = model.generate(**_inp, max_new_tokens=80, do_sample=True, temperature=0.7)
_resp = tokenizer.decode(_out[0][_inp.input_ids.shape[1]:], skip_special_tokens=True)
_p    = parse_llm_output(_resp)
print(f"   Response  : {_resp[:150]}")
print(f"   Parse type: {_p['parse_type']}")
if _p["parse_type"] == "final_decision":
    print("   🎉 submit_decision() detected at cold start — GRPO will work!")
else:
    print("   ⚠️  Not yet using submit_decision — will learn during training")
print("✅ Smoke test done")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 9: LOAD DATASET ═══════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading dataset: {HF_DATASET}...")
import json as _json

jsonl_path = hf_hub_download(
    repo_id   = HF_DATASET,
    filename  = "grpo_dataset.jsonl",
    repo_type = "dataset",
)

rows = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        row  = _json.loads(line)
        meta = row.get("metadata", row)
        hr   = meta.get("hard_rules", [])
        rows.append({
            "prompt"          : row["prompt"],
            "task_id"         : meta.get("task_id", "task1"),
            "ground_truth_pd" : float(meta.get("ground_truth_pd", 0.5)),
            "optimal_action"  : int(meta.get("optimal_action", 2)),
            "hard_rules"      : _json.dumps(hr) if isinstance(hr, list) else str(hr),
            "has_red_alerts"  : bool(meta.get("has_red_alerts", False)),
            "npa_rate"        : float(meta.get("npa_rate", 0.02)),
            "crar"            : float(meta.get("crar", 0.18)),
            "sector"          : str(meta.get("sector", "Unknown")),
            "company_name"    : str(meta.get("company_name", "Company")),
        })

print(f"   Total samples  : {len(rows)}")
tc = defaultdict(int)
for r in rows: tc[r["task_id"]] += 1
print(f"   Tasks          : {dict(sorted(tc.items()))}")
print("✅ Dataset ready")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 10: EPISODE RUNNER ════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

def run_episode(sample: Dict, model, tokenizer,
                temperature: float = 1.0,
                memory_bank: Optional[MemoryBank] = None) -> Dict:
    prompt_text     = sample["prompt"]
    sys_prompt      = build_system_prompt(MAX_TOOL_TURNS, memory_bank)
    tool_transcript = []
    tool_calls_made = 0
    last_completion = ""
    last_parsed     = {}
    all_turns       = []

    for turn in range(MAX_TOOL_TURNS + 1):
        messages  = build_messages(prompt_text, tool_transcript, sys_prompt)
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if turn == MAX_TOOL_TURNS:
            chat_text += "\n[SYSTEM: You MUST now submit your final decision using submit_decision().]"

        inputs = tokenizer(
            chat_text, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                min_new_tokens = 15,
                do_sample      = True,
                temperature    = temperature,
                top_p          = 0.92,
                repetition_penalty = 1.15,   # ← prevents repetition loops
                pad_token_id   = tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        if not completion:
            completion = 'submit_decision("REJECT", "Insufficient information to approve. Rejecting as a precaution.")'

        all_turns.append((chat_text, completion))
        parsed          = parse_llm_output(completion)
        last_completion = completion
        last_parsed     = parsed

        if parsed["parse_type"] == "tool_call":
            tool_result = synthesize_tool_result(
                parsed.get("tool_name", ""),
                parsed.get("tool_arg", ""),
                prompt_text,
            )
            tool_transcript.append({
                "assistant":   completion,
                "tool_result": f"[TOOL RESULT]\n{tool_result}",
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
# ═══ CELL 11: GRPO LOSS ═════════════════════════════════════════════
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
    shift_logits = logits[0, prompt_len - 1:-1, :]
    shift_labels = full_ids[0, prompt_len:]
    log_probs    = F.log_softmax(shift_logits.float(), dim=-1)
    token_lps    = log_probs[
        torch.arange(shift_labels.shape[0], device=model.device), shift_labels
    ]
    return token_lps.sum()


def grpo_loss_step(model, tokenizer, samples: List[Dict],
                   temperature: float, memory_bank: MemoryBank,
                   ref_cache: Dict) -> Tuple[torch.Tensor, Dict]:
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

        sys_p = build_system_prompt(MAX_TOOL_TURNS, memory_bank)

        for ep, adv in zip(episodes, advantages):
            # Single-turn GRPO (MULTI_STEP_GRPO=False) — stable for cold start
            p_str = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_p},
                 {"role": "user",   "content": ep["prompt_text"]}],
                tokenize=False, add_generation_prompt=True
            )
            lp  = compute_log_probs(model, tokenizer, p_str, ep["completion"])

            # ── Reference log-probs: base model (LoRA disabled) ──────────
            # FIX: previously ref_cache used the CURRENT model → kl always 0.
            # Now we disable the LoRA adapter to get the frozen base model's
            # log-probs — this is the correct GRPO reference policy.
            with torch.no_grad(), model.disable_adapter():
                ref_lp = compute_log_probs(
                    model, tokenizer, p_str, ep["completion"]
                ).detach()

            kl        = (lp - ref_lp).clamp(min=0)
            adv_t     = torch.tensor(float(adv), device=model.device)
            step_loss = -(adv_t * lp) + KL_BETA * kl
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

print("✅ GRPO loss ready")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 12: STAGE 0 — SFT WARMUP (5 steps, cross-entropy) ════════
# ════════════════════════════════════════════════════════════════════
#
# WHY: GRPO requires variance (std > 0) across the N generations for
# each sample. The cold Qwen-3B model outputs the same vague text for
# all 6 generations → rewards identical → advantages = 0 → loss = 0.
#
# FIX (used by DeepSeek-R1, Kimi, etc.): run a short SFT warmup on
# hand-crafted gold examples that explicitly contain submit_decision().
# After just 5 CE steps the model produces the format naturally at
# temp=1.5, giving the reward variance GRPO needs.

SFT_GOLD_EXAMPLES = [
    # ── REJECT (HR triggered) ────────────────────────────────────────
    {
        "user": (
            "Application: TechTex Pvt Ltd. Loan: ₹5 Cr.\n"
            "DSCR: 0.72x | Current Ratio: 0.9x | CIBIL: 610\n"
            "Cheque Bounce Rate: 8% | Hard Rules: HR-01 triggered (DSCR<1.0)"
        ),
        "assistant": (
            "Reviewing the application: DSCR is 0.72x which is below the mandatory "
            "1.0x minimum, directly triggering Hard Rule HR-01. This is a mandatory "
            "reject condition regardless of other factors.\n"
            'submit_decision("REJECT", "Hard rule HR-01 triggered: DSCR 0.72x is below '
            'the mandatory 1.0x minimum. Application declined per RBI compliance policy.")'
        ),
    },
    {
        "user": (
            "Application: Sunrise Exports. Loan: ₹2 Cr.\n"
            "DSCR: 0.88x | Cheque Bounce Rate: 32% | CIBIL: 640\n"
            "Hard Rules: HR-01 (DSCR<1.0), HR-04 (bounce>25%) both triggered."
        ),
        "assistant": (
            "Two hard rules are triggered simultaneously: HR-01 due to DSCR of 0.88x "
            "and HR-04 due to cheque bounce rate of 32%. Both are mandatory reject conditions.\n"
            'submit_decision("REJECT", "Two hard rules triggered: HR-01 (DSCR 0.88x < 1.0) '
            'and HR-04 (bounce rate 32% > 25%). Mandatory reject under RBI guidelines.")'
        ),
    },
    {
        "user": (
            "Application: Nova Steel Ltd. Loan: ₹8 Cr.\n"
            "DSCR: 0.95x | EBITDA Margin: 12% | CIBIL: 580\n"
            "Hard Rules: HR-01 triggered. NPA Rate: 5.1% | CRAR: 13%"
        ),
        "assistant": (
            "DSCR of 0.95x falls below the mandatory 1.0x threshold, triggering HR-01. "
            "Additionally, NPA rate 5.1% is above the 5% limit and CIBIL 580 is weak.\n"
            'submit_decision("REJECT", "HR-01 triggered: DSCR 0.95x below 1.0x minimum. '
            'Compounded by weak CIBIL 580 and NPA rate 5.1% above institutional limit.")'
        ),
    },
    # ── APPROVE (strong financials, no hard rules) ───────────────────
    {
        "user": (
            "Application: GreenLeaf Foods. Loan: ₹3 Cr.\n"
            "DSCR: 1.85x | Current Ratio: 1.6x | CIBIL: 762\n"
            "Collateral: 1.45x | D/E: 1.1x | Hard Rules: None triggered.\n"
            "Revenue CAGR: +14% | NPA Rate: 1.8% | CRAR: 19%"
        ),
        "assistant": (
            "All financial metrics are strong: DSCR 1.85x well above minimum, CIBIL 762, "
            "collateral coverage 1.45x, no hard rules triggered, and healthy revenue CAGR.\n"
            'submit_decision("APPROVE", "Strong application: DSCR 1.85x, CIBIL 762, '
            'collateral 1.45x, no compliance issues, 14% revenue CAGR. Risk profile is sound.")'
        ),
    },
    {
        "user": (
            "Application: Precision Auto Parts. Loan: ₹1.5 Cr.\n"
            "DSCR: 2.1x | CIBIL: 798 | Collateral: 1.8x | D/E: 0.8x\n"
            "Hard Rules: None. Cheque Bounce: 1%. OD Utilisation: 35%\n"
            "NPA Rate: 1.2% | CRAR: 21%"
        ),
        "assistant": (
            "Exceptional financials across all metrics. DSCR 2.1x is very strong, CIBIL 798 "
            "is excellent, collateral is well-covered, and banking behaviour is clean.\n"
            'submit_decision("APPROVE", "Excellent credit profile: DSCR 2.1x, CIBIL 798, '
            'collateral 1.8x, 1% bounce rate. No compliance concerns. Strong approval candidate.")'
        ),
    },
    {
        "user": (
            "Application: BlueBird Logistics. Loan: ₹4 Cr.\n"
            "DSCR: 1.52x | CIBIL: 731 | Collateral: 1.3x\n"
            "Hard Rules: None triggered. Revenue: ₹28 Cr | NPA: 2.1%"
        ),
        "assistant": (
            "DSCR 1.52x is adequately above the 1.0x minimum, CIBIL is good at 731, "
            "collateral provides reasonable coverage, and no hard rules are triggered.\n"
            'submit_decision("APPROVE", "Adequate financials: DSCR 1.52x above minimum, '
            'CIBIL 731, collateral 1.3x, no compliance violations. Application approved.")'
        ),
    },
    # ── CONDITIONAL (borderline, needs conditions) ───────────────────
    {
        "user": (
            "Application: Vertex Pharma. Loan: ₹6 Cr.\n"
            "DSCR: 1.18x | CIBIL: 715 | Collateral: 0.92x | D/E: 2.3x\n"
            "Hard Rules: None triggered. OD Utilisation: 71%"
        ),
        "assistant": (
            "DSCR 1.18x is just above the minimum, CIBIL 715 is acceptable, but collateral "
            "of 0.92x is below the preferred 1.25x, and D/E of 2.3x indicates high leverage.\n"
            'submit_decision("CONDITIONAL", "Borderline case: DSCR 1.18x barely adequate, '
            'collateral 0.92x below preferred 1.25x, D/E 2.3x high. Approve subject to '
            'additional collateral or personal guarantee from promoter.")'
        ),
    },
    {
        "user": (
            "Application: Horizon Textiles. Loan: ₹2.5 Cr.\n"
            "DSCR: 1.25x | CIBIL: 688 | Collateral: 1.05x\n"
            "Hard Rules: None. Revenue declining -3% CAGR. OD: 68%"
        ),
        "assistant": (
            "DSCR 1.25x is adequate, but CIBIL 688 is slightly below the preferred 700, "
            "collateral at 1.05x is thin, and revenue shows a declining trend of -3% CAGR.\n"
            'submit_decision("CONDITIONAL", "Adequate DSCR 1.25x but CIBIL 688 below '
            'preferred threshold, collateral 1.05x thin, and -3% revenue decline. '
            'Approve only with additional security and quarterly monitoring.")'
        ),
    },
    {
        "user": (
            "Application: Radiant Construction. Loan: ₹10 Cr.\n"
            "DSCR: 1.35x | CIBIL: 725 | Collateral: 1.15x | D/E: 1.9x\n"
            "Hard Rules: None. Sector: Construction (high stress). NPA: 3.8%"
        ),
        "assistant": (
            "Financial metrics are reasonable, but the construction sector carries elevated "
            "stress, D/E is high at 1.9x, and collateral coverage at 1.15x is below ideal.\n"
            'submit_decision("CONDITIONAL", "Acceptable financials but construction sector '
            'stress, D/E 1.9x, and collateral 1.15x below preferred 1.25x. Approve subject '
            'to phased disbursement and enhanced monitoring given sector risk.")'
        ),
    },
]

# ── SFT Warmup Training ───────────────────────────────────────────────
SFT_STEPS      = 15   # more steps → format ingrained for complex prompts too
SFT_LR         = 2e-5
SFT_EPOCHS     = 2   # cycle through gold examples multiple times per step

print("\n" + "═" * 68)
print("  STAGE 0 — SFT WARMUP (primes submit_decision format)")
print(f"  Gold examples: {len(SFT_GOLD_EXAMPLES)} | Steps: {SFT_STEPS} | LR: {SFT_LR}")
print("  This teaches the model the output format before GRPO begins.")
print("═" * 68)

sft_optimizer = _get_optimizer(model, SFT_LR) if '_get_optimizer' in dir() else \
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=SFT_LR)

def _get_optimizer(model, lr: float):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=0.01)

sft_optimizer = _get_optimizer(model, SFT_LR)
sys_p         = build_system_prompt(MAX_TOOL_TURNS, None)
model.train()
t0_sft = time.time()

for sft_step in range(1, SFT_STEPS + 1):
    step_loss = torch.tensor(0.0, device=model.device)
    n_tokens  = 0

    for _ in range(SFT_EPOCHS):
        random.shuffle(SFT_GOLD_EXAMPLES)
        for ex in SFT_GOLD_EXAMPLES:
            msgs = [
                {"role": "system",    "content": sys_p},
                {"role": "user",      "content": ex["user"]},
                {"role": "assistant", "content": ex["assistant"]},
            ]
            full_text  = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_txt = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True
            )
            full_ids   = tokenizer(full_text,  return_tensors="pt",
                                   truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(model.device)
            prompt_ids = tokenizer(prompt_txt, return_tensors="pt",
                                   truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(model.device)
            prompt_len = prompt_ids.shape[1]

            if full_ids.shape[1] <= prompt_len:
                continue

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                logits = model(full_ids).logits

            # Compute CE loss only on the ASSISTANT tokens (not prompt)
            shift_logits = logits[0, prompt_len - 1:-1, :]
            shift_labels = full_ids[0, prompt_len:]
            ce = F.cross_entropy(shift_logits.float(), shift_labels, reduction="sum")
            step_loss = step_loss + ce
            n_tokens  = n_tokens  + shift_labels.shape[0]

    if n_tokens > 0:
        avg_loss = step_loss / n_tokens
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sft_optimizer.step()
        sft_optimizer.zero_grad()
        print(f"  [SFT] Step {sft_step}/{SFT_STEPS} | ce_loss={avg_loss.item():.4f} | "
              f"tokens={n_tokens} | ETA {(time.time()-t0_sft)/sft_step*(SFT_STEPS-sft_step)/60:.1f}m")

# Verify the warmup worked with a quick generation test
model.eval()
_wm_msgs = [
    {"role": "system", "content": sys_p},
    {"role": "user",   "content": "DSCR=0.85x (HR-01 triggered). CIBIL=620. What is your decision?"},
]
_wm_txt = tokenizer.apply_chat_template(_wm_msgs, tokenize=False, add_generation_prompt=True)
_wm_inp = tokenizer(_wm_txt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
with torch.no_grad():
    _wm_out = model.generate(**_wm_inp, max_new_tokens=100, do_sample=True, temperature=0.8)
_wm_resp = tokenizer.decode(_wm_out[0][_wm_inp.input_ids.shape[1]:], skip_special_tokens=True)
_wm_p    = parse_llm_output(_wm_resp)
print(f"\n  Post-SFT check: parse_type={_wm_p['parse_type']} | action={_wm_p['action']}")
print(f"  Response: {_wm_resp[:120]}")
if _wm_p["parse_type"] == "final_decision":
    print("  🎉 SFT warmup successful — model now outputs submit_decision()!")
elif _wm_p["parse_type"] in ("structured_label", "sentence_decision"):
    print("  ✅ SFT warmup partial — model improved, GRPO will finish the job.")
else:
    print("  ⚠️  SFT warmup may need more steps — continuing to GRPO anyway.")
model.train()

sft_time = time.time() - t0_sft
print(f"  ✅ Stage 0 SFT done in {sft_time/60:.1f} min")
print("═" * 68)


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 13: GRPO TRAINING STAGES 1-3 ═════════════════════════════
# ════════════════════════════════════════════════════════════════════

STAGE_CONFIGS = {
    # Stage 1: FORMAT — GRPO on top of SFT-warmed model
    1: {"name": "Stage 1 — Format",     "tasks": ["task1"],
        "lr": 5e-5,  "steps": 15, "temp": 1.5},
    # Stage 2: HARD RULES — HR-01/HR-04 compliance
    2: {"name": "Stage 2 — Hard Rules", "tasks": ["task1", "task2"],
        "lr": 3e-5,  "steps": 15, "temp": 1.2},
    # Stage 3: PORTFOLIO — all tasks, low LR, low temp
    3: {"name": "Stage 3 — Portfolio",  "tasks": None,
        "lr": 1e-5,  "steps": 20, "temp": 1.0},
}

all_logs    = {1: [], 2: [], 3: []}
stage_times = {}

print("=" * 68)
print(f"  STARTING GRPO STAGES — {MODEL_NAME.split('/')[-1]} (A100)")
print(f"  KL_BETA={KL_BETA} | NUM_GEN={NUM_GENERATIONS} | Total steps={sum(c['steps'] for c in STAGE_CONFIGS.values())}")
print("=" * 68)

for stage_num in [1, 2, 3]:
    cfg = STAGE_CONFIGS[stage_num]
    stage_data = rows if cfg["tasks"] is None \
        else [r for r in rows if r["task_id"] in cfg["tasks"]]
    random.shuffle(stage_data)

    print(f"\n{'─'*68}")
    print(f"  {cfg['name']}")
    print(f"  Samples: {len(stage_data)} | LR: {cfg['lr']} | "
          f"Steps: {cfg['steps']} | Temp: {cfg['temp']}")
    print(f"{'─'*68}")

    optimizer = _get_optimizer(model, cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["steps"], eta_min=cfg["lr"] * 0.1
    )
    ref_cache: Dict = {}
    model.train()
    t_start  = time.time()
    data_idx = 0

    for step in range(1, cfg["steps"] + 1):
        batch = [stage_data[data_idx % len(stage_data)]
                 for _ in range(BATCH_SIZE)]
        data_idx += BATCH_SIZE

        loss, metrics = grpo_loss_step(
            model, tokenizer, batch, cfg["temp"], MEMORY_BANK, ref_cache
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

        avg_reward = float(np.mean(metrics.get("reward", [0])))
        std_reward = float(np.std(metrics.get("reward", [0])))
        avg_kl     = float(np.mean(metrics.get("kl", [0])))
        avg_tools  = float(np.mean(metrics.get("tool_calls", [0])))
        pt_counts  = defaultdict(int)
        for pt in metrics.get("parse_types", []): pt_counts[pt] += 1
        total_pt   = max(sum(pt_counts.values()), 1)
        submit_pct = pt_counts.get("final_decision", 0) / total_pt * 100
        elapsed    = time.time() - t_start
        eta        = (elapsed / step) * (cfg["steps"] - step)

        all_logs[stage_num].append({
            "step": step, "stage": stage_num,
            "loss": float(loss.detach()), "reward": avg_reward,
            "reward_std": std_reward, "kl": avg_kl,
            "tool_calls": avg_tools, "submit_pct": submit_pct,
        })

        # Build parse_type breakdown for diagnostics
        pt_str = " ".join(f"{k[:4]}:{v}" for k, v in sorted(pt_counts.items()) if v > 0)
        print(f"  [{stage_num}] Step {step:2d}/{cfg['steps']} | "
              f"loss={float(loss.detach()):+.6f} | "
              f"reward={avg_reward:+.2f}±{std_reward:.2f} | "
              f"submit={submit_pct:.0f}% | "
              f"kl={avg_kl:.4f} | "
              f"types=[{pt_str}] | "
              f"ETA {eta/60:.1f}m")

    elapsed = time.time() - t_start
    stage_times[stage_num] = elapsed

    ckpt_dir = f"{OUTPUT_BASE}/stage_{stage_num}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"\n  ✅ Stage {stage_num} done in {elapsed/60:.1f} min")
    print(f"  💾 Checkpoint: {ckpt_dir}")

    print(f"\n  🔍 Inferences after Stage {stage_num}:")
    model.eval()
    for idx in range(min(3, len(stage_data))):
        s  = stage_data[idx]
        ep = run_episode(s, model, tokenizer, 0.7, MEMORY_BANK)
        print(f"  [{idx+1}] PD={s['ground_truth_pd']:.2f} | [{ep['parse_type']}] "
              f"tools={ep['tool_calls_made']} | {ep['completion'][:100]}...")
    model.train()

total_time = sum(stage_times.values())
print(f"\n{'='*68}")
print(f"  ALL 3 STAGES COMPLETE ✅")
print(f"  Total time  : {total_time/60:.1f} min")
print(f"  Total cost  : ~${total_time/3600*2.5:.2f} at $2.50/hr")
print(f"{'='*68}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 13: LEARNING CURVES ═══════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

print("\n📊 Generating charts...")

all_steps, all_losses, all_rewards, all_kl, all_submit = [], [], [], [], []
stage_boundaries = []
offset = 0

for stage in [1, 2, 3]:
    logs = all_logs[stage]
    if not logs: continue
    stage_boundaries.append(offset)
    for e in logs:
        s = e["step"] + offset
        all_steps.append(s)
        all_losses.append((s, e["loss"]))
        all_rewards.append((s, e["reward"]))
        all_kl.append((s, e["kl"]))
        all_submit.append((s, e["submit_pct"]))
    offset += max(e["step"] for e in logs) + 1

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("IntelliCredit GRPO v2 — Qwen2.5-3B (A100) — Fixed",
             fontsize=13, fontweight="bold")

def _plot(ax, data, color, title, ylabel, ylim=None):
    if not data: return
    xs, ys = zip(*data)
    ax.plot(xs, ys, color=color, alpha=0.3, linewidth=0.8)
    if len(ys) > 4:
        w  = min(5, len(ys) // 2)
        sm = np.convolve(ys, np.ones(w)/w, mode="valid")
        ax.plot(xs[:len(sm)], sm, color=color, linewidth=2.5, label="Smoothed")
    for b in stage_boundaries[1:]:
        ax.axvline(x=b, color="gray", linestyle="--", alpha=0.4, label="Stage →")
    ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel)
    ax.set_xlabel("Step"); ax.grid(True, alpha=0.3)
    if ylim: ax.set_ylim(ylim)
    ax.legend(fontsize=8)

_plot(axes[0, 0], all_losses,  "#E53935", "GRPO Loss",         "Loss")
_plot(axes[0, 1], all_rewards, "#1976D2", "Mean Reward ↑",     "Reward")
_plot(axes[1, 0], all_kl,      "#7B1FA2", "KL Divergence",     "KL",  ylim=[0, None])
_plot(axes[1, 1], all_submit,  "#00796B", "submit_pct ↑",      "%",   ylim=[0, 105])

plt.tight_layout()
chart_path = f"{OUTPUT_BASE}/charts/v2_curves.png"
plt.savefig(chart_path, dpi=130)
plt.show()
print(f"✅ Charts saved → {chart_path}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 14: PUSH TO HF HUB ════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

HF_USERNAME = "vssksn"   # ← change if needed
PUSH_REPO   = f"{HF_USERNAME}/intellicredit-mistral-7b-grpo"

print(f"\n🚀 Pushing LoRA adapter → {PUSH_REPO}")
final_dir = f"{OUTPUT_BASE}/final"
os.makedirs(final_dir, exist_ok=True)
model.eval()
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

try:
    model.push_to_hub(PUSH_REPO, private=True)
    tokenizer.push_to_hub(PUSH_REPO, private=True)
    print(f"✅ Pushed → https://huggingface.co/{PUSH_REPO}")
except Exception as e:
    print(f"⚠️  Push failed ({e})")
    print(f"   Adapter available locally at: {final_dir}/")

print("\n🎉 v2 training complete!")
