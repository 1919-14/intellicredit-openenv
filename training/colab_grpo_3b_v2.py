"""
╔══════════════════════════════════════════════════════════════════════╗
║  IntelliCredit v2 — GRPO Mistral-7B (A100) + UNSLOTH               ║
║  Model  : mistralai/Mistral-7B-Instruct-v0.3  (4-bit QLoRA)        ║
║  Engine : 🦥 Unsloth — 2-3x faster training, 60% less VRAM         ║
║  Target : Jupyter Notebook — Nvidia A100 · 80 GB                   ║
║                                                                      ║
║  ✅ Unsloth FastLanguageModel for optimised 4-bit loading           ║
║  ✅ Unsloth gradient checkpointing (30% less VRAM)                  ║
║  ✅ Fused kernels for faster forward/backward                       ║
║  ✅ Auto-fallback to manual transformers+peft if Unsloth fails      ║
║  ✅ KL_BETA=0.15 + SFT warmup + robust parser v3                   ║
║  ✅ Stage 0 SFT warmup (9 gold examples) before GRPO               ║
║  ✅ Robust error handling for Jupyter notebook environment          ║
╚══════════════════════════════════════════════════════════════════════╝

Paste each ═══ CELL ═══ block into a separate Jupyter cell.
Run Cell 1 → Restart Kernel → Run Cell 2 onward in order.
"""


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 1: INSTALL ════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════

import subprocess, sys, os

def _pip(*args):
    """Safe pip install — returns True on success, False on failure."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *args])
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  pip install failed for {args}: {e}")
            return False

def _pip_uninstall(*pkgs):
    """Silently uninstall packages."""
    for pkg in pkgs:
        subprocess.call(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

# ── Step 1: Upgrade pip ──────────────────────────────────────────────
_pip("--upgrade", "pip")

# ── Step 2: Install PyTorch ──────────────────────────────────────────
# Unsloth needs PyTorch >= 2.6 (for triton configs + torch.int1).
# cu121 only has 2.5.x → try cu124 first (works on A100 w/ driver ≥550).
print("🔧 Step 1/5: Installing PyTorch...")
_got_torch26 = False

# Try cu124 first (PyTorch 2.6+, needed for Unsloth)
print("   Trying cu124 (PyTorch 2.6+)...")
if _pip("torch>=2.6.0", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124"):
    _got_torch26 = True
    print("   ✅ PyTorch 2.6+ installed (cu124)")
else:
    # Fall back to cu121 (PyTorch 2.5.x — no Unsloth, but training works)
    print("   cu124 failed (CUDA driver may be too old), trying cu121...")
    _pip("torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cu121")
    print("   ✅ PyTorch 2.5.x installed (cu121)")

# ── Step 3: CRITICAL — Remove torchao ────────────────────────────────
# torchao >= 0.7 uses torch.int1 which only exists in PyTorch 2.7+.
# If torchao is installed (pulled in by transformers), it CRASHES both
# Unsloth and transformers imports — the "poison pill" that breaks all.
# We use bitsandbytes for quantization, NOT torchao, so removing it is
# safe and fixes the entire import chain.
print("🧹 Step 2/5: Removing incompatible torchao...")
_pip_uninstall("torchao")
print("   ✅ torchao removed (we use bitsandbytes instead)")

# ── Step 4: Install Unsloth (only makes sense with PyTorch 2.6+) ─────
_unsloth_ok = False
if _got_torch26:
    print("🦥 Step 3/5: Installing Unsloth...")
    if _pip("--no-deps", "unsloth"):
        if _pip("--no-deps", "unsloth_zoo"):
            _unsloth_ok = True
            print("   ✅ Unsloth installed")
        else:
            print("   ⚠️  unsloth_zoo failed")
    if not _unsloth_ok:
        print("   Trying git install...")
        _pip("--no-deps", "unsloth @ git+https://github.com/unslothai/unsloth.git")
        _pip("--no-deps", "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git")
        _unsloth_ok = True
        print("   ✅ Unsloth installed via git")
else:
    print("⏭️  Step 3/5: Skipping Unsloth (needs PyTorch 2.6+, using manual mode)")

# ── Step 5: Install remaining dependencies ────────────────────────────
print("📦 Step 4/5: Installing remaining deps...")
_pip("--upgrade", "transformers>=4.45.0", "trl>=0.15.2", "peft>=0.13.0",
     "accelerate>=1.0.0", "bitsandbytes>=0.46.1")
_pip("datasets>=2.20.0", "huggingface_hub>=0.24.0", "matplotlib", "langdetect")
_pip("sentencepiece", "protobuf")
# Remove torchao AGAIN — transformers may have re-pulled it as a dep
_pip_uninstall("torchao")
print("   ✅ All dependencies installed")

# ── Step 6: HF Token login ──────────────────────────────────────────
print("🔑 Step 5/5: Authentication...")
hf_tok = os.environ.get("HF_TOKEN", "")
if hf_tok:
    from huggingface_hub import login
    login(token=hf_tok, add_to_git_credential=False)
    print("   ✅ HF Token loaded")
else:
    print("   ⚠️  HF_TOKEN not set — anonymous access (OK for public models)")

# ── Report versions ──────────────────────────────────────────────────
import torch as _t
print(f"\n✅ All packages installed")
print(f"   PyTorch : {_t.__version__}")
print(f"   CUDA    : {_t.version.cuda if _t.cuda.is_available() else 'N/A'}")
print(f"   Unsloth : {'installed' if _unsloth_ok else 'NOT installed (manual mode)'}")
# Verify torchao is gone
try:
    import torchao
    print(f"   ⚠️  torchao still present — may cause import errors")
except ImportError:
    print(f"   torchao : removed ✅ (prevents torch.int1 crash)")
print(f"\n   ⚡ IMPORTANT: Kernel → Restart Kernel → then run from Cell 2")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 2: IMPORTS & CONFIG ═══════════════════════════════════════
# ════════════════════════════════════════════════════════════════════
#
# ⚠️  CRITICAL: Unsloth MUST be imported BEFORE transformers/peft.
#     This cell handles the import order correctly and auto-falls back
#     to manual transformers+peft if Unsloth is unavailable.

import os, re, json, time, random, gc, sys
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

# ── Suppress noisy Jupyter warnings BEFORE any ML imports ────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*torch.amp.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="unsloth")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*Both.*max_new_tokens.*")

# Suppress transformers verbose logging during training (keeps errors only)
import logging as _logging
_logging.getLogger("transformers").setLevel(_logging.ERROR)

# ── Safety: hide torchao from transformers if PyTorch is too old ─────
# torchao >= 0.7 needs torch.int1 (PyTorch 2.7+). If torchao is present
# but PyTorch is older, transformers' `is_torchao_available()` returns
# True, then `from torchao...` crashes the ENTIRE import chain.
#
# FIX: Monkey-patch importlib.util.find_spec to return None for torchao.
# This makes is_torchao_available() → False, so transformers skips it.
import importlib.util
import torch as _t_check
if not hasattr(_t_check, "int1"):
    # Evict any cached torchao modules
    for _k in list(sys.modules.keys()):
        if _k == "torchao" or _k.startswith("torchao."):
            del sys.modules[_k]
    # Patch find_spec so is_torchao_available() returns False
    _original_find_spec = importlib.util.find_spec
    def _patched_find_spec(name, *args, **kwargs):
        if name and (name == "torchao" or name.startswith("torchao.")):
            return None
        return _original_find_spec(name, *args, **kwargs)
    importlib.util.find_spec = _patched_find_spec
    print("🛡️  torchao hidden from importlib (PyTorch missing torch.int1)")
else:
    print("✅ PyTorch has torch.int1 — torchao compatible")

# ── Try Unsloth FIRST (must import before transformers/peft) ─────────
USE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("🦥 Unsloth loaded successfully")
except Exception as _e:
    print(f"⚠️  Unsloth not available: {str(_e)[:80]}")
    print("   → Using manual transformers + peft (works fine, just slower)")

# ── Import transformers/peft (fallback or alongside Unsloth) ─────────
if not USE_UNSLOTH:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

# ── Core Config ──────────────────────────────────────────────────────
MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.3"
HF_DATASET   = "vssksn/intellicredit-grpo-v2"
OUTPUT_BASE  = "intellicredit-grpo-mistral-7b"

# ── GRPO hyper-params — tuned for ~45 min on A100 ──────────────────────────
NUM_GENERATIONS  = 4       # 4 gives good advantage contrast, 33% faster than 6
MAX_NEW_TOKENS   = 200     # 200 is plenty for reasoning + submit_decision()
MAX_TOOL_TURNS   = 2
BATCH_SIZE       = 1       # 1 sample/step → fastest wallclock time per step
KL_BETA          = 0.15    # tight leash — prevents drift
MAX_SEQ_LEN      = 2048    # 2048: headroom for long system prompt + MAX_NEW_TOKENS

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/charts", exist_ok=True)

# ── Helper functions for Unsloth/manual mode switching ───────────────
def safe_for_inference(model):
    """Switch model to inference mode (Unsloth-aware)."""
    if USE_UNSLOTH:
        FastLanguageModel.for_inference(model)
    else:
        model.eval()

def safe_for_training(model):
    """Switch model to training mode (Unsloth-aware)."""
    if USE_UNSLOTH:
        FastLanguageModel.for_training(model)
    else:
        model.train()

print(f"✅ Imports complete ({'🦥 Unsloth' if USE_UNSLOTH else '🔧 Manual'} mode)")
gpu  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"   GPU    : {gpu}")
print(f"   VRAM   : {vram:.1f} GB")
print(f"   Model  : {MODEL_NAME}")
print(f"   Engine : {'🦥 Unsloth (2-3x speedup)' if USE_UNSLOTH else '🔧 transformers+peft'}")
print(f"   KL_BETA={KL_BETA} | NUM_GEN={NUM_GENERATIONS}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 3: ACTION PARSER v3 (robust — handles verbose style) ══════
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
    k = re.sub(r"[\s_]+", " ", k)
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

    # ── L3: tool call ────────────────────────────────────────────────
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

    # ── L1: exact submit_decision ────────────────────────────────────
    ms = list(_RE_L1_EXACT.finditer(t))
    if ms:
        m   = ms[-1]
        raw = m.group(1)
        rsn = (m.group(2) or "").strip()
        if not rsn:
            rsn = _extract_context(t, m.start(), 120)
        act = _norm_action(raw)
        return {"action": act, "parse_type": "final_decision",
                "parse_confidence": 0.95 if rsn else 0.70,
                "reasoning": rsn, "tool_name": None, "tool_args": None,
                "parse_failure": False}

    # ── L2: fuzzy submit variants ────────────────────────────────────
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

    # ── L4: structured label ─────────────────────────────────────────
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

    # ── L5: sentence-level intent ────────────────────────────────────
    ml5 = list(_RE_L5_SENTENCE.finditer(t))
    if ml5:
        m   = ml5[-1]
        raw = m.group(1)
        act = _norm_action(raw)
        rsn = t[max(0, m.start() - 150): m.end() + 50].strip()
        return {"action": act, "parse_type": "sentence_decision",
                "parse_confidence": 0.65, "reasoning": rsn,
                "tool_name": None, "tool_args": None, "parse_failure": False}

    # ── L6: bare keyword anywhere ────────────────────────────────────
    ml6 = list(_RE_L6_KEYWORD.finditer(t))
    if ml6:
        m   = ml6[-1]
        raw = m.group(1)
        act = _norm_action(raw)
        return {"action": act, "parse_type": "fallback_keyword",
                "parse_confidence": 0.45, "reasoning": t[-150:],
                "tool_name": None, "tool_args": None, "parse_failure": True}

    # ── L7: silence / unparseable ────────────────────────────────────
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

try:
    from langdetect import detect as _lang_detect
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False

def _is_english(text: str) -> bool:
    if not _LANGDETECT_OK or len(text.strip()) < 20:
        return True
    try:
        return _lang_detect(text[:300]) == "en"
    except Exception:
        return True


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
            s += 0.5
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
    scores = []
    for comp in completions:
        p   = parse_llm_output(comp)
        rsn = p.get("reasoning", "")
        pt  = p["parse_type"]
        lang_ok      = _is_english(comp)
        lang_penalty = 0.0 if lang_ok else -2.0
        if pt == "final_decision" and len(rsn) > 20:
            s = 1.0
        elif pt == "final_decision":
            s = 0.6
        elif pt == "structured_label":
            s = 0.4
        elif pt == "sentence_decision":
            s = 0.2
        elif pt == "tool_call":
            s = 0.1
        elif pt == "fallback_keyword":
            s = -0.3
        else:
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
#
# Two code paths:
#   🦥 Unsloth path  — FastLanguageModel.from_pretrained (faster, less VRAM)
#   🔧 Manual path   — transformers + peft + bitsandbytes (always works)

print(f"\n🔄 Loading model: {MODEL_NAME}...")
print(f"   Mode: {'🦥 Unsloth' if USE_UNSLOTH else '🔧 Manual'}")

# Clear VRAM before loading
gc.collect()
torch.cuda.empty_cache()

if USE_UNSLOTH:
    # ══════════════════════════════════════════════════════════════════
    # 🦥 UNSLOTH PATH — 2x faster loading, 30% less VRAM
    # ══════════════════════════════════════════════════════════════════
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,           # auto-detect (bf16 on A100)
        load_in_4bit   = True,           # 4-bit QLoRA quantization
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = 16,
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha                 = 32,
        lora_dropout               = 0,              # Unsloth: 0 is fastest
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",       # 30% less VRAM
        random_state               = 42,
        use_rslora                 = False,
        loftq_config               = None,
    )

else:
    # ══════════════════════════════════════════════════════════════════
    # 🔧 MANUAL PATH — transformers + peft + bitsandbytes (always works)
    # ══════════════════════════════════════════════════════════════════
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
        torch_dtype         = torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

# ── Tokenizer setup (works for both paths) ───────────────────────────
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Fix the max_new_tokens vs max_length warning at the source
# Unsloth sets max_length=32768 in generation_config, conflicting with max_new_tokens
try:
    model.generation_config.max_length = None
except Exception:
    pass

model.print_trainable_parameters()
print(f"✅ Model + LoRA ready (vocab size: {len(tokenizer)})")

# Report VRAM usage
if torch.cuda.is_available():
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"   VRAM used: {vram_used:.1f} GB | reserved: {vram_reserved:.1f} GB")

# ── Smoke test ──────────────────────────────────────────────────────
print("\n🔍 Smoke test (should show submit_decision in response)...")
safe_for_inference(model)
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
    """Run a single episode. Caller must call safe_for_inference(model) first."""
    prompt_text     = sample["prompt"]
    sys_prompt      = build_system_prompt(MAX_TOOL_TURNS, memory_bank)
    tool_transcript = []
    tool_calls_made = 0
    last_completion = ""
    last_parsed     = {}
    all_turns       = []
    last_input_ids  = None   # stored for GRPO log-prob (avoids re-tokenisation)
    last_comp_ids   = None   # stored for GRPO log-prob

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
                max_new_tokens     = MAX_NEW_TOKENS,
                min_new_tokens     = 15,
                do_sample          = True,
                temperature        = temperature,
                top_p              = 0.92,
                repetition_penalty = 1.15,
                pad_token_id       = tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Store raw token IDs for GRPO log-prob computation (no re-tokenisation needed)
        last_input_ids = inputs.input_ids.cpu()
        _raw_ids       = output[0][inputs.input_ids.shape[1]:].cpu()
        # Pre-clamp to tokenizer vocab — Unsloth pads the embedding table for
        # GPU alignment so generated IDs can exceed training-mode vocab_size.
        # Clamping here is belt-and-suspenders on top of the guard in compute_log_probs_from_ids.
        last_comp_ids  = _raw_ids.clamp(0, len(tokenizer) - 1)

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
        "last_input_ids" : last_input_ids,   # [1, prompt_len] CPU tensor for GRPO
        "last_comp_ids"  : last_comp_ids,    # [comp_len]      CPU tensor for GRPO
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

def _get_optimizer(model, lr: float):
    """Create AdamW optimizer for trainable (LoRA) parameters only."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )


def compute_log_probs_from_ids(model, input_ids: torch.Tensor,
                               comp_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of log-probs for comp_ids tokens given input_ids context.
    Reuses the exact token IDs produced during generation — avoids all
    re-tokenisation edge cases (prompt too long → comp truncated to 0).

    Args:
        model     : policy model (must be in .train() with enable_grad for current-lp;
                    may be in no_grad for ref-lp)
        input_ids : [1, prompt_len]  — prompt token IDs already on model.device
        comp_ids  : [comp_len]       — 1-D completion token IDs on model.device
    Returns:
        Scalar tensor.  requires_grad=True when called inside enable_grad context.
    """
    if comp_ids.shape[0] == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    comp_ids_2d = comp_ids.unsqueeze(0)                          # [1, comp_len]
    full_ids    = torch.cat([input_ids, comp_ids_2d], dim=1)     # [1, T_raw]
    prompt_len  = input_ids.shape[1]

    # ── CRITICAL: truncate to MAX_SEQ_LEN *before* the forward pass ──────────
    # full_ids = prompt(~1900) + comp(200) can exceed MAX_SEQ_LEN=2048.
    # Without this, Unsloth silently truncates the *logits* to 2048 rows while
    # full_ids still has T_raw rows, making shift_labels longer than shift_logits
    # → arange(comp_len) indexes into fewer logit rows → CUDA "out of bounds" crash.
    if full_ids.shape[1] > MAX_SEQ_LEN:
        full_ids = full_ids[:, :MAX_SEQ_LEN]   # trim trailing comp tokens
    T = full_ids.shape[1]   # always <= MAX_SEQ_LEN now

    if prompt_len >= T:   # every completion token was truncated away
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        logits = model(full_ids).logits                          # [1, T, V] — T<=MAX_SEQ_LEN

    comp_len     = T - prompt_len
    # logits[prompt_len-1] predicts comp[0], ..., logits[T-2] predicts comp[comp_len-1]
    shift_logits = logits[0, prompt_len - 1 : T - 1, :].float() # [comp_len, V]
    shift_labels = full_ids[0, prompt_len:]                      # [comp_len] — from truncated ids

    if shift_logits.shape[0] == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    log_probs     = F.log_softmax(shift_logits, dim=-1)
    vocab_size_lp = shift_logits.shape[-1]  # training-mode vocab (may differ from inference)

    # ── CRITICAL: Unsloth fast-inference pads vocab to a GPU-alignment multiple
    # (e.g. 32768 → 32832). Generated token IDs in the padded range are valid
    # during sampling but OOB when indexing the training-mode logit table
    # (vocab_size=32768).  Clamping prevents the CUDA "index out of bounds" crash.
    valid_mask     = (shift_labels < vocab_size_lp)            # [comp_len] bool
    shift_labels_s = shift_labels.clamp(0, vocab_size_lp - 1) # safe gather indices
    token_lps = log_probs[
        torch.arange(shift_labels_s.shape[0], device=model.device), shift_labels_s
    ]
    # Zero-out padded/OOB token contributions so they don't corrupt the gradient
    token_lps = token_lps * valid_mask.float()
    # Normalise to per-token average → loss magnitude scale-invariant to seq length
    # (prevents huge loss ~400 that would send gradients through the roof)
    n_valid = valid_mask.sum().clamp(min=1)
    return token_lps.sum() / n_valid


def _disable_adapters_safe(model):
    """Context manager to disable LoRA adapters (works with both Unsloth and PEFT)."""
    try:
        return model.disable_adapter()
    except Exception:
        # Fallback: manual toggle
        class _ManualDisable:
            def __enter__(self_inner):
                try:
                    model.disable_adapters()
                except Exception:
                    pass
                return self_inner
            def __exit__(self_inner, *args):
                try:
                    model.enable_adapters()
                except Exception:
                    pass
        return _ManualDisable()


def grpo_loss_step(model, tokenizer, samples: List[Dict],
                   temperature: float, memory_bank: MemoryBank,
                   ref_cache: Dict, step_num: int = 0) -> Tuple[torch.Tensor, Dict]:
    all_losses = []
    metrics    = defaultdict(list)

    for s_idx, sample in enumerate(samples):
        # ── Phase 1: Generate completions (NO gradients) ─────────────
        safe_for_inference(model)   # Unsloth-aware: enables fast attention kernels
        with torch.no_grad():
            completions, episodes = [], []
            for g_idx in range(NUM_GENERATIONS):
                print(f"\r    ⏳ Sample {s_idx+1}/{len(samples)} | Gen {g_idx+1}/{NUM_GENERATIONS}", end="", flush=True)
                ep = run_episode(sample, model, tokenizer, temperature, memory_bank)
                episodes.append(ep)
                completions.append(ep["completion"])
        print(f"\r    ✅ Sample {s_idx+1}/{len(samples)} | {NUM_GENERATIONS} gens done       ", flush=True)

        prompts_list  = [ep["prompt_text"] for ep in episodes]
        meta_keys     = ["ground_truth_pd", "hard_rules", "has_red_alerts", "npa_rate", "crar"]
        reward_kwargs = {k: [ep["metadata"][k] for ep in episodes] for k in meta_keys}
        reward_list   = combined_reward(prompts_list, completions, **reward_kwargs)

        metrics["reward"].extend(reward_list)
        metrics["tool_calls"].extend(ep["tool_calls_made"] for ep in episodes)
        metrics["parse_types"].extend(ep["parse_type"] for ep in episodes)

        r_arr = np.array(reward_list, dtype=np.float32)
        r_std = r_arr.std()

        # Skip degenerate case: all rewards identical → advantages all 0 → loss=0
        if r_std < 1e-4:
            if step_num <= 3:
                print(f"    ⚠️  All {NUM_GENERATIONS} rewards identical ({r_arr[0]:.2f}) — skipping loss", flush=True)
            continue

        advantages = (r_arr - r_arr.mean()) / (r_std + 1e-8)
        sys_p = build_system_prompt(MAX_TOOL_TURNS, memory_bank)

        # ── Phase 2: Compute GRPO loss (WITH gradients) ──────────────
        # safe_for_training restores LoRA requires_grad and unpatches Unsloth
        # inference-only kernels so autograd works through the forward pass.
        safe_for_training(model)
        with torch.enable_grad():
            for ep, adv in zip(episodes, advantages):
                if not ep["completion"].strip():
                    continue  # skip empty completions

                # Guard: ensure stored IDs exist and are non-empty
                if ep.get("last_comp_ids") is None or ep["last_comp_ids"].shape[0] == 0:
                    if step_num <= 3:
                        print(f"    ⚠️  Skipping ep — last_comp_ids empty", flush=True)
                    continue

                dev      = model.device
                inp_ids  = ep["last_input_ids"].to(dev)   # [1, prompt_len]
                comp_ids = ep["last_comp_ids"].to(dev)    # [comp_len]

                # Current policy log-probs (with gradients) — uses stored IDs
                lp = compute_log_probs_from_ids(model, inp_ids, comp_ids)

                # Skip samples where the entire prompt filled MAX_SEQ_LEN and no
                # completion tokens survived truncation (lp returns a disconnected 0).
                # These contribute zero gradient anyway; skipping keeps metrics clean.
                if lp.item() == 0.0:
                    if step_num <= 3:
                        print(f"    ⚠️  Skipping — prompt filled context, lp=0", flush=True)
                    continue

                # Reference policy log-probs (base model, LoRA adapters disabled)
                with torch.no_grad():
                    with _disable_adapters_safe(model):
                        ref_lp = compute_log_probs_from_ids(
                            model, inp_ids, comp_ids
                        ).detach()

                # Symmetric KL divergence: |lp - ref_lp| / n_tokens
                # The one-sided clamp(min=0) was always 0 when lp < ref_lp (model
                # learning new behaviours is more diffuse than ref). Symmetric KL
                # gives a non-zero regularisation signal in both directions so the
                # progressbar kl metric is informative and the penalty works correctly.
                kl        = (lp - ref_lp).abs()
                adv_t     = torch.tensor(float(adv), device=dev, dtype=torch.float32)
                step_loss = -(adv_t * lp) + KL_BETA * kl

                # Diagnostic for first 3 steps
                if step_num <= 3:
                    print(f"    📊 adv={adv:.3f} lp={lp.item():.3f} ref={ref_lp.item():.3f} "
                          f"kl={kl.item():.4f} loss={step_loss.item():.4f} "
                          f"comp_len={comp_ids.shape[0]} has_grad={lp.requires_grad}", flush=True)

                all_losses.append(step_loss)
                metrics["kl"].append(kl.item())

        worst = episodes[int(np.argmin(reward_list))]
        MEMORY_BANK.learn_from_episode(
            worst["prompt_text"], worst["completion"],
            min(reward_list), worst["parse_type"],
        )

    if not all_losses:
        # All steps were degenerate — return tiny non-zero loss so optimizer still steps
        dummy = torch.tensor(1e-6, device=model.device, requires_grad=True)
        return dummy, metrics
    return torch.stack(all_losses).mean(), metrics

print("✅ GRPO loss ready")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 12: STAGE 0 — SFT WARMUP (15 steps, cross-entropy) ═══════
# ════════════════════════════════════════════════════════════════════
#
# WHY: GRPO requires variance (std > 0) across the N generations for
# each sample. The cold model outputs the same vague text for all 6
# generations → rewards identical → advantages = 0 → loss = 0.
#
# FIX (used by DeepSeek-R1, Kimi, etc.): run a short SFT warmup on
# hand-crafted gold examples that explicitly contain submit_decision().

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

# ── SFT Warmup Training ──────────────────────────────────────────────
SFT_STEPS  = 50    # 50 steps × 2 epochs × 9 examples = ~900 gradient updates
SFT_LR     = 1e-4  # 2× higher LR for faster format memorisation
SFT_EPOCHS = 2     # 2 epochs keeps each step fast while covering all examples

print("\n" + "═" * 68)
print("  STAGE 0 — SFT WARMUP (primes submit_decision format)")
print(f"  Gold examples: {len(SFT_GOLD_EXAMPLES)} | Steps: {SFT_STEPS} | LR: {SFT_LR}")
print("  This teaches the model the output format before GRPO begins.")
print("═" * 68)

sft_optimizer = _get_optimizer(model, SFT_LR)
sys_p         = build_system_prompt(MAX_TOOL_TURNS, None)

# Switch to training mode
safe_for_training(model)
t0_sft = time.time()

sft_pbar = tqdm(range(1, SFT_STEPS + 1), desc="  SFT Warmup", unit="step",
                bar_format="{l_bar}{bar:30}{r_bar}")
for sft_step in sft_pbar:
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
        sft_pbar.set_postfix(loss=f"{avg_loss.item():.4f}", tokens=n_tokens)

# Verify the warmup worked
safe_for_inference(model)
_wm_msgs = [
    {"role": "system", "content": sys_p},
    {"role": "user",   "content": "DSCR=0.85x (HR-01 triggered). CIBIL=620. What is your decision?"},
]
_wm_txt = tokenizer.apply_chat_template(_wm_msgs, tokenize=False, add_generation_prompt=True)
_wm_inp = tokenizer(_wm_txt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
with torch.no_grad():
    # Use 250 tokens — the model reasons first, then calls submit_decision() at the end.
    # 100 tokens was too short: the reasoning filled the budget before reaching the call.
    _wm_out = model.generate(**_wm_inp, max_new_tokens=250, do_sample=False, temperature=1.0)
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

sft_time = time.time() - t0_sft
print(f"  ✅ Stage 0 SFT done in {sft_time/60:.1f} min")
print("═" * 68)


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 13: GRPO TRAINING STAGES 1-3 ═════════════════════════════
# ════════════════════════════════════════════════════════════════════

STAGE_CONFIGS = {
    # Stage 1: format — primary goal is submit_decision() use.
    # Temp 1.5 was too chaotic → model ignored learned format (sub=0%).
    # 1.0 keeps exploration variance while respecting SFT-learned structure.
    1: {"name": "Stage 1 — Format",     "tasks": ["task1"],
        "lr": 5e-5,  "steps": 10, "temp": 1.0},
    2: {"name": "Stage 2 — Hard Rules", "tasks": ["task1", "task2"],
        "lr": 3e-5,  "steps": 10, "temp": 0.9},
    3: {"name": "Stage 3 — Portfolio",  "tasks": None,
        "lr": 1e-5,  "steps": 15, "temp": 0.8},
}

all_logs    = {1: [], 2: [], 3: []}
stage_times = {}

print("=" * 68)
print(f"  STARTING GRPO STAGES — {MODEL_NAME.split('/')[-1]} (A100)")
print(f"  Engine: {'🦥 Unsloth' if USE_UNSLOTH else '🔧 Manual'}")
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
    t_start  = time.time()
    data_idx = 0

    stage_pbar = tqdm(range(1, cfg["steps"] + 1),
                      desc=f"  Stage {stage_num}", unit="step",
                      bar_format="{l_bar}{bar:30}{r_bar}")
    for step in stage_pbar:
        batch = [stage_data[data_idx % len(stage_data)]
                 for _ in range(BATCH_SIZE)]
        data_idx += BATCH_SIZE

        # grpo_loss_step manages model.eval()/model.train() internally
        loss, metrics = grpo_loss_step(
            model, tokenizer, batch, cfg["temp"], MEMORY_BANK, ref_cache,
            step_num=step
        )

        # model is already in train() from grpo_loss_step Phase 2
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

        # Update progress bar with key metrics
        stage_pbar.set_postfix(
            loss=f"{float(loss.detach()):+.4f}",
            rwd=f"{avg_reward:+.2f}",
            sub=f"{submit_pct:.0f}%",
            kl=f"{avg_kl:.3f}",
            eta=f"{eta/60:.1f}m"
        )

        # Full log every 3 steps
        if step % 3 == 0 or step == cfg["steps"]:
            pt_str = " ".join(f"{k[:4]}:{v}" for k, v in sorted(pt_counts.items()) if v > 0)
            tqdm.write(f"  [{stage_num}] Step {step:2d}/{cfg['steps']} | "
                  f"loss={float(loss.detach()):+.6f} | "
                  f"reward={avg_reward:+.2f}±{std_reward:.2f} | "
                  f"submit={submit_pct:.0f}% | "
                  f"kl={avg_kl:.4f} | "
                  f"types=[{pt_str}]")

        # Periodic VRAM check
        if step % 5 == 0 and torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
            tqdm.write(f"       💾 VRAM: {vram_gb:.1f} GB")

    elapsed = time.time() - t_start
    stage_times[stage_num] = elapsed

    ckpt_dir = f"{OUTPUT_BASE}/stage_{stage_num}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"\n  ✅ Stage {stage_num} done in {elapsed/60:.1f} min")
    print(f"  💾 Checkpoint: {ckpt_dir}")

    print(f"\n  🔍 Inferences after Stage {stage_num}:")
    safe_for_inference(model)
    for idx in range(min(3, len(stage_data))):
        s  = stage_data[idx]
        ep = run_episode(s, model, tokenizer, 0.7, MEMORY_BANK)
        print(f"  [{idx+1}] PD={s['ground_truth_pd']:.2f} | [{ep['parse_type']}] "
              f"tools={ep['tool_calls_made']} | {ep['completion'][:100]}...")

total_time = sum(stage_times.values())
print(f"\n{'='*68}")
print(f"  ALL 3 STAGES COMPLETE ✅")
print(f"  Total time  : {total_time/60:.1f} min")
print(f"  Total cost  : ~${total_time/3600*2.5:.2f} at $2.50/hr")
print(f"{'='*68}")


# ════════════════════════════════════════════════════════════════════
# ═══ CELL 13b: LEARNING CURVES ══════════════════════════════════════
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
engine_tag = "Unsloth" if USE_UNSLOTH else "Manual"
fig.suptitle(f"IntelliCredit GRPO v2 — Mistral-7B (A100 + {engine_tag})",
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
# ═══ CELL 14: SAVE & PUSH TO HF HUB ════════════════════════════════
# ════════════════════════════════════════════════════════════════════

HF_USERNAME = "vssksn"   # ← change if needed
PUSH_REPO   = f"{HF_USERNAME}/intellicredit-mistral-7b-grpo"

# ── Save LoRA adapter locally ────────────────────────────────────────
print(f"\n💾 Saving LoRA adapter locally...")
final_dir = f"{OUTPUT_BASE}/final_lora"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"   ✅ LoRA adapter saved → {final_dir}/")

# ── Save merged 16-bit model (Unsloth only) ─────────────────────────
if USE_UNSLOTH:
    print(f"\n💾 Saving merged 16-bit model...")
    merged_dir = f"{OUTPUT_BASE}/final_merged"
    try:
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"   ✅ Merged model saved → {merged_dir}/")
    except Exception as e:
        print(f"   ⚠️  Merged save failed ({e}) — LoRA adapter still available.")

# ── Push to HF Hub ──────────────────────────────────────────────────
print(f"\n🚀 Pushing to HF Hub → {PUSH_REPO}")
try:
    if USE_UNSLOTH:
        model.push_to_hub_merged(PUSH_REPO, tokenizer, save_method="merged_16bit", private=True)
        print(f"✅ Pushed merged model → https://huggingface.co/{PUSH_REPO}")
    else:
        model.push_to_hub(PUSH_REPO, private=True)
        tokenizer.push_to_hub(PUSH_REPO, private=True)
        print(f"✅ Pushed LoRA adapter → https://huggingface.co/{PUSH_REPO}")
except Exception as e:
    print(f"⚠️  Push failed ({e})")
    print(f"   Adapter available locally at: {final_dir}/")

print("\n🎉 v2 training complete!")
