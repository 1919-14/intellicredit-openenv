"""
╔══════════════════════════════════════════════════════════════════════╗
║  IntelliCredit — ONLINE GRPO Training (Full Environment Showcase)  ║
║  Model: mistralai/Mistral-7B-Instruct-v0.3 (4-bit QLoRA)          ║
║  Environment: https://vssksn-intellicredit-openenv.hf.space        ║
║  Features: Tool Calling, Multi-Agent, Reflection, 50-step Episodes ║
║  Stage 2 of 2-stage GRPO pipeline — trains on LIVE environment     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ═══ CELL 1: INSTALL (uncomment in Colab) ═══
# !pip install "transformers>=4.45.0" peft accelerate bitsandbytes
# !pip install "trl>=0.15.2" datasets matplotlib huggingface_hub requests

# ═══ CELL 2: IMPORTS & CONFIG ═══
import os, re, json, time, uuid, random, math
import torch
import requests
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.3"
ENV_BASE_URL = "https://vssksn-intellicredit-openenv.hf.space"
OUTPUT_BASE  = "intellicredit-online-grpo"
FINAL_MODEL  = "intellicredit-mistral-7b-grpo"

MAX_STEPS_PER_EP     = 50
MAX_TOOL_CALLS       = 4
MAX_NEW_TOKENS       = 200
NUM_UPDATES          = 50
EPISODES_PER_UPD     = 4
LR                   = 5e-5
BETA                 = 0.04
GRAD_ACCUM           = 4

CURRICULUM = {
    1: {"updates": (1,15),  "tasks": ["task1"],                          "temp": (1.2,1.0)},
    2: {"updates": (16,35), "tasks": ["task1","task2","task3"],           "temp": (1.0,0.9)},
    3: {"updates": (36,50), "tasks": ["task1","task2","task3","task4","task5"], "temp": (0.9,0.8)},
}

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/charts", exist_ok=True)

print("✅ Config ready")
print(f"   Model: {MODEL_NAME} | Env: {ENV_BASE_URL}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")


# ═══ CELL 3: ENV CLIENT ═══
class IntelliCreditClient:
    HEADERS = {"Content-Type": "application/json"}
    TIMEOUT = 30

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base}/health", timeout=5).json().get("status") == "healthy"
        except: return False

    def reset(self, task_id="task1", seed=None) -> Dict:
        payload = {"episode_id": str(uuid.uuid4()), "task_id": task_id,
                    "seed": seed or int(time.time() % 100000)}
        r = requests.post(f"{self.base}/reset", json=payload, headers=self.HEADERS, timeout=self.TIMEOUT)
        r.raise_for_status()
        d = r.json()
        return {"episode_id": payload["episode_id"],
                "observation": d.get("observation", []), "prompt": d.get("prompt", d.get("text", "")),
                "info": d.get("info", {})}

    def step(self, episode_id, decision, reasoning="") -> Dict:
        payload = {"episode_id": episode_id, "action": {"decision": decision, "reasoning": reasoning}, "timeout_s": self.TIMEOUT}
        r = requests.post(f"{self.base}/step", json=payload, headers=self.HEADERS, timeout=self.TIMEOUT)
        r.raise_for_status()
        d = r.json()
        obs = d.get("observation", d)
        return {"reward": float(d.get("reward", 0.0)), "observation": obs,
                "prompt": d.get("prompt", d.get("text", "")),
                "done": bool(d.get("done", False) or obs.get("done", False)),
                "info": d.get("info", {}), "obs_data": obs}

env_client = IntelliCreditClient(ENV_BASE_URL)
print(f"🌐 Env healthy: {env_client.health()}")


# ═══ CELL 4: FULL ACTION PARSER (4 parse paths) ═══
ACTION_MAP = {"APPROVE":0,"APPROVED":0,"CONDITIONAL":1,"CONDITIONAL_APPROVE":1,
              "CONDITIONAL APPROVE":1,"REJECT":2,"REJECTED":2,"DECLINE":2,"DENY":2}
TOOL_NAMES = {"get_financial_report","check_compliance_status","get_market_intelligence"}

_RE_TOOL = re.compile(r"\b(get_financial_report|check_compliance_status|get_market_intelligence)\s*\(([^)]*)\)", re.I)
_RE_SUBMIT = re.compile(r"submit_decision\s*\(\s*['\"]?([A-Z_]+)['\"]?\s*,\s*['\"]?(.*?)['\"]?\s*\)", re.I|re.DOTALL)
_RE_KEYWORD = re.compile(r"\b(APPROVE(?:D)?|CONDITIONAL(?:_APPROVE)?|REJECT(?:ED)?|DECLINE|DENY)\b", re.I)
_RE_QUOTED = re.compile(r"['\"]([^'\"]*)['\"]")

def _parse_tool_args(raw, tool_name):
    raw = raw.strip()
    if not raw: return {}
    quoted = _RE_QUOTED.findall(raw)
    if quoted:
        param = {"get_financial_report":"company_id","check_compliance_status":"company_id",
                 "get_market_intelligence":"sector"}.get(tool_name,"arg0")
        return {param: quoted[0]}
    bare = raw.strip("'\"").strip()
    if bare:
        param = {"get_financial_report":"company_id","check_compliance_status":"company_id",
                 "get_market_intelligence":"sector"}.get(tool_name,"arg0")
        return {param: bare}
    return {"raw": raw}

def parse_llm_output(text):
    """Parse LLM output → tool_call / final_decision / fallback_keyword / default_reject"""
    if not text or not text.strip():
        return {"parse_type":"default_reject","tool_name":None,"tool_args":None,
                "action":2,"reasoning":"[EMPTY OUTPUT]","parse_confidence":0.0,"parse_failure":True}
    text = text.strip()
    # Priority 1: Tool calls
    tools = list(_RE_TOOL.finditer(text))
    if tools:
        m = tools[0]
        tn = m.group(1).lower()
        args = _parse_tool_args(m.group(2), tn)
        return {"parse_type":"tool_call","tool_name":tn,"tool_args":args,
                "action":2,"reasoning":f"Calling {tn}","parse_confidence":0.95,"parse_failure":False}
    # Priority 2: submit_decision
    subs = list(_RE_SUBMIT.finditer(text))
    if subs:
        m = subs[-1]
        raw_act = m.group(1).upper().strip()
        reason = (m.group(2) or "").strip()
        act = ACTION_MAP.get(raw_act, ACTION_MAP.get(raw_act.replace("_"," "), 2))
        conf = 0.90 if reason else 0.65
        if not reason: reason = "[NO REASONING]"
        return {"parse_type":"final_decision","tool_name":None,"tool_args":None,
                "action":act,"reasoning":reason,"parse_confidence":conf,"parse_failure":False}
    # Priority 3: Keyword fallback
    kws = list(_RE_KEYWORD.finditer(text))
    if kws:
        kw = kws[-1].group(1).upper()
        return {"parse_type":"fallback_keyword","tool_name":None,"tool_args":None,
                "action":ACTION_MAP.get(kw,2),"reasoning":f"[KEYWORD: {kw}]",
                "parse_confidence":0.55,"parse_failure":False}
    # Priority 4: Default REJECT
    return {"parse_type":"default_reject","tool_name":None,"tool_args":None,
            "action":2,"reasoning":"[PARSE FAILURE]","parse_confidence":0.0,"parse_failure":True}

print("✅ Action parser ready (4 paths: tool_call → submit_decision → keyword → default)")


# ═══ CELL 5: CLIENT-SIDE TOOL SIMULATOR ═══
def simulate_tool_result(tool_name, tool_args, obs_data):
    """Simulate tool results from observation data (tools are read-only in env)."""
    app = obs_data.get("application_summary", {})
    summary = app.get("text_summary","") if isinstance(app,dict) else str(app)
    port = obs_data.get("portfolio_state", [0]*10)
    macro = obs_data.get("macro_state", [0]*5)
    alerts = obs_data.get("alert_state", [0]*5)
    mem = obs_data.get("memory_features", [0]*10)

    if tool_name == "get_financial_report":
        # Extract financial data from observation text
        lines = [
            "═══ FINANCIAL REPORT ═══",
            f"Application Summary: {summary[:300]}",
            "── Key Ratios (from observation) ──",
            f"  NPA Rate     : {port[2]*100:.1f}%" if len(port)>2 else "",
            f"  CRAR         : {port[9]*100:.1f}%" if len(port)>9 else "",
            f"  Capital Used : {port[1]*100:.0f}%" if len(port)>1 else "",
            "── Macro Context ──",
            f"  Stress Level : {macro[0]:.2f}" if len(macro)>0 else "",
            f"  Shock Active : {'YES ⚠️' if (len(macro)>1 and macro[1]>0.5) else 'No'}",
            "═══ END REPORT ═══"
        ]
        return "\n".join(l for l in lines if l)

    elif tool_name == "check_compliance_status":
        alert_labels = ["CC_SPIKE","BOUNCE_SURGE","GST_FILING_MISS","ADVERSE_MEDIA","CREDIT_DEGRADATION"]
        active = [alert_labels[i] for i,v in enumerate(alerts[:5]) if v > 0.5]
        has_red = any(v > 0.8 for v in alerts[:5])
        lines = [
            "═══ COMPLIANCE STATUS ═══",
            f"  Active Alerts: {', '.join(active) if active else 'None ✅'}",
            f"  RED Alert Present: {'🔴 YES — MANDATORY REJECT' if has_red else '✅ No'}",
            f"  Repeat Applicant Score: {mem[4]:.2f}" if len(mem)>4 else "",
            f"  Audit Risk: {mem[5]:.2f}" if len(mem)>5 else "",
            "═══ END COMPLIANCE ═══"
        ]
        return "\n".join(l for l in lines if l)

    elif tool_name == "get_market_intelligence":
        stress = macro[0] if len(macro)>0 else 0.2
        level = "HIGH STRESS ⚠️" if stress>0.6 else ("MODERATE" if stress>0.35 else "STABLE ✅")
        lines = [
            "═══ MARKET INTELLIGENCE ═══",
            f"  Macro Stress: {level} ({stress:.2f})",
            f"  Shock Active: {'YES ⚠️' if (len(macro)>1 and macro[1]>0.5) else 'No'}",
            f"  GDP Growth  : {macro[2]:.2f}" if len(macro)>2 else "",
            f"  Inflation   : {macro[3]:.2f}" if len(macro)>3 else "",
            "── Portfolio ──",
            f"  NPA Rate    : {port[2]*100:.1f}%" if len(port)>2 else "",
            f"  CRAR        : {port[9]*100:.1f}%" if len(port)>9 else "",
            f"  Max Sector  : {mem[2]*100:.0f}%" if len(mem)>2 else "",
            f"  {'⚠️ NEAR SECTOR LIMIT (>25%)' if (len(mem)>2 and mem[2]>0.25) else ''}",
            "═══ END INTELLIGENCE ═══"
        ]
        return "\n".join(l for l in lines if l)

    return f"[TOOL ERROR] Unknown tool: {tool_name}"

print("✅ Tool simulator ready (3 tools: financial, compliance, market)")


# ═══ CELL 6: 7-LAYER SYSTEM PROMPT BUILDER ═══
def build_system_prompt(obs_data, step_num, total_steps=50, memory_bank=None):
    """Build the full 7-layer prompt matching agent_loop.py"""
    app = obs_data.get("application_summary", {})
    summary = app.get("text_summary","No application data.") if isinstance(app,dict) else str(app)
    port = obs_data.get("portfolio_state", [0]*10)
    mem = obs_data.get("memory_features", [0]*10)

    npa = mem[0] if len(mem)>0 else 0.0
    approval_rate = mem[1] if len(mem)>1 else 0.5
    audit_risk = mem[5] if len(mem)>5 else 0.0
    macro_trend = mem[3] if len(mem)>3 else 0.0
    repeat_score = mem[4] if len(mem)>4 else 0.0

    trend_label = "↑ Worsening" if macro_trend>0.05 else ("↓ Improving" if macro_trend<-0.05 else "Stable")
    risk_label = "HIGH ⚠️" if audit_risk>0.7 else ("MODERATE" if audit_risk>0.3 else "Low")

    # Repeat applicant warning
    repeat_banner = ""
    if repeat_score > 0.3:
        attempt = int(repeat_score * 3) + 1
        repeat_banner = f"\n⚠️ REPEAT APPLICANT — Attempt #{attempt}. Surface metrics may have improved but underlying risk likely UNCHANGED or HIGHER."

    # Lessons from memory bank
    lessons_block = ""
    if memory_bank and memory_bank.lesson_count > 0:
        lessons_block = "\n" + memory_bank.get_lessons_text(5) + "\n"

    return f"""You are a Senior Credit Officer at IntelliCredit Bank.
You review MSME loan applications and must make credit decisions balancing yield, risk, and regulatory compliance.

═══ CURRENT STATE ═══
Episode Step   : {step_num} / {total_steps}
NPA Rate       : {npa:.1%} (rolling 10-step)
Approval Rate  : {approval_rate:.0%} (last 10 decisions)
Audit Risk     : {risk_label} (next audit in ~{max(0,int((1-audit_risk)*10))} steps)
Macro Trend    : {trend_label}{repeat_banner}
{lessons_block}
═══ CURRENT APPLICATION ═══
{summary[:600]}

═══ AVAILABLE TOOLS ═══
You may call up to {MAX_TOOL_CALLS} tools before submitting your decision.
Each tool call returns information. The step does NOT advance until you call submit_decision().

TOOL 1 — get_financial_report("company_name")
  Returns: Revenue trends, EBITDA margins, debt schedule, auditor remarks, cash flow.
  Example: get_financial_report("Bharat Industries Pvt. Ltd.")

TOOL 2 — check_compliance_status("company_name")
  Returns: Hard rule status, forensic alerts (RED/YELLOW), DIN status, CIBIL, prior defaults.
  Example: check_compliance_status("Bharat Industries Pvt. Ltd.")

TOOL 3 — get_market_intelligence("sector_name")
  Returns: Macro stress, sector exposure, peer NPA rate, headwinds/tailwinds.
  Example: get_market_intelligence("Manufacturing")

═══ SUBMITTING YOUR DECISION ═══
submit_decision("APPROVE", "Detailed reasoning here (minimum 50 characters)")
submit_decision("CONDITIONAL", "Approve with conditions: require quarterly cash-flow...")
submit_decision("REJECT", "Rejecting due to HR-01 DSCR below 1.0 and RED alert...")

HARD RULES (ANY = MANDATORY REJECT):
  HR-01: DSCR < 1.0           HR-04: Cheque bounce rate > 25%
  HR-02: Director disqualified HR-05: GST compliance < 40%
  HR-03: RED forensic alert    HR-06: Severe adverse media (>0.80)

ANTI-GAMING: Redundant tool calls = penalty. >4 tools = forced CONDITIONAL. Empty reasoning = rejected.

Think step by step. Use tools when uncertain. Then submit your decision."""

print("✅ System prompt builder ready (7-layer)")


# ═══ CELL 7: MEMORY BANK (Lightweight Reflection) ═══
class MemoryBank:
    """Cross-episode lesson storage for self-improvement."""
    MAX = 20
    SEVERITY = {"critical":0,"high":1,"medium":2,"low":3}

    def __init__(self):
        self.lessons = []
        self.scores = []

    @property
    def lesson_count(self): return len(self.lessons)

    def add_episode(self, score, step_rewards, actions, obs_list):
        """Extract and store lessons from a completed episode."""
        self.scores.append(score)
        new_lessons = []

        for i, (r, a) in enumerate(zip(step_rewards, actions)):
            if r >= 0: continue
            obs = obs_list[i] if i < len(obs_list) else {}
            alerts = obs.get("alert_state", [0]*5)
            has_red = any(v > 0.8 for v in alerts[:5])

            if has_red and a == 0:
                new_lessons.append({"type":"hard_rule","severity":"critical",
                    "lesson":f"RULE: RED alert at step {i+1} — approved and lost {r:.1f}. Always REJECT on RED.",
                    "reward_lost":r})
            elif a == 0 and r < -1.0:
                new_lessons.append({"type":"bad_approve","severity":"high",
                    "lesson":f"CAUTION: Risky approval at step {i+1} lost {r:.1f}. Be more conservative.",
                    "reward_lost":r})

        if score < 0:
            new_lessons.append({"type":"poor_episode","severity":"medium",
                "lesson":f"PORTFOLIO: Episode scored {score:.2f}. Reduce risky approvals.",
                "reward_lost":score})

        # Dedup and add
        for nl in new_lessons:
            dup = False
            for ex in self.lessons:
                if ex["type"]==nl["type"] and ex["lesson"][:30]==nl["lesson"][:30]:
                    ex["seen_count"] = ex.get("seen_count",1)+1; dup=True; break
            if not dup:
                nl["seen_count"] = 1
                self.lessons.append(nl)

        # Sort: critical first, then high, medium, low
        self.lessons.sort(key=lambda l: self.SEVERITY.get(l["severity"],9))
        # FIFO eviction
        while len(self.lessons) > self.MAX:
            self.lessons.pop()

    def get_lessons_text(self, n=5):
        if not self.lessons: return ""
        icons = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢"}
        lines = ["═══ PAST LESSONS LEARNED ═══","(Apply these to avoid repeating mistakes)",""]
        for i, l in enumerate(self.lessons[:n], 1):
            icon = icons.get(l["severity"],"⚪")
            count = f" (seen {l['seen_count']}x)" if l.get("seen_count",1)>1 else ""
            lines.append(f"  {i}. {icon} {l['lesson'][:100]}{count}")
        lines.append("")
        return "\n".join(lines)

memory_bank = MemoryBank()
print("✅ Memory bank ready (cross-episode reflection)")

# ═══ CELL 8: LOAD MODEL ═══
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch.nn.functional as F
import warnings

print(f"\n🔄 Loading {MODEL_NAME}...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True)

# FIX 1: prepare for kbit training (enables gradient checkpointing)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# FIX 2: CRITICAL — without this, gradient checkpointing cannot propagate
# gradients through quantized embeddings → all gradients are None → no learning
model.enable_input_require_grads()

# FIX 3: explicitly disable KV cache (incompatible with gradient checkpointing)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

lora_cfg = LoraConfig(r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Suppress the use_reentrant checkpoint warning (safe to ignore with our setup)
warnings.filterwarnings("ignore", message=".*use_reentrant.*", category=UserWarning)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
print("✅ Model + optimizer ready (gradient checkpointing + input_require_grads enabled)")

# ═══ CELL 9: REFERENCE MODEL ═══
print("\n🔄 Loading reference model (frozen)...")
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True)
for p in ref_model.parameters(): p.requires_grad = False
ref_model.eval()
print("✅ Reference model ready")


# ═══ CELL 10: GRPO CORE ═══
@torch.no_grad()
def generate_with_logprobs(prompt_text, temperature=1.0):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    input_len = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
        temperature=max(temperature, 0.1), top_p=0.9, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True, output_scores=True)
    gen_ids = outputs.sequences[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    log_prob = 0.0
    if outputs.scores:
        for idx, score in enumerate(outputs.scores):
            if idx >= len(gen_ids): break
            lp = F.log_softmax(score[0], dim=-1)
            log_prob += lp[gen_ids[idx]].item()
    return text, log_prob, inputs.input_ids, outputs.sequences

@torch.no_grad()
def ref_log_prob(full_ids, gen_len):
    logits = ref_model(full_ids).logits[0]
    input_len = full_ids.shape[1] - gen_len
    gen_logits = logits[input_len-1:input_len-1+gen_len]
    gen_tokens = full_ids[0, input_len:input_len+gen_len]
    lps = F.log_softmax(gen_logits, dim=-1)
    return lps[range(gen_len), gen_tokens].sum().item()

def compute_grpo_loss(trajectories, beta=BETA):
    if not trajectories: return torch.tensor(0.0), {}
    rewards = torch.tensor([t["episode_reward"] for t in trajectories], dtype=torch.float32)
    mean_r, std_r = rewards.mean(), rewards.std() + 1e-8
    advs = (rewards - mean_r) / std_r
    total_loss = torch.tensor(0.0, requires_grad=True)
    total_kl = 0.0
    for traj, adv in zip(trajectories, advs):
        full_text = traj["prompt_text"] + traj["completion_text"]
        inp_full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=900).to(model.device)
        inp_prompt = tokenizer(traj["prompt_text"], return_tensors="pt", truncation=True, max_length=768).to(model.device)
        prompt_len = inp_prompt.input_ids.shape[1]
        gen_len = min(inp_full.input_ids.shape[1] - prompt_len, MAX_NEW_TOKENS)
        if gen_len <= 0: continue
        logits = model(inp_full.input_ids).logits[0]
        gen_logits = logits[prompt_len-1:prompt_len-1+gen_len]
        gen_tokens = inp_full.input_ids[0, prompt_len:prompt_len+gen_len]
        lps = F.log_softmax(gen_logits, dim=-1)
        policy_lp = lps[range(gen_len), gen_tokens].sum()
        kl = float(policy_lp.item()) - traj["ref_log_prob"]
        total_kl += kl
        total_loss = total_loss + (-adv * policy_lp + beta * max(0.0, kl))
    n = max(len(trajectories), 1)
    return total_loss/n, {"mean_reward":float(mean_r),"reward_std":float(std_r-1e-8),
                          "mean_kl":total_kl/n,"n_trajectories":n}

print("✅ GRPO core ready")


# ═══ CELL 11: MULTI-TURN AGENT STEP (THE KEY CHANGE) ═══
def run_agent_step(obs_data, episode_id, step_num, temperature=1.0):
    """
    Run one environment step with multi-turn tool calling.
    Agent can call tools (step doesn't advance) then submit_decision (step advances).
    Returns: (action, reasoning, reward, done, prompt_text, completion_text, log_prob, ref_lp, tool_calls_made, parse_types)
    """
    tool_calls_made = 0
    conversation_context = ""
    all_completions = ""
    parse_types = []
    best_log_prob = 0.0
    best_ref_lp = 0.0

    for turn in range(MAX_TOOL_CALLS + 1):  # max 4 tools + 1 decision
        # Build prompt with current context
        sys_prompt = build_system_prompt(obs_data, step_num, MAX_STEPS_PER_EP, memory_bank)
        if conversation_context:
            full_prompt = sys_prompt + "\n" + conversation_context + "\nContinue your analysis or submit your decision."
        else:
            full_prompt = sys_prompt

        # Apply chat template
        messages = [{"role":"system","content":"You are an AI credit analyst."},
                     {"role":"user","content":full_prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate
        completion, log_prob, prompt_ids, full_ids = generate_with_logprobs(prompt_text, temperature)
        gen_len = full_ids.shape[1] - prompt_ids.shape[1]
        r_lp = ref_log_prob(full_ids, gen_len) if gen_len > 0 else 0.0

        if abs(log_prob) > abs(best_log_prob):
            best_log_prob = log_prob
            best_ref_lp = r_lp

        all_completions += completion + "\n"

        # Parse
        parsed = parse_llm_output(completion)
        parse_types.append(parsed["parse_type"])

        # Tool call branch — simulate tool, add result to context, don't send to env
        if parsed["parse_type"] == "tool_call" and tool_calls_made < MAX_TOOL_CALLS:
            tool_result = simulate_tool_result(parsed["tool_name"], parsed["tool_args"], obs_data)
            conversation_context += f"\n[Agent]: {completion}\n[Tool Result: {parsed['tool_name']}] ({tool_calls_made+1}/{MAX_TOOL_CALLS}):\n{tool_result}\n"
            tool_calls_made += 1
            continue

        # Decision branch — send to environment
        action = parsed["action"]
        reasoning = parsed["reasoning"]

        try:
            step_result = env_client.step(episode_id, action, reasoning[:200])
            reward = step_result["reward"]
            done = step_result["done"]
            new_obs = step_result.get("obs_data", obs_data)
        except Exception as e:
            print(f"   ⚠️ API error: {e}")
            reward, done, new_obs = 0.0, True, obs_data

        return {
            "action": action, "reasoning": reasoning, "reward": reward, "done": done,
            "new_obs": new_obs, "prompt_text": prompt_text, "completion_text": all_completions,
            "log_prob": best_log_prob, "ref_log_prob": best_ref_lp,
            "tool_calls": tool_calls_made, "parse_types": parse_types
        }

    # Fallback: exceeded tool limit, force CONDITIONAL
    try:
        step_result = env_client.step(episode_id, 1, "Forced CONDITIONAL — exceeded tool limit")
        return {"action":1,"reasoning":"FORCED","reward":step_result["reward"],
                "done":step_result["done"],"new_obs":step_result.get("obs_data",obs_data),
                "prompt_text":prompt_text,"completion_text":all_completions,
                "log_prob":best_log_prob,"ref_log_prob":best_ref_lp,
                "tool_calls":tool_calls_made,"parse_types":parse_types}
    except:
        return {"action":1,"reasoning":"FORCED","reward":-0.5,"done":True,
                "new_obs":obs_data,"prompt_text":prompt_text,"completion_text":all_completions,
                "log_prob":best_log_prob,"ref_log_prob":best_ref_lp,
                "tool_calls":tool_calls_made,"parse_types":parse_types}

print("✅ Multi-turn agent loop ready (tool calling + decision)")


# ═══ CELL 12: EPISODE COLLECTION ═══
def collect_episode(task_id="task1", temperature=1.0):
    """Run one full 50-step episode with multi-turn tool calling."""
    ep_data = env_client.reset(task_id=task_id)
    episode_id = ep_data["episode_id"]
    obs_data = ep_data.get("observation", ep_data.get("info", {}))
    if isinstance(obs_data, list): obs_data = {"observation": obs_data}

    cumulative_reward = 0.0
    step_rewards, actions_list, obs_list = [], [], []
    all_parse_types, total_tool_calls = [], 0
    best_traj = {"prompt_text":"","completion_text":"","log_prob":0.0,"ref_log_prob":0.0}

    for step in range(MAX_STEPS_PER_EP):
        result = run_agent_step(obs_data, episode_id, step+1, temperature)

        reward = result["reward"]
        cumulative_reward += reward
        step_rewards.append(reward)
        actions_list.append(result["action"])
        obs_list.append(obs_data)
        all_parse_types.extend(result["parse_types"])
        total_tool_calls += result["tool_calls"]

        # Track best trajectory for GRPO
        if abs(result["log_prob"]) > abs(best_traj["log_prob"]):
            best_traj = {"prompt_text": result["prompt_text"],
                         "completion_text": result["completion_text"],
                         "log_prob": result["log_prob"],
                         "ref_log_prob": result["ref_log_prob"]}

        obs_data = result["new_obs"]
        if result["done"]: break

    # Post-episode reflection
    memory_bank.add_episode(cumulative_reward, step_rewards, actions_list, obs_list)

    # Stats
    tool_usage_rate = total_tool_calls / max(len(step_rewards), 1)
    parse_success = sum(1 for p in all_parse_types if p != "default_reject") / max(len(all_parse_types), 1)

    return {
        "episode_id": episode_id, "task_id": task_id,
        "steps": len(step_rewards), "episode_reward": cumulative_reward,
        "step_rewards": step_rewards,
        "prompt_text": best_traj["prompt_text"],
        "completion_text": best_traj["completion_text"],
        "log_prob": best_traj["log_prob"],
        "ref_log_prob": best_traj["ref_log_prob"],
        "tool_usage_rate": tool_usage_rate,
        "parse_success_rate": parse_success,
        "total_tool_calls": total_tool_calls,
    }

print("✅ Episode collector ready (50-step, multi-turn, reflection)")

# ═══ CELL 13: MAIN TRAINING LOOP (3-Phase Curriculum) ═══
print("\n" + "="*70)
print("  🚀 ONLINE GRPO TRAINING — Full Environment Showcase")
print("="*70)
print(f"   Updates: {NUM_UPDATES} | Episodes/update: {EPISODES_PER_UPD}")
print(f"   Curriculum: 3 phases | Max steps/ep: {MAX_STEPS_PER_EP}")
print(f"   Features: Tool calling, Multi-agent, Reflection, 5 task levels")
print("="*70)

training_logs = []
best_reward = -float("inf")
t_total_start = time.time()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_UPDATES, eta_min=1e-6)

def get_curriculum_phase(update):
    for phase, cfg in CURRICULUM.items():
        lo, hi = cfg["updates"]
        if lo <= update <= hi: return phase, cfg
    return 3, CURRICULUM[3]

for update in range(1, NUM_UPDATES + 1):
    t_upd = time.time()
    model.train()

    phase, phase_cfg = get_curriculum_phase(update)
    tasks = phase_cfg["tasks"]
    temp_lo, temp_hi = phase_cfg["temp"]
    # Interpolate temperature within phase
    lo_upd, hi_upd = phase_cfg["updates"]
    progress = (update - lo_upd) / max(hi_upd - lo_upd, 1)
    temperature = temp_lo + (temp_hi - temp_lo) * progress
    task_id = random.choice(tasks)

    print(f"\n{'─'*70}")
    print(f"  Update {update}/{NUM_UPDATES} | Phase {phase} | Task: {task_id} | Temp: {temperature:.2f}")
    print(f"  Tasks: {tasks} | Memory: {memory_bank.lesson_count} lessons")
    print(f"{'─'*70}")

    # Collect episodes
    trajectories = []
    for ep_idx in range(EPISODES_PER_UPD):
        ep = collect_episode(task_id=task_id, temperature=temperature)
        trajectories.append(ep)
        print(f"   Ep {ep_idx+1}: steps={ep['steps']}, reward={ep['episode_reward']:.3f}, "
              f"tools={ep['total_tool_calls']}, parse_ok={ep['parse_success_rate']:.0%}")

    # GRPO gradient update
    optimizer.zero_grad()
    loss, stats = compute_grpo_loss(trajectories)
    if loss.requires_grad:
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
    scheduler.step()

    t_elapsed = time.time() - t_upd

    # Log
    avg_tools = np.mean([t["tool_usage_rate"] for t in trajectories])
    avg_parse = np.mean([t["parse_success_rate"] for t in trajectories])
    log_entry = {
        "update": update, "phase": phase, "task": task_id,
        "loss": float(loss.item()) if loss.requires_grad else 0.0,
        "mean_reward": stats.get("mean_reward", 0), "reward_std": stats.get("reward_std", 0),
        "mean_kl": stats.get("mean_kl", 0), "lr": scheduler.get_last_lr()[0],
        "tool_usage_rate": float(avg_tools), "parse_success_rate": float(avg_parse),
        "memory_lessons": memory_bank.lesson_count, "time_s": t_elapsed,
    }
    training_logs.append(log_entry)

    print(f"\n   📊 loss={log_entry['loss']:.4f} | reward={log_entry['mean_reward']:.3f} | "
          f"kl={log_entry['mean_kl']:.4f} | tools={avg_tools:.2f}/step | "
          f"parse={avg_parse:.0%} | {t_elapsed/60:.1f}m")

    if stats.get("mean_reward", -999) > best_reward:
        best_reward = stats["mean_reward"]
        model.save_pretrained(f"{OUTPUT_BASE}/best_model")
        tokenizer.save_pretrained(f"{OUTPUT_BASE}/best_model")
        print(f"   💾 NEW BEST reward={best_reward:.3f}")

    if update % 10 == 0:
        ckpt = f"{OUTPUT_BASE}/checkpoint_{update}"
        model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
        with open(f"{OUTPUT_BASE}/training_logs.json", "w") as f:
            json.dump(training_logs, f, indent=2, default=str)
        print(f"   💾 Checkpoint + logs saved")

t_total = time.time() - t_total_start
print(f"\n{'='*70}")
print(f"  ✅ ONLINE GRPO TRAINING COMPLETE — {t_total/3600:.1f} hrs | Best: {best_reward:.3f}")
print(f"{'='*70}")

model.save_pretrained(FINAL_MODEL); tokenizer.save_pretrained(FINAL_MODEL)
with open(f"{OUTPUT_BASE}/training_logs.json", "w") as f:
    json.dump(training_logs, f, indent=2, default=str)


# ═══ CELL 14: LEARNING CURVES (6-panel) ═══
print("\n📊 Generating learning curves...")
updates = [l["update"] for l in training_logs]
rewards = [l["mean_reward"] for l in training_logs]
losses  = [l["loss"] for l in training_logs]
kls     = [l["mean_kl"] for l in training_logs]
lrs     = [l["lr"] for l in training_logs]
tools   = [l["tool_usage_rate"] for l in training_logs]
parses  = [l["parse_success_rate"] for l in training_logs]

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("IntelliCredit Online GRPO — Full Environment Training", fontsize=14, fontweight="bold")

def _plot(ax, xs, ys, color, title, ylabel):
    ax.plot(xs, ys, color=color, alpha=0.4, linewidth=1)
    if len(ys) > 3:
        w = min(5, len(ys)//2)
        sm = np.convolve(ys, np.ones(w)/w, mode="valid")
        ax.plot(xs[:len(sm)], sm, color=color, linewidth=2.5, label="Smoothed")
    ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel); ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3); ax.legend()
    # Add phase boundaries
    for cfg in CURRICULUM.values():
        ax.axvline(x=cfg["updates"][0], color="gray", linestyle="--", alpha=0.3)

_plot(axes[0,0], updates, losses,  "#E53935", "GRPO Loss",              "Loss")
_plot(axes[0,1], updates, rewards, "#1976D2", "Episode Reward ↑",       "Reward")
_plot(axes[1,0], updates, kls,     "#7B1FA2", "KL Divergence",          "KL")
_plot(axes[1,1], updates, tools,   "#FF6F00", "Tool Usage Rate ↑",      "Tools/Step")
_plot(axes[2,0], updates, parses,  "#00796B", "Parse Success Rate ↑",   "Success %")
_plot(axes[2,1], updates, lrs,     "#455A64", "Learning Rate (Cosine)", "LR")

plt.tight_layout()
p = f"{OUTPUT_BASE}/charts/online_grpo_curves.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show()
print(f"  💾 {p}")


# ═══ CELL 15: FINAL EVALUATION (All 5 Tasks) ═══
print("\n🧪 Final evaluation — 2 episodes × 5 task levels...")
model.eval()
eval_results = []

for task in ["task1","task2","task3","task4","task5"]:
    for run in range(2):
        ep = collect_episode(task_id=task, temperature=0.7)
        eval_results.append(ep)
        print(f"   {task} run {run+1}: steps={ep['steps']}, reward={ep['episode_reward']:.3f}, "
              f"tools={ep['total_tool_calls']}, parse={ep['parse_success_rate']:.0%}")

avg_reward = np.mean([e["episode_reward"] for e in eval_results])
per_task = {}
for task in ["task1","task2","task3","task4","task5"]:
    task_eps = [e for e in eval_results if e["task_id"]==task]
    if task_eps: per_task[task] = np.mean([e["episode_reward"] for e in task_eps])

print(f"\n{'='*70}")
print(f"  📊 ONLINE GRPO TRAINING SUMMARY")
print(f"{'='*70}")
print(f"  Updates completed  : {len(training_logs)}")
print(f"  Best train reward  : {best_reward:.3f}")
print(f"  Final eval reward  : {avg_reward:.3f}")
print(f"  Per-task rewards   : {json.dumps({k:round(v,3) for k,v in per_task.items()})}")
print(f"  Memory lessons     : {memory_bank.lesson_count}")
print(f"  Total time         : {t_total/3600:.1f} hrs")
print(f"  Model saved at     : {FINAL_MODEL}/")
print(f"{'='*70}")
print(f"\n  🎯 FEATURES EXERCISED:")
print(f"     ✅ Tool Calling (get_financial_report, check_compliance, get_market_intelligence)")
print(f"     ✅ Multi-Turn Agent Loop (tools → gather info → decide)")
print(f"     ✅ 7-Layer System Prompt (role, state, lessons, app, tools, rules, action)")
print(f"     ✅ Cross-Episode Reflection (memory bank with {memory_bank.lesson_count} lessons)")
print(f"     ✅ Multi-Agent (BorrowerAgent re-applicants, RegulatorAgent audits)")
print(f"     ✅ 50-Step Episodes (macro shocks, delayed NPAs, audits at 10/20/30/40/50)")
print(f"     ✅ 5 Task Levels (task1→task5, easy→master)")
print(f"     ✅ 3-Phase Curriculum Training")
print(f"     ✅ 4 Reward Components (correctness, hard rules, format, portfolio)")
print(f"{'='*70}")
