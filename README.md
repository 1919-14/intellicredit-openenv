---
title: IntelliCredit CreditAppraisal v2
emoji: 🏦
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🏦 IntelliCredit-X — Teaching an LLM to Think Like a Credit Officer

<div align="center">

[![HF Space](https://img.shields.io/badge/🤗_Space-Live_Demo-blue)](https://huggingface.co/spaces/vssksn/intellicredit-openenv)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-GRPO_Training_Data-green)](https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset)
[![Model](https://img.shields.io/badge/🤗_Model-Mistral--7B_GRPO-orange)](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo)
[![GitHub](https://img.shields.io/badge/GitHub-intellicredit--openenv-black)](https://github.com/1919-14/intellicredit-openenv)
[![API Docs](https://img.shields.io/badge/API-Swagger_UI-purple)](https://vssksn-intellicredit-openenv.hf.space/docs)
[![Blog](https://img.shields.io/badge/📖_Blog-Full_Technical_Writeup-teal)](./docs/blog.md)
[![Colab](https://img.shields.io/badge/Colab-GRPO_Training_Notebook-yellow)](https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0-orange)](./PROJECT_SUMMARY.md)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blueviolet)](https://github.com/meta-pytorch/openenv)

**By V S S K Sai Narayana & Sujeet Jaiswal**  
*Meta × Hugging Face OpenEnv Hackathon 2026*

</div>

---

## 📖 Want the Full Story? Read the Blog

> ### ➡️ [IntelliCredit-X: Teaching an LLM to Think Like a Credit Officer Using Multi-Agent RL and GRPO](./docs/blog.md)
>
> *~5,000 words · Fully illustrated · Deep technical walkthrough*  
> Covers: the real-world MSME problem → 3-agent architecture → 55D observation space → tool calling system → sparse reward design → GRPO 3-stage curriculum → what the training curves actually tell you → qualitative before/after examples → 4 critical bugs we had to fix → honest limitations.
>
> **[🚀 Read the Blog →](./docs/blog.md)**

---

> **IntelliCredit-X** is an OpenEnv-compliant multi-agent reinforcement learning environment where an LLM learns to act as a regulatory-compliant Senior Credit Officer — investigating fraud signals via tool calls, managing a live loan portfolio across 50-step episodes, and respecting hard RBI mandates enforced by a RegulatorAgent. After GRPO fine-tuning of Mistral-7B, NPA rate halved on the hardest task and total reward improved 10×.

---

## 📊 Results at a Glance

<img width="1600" height="884" alt="051701af-7e29-4e6c-8895-f0c9b6569cf2" src="https://github.com/user-attachments/assets/1e4e4a05-9327-45c4-aed7-239ab8d74bbc" />


*Baseline Mistral-7B-Instruct-v0.3 (blue) vs. GRPO-trained IntelliCredit model (green) — **zero regressions across all 24 metric-task combinations.***

| Task | Metric | Base Model | GRPO Model | Δ |
|------|--------|-----------|-----------|---|
| Task 1 (Easy) | Score | 0.900 | **0.955** | **+0.055 ✅** |
| | Accuracy | 80.0% | **86.7%** | **+6.7% ✅** |
| | Capital Util | 40.0% | **60.0%** | **+20.0% ✅** |
| Task 2 (Medium) | Score | 1.000 | 1.000 | ceiling ✅ |
| | Total Reward | 10.305 | **10.584** | **+0.279 ✅** |
| Task 3 (Hard) | Score | 0.767 | **0.833** | **+0.067 ✅** |
| | Total Reward | 0.215 | **2.491** | **+2.276 ✅ (10×!)** |
| | **NPA Rate** | **16.7%** | **8.3%** | **−8.3% ✅ (halved!)** |

---

## 🎯 Core Motivation

The MSME lending sector in India processes over **100,000 loan applications daily**. Current bottlenecks:

- A senior loan officer reviews **~16 applications/day** — 0.016% of total volume by human experts
- **12–15% annual default rates** due to poor risk assessment
- **Manual cross-referencing** of GST, MCA, CIBIL, court records takes days per application
- **No explainable audit trail** — decisions based on "gut feeling" under time pressure

**Our approach:** Create a training ground where an AI learns to *think* like the best credit officers — gathering evidence, detecting hidden fraud, respecting non-negotiable regulations, and managing portfolio risk across time.

---

## ⚙️ How the Environment Works (v2.0)

An agent plays a **50-step Credit Committee Episode**:

```
Step T = 1..50:
  1. Environment generates an MSME application (Anchor × Sector × Size × Tier)
  2. Agent sees 55D observation (application + portfolio + macro + memory)
  3. Agent may call up to 4 investigation tools
  4. Agent submits: APPROVE (0) | CONDITIONAL (1) | REJECT (2)
  5. Reward computed: R1 (correctness) + R2 (hard rules) + R3 (format) + R4 (portfolio)
  6. Approved loans join portfolio
  7. RegulatorAgent audits at jittered steps ≈ 10/20/30/40/50
  8. Loan maturity events fire T+10 to T+30 (delayed NPA consequences)
  9. At step 50: settlement reward + Reflection Module activates
```

### Multi-Agent System

| Agent | Simulated By | Responsibility |
|-------|-------------|----------------|
| **Credit Officer** | LLM (GRPO fine-tuned) | Reviews applications, calls tools, makes decisions |
| **BorrowerAgent** | Programmatic | Reapplies after rejection with improved *surface* metrics (hidden PD unchanged or worse) |
| **RegulatorAgent** | Programmatic | Audits at ≈steps 10/20/30/40/50 (±1 jitter), shuts down after 3 consecutive failures |

#### BorrowerAgent — Fraud Simulation Detail

When rejected, BorrowerAgent: (1) waits 3–5 steps, (2) reapplies with DSCR +8%, Collateral +15%, Director guarantee added. **Hidden PD stays the same or worsens.** Max 3 attempts.

Detection signals — `Dim 49: borrower_persistence_score` (0.0=1st, 0.5=2nd, **1.0=3rd attempt = maximum manipulation signal**), `alert_state[4]` REPEAT_APPLICANT flag, and `check_compliance_status()` reveals unchanged GST/MCA history.

#### RegulatorAgent — Audit Mechanics

Audits fire at ≈steps 10/20/30/40/50 (±1 jitter to prevent timing exploitation).

| Constraint | Clean | Warning | Violation | Penalty |
|-----------|-------|---------|-----------|--------|
| NPA Rate | <3% | 3–5% | ≥5% | **−8.0** |
| CRAR | >15% | 12.5–15% | <12.5% | **−15.0 + possible termination** |
| Sector Concentration | <25% | 25–30% | >30% | **−8.0** |
| Single Borrower Limit | <12% | 12–15% | >15% | **−5.0** |

Escalation: 0 failures=normal → 1=warning → 2=capital penalty (−10% available capital) → 3=**REGULATORY_SHUTDOWN** (−50.0 terminal penalty)

### 📅 50-Step Episode Lifecycle

```
Steps  1–10  │ EARLY PHASE   │ Clean profiles, build portfolio baseline
   Step ~10  │ AUDIT #1      │ NPA / CRAR / sector concentration checked
Steps 11–20  │ MIDDLE PHASE  │ Forensic RED alerts appear; repeat applicants reapply
   Step ~20  │ MACRO SHOCK   │ GDP contraction fires; 1–2 sectors enter stressed state
   Step ~20  │ AUDIT #2      │
Steps 21–30  │ CRISIS PHASE  │ Maturity events fire: Repaid +10.0 / Defaulted −15.0
   Step ~30  │ AUDIT #3      │
Steps 31–40  │ RECOVERY      │ Balance new approvals vs NPA cleanup
   Step ~40  │ AUDIT #4      │
Steps 41–50  │ FINAL PHASE   │ Survival, capital preservation
   Step  50  │ SETTLEMENT    │ score = 0.30×yield + 0.30×(1−npa) + 0.20×compliance + 0.20×capital_util
             │ REFLECTION    │ Lesson extraction activates for next episode
```

---

## 🧠 Training Curves

<img width="1600" height="1142" alt="c54ed1cb-564e-40bd-81be-d56a76d9713f" src="https://github.com/user-attachments/assets/dd6b90ad-60d5-432e-9e1b-47cbcdab183e" />


*IntelliCredit GRPO v2 training across 3 curriculum stages. Note the key inflection points at stage transitions (dashed lines):*

- **GRPO Loss (red):** Controlled upward drift from ~0 → 0.05 — policy is meaningfully diverging from base model
- **Mean Reward (blue):** Starts at −2.0 (random violations), crosses zero by step 10, stabilizes near +0.5–+1.0 — **the environment is learnable**
- **KL Divergence (purple):** Grows to ~0.04–0.08 — model learned new behaviors while preserving language capability
- **`submit_pct` (teal):** Format compliance climbs from 0% → 40–65% — model acquires the task's vocabulary

---

## 🛑 Regulatory Rules (6 Non-Negotiable Hard Rules)

| Rule | Condition | Action |
|------|-----------|--------|
| **HR-01** | DSCR < 1.0 | Mandatory REJECT + −2.0 penalty |
| **HR-02** | Director disqualified (DIN < 0.1) | Mandatory REJECT + −2.0 penalty |
| **HR-03** | RED forensic alert present | Mandatory REJECT + −2.0 penalty |
| **HR-04** | Cheque bounce rate > 25% | Mandatory REJECT + −2.0 penalty |
| **HR-05** | GST compliance < 40% | Mandatory REJECT + −2.0 penalty |
| **HR-06** | Severe adverse media (> 0.80) | Mandatory REJECT + −2.0 penalty |

### Portfolio Constraints

| Constraint | Threshold | Consequence |
|------------|-----------|-------------|
| CRAR | > 12.5% | Episode terminates if breached |
| NPA Rate | < 5% | Episode terminates if breached |
| Sector Concentration | < 30% | −8.0 penalty per audit |
| Single Borrower | < 15% | −5.0 penalty per audit |

---

## 👁️ Observation Space (55D)

The agent observes a **55-dimensional vector** bounded `[−1.0, +1.0]`.
*(−1.0 = sentinel for missing/masked data — teaching the agent that data absence itself is a risk signal.)*

| Group | Dims | Description |
|-------|------|-------------|
| Application Features | 0–24 | 25 financial/forensic/governance ratios |
| Portfolio State | 25–34 | Capital deployed, NPA rate, CRAR, provisioning coverage, sector flags |
| Macro State | 35–39 | Systemic stress, GDP growth, inflation, credit cycle phase |
| Alert State | 40–44 | Running RED/YELLOW alert tallies from episode |
| **Memory Features** *(v2 NEW)* | **45–54** | **Agent's own behavioral history encoded as state** |

### Application Features (Dims 0–24) — Key Metrics

| Category | Metrics |
|----------|---------|
| Debt Serviceability | DSCR, Current Ratio, Debt-to-Equity, EBITDA Margin |
| Collateral | Collateral Coverage Ratio, RONW |
| Banking Behavior | OD Utilisation, CC Volatility, Cheque Bounce Rate, Working Capital Cycle |
| GST/Tax | GST CAGR, GST 2A-3B Gap, ITC Mismatch, GST Alignment Score |
| Fraud Signals | Related-Party Transactions, Circular Trading Score |
| Governance | Promoter Litigation Count, MCA Charges, Adverse Media Sentiment, DIN Score |

**Key memory dimensions (Dims 45–54):**
- `Dim 49: borrower_persistence_score` — 0.0=1st attempt, 0.5=2nd, **1.0=3rd attempt (maximum manipulation signal)**
- `Dim 50: audit_risk_score` — proximity to next regulator audit
- `Dim 51: capital_buffer_ratio` — headroom above minimum CRAR
- `Dim 53: episode_progress` — normalized step count (0.0–1.0)

---

## 🕹️ Action Space + Tool Calling

**Discrete(3):** APPROVE(0) | CONDITIONAL(1) | REJECT(2) — plus optional tool calls before deciding.

### Investigation Tools (up to 4 per step)

| Tool | Returns | Best Used When |
|------|---------|----------------|
| `get_financial_report(company_id)` | 3yr revenue trend, EBITDA, auditor remarks, related-party txns | Borderline financials, need trend confirmation |
| `check_compliance_status(company_id)` | DIN status, NCLT cases, GST filings, CIBIL, prior defaults | RED alert present, low governance score |
| `get_market_intelligence(sector)` | Sector stress, RBI advisory, portfolio exposure, peer NPA rate | Approaching 30% concentration limit |
| `submit_decision(action, reasoning)` | Finalizes step (reasoning ≥ 50 chars required) | After investigation complete |

### Action Parser — Priority Order (`server/action_parser.py`)

The LLM outputs free-form text. Parsed in strict priority:

1. **Tool call detected** → `get_financial_report(...)` → executes tool, does **not** advance step
2. **`submit_decision(action, reasoning)`** → validates format, advances step counter
3. **Standalone keyword** → `APPROVE` / `CONDITIONAL` / `REJECT` scanned in text
4. **Default fallback** → REJECT (safe default) + logs `parse_failure=True`

Anti-abuse: multiple decisions → last wins; reasoning < 50 chars → penalty; empty reasoning → decision blocked.

---

## 📈 Reward System

| Component | Weight | Range | Description |
|-----------|--------|-------|-------------|
| R1: Decision Correctness | 40% | [−2.0, +1.0] | PD-based: low PD+APPROVE=+1.0; high PD+APPROVE=−2.0 |
| R2: Hard Rule Compliance | 30% | [−2.0, +0.5] | HR+REJECT=+0.5; HR+APPROVE=−2.0 |
| R3: Format Compliance | 15% | [−0.3, +0.3] | `submit_decision()` used=+0.3; parse failure=−0.3 |
| R4: Portfolio Awareness | 15% | [−0.8, +0.3] | NPA>8%+risky approve=−0.5; healthy approve=+0.2 |

**Delayed Events:** Loan maturity fires T+10 to T+30 after approval (Repaid: +10.0, Defaulted: −15.0×(1−recovery))  
**Audit Bonus:** +2.0 clean audit / −8.0 violation / −15.0 capital breach / −50.0 shutdown (3rd failure)  
**Settlement (step 50):** `0.30×yield + 0.30×(1−npa) + 0.20×compliance + 0.20×capital_util`

### Survival Bonus (Every 10 Steps)

| CRAR Level | Bonus | Meaning |
|-----------|-------|--------|
| ≥ 15% | +0.10 | Healthy capital buffer |
| 12.5–15% | +0.05 | Marginal — caution signal |
| < 12.5% | Episode terminates | Capital inadequacy = bank failure |

## 🛡️ Anti-Gaming Mechanisms (10 Independent Safeguards)

| # | Mechanism | What It Prevents |
|---|-----------|------------------|
| 1 | Hidden PD — agent cannot see true default probability | Cannot directly optimize against ground truth |
| 2 | Read-only tools — cannot mutate environment state | Tool calls cannot manipulate outcomes |
| 3 | Max 4 tool calls enforced at env level (not agent) | Cannot bypass limit via prompt tricks |
| 4 | Reasoning quality check — empty text blocks decision | Cannot submit empty reasoning for format reward |
| 5 | Redundant tool call penalty (−0.1 each) | Prevents information-flooding strategy |
| 6 | Delayed NPA — defaults arrive T+10 to T+30 | Cannot see future consequences to optimize backward |
| 7 | World state locked — agent has no write access | Cannot modify portfolio variables directly |
| 8 | Deterministic episode seeds | No lucky randomness — same episode every run |
| 9 | Multiple independent reward functions | Gaming one component doesn't win overall |
| 10 | Jittered audit timing (±1 step) | Cannot predict exact audit step to game timing |

---

## 🤖 GRPO Training Pipeline — 2-Stage Approach

The final model [`vssksn/intellicredit-mistral-7b-grpo`](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo) was trained using a **2-stage pipeline**: offline GRPO for speed and domain knowledge, then **online GRPO directly against the live IntelliCredit environment** for true behavioral alignment.

```
┌─────────────────────────┐   ┌─────────────────────────┐
│  STAGE 1 — Offline GRPO      │   │  STAGE 2 — Online GRPO       │
│  (Speed-Optimised)           │   │  (Environment-Native)        │
│                             │   │                             │
│  Model : Mistral-7B-v0.3    │   │  Model : Mistral-7B (Stg 1) │
│  Engine: Unsloth + TRL      │   │  Env   : Live HF Space       │
│  Data  : 2,000 prompts      │   │  Data  : Real episodes       │
│  Reward: 4 local functions  │   │  Reward: /step endpoint 100% │
│  Speed : ~45 minutes        │   │  Type  : True Online RL      │
│  Goal  : Domain knowledge   │   │  Goal  : True env alignment  │
└──────────────┬──────────┘   └──────────────┬──────────┘
               │                               │
               └──────────────┬──────────────┘
                              │
                              ▼
         vssksn/intellicredit-mistral-7b-grpo
         Post-trained on live environment interactions
```

### Stage 1 — Offline GRPO (Speed-Optimised) 🚀

**[📓 Stage 1 Colab Notebook](https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing)** — Mistral-7B + Unsloth, A100, ~45 minutes

Pre-trains on a curated 2,000-prompt dataset for maximum training speed and domain knowledge transfer.

**Training Dataset:**
- **2,000 prompts** — 400 per task level (task1–task5), ~2,400 chars each
- Ground truth metadata: hidden PD, optimal action, hard rules, alerts, sector, CRAR, NPA
- Distribution: **47.2%** hard rules triggered | **28.1%** RED forensic alerts
- Published: [vssksn/intellicredit-grpo-dataset](https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset)

**3-Stage Curriculum:**

| Stage | Data | LR | Temperature | Goal |
|-------|------|----|-------------|------|
| Stage 0 (SFT Warmup) | Mixed | 5e-5 | — | Bootstrap `submit_decision()` format compliance |
| Stage 1 | task1 (Easy) | 5e-6 | 0.9 | Hard rule recognition on clean profiles |
| Stage 2 | task1 + task2 | 5e-6 | 0.9 | Forensic alert detection, tool call initiation |
| Stage 3 | All tasks | 2e-6 | 0.8 | Long-horizon portfolio management |

```
Config: rank=16 QLoRA (Unsloth), seq_len=2048, 8 generations/prompt
        batch=2 + grad_accum=8 (effective=16), KL β=0.001
```

### Stage 2 — Online GRPO (Environment-Native) 🌍

**[🌍 Stage 2 Notebook — Online Training (Colab)](https://colab.research.google.com/github/1919-14/intellicredit-openenv/blob/v2/training/colab_online_grpo.ipynb)** — Live env, 50-step episodes, real rewards, Mistral-7B

Post-trains the Stage 1 model by **directly interacting with the live IntelliCredit environment**. Every single reward signal comes from the actual `/step` endpoint — this is true online RL, not a proxy.

| Feature | Detail |
|---------|--------|
| Environment | [vssksn-intellicredit-openenv.hf.space](https://vssksn-intellicredit-openenv.hf.space) (live HTTP) |
| Episode length | **50 steps** — full credit committee lifecycle |
| **Reward source** | **`/step` endpoint — 100% environment-native** |
| Tool calling | Multi-turn: tools → evidence → `submit_decision()` |
| Reflection | Cross-episode memory bank (6 lesson categories, FIFO 20) |
| Curriculum | 3 phases: task1 → task3 → all 5 tasks, temp 1.2→0.8 |
| Model published | [vssksn/intellicredit-mistral-7b-grpo](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo) |

### 🔧 Critical Training Bug Fixes

| Bug | Root Cause | Fix Applied |
|-----|-----------|-------------|
| CUDA Index OOB | Unsloth pads vocab 32768→32832; padded token IDs indexed into smaller training logits | Clamp all IDs to `vocab_size−1` + `valid_mask` to skip OOB |
| Sequence Mismatch | `full_ids` exceeded 2048 before forward pass; logits truncated → shape crash | Enforce `full_ids = full_ids[:, :MAX_SEQ_LEN]` before forward |
| Loss Scale Instability | Raw log-prob sum scaled with sequence length → exploding gradients | Switch to per-token average: `loss = -sum(log_probs) / n_valid_tokens` |
| Flat KL Divergence | `clamp(min=0)` → KL=0 when new policy more confident than reference | Changed to `abs()` for symmetric KL — always non-zero |
| Zero-LP Episodes | Prompt filled entire 2048-token context → 0 completion tokens | Skip with `continue` when `sum(log_probs) == 0` |

---

## 🪞 Self-Improvement Reflection System

GRPO updates weights. The Reflection Module improves the model **without retraining** — by injecting structured lessons from episode failures into the next episode's system prompt.

```
Episode N → Analyze all steps where reward < 0
          → Extract lessons by failure type (6 categories)
          → Store top 20 lessons in memory_bank.json (FIFO eviction, deduplicated)
Episode N+1 → Inject top 5 lessons into system prompt Layer 3 → better decisions
```

### 6 Lesson Trigger Types

| Trigger | Lesson Injected | Severity |
|---------|-----------------|----------|
| Hard Rule Violation | `RULE: When [condition], always REJECT` | Critical |
| Delayed Default | `CAUTION: Loans with [pattern] defaulted T+N steps later` | High |
| Audit Failure | `COMPLIANCE: Audit failed due to [metric breach]` | High |
| Borrower Manipulation | `FRAUD RISK: Repeat applicant with [pattern] defaulted` | Critical |
| Macro Shock Loss | `MACRO: During [state], be conservative with [sector]` | Medium |
| Portfolio Overexposure | `PORTFOLIO: NPA reached X%. Tighten approvals.` | High |

**Verified result (base model, no fine-tuning, 3 consecutive episodes):**

| Episode | Score | Improvement |
|---------|-------|-------------|
| 1 | 0.213 | Baseline |
| 2 | 0.265 | **+24.4% ✅** |
| 3 | 0.304 | **+43.2% ✅** |

43% improvement purely through in-context lesson injection — zero weight changes.

---

## 🏆 Task Descriptions

| Task | Difficulty | Steps | Key Challenge |
|------|-----------|-------|---------------|
| `task1` | 🟢 Easy | 50 | Clean profiles, basic APPROVE/REJECT |
| `task2` | 🟡 Medium | 50 | Forensic alerts (YELLOW/RED), tool investigation |
| `task3` | 🔴 Hard | 50 | Macro shocks + missing data + repeat applicants |
| `task4` | 🔥 Expert | 50 | Hard-rule violations + all adversarial patterns |
| `task5` | ⚡ Master | 50 | Full: CRAR limits + cascading NPAs + 5 audits |

---

## 💻 Quick Start

### Try the Live API

```bash
# Start an episode
curl -X POST https://vssksn-intellicredit-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "demo-001", "seed": 42, "task_id": "task2"}'

# Submit a decision (0=APPROVE, 1=CONDITIONAL, 2=REJECT)
curl -X POST https://vssksn-intellicredit-openenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "demo-001", "action": {"decision": 2}}'
```

**→ [Full Swagger UI](https://vssksn-intellicredit-openenv.hf.space/docs)**

### Local Setup

```bash
git clone https://github.com/1919-14/intellicredit-openenv.git --branch v2
cd intellicredit-openenv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m server.app          # → http://localhost:7860/docs
```

### Evaluate the GRPO Model

```bash
# Run GRPO model against environment
python eval_llm.py \
  --model vssksn/intellicredit-mistral-7b-grpo \
  --env-url http://localhost:7860 \
  --out grpo_results.json

# Compare vs base model
python eval_llm.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --env-url http://localhost:7860 \
  --out base_results.json

# Generate comparison chart
python compare_results.py \
  --baseline base_results.json \
  --after grpo_results.json \
  --out comparison.png
```

### Docker

```bash
docker build -t intellicredit-v2 .
docker run -p 7860:7860 intellicredit-v2
```

---

## 📁 Project Structure

```
intellicraft-openenv/
├── server/
│   ├── app.py                 # FastAPI server — /reset, /step, /info, /health
│   ├── intellicredit_env.py   # v2 core: WorldState, 50-step lifecycle, multi-agent
│   ├── dataset.py             # Application generator (Anchor × Sector × Size × Tier)
│   ├── reward.py              # R1-R4 reward engine + settlement grader
│   ├── action_parser.py       # LLM text → tool call / decision parser (6-level)
│   ├── tool_executor.py       # Read-only tool execution (financial, compliance, market)
│   ├── agent_loop.py          # Agent orchestrator + prompt injection + step logger
│   └── reflection.py          # Self-improvement + memory bank system
│
├── training/
│   ├── colab_grpo_3b_v2.py    # ← PRIMARY: Unsloth GRPO training (A100, ~45 min)
│   ├── generate_dataset.py    # 2000-prompt GRPO dataset generator
│   ├── grpo_rewards.py        # 4 GRPO reward functions (R1-R4)
│   └── train_grpo.py          # 3-stage curriculum pipeline
│
├── evaluation/
│   ├── evaluate.py            # Multi-mode evaluation engine (baseline/reflection/GRPO)
│   └── compare.py             # Comparison tables + reward curves (4-panel PNG)
│
├── docs/
│   ├── blog.md                # Full technical blog post (~5,000 words)
│   └── assets/
│       ├── comparison.png     # Baseline vs GRPO results chart
│       └── training_curves.png # GRPO training curves (Mistral-7B, A100)
│
├── eval_llm.py                # LLM evaluation via HTTP (base vs trained)
├── compare_results.py         # Bar chart generator (8 metrics × 3 tasks)
├── baseline_results.json      # RuleBasedAgent reference scores
├── memory_bank.json           # Persistent cross-episode lesson storage (auto-generated)
├── inference.py               # LLM inference wrapper (HF API)
├── models.py                  # Pydantic schemas (55D observation, action)
├── client.py                  # HTTP client for environment interaction
├── openenv.yaml               # OpenEnv framework config
├── PROJECT_SUMMARY.md         # Complete project summary (all 8 phases)
├── Dockerfile                 # HF Spaces Docker deployment
└── requirements.txt           # Python dependencies
```

## 🧪 Evaluation Methodology

Two evaluation approaches:

**Approach 1 — Direct Python (`evaluation/evaluate.py`):** Tests agents by calling `IntelliCreditEnvironment` directly. Agents: `RuleBasedAgent` (optimal), `RandomAgent` (lower bound), `GreedyApproveAgent`. Output: `baseline_results.json`.

**Approach 2 — HTTP API (`eval_llm.py`):** Tests actual LLM via running server. Since `/step` returns only `{observation, reward, done}`, scores computed locally:

| Metric | Formula | Weight |
|--------|---------|--------|
| Accuracy | steps with positive reward / total steps | 0.5 |
| HR Compliance | 1 − (steps with reward < −5) / total steps | 0.3 |
| Survival Rate | 1.0 if all 50 steps completed without shutdown | 0.2 |
| **Final Score** | accuracy×0.5 + hr_compliance×0.3 + survival×0.2 | — |

## 📋 Version History

| Feature | v1.0 | v2.0 (Current) |
|---------|------|----------------|
| Episode Length | 12 steps | **50 steps** (4×) |
| Observation Dims | 45D | **55D** (+10 memory features) |
| Agent Count | 1 | **3** (Credit Officer + Borrower + Regulator) |
| Reward Type | Dense per-step | **Delayed + sparse** (realistic credit risk) |
| Tool Calling | ❌ | **✅ 3 tools, max 4 calls/step** |
| Self-Improvement | ❌ | **✅ Cross-episode reflection module** |
| GRPO Fine-Tuning | ❌ | **✅ Mistral-7B, A100, ~45 min** |
| Deployment | Local only | **✅ Docker + HF Spaces** |

## 📊 Baseline Agent Results (RuleBasedAgent — 25 episodes)

| Task | Avg Score | Accuracy | NPA Rate |
|------|-----------|----------|----------|
| task1 (Easy) | 0.389 | 77.9% | 4.8% |
| task2 (Medium) | 0.325 | 66.6% | 8.9% |
| task3 (Hard) | 0.288 | 81.5% | 20.2% |
| task4 (Expert) | 0.265 | 85.9% | 26.7% |
| task5 (Master) | 0.251 | 77.8% | 6.7% |
| **Overall** | **0.304** | **77.9%** | **13.4%** |

---

## 🔗 All Links

| Resource | Link |
|----------|------|
| 🤗 **Live Environment** | [huggingface.co/spaces/vssksn/intellicredit-openenv](https://huggingface.co/spaces/vssksn/intellicredit-openenv) |
| 🤗 **GRPO Model** | [huggingface.co/vssksn/intellicredit-mistral-7b-grpo](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo) |
| 🤗 **Training Dataset** | [huggingface.co/datasets/vssksn/intellicredit-grpo-dataset](https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset) |
| 💻 **GitHub (v2 branch)** | [github.com/1919-14/intellicredit-openenv/tree/v2](https://github.com/1919-14/intellicredit-openenv/tree/v2) |
| 📖 **API Swagger** | [vssksn-intellicredit-openenv.hf.space/docs](https://vssksn-intellicredit-openenv.hf.space/docs) |
| 📝 **Full Blog Post** | [docs/blog.md](./docs/blog.md) |
| 📓 **Colab Training Notebook** | [Open in Colab](https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing) |
| 📊 **Project Summary** | [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) |
| 📋 **Env Info API** | [/info endpoint](https://vssksn-intellicredit-openenv.hf.space/info) |

---

## 🐳 Docker Deployment

```bash
docker build -t intellicredit-v2 .
docker run -p 7860:7860 intellicredit-v2
# With HF token for LLM inference:
docker run -p 7860:7860 -e HF_TOKEN="your-token" intellicredit-v2
```

## 💻 Hardware Requirements

| Component | Environment Server | GRPO Training |
|-----------|-------------------|---------------|
| CPU | 2 vCPUs minimum | 8+ cores |
| RAM | 2 GB minimum | 32 GB minimum |
| GPU | **Not required** | **A100 80GB mandatory** |
| Storage | 500 MB | ~30 GB (model checkpoints) |
| Training Time | — | ~45 minutes |

## 🔐 Environment Variables

| Variable | Description | Default |
|----------|-------------|--------|
| `HF_TOKEN` | Hugging Face API token | Required for `inference.py` |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | LLM model for inference | `meta-llama/Llama-3.3-70B-Instruct` |
| `ENV_URL` | Environment server URL (for eval/training) | `http://localhost:7860` |

---

## 📚 Citation

```bibtex
@article{intellicredit2025,
  title   = {IntelliCredit-X: A Multi-Agent Constrained MDP for MSME Credit
             Appraisal with GRPO Fine-Tuning},
  author  = {Narayana, V S S K Sai and Jaiswal, Sujeet},
  year    = {2026},
  note    = {OpenEnv Hackathon Submission — Meta × Hugging Face},
  url     = {https://huggingface.co/spaces/vssksn/intellicredit-openenv}
}
```

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

*Built by **V S S K Sai Narayana** & **Sujeet Jaiswal** for the Meta × Hugging Face OpenEnv Hackathon 2026.*
