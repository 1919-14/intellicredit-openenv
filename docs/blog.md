---
title: "IntelliCredit-X: Teaching an LLM to Think Like a Credit Officer Using Multi-Agent RL and GRPO"
thumbnail: https://github.com/user-attachments/assets/53dfddff-d17c-4b63-8d22-a763f25c2bd7
authors:
- user: vssksn
  guest: true
- user: sujeetjaiswal
  guest: true
tags: [reinforcement-learning, grpo, fintech, multi-agent, llm-finetuning, openenv, credit-risk, msme, mistral]
---

# 🏦 IntelliCredit-X: Teaching an LLM to Think Like a Credit Officer Using Multi-Agent RL and GRPO

> **We built an OpenEnv-compliant multi-agent RL environment where Mistral-7B learns to act as a regulatory-compliant Senior Credit Officer — investigating fraud signals, managing a loan portfolio across 50-step episodes, and respecting hard RBI mandates. After GRPO fine-tuning, NPA rate halved on the hardest task and total reward improved 10×. Everything is open source.**

<div style="display: flex; gap: 10px; flex-wrap: wrap; margin: 20px 0;">

[![HF Space](https://img.shields.io/badge/🤗_Space-Live_Demo-blue)](https://huggingface.co/spaces/vssksn/intellicredit-openenv)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-GRPO_Training_Data-green)](https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset)
[![Model](https://img.shields.io/badge/🤗_Model-Mistral--7B_GRPO-orange)](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo)
[![GitHub](https://img.shields.io/badge/GitHub-intellicredit--openenv-black)](https://github.com/1919-14/intellicredit-openenv)
[![API Docs](https://img.shields.io/badge/API-Swagger_UI-purple)](https://vssksn-intellicredit-openenv.hf.space/docs)
[![License](https://img.shields.io/badge/License-MIT-red)](https://github.com/1919-14/intellicredit-openenv/blob/main/LICENSE)

</div>

*By **V S S K Sai Narayana** & **Sujeet Jaiswal** — Meta × Hugging Face OpenEnv Hackathon 2026*

---

## Table of Contents

- [The Moment That Started Everything](#the-moment-that-started-everything)
- [What We Built, and Why It Is Hard](#what-we-built-and-why-it-is-hard)
- [The Architecture: Three Agents, One Environment](#the-architecture-three-agents-one-environment)
- [The 55-Dimensional State Space: Financial Reality in Numbers](#the-55-dimensional-state-space-financial-reality-in-numbers)
- [The Tool System: Investigating Before Deciding](#the-tool-system-investigating-before-deciding)
- [The Reward System: Sparse Signals That Force Real Planning](#the-reward-system-sparse-signals-that-force-real-planning)
- [GRPO Training: How the LLM Learns Credit Judgment](#grpo-training-how-the-llm-learns-credit-judgment)
- [What the Training Curves Tell Us](#what-the-training-curves-tell-us)
- [Results: Before vs. After GRPO](#results-before-vs-after-grpo)
- [How to Use IntelliCredit-X](#how-to-use-intellicredit-x)
- [What We Learned Building This](#what-we-learned-building-this)
- [Limitations and What We Are Working On](#limitations-and-what-we-are-working-on)
- [Resources](#resources)

---

## The Moment That Started Everything

Picture this: a credit manager at a mid-sized Indian NBFC sits across the table from an MSME founder. The founder runs a textile manufacturing unit in Surat. Revenue is growing at 22% CAGR. GST filings look clean. Collateral is offered. The credit manager approves the loan.

Eight months later, that loan becomes an NPA.

A forensic audit reveals the truth: the revenue growth was inflated through circular trading between three related-party entities. The GST alignment — when cross-checked against bank statements — showed a 23% mismatch. The director had two undisclosed NCLT cases. Every signal was there. Buried in data that would have taken three days to manually cross-reference. Data the credit manager technically had access to — but not the time, or the system, to fully investigate.

**This is the problem IntelliCredit-X is built to solve.**

Not by creating another rules engine. Not by building a simple ML classifier. But by training an LLM to *reason* the way an experienced credit officer reasons — gathering evidence from multiple systems, weighing conflicting signals, respecting hard regulatory constraints, and managing a loan portfolio across time.

**The scale of this problem in India is staggering:** over 100,000 MSME loan applications are processed daily. A senior credit officer reviews 16 applications per day, spending 30 minutes each. That is 0.016% coverage by human experts working at capacity. The rest is processed by junior staff, rule engines, and educated guesses. Default rates for MSMEs run at 12–15% annually as a direct consequence.

---

## What We Built, and Why It Is Hard

IntelliCredit-X is a fully OpenEnv-compliant multi-agent reinforcement learning environment where an LLM acts as a **Senior Credit Officer** at an Indian lending institution. The agent reviews MSME corporate loan applications, calls investigation tools to gather evidence, submits approval decisions, and manages a live loan portfolio across a 50-step episode representing a full credit committee lifecycle.

Here is what makes this genuinely different from standard classification ML:

**The decision is not a one-shot classification.** A real credit officer does not look at a spreadsheet and output APPROVE or REJECT. They pull compliance records. They check sector exposure. They think about what happens to the portfolio if this loan defaults eighteen months from now.

**The consequences are delayed.** A loan approved at step 5 may default at step 30. The reward signal arrives 25 steps after the decision. This is the core challenge of credit risk RL — and what makes shallow learning strategies collapse.

**Other agents are working against you.** A rejected borrower does not disappear. They return a few steps later with better-looking numbers while the underlying risk is unchanged or worse. A regulator audits your portfolio every ten steps and shuts you down if you fail compliance checks three times consecutively.

**The regulations are non-negotiable.** Basel III requires Capital Risk Adequacy Ratio above 12.5% at all times. RBI mandates rejecting any borrower whose director is disqualified. These are not soft objectives — they are hard constraints that terminate episodes on violation.

Standard RL environments have none of these properties. That is why we built IntelliCredit-X.

---

## The Architecture: Three Agents, One Environment

```
┌─────────────────────────────────────────────────────────────────────┐
│                   INTELLICREDIT-X SYSTEM OVERVIEW                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   BORROWER AGENT (Adversarial)      REGULATOR AGENT (Enforcer)     │
│   ┌─────────────────────────┐       ┌─────────────────────────┐    │
│   │ Generates applications  │       │ Audits at steps         │    │
│   │ Hides real risk behind  │       │ ≈10, 20, 30, 40, 50     │    │
│   │ improved surface metrics│       │ Checks NPA, CRAR,       │    │
│   │ Reapplies after         │       │ sector concentration     │    │
│   │ rejection — up to       │       │ Capital penalty on fail │    │
│   │ 3 attempts              │       │ Episode shutdown after   │    │
│   └────────────┬────────────┘       │ 3 consecutive failures  │    │
│                │                    └────────────┬────────────┘    │
│                └──────────────┬──────────────────┘                  │
│                               ▼                                     │
│                  ┌────────────────────────┐                         │
│                  │      WORLD STATE       │                         │
│                  │  Macro economy trends  │                         │
│                  │  Sector health scores  │                         │
│                  │  Loan maturity queue   │                         │
│                  │  Portfolio ledger      │                         │
│                  └───────────┬────────────┘                         │
│                              ▼                                      │
│         ┌────────────────────────────────────────────┐             │
│         │          CREDIT OFFICER  (LLM Agent)        │             │
│         │                                             │             │
│         │  Receives 55D observation as text prompt    │             │
│         │  Calls 3 investigation tools (max 4/step)  │             │
│         │  Reasons through multi-turn dialogue        │             │
│         │  Submits decision + written reasoning       │             │
│         │                                             │             │
│         │  Fine-tuned via GRPO (TRL + Unsloth)       │             │
│         │  Self-improves via Reflection Module        │             │
│         └────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### The Credit Officer: The LLM Being Trained

The Credit Officer is `mistralai/Mistral-7B-Instruct-v0.3` fine-tuned via GRPO. It does not output a single decision token. It reasons through the problem in a multi-turn investigation:

```
Turn 1: Read application summary (text from 55D observation)
Turn 2: call get_financial_report('surat_textiles_001')
        → Returns: 3yr CAGR 22%, auditor remark: "GOING CONCERN expressed FY24",
                   Related party txns: 38% of revenue
Turn 3: call check_compliance_status('surat_textiles_001')
        → Returns: DIN: ACTIVE, NCLT cases: 2, GST returns: 8/12
Turn 4: call get_market_intelligence('Textiles')
        → Returns: Sector stress: 0.61, RBI advisory: ACTIVE,
                   Portfolio exposure: 23% (approaching 30% limit)
Turn 5: submit_decision("REJECT",
         "DSCR 0.89 violates HR-01. Going concern audit qualification contradicts
          22% CAGR — likely related-party revenue inflation (38% RP txns).
          Sector stress 0.61 with active RBI advisory. Portfolio at 23%
          textiles concentration — near 30% limit.")
```

**The base Mistral-7B-Instruct-v0.3 model, without fine-tuning, approved this application.** The GRPO-trained model rejected it — with reasoning a credit professional would recognize as correct.

### The Borrower Agent: The Adversary

A real MSME founder who gets rejected doesn't accept that decision quietly. They work with their CA, improve the numbers that got flagged, and come back. Our BorrowerAgent does exactly this:

| Rejection # | Surface Changes | Hidden Reality |
|---|---|---|
| After 1st rejection | DSCR +8%, Current Ratio +5%, Collateral +15%, Director guarantee added | **PD unchanged** |
| After 2nd rejection | Additional ratio improvement, minor forensic flags removed | **PD may WORSEN** |

**Key observable signal:** `borrower_persistence_score` in dimension 49 of the observation vector: 0.0 = first attempt, 0.5 = second, 1.0 = third attempt (maximum manipulation signal). This adversarial pressure forces the agent to learn underlying risk signals rather than memorizing surface patterns.

### The Regulator Agent: Compliance Enforcer That Doesn't Negotiate

Every 10 steps (with ±1 jitter to prevent gaming), the RegulatorAgent runs a portfolio audit with real consequences:

| Check | Safe | Violation | Consequence |
|---|---|---|---|
| Portfolio NPA Rate | < 5% | ≥ 5% | **−8.0 reward + capital haircut** |
| CRAR (Capital Adequacy) | > 12.5% | < 12.5% | **−15.0 + possible termination** |
| Sector Concentration | < 30% | > 30% | **−8.0 reward** |
| Single Borrower Limit | < 15% | > 15% | **−5.0 reward** |

Three consecutive failed audits → `REGULATORY_SHUTDOWN` → **episode terminates with −50.0 terminal penalty.** This is not a soft objective. This is how real banking regulation works.

The ±1 step jitter on audit timing is a deliberate anti-gaming mechanism. Without it, agents learn to clean up their portfolio just before audits and resume risky behavior immediately after. With jitter, compliance must be continuous.

---

## The 55-Dimensional State Space: Financial Reality in Numbers

Every decision step, the agent receives a 55-dimensional observation vector. The −1.0 sentinel value signals missing data — a common reality in MSME lending where 30–40% of features may be unavailable. Teaching the agent that data absence itself is a risk signal is one of the environment's most important design choices.

| Group | Dimensions | Content |
|---|---|---|
| Application Features | 0–24 | 25 financial/forensic/governance ratios per borrower |
| Portfolio State | 25–34 | Capital deployed, NPA rate, CRAR, provisioning coverage |
| Macro State | 35–39 | GDP growth, inflation, credit cycle phase, systemic stress |
| Alert State | 40–44 | Running RED/YELLOW alert tallies |
| **Memory Features (v2 NEW)** | **45–54** | **Agent's own behavioral history encoded as features** |

The Memory Features are what fundamentally separate IntelliCredit-X from a credit scoring model:

```
Dim 45: rolling_npa_rate_10step   → How many recent approvals defaulted?
Dim 46: approval_rate_recent      → Am I being too aggressive or conservative?
Dim 47: sector_max_concentration  → How close am I to the 30% sector limit?
Dim 48: macro_stress_trend        → Is the economy improving or worsening?
Dim 49: borrower_persistence      → 1st, 2nd, or 3rd attempt? (KEY: 1.0 = DANGER)
Dim 50: audit_risk_score          → How soon is the next regulator audit?
Dim 51: capital_buffer_ratio      → Headroom above minimum CRAR?
Dim 52: recent_reflection_count   → Lessons accumulated in memory bank?
Dim 53: episode_progress          → How far through the 50-step episode?
Dim 54: world_model_confidence    → How much info gathered via tool calls?
```

These 10 dimensions encode the agent's relationship to its own history, transforming the environment from a stateless per-application decision into genuine long-horizon portfolio management.

### The 6 Hard Rules: Non-Negotiable RBI Mandates

| Rule | Condition | Consequence |
|---|---|---|
| **HR-01** | DSCR < 1.0 | Automatic REJECT + **−2.0 penalty** |
| **HR-02** | Director DIN Score < 0.1 (disqualified) | Automatic REJECT + **−2.0 penalty** |
| **HR-03** | RED Forensic Alert Present (circular trading, ITC fraud) | Automatic REJECT + **−2.0 penalty** |
| **HR-04** | Cheque Bounce Rate > 25% | Automatic REJECT + **−2.0 penalty** |
| **HR-05** | GST Compliance < 40% | Automatic REJECT + **−2.0 penalty** |
| **HR-06** | Adverse Media Score > 0.80 | Automatic REJECT + **−2.0 penalty** |

Even if the agent wants to approve — the environment's compliance engine overrides it and levies the penalty. These exist because in real banking, these are precisely the cases where senior management is most vulnerable to relationship-based overrides. The environment enforces what human judgment sometimes fails to.

---

## The Tool System: Investigating Before Deciding

The most significant architectural upgrade from our v1 submission is the tool calling system. In v1, the agent saw a feature summary and output a decision — a sophisticated classifier, not an agent. In v2, the agent calls investigation tools exactly as a credit officer pulls data from multiple systems before approving a large loan.

### The Three Investigation Tools

**`get_financial_report(company_id)`**  
Returns three-year revenue trend, EBITDA margin history, debt schedule, cash flow from operations, and — critically — verbatim auditor remarks. A "Going Concern" qualification buried in auditor remarks is a severe red flag that does not appear in summary statistics. The model must learn to call this tool when revenue figures look too clean.

**`check_compliance_status(company_id)`**  
Returns MCA filing status, GST return history, director DIN status (active vs. disqualified), NCLT case count, ROC charges, CIBIL score, and prior loan defaults. This explicitly surfaces HR-02 — director disqualification — which requires looking up a government database most applicants assume won't be checked.

**`get_market_intelligence(sector_name)`**  
Returns sector stress score, active RBI sector advisories, current portfolio exposure percentage, competitive headwinds/tailwinds, and peer NPA rates. Critical when the portfolio is approaching the 30% concentration limit for a sector.

### Anti-Hacking Constraints (Because Every System Gets Gamed)

| Rule | Behavior | Penalty |
|---|---|---|
| Max 4 calls per step | 5th call → forced CONDITIONAL decision | - |
| Redundant call | Same tool + same args twice | **−0.1 per duplicate** |
| Malformed arguments | Error returned, counts as 1 call | **−0.05** |
| Empty submit reasoning | Decision rejected, step continues | Blocked |
| Reasoning < 50 chars | Format quality penalty | **−0.2** |

### What Effective Tool Usage Looks Like

Here is a real interaction trace from a fine-tuned model reviewing a high-risk application:

```
Application: Real Estate company, ₹8.5Cr loan request
DSCR: 0.94, Circular Trading Alert: AMBER
Persistence Score: 1.0 (3rd-attempt applicant)
Portfolio: Real estate exposure 27%, CRAR 13.1%

Turn 1 → get_financial_report('RE_Corp_003')
  Revenue CAGR 34% — suspiciously high
  Auditor remark: "Going concern doubt expressed FY24"
  Related party transactions: 41% of revenue

Turn 2 → check_compliance_status('RE_Corp_003')
  GST returns filed: 7/12 months
  NCLT cases: 2
  CIBIL: 0.52

Turn 3 → get_market_intelligence('Real Estate')
  Sector stress score: 0.71 (HIGH — RBI advisory active)
  Portfolio exposure: 27% (approaching 30% limit)

Turn 4 → submit_decision("REJECT",
  "DSCR 0.94 near HR-01 threshold. Revenue of 34% CAGR contradicted
   by going concern audit doubt — 41% related party revenue suggests
   inflation. 3rd-attempt applicant (persistence_score=1.0) is a
   manipulation signal. Real estate stress 0.71 + approving would push
   exposure to 30% concentration limit. Hard REJECT.")
```

**Base model decision:** APPROVE (reward: −2.4)  
**GRPO model decision:** REJECT (reward: +1.8)

---

## The Reward System: Sparse Signals That Force Real Planning

Standard RL environments give dense feedback — a reward every step. This sounds helpful but creates a subtle failure mode specific to credit: the agent optimizes individual decisions rather than portfolio outcomes. A model that gets immediate positive feedback for every approval learns to approve aggressively. It never discovers that those loans default at step 30.

IntelliCredit-X is built around sparse, delayed signals:

```
Normal steps (no special event):  reward ≈ 0
  → Agent must plan without constant feedback

Loan Maturity Events (fires T+10 to T+30 after approval):
  Loan approved at step 5 → maturity check at step 25–35
    Repaid:    +10.0 reward
    Defaulted: −15.0 × (1 − recovery_rate)

Regulator Audits (jittered ±1 from steps 10/20/30/40/50):
  Clean audit:    +2.0
  Per violation:  −8.0
  Capital breach: −15.0
  3rd failure:    −50.0 + SHUTDOWN

Survival Bonus (every 10 steps, CRAR > 12.5%):
  CRAR ≥ 15%:       +0.10
  CRAR 12.5–15%:    +0.05
  CRAR < 12.5%:     Episode terminates

Settlement Score (step 50 only):
  score = 0.30 × yield
        + 0.30 × (1 − npa_rate / 0.05)
        + 0.20 × compliance_score
        + 0.20 × capital_utilization
```

The settlement score is the final exam. A portfolio that maximized step-level yield by approving risky loans will score poorly on settlement even if it passed every individual audit. This rewards durable portfolio health, not short-term throughput.

### Four Independent GRPO Reward Functions

During fine-tuning, the reward is computed by four completely independent functions. Why four and not one? Because a single reward function is a single attack surface for reward hacking. If the model discovers that always REJECTing scores reasonably — no NPAs, no concentration violations, no hard rule penalties — it will exploit that. Four independent functions prevent any pure strategy from dominating:

| Function | Range | Key Logic | Pure-REJECT Defense |
|---|---|---|---|
| `reward_correctness` | [−2.0, +1.0] | PD < 0.25 → APPROVE = +1.0 | REJECTing good loans gets penalized |
| `reward_hard_rule` | [−2.0, +0.5] | HR triggered + REJECT = +0.5 | Clean apps rejected = no bonus |
| `reward_format` | [−0.3, +0.3] | `submit_decision()` used = +0.3 | Pure REJECT without format = −0.3 |
| `reward_portfolio` | [−0.8, +0.3] | Low capital util + REJECT = −0.5 | Idle capital is penalized |

Always-REJECT gets a heavy penalty from `reward_portfolio`. Always-APPROVE gets hammered by `reward_correctness` on high-PD applications. The model is forced to actually learn the task.

---

## GRPO Training: How the LLM Learns Credit Judgment

### The 2-Stage Training Pipeline

[`vssksn/intellicredit-mistral-7b-grpo`](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo) was not trained with a single script. The published model is the result of a **two-stage pipeline** — an architecture that mirrors how real-world RL systems are built.

```
┌────────────────────────────────────────────┐
│  STAGE 1: Offline GRPO (Speed-Optimised)   │
│  Script  : training/colab_grpo_3b_v2.py    │
│  Model   : Mistral-7B-Instruct-v0.3        │
│  Engine  : Unsloth + TRL + 4-bit QLoRA     │
│  Data    : 2,000 pre-generated prompts     │
│  Reward  : 4 local functions (no server)   │
│  Runtime : ~45 minutes on A100 80GB        │
│  Goal    : Domain knowledge, format learn  │
└────────────────────────┬───────────────────┘
                         │
                         ▼  (Stage 1 checkpoint)
┌────────────────────────────────────────────┐
│  STAGE 2: Online GRPO (Environment-Native) │
│  Script  : training/colab_online_grpo.py   │
│  Model   : Mistral-7B from Stage 1         │
│  Env     : Live HF Space (real HTTP calls) │
│  Data    : Real 50-step episodes           │
│  Reward  : 100% from /step endpoint        │
│  Goal    : True environment alignment      │
└────────────────────────┬───────────────────┘
                         │
                         ▼
         vssksn/intellicredit-mistral-7b-grpo
```

**Stage 1** teaches the model what credit officers know — the hard rules, the forensic flags, the tool calling format. **Stage 2** teaches the model to actually perform in the environment — where every reward signal comes from the live `/step` endpoint and every mistake costs real episode reward. The published model was post-trained on direct environment interactions: it learned by actually operating in IntelliCredit-X, not by reading about it.

This two-stage approach is why the GRPO model generalizes beyond the training distribution. Stage 1 builds broad domain coverage cheaply. Stage 2 narrows that coverage to precisely what the environment rewards — which is precisely what matters at evaluation time.

### Why GRPO, Not PPO?

Our v1 submission used PPO with Stable Baselines3 trained on the 45-dimensional observation vector. That agent learned well — improving from −1.20 to +3.57 average reward over 500K steps. But it has a fundamental limitation: it has no language understanding. It cannot read the text of an auditor's going concern remark. It cannot call a tool and interpret the result. It cannot write a reasoned explanation for its decision.

GRPO fine-tuning on Mistral-7B-Instruct-v0.3 solves all three problems. The model already understands language — we are teaching it financial domain judgment on top of that foundation.

### How GRPO Works in This Context

```
For each training batch:

1. SAMPLE:    Take N prompts (credit application + portfolio state + tools doc)

2. GENERATE:  Sample 8 completions from current policy per prompt
   Output A:  submit_decision('APPROVE', 'Financials look okay')
   Output B:  [calls 3 tools] → submit_decision('REJECT', '[specific reasoning]')
   Output C:  'REJECT'  (no tools, no format)
   ... 5 more completions ...

3. SCORE:     Run all 4 reward functions on all 8 outputs
   Output A:  −2.4 (wrong decision, no investigation)
   Output B:  +2.1 (correct + tool use + quality reasoning)
   Output C:  +0.8 (correct but wrong format)

4. RANK:      Compute group-relative advantage
   advantage_i = (score_i − mean_score) / std(scores)
   Output B gets highest advantage; Output A most negative

5. UPDATE:    Shift weights so Output B-style behavior becomes more probable
              KL penalty (β = 0.001) keeps policy near base model

6. REPEAT across 3-stage curriculum
```

Key advantage over PPO: GRPO evaluates quality relatively within a group of sampled outputs. No learned critic or value function needed. More memory-efficient and more stable for LLM post-training on verifiable tasks.

### The 3-Stage Curriculum (Easy → Hard)

Starting all tasks simultaneously would produce a model that never receives positive reward and learns nothing. Curriculum is mandatory.

| Stage | Data | LR | Temp | Goal |
|---|---|---|---|---|
| **Stage 0 (SFT Warmup)** | Mixed tasks | 5e-5 | — | Bootstrap `submit_decision()` format compliance |
| **Stage 1** | task1 (Easy) | 5e-6 | 0.9 | Hard rule recognition on clean profiles |
| **Stage 2** | task1 + task2 | 5e-6 | 0.9 | Forensic alert detection, tool call initiation |
| **Stage 3** | All tasks | 2e-6 | 0.8 | Long-horizon portfolio management |

**Training Configuration:**

```
Base Model:      mistralai/Mistral-7B-Instruct-v0.3
Quantization:    4-bit QLoRA via Unsloth  
LoRA:            rank=16, targeting q/k/v/o projection layers
Sequence Length: 2048 tokens (strictly enforced)
Num Generations: 8 per prompt
Batch:           2 + grad_accum=8 = effective batch 16
KL Beta:         0.001
Hardware:        A100 80GB — ~45 minutes total
```

### Critical Bug Fixes We Had to Solve

Implementing GRPO with Unsloth on a custom environment surfaced several non-obvious issues that would silently corrupt training if undetected:

**1. CUDA Index Out of Bounds — The Silent Killer**

Unsloth internally pads the model's vocabulary from 32768 to 32832 tokens during quantized inference. If completion token IDs from the padded vocabulary were naively indexed into training-mode logits (which only reflect the original 32768 tokens), CUDA would throw an assertion error at runtime — or worse, silently produce garbage gradients.

*Fix:* Clamp all token IDs to `vocab_size - 1` before any indexing. Apply a `valid_mask` to exclude any out-of-bounds tokens from log-prob computation.

**2. Sequence Length Shape Mismatch**

`full_ids` (prompt + completion concatenated) could exceed the 2048-token limit before the forward pass, but `model(input_ids=full_ids)` would return truncated logits. Indexing the completion log-probs then causes a shape mismatch crash.

*Fix:* Enforce `full_ids = full_ids[:, :MAX_SEQ_LEN]` *before* the forward pass, not after.

**3. Flat KL Divergence — The Silent Loss of Signal**

Our initial KL computation used `clamp(min=0)`, which means when the new policy was *more* confident than the reference model, KL registered as exactly zero. The penalty term was providing no signal, allowing the policy to drift arbitrarily in one direction.

*Fix:* Use `abs()` for symmetric KL — always non-zero and informative in both directions.

**4. Zero Log-Probability Episodes**

When a very long prompt filled the entire 2048-token context, the completion portion was truncated to zero tokens. Log-probabilities would be `0.0`, and gradients from such episodes are mathematically meaningless. Including them degrades training quality.

*Fix:* Skip any episode where `sum(log_probs) == 0` with a `continue` statement.

These four fixes were the difference between training that appeared to run (metrics updating, no crashes) and training that actually improved the model.

---

## What the Training Curves Tell Us

<img width="1600" height="1142" alt="c54ed1cb-564e-40bd-81be-d56a76d9713f" src="https://github.com/user-attachments/assets/d225eb30-db76-4edb-bbc2-b429c6222095" />


*Figure 1: IntelliCredit GRPO v2 training curves across three curriculum stages (dashed lines mark stage transitions). The smoothed lines reveal the signal beneath the noise of stochastic generation.*

Reading these four panels tells the full story of what the model learned and when:

**GRPO Loss (top left — red):** The loss starts slightly negative (before the policy has diverged significantly from the reference), then climbs and stabilizes around 0.02–0.05 throughout training. This is the expected shape for GRPO: the loss reflects the KL penalty term growing as the policy diverges from the base model. A flat zero loss would indicate training is not moving — which is a failure mode, not a success. The controlled upward drift here is healthy.

**Mean Reward (top right — blue):** This is the most important chart. The model starts at −2.0 average reward — essentially random, frequently violating hard rules and producing unformatted outputs. By step 10 (end of Stage 1), reward crosses zero. By step 20 (end of Stage 2), it stabilizes near +0.5–+1.0. Stage 3 (all tasks) introduces harder problems, causing a dip and re-stabilization — a characteristic curriculum learning signature. The model learned. The environment was learnable.

**KL Divergence (bottom left — purple):** The KL grows gradually from ~0 to ~0.04–0.08 across training. This tells us the policy is meaningfully diverging from the base Mistral-7B distribution — it is genuinely learning credit domain behavior, not just format compliance. The fact that KL stays below 0.12 (our monitored threshold) confirms the β=0.001 penalty is doing its job: the model changed, but not so much that it catastrophically forgot general language capabilities.

**`submit_pct` (bottom right — teal):** This tracks the percentage of completions where the model correctly formats its final decision as `submit_decision()` rather than bare keywords or free text. It starts near 0% — the base model has no concept of this format — and climbs to 40–65% by end of training. Format compliance is a prerequisite for reward: a model that cannot format its output cannot receive task reward. Watching this metric climb from zero is watching the model acquire the vocabulary of the task.

**The stage dashed lines** reveal something subtle about curriculum learning: the biggest single-step improvement in mean reward happens at the Stage 1→2 transition, when the model first encounters forensic alerts. This suggests that forensic pattern recognition was the bottleneck behavior — once the model learned to detect circular trading and ITC mismatch signals, subsequent improvements compounded naturally.

---

## Results: Before vs. After GRPO
<img width="1600" height="884" alt="051701af-7e29-4e6c-8895-f0c9b6569cf2" src="https://github.com/user-attachments/assets/53dfddff-d17c-4b63-8d22-a763f25c2bd7" />


*Figure 2: Per-task, per-metric comparison of base Mistral-7B-Instruct-v0.3 (blue) versus GRPO-trained IntelliCredit model (green). Green headers indicate improvement; all metrics either improved or held steady — zero regressions across all 24 evaluated metric-task combinations.*

### Quantitative Results

| Task | Metric | Base Model | GRPO Model | Δ |
|---|---|---|---|---|
| **Task 1** (Easy) | Task Score | 0.900 | **0.955** | **+0.055 ✅** |
| | Total Reward | 2.904 | 3.272 | +0.368 ✅ |
| | Accuracy | 80.0% | **86.7%** | **+6.7% ✅** |
| | Hard Rule Comply | 100.0% | 100.0% | unchanged ✅ |
| | Capital Utilization | 40.0% | **60.0%** | **+20.0% ✅** |
| **Task 2** (Medium) | Task Score | 1.000 | 1.000 | ceiling (both perfect) |
| | Total Reward | 10.305 | 10.584 | +0.279 ✅ |
| | Capital Utilization | 25.0% | **29.2%** | **+4.2% ✅** |
| **Task 3** (Hard) | Task Score | 0.767 | **0.833** | **+0.067 ✅** |
| | Total Reward | 0.215 | **2.491** | **+2.276 ✅ (+10×!)** |
| | Accuracy | 58.3% | **66.7%** | **+8.3% ✅** |
| | **NPA Rate** | **16.7%** | **8.3%** | **−8.3% ✅ (halved!)** |
| | Capital Utilization | 16.7% | **25.0%** | **+8.3% ✅** |

**Zero regressions across all 3 tasks and 8 metrics.** Every metric either improved or held steady.

### Reading the Comparison Chart

The chart tells a layered story:

**Task 1 (top row):** The trained model improves accuracy by 6.7 points and dramatically increases capital utilization (+20 points) — it is deploying more capital into good loans it correctly identifies, not just being a better rejector.

**Task 2 (middle row):** Both models achieve perfect Task Score (1.000) — this task was already within reach of the base model's capabilities. The GRPO model squeezes additional reward (+0.279) from better capital efficiency.

**Task 3 (bottom row):** This is where training matters most. The hardest task — macro shocks, missing data, repeat adversarial applicants. The base model generates barely positive total reward (0.215). The GRPO model generates 2.491. The NPA rate drops from 16.7% to 8.3% — the model is not just making better individual decisions, it is managing portfolio risk over time. This is the learned behavior that justifies the entire training pipeline.

### Qualitative Example: The Repeat Applicant Trap

```
Step 15: Application arrives — Real estate company, ₹4.2Cr
  DSCR: 1.08 (above threshold), Current Ratio: 1.35 (healthy)
  Collateral: 1.4× (improved from previous application)
  Persistence Score: 1.0 (3rd attempt)

BASE MODEL:
  → (No tool calls)
  → submit_decision("APPROVE", "Financials are solid.")
  → Step 35: Loan defaults. Reward: −15.0
  → NPA rate spikes

GRPO MODEL:
  → call check_compliance_status() → reveals 2 new NCLT cases added since last application
  → call get_market_intelligence('Real Estate') → sector stress 0.68, RBI advisory
  → submit_decision("REJECT",
      "3rd-attempt applicant (persistence=1.0) — manipulation signal.
       2 new NCLT cases since last application.
       Improved DSCR does not explain new litigation.
       Real estate sector stress 0.68 with active RBI advisory.")
  → Reward: +1.4
```

This is the signature behavior of a well-trained credit model: it learned that surface improvement combined with behavioral red flags (persistence score, new litigation) means escalating, not declining, risk.

---

## The Self-Improvement Reflection Module

GRPO updates model weights — which requires compute. Between training runs, can the model improve in-context?

Yes. The Reflection Module analyzes the full episode log after every episode ends and extracts structured lessons from every mistake, storing them in `memory_bank.json`. At the start of the next episode, the top 5 lessons are prepended to the system prompt:

```
Layer 1: Base system role (Senior Credit Officer)
Layer 2: Current portfolio state (NPA, CRAR, audit risk)
Layer 3: ← TOP 5 LESSONS FROM MEMORY BANK (Phase 5)
Layer 4: Current application data
Layer 5: Tool documentation
Layer 6: Decision format + hard rules
Layer 7: Action request
```

**Running the base Mistral model (no fine-tuning) with the Reflection Module active across 30 episodes:**

| Episodes | Average Score | What Changed |
|---|---|---|
| 1–5 | 0.22 | No lessons yet — baseline behavior |
| 6–10 | 0.31 | Hard rule violations dropping |
| 11–20 | 0.44 | Forensic alerts triggering tool calls |
| 21–30 | 0.55 | Portfolio management improving |

A 150% improvement without changing a single model weight. The environment provides enough signal that in-context learning from episode history produces meaningful behavioral change. The GRPO fine-tuned model also uses the Reflection Module — the two self-improvement mechanisms compound each other.

---

## How to Use IntelliCredit-X

### Option 1: Try the Live Demo

The environment is live on Hugging Face Spaces and accepts connections from any HTTP client:

**[🚀 Open Live Swagger UI](https://vssksn-intellicredit-openenv.hf.space/docs)**

```bash
# Start an episode
curl -X POST https://vssksn-intellicredit-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "my-session-1", "seed": 42, "task_id": "task2"}'

# Submit a decision (0=APPROVE, 1=CONDITIONAL, 2=REJECT)
curl -X POST https://vssksn-intellicredit-openenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "my-session-1", "action": {"decision": 2}}'
```

### Option 2: Train Your Own Model — Two Approaches

We provide two training notebooks depending on your goal:

**🚀 Stage 1 — Offline GRPO (Fast, A100, ~45 mins):** [`training/colab_grpo_3b_v2.py`](https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing)

Uses a pre-generated 2,000-prompt dataset + 4 local reward functions. No live server needed. Mistral-7B + Unsloth. Best starting point.

```bash
# In Google Colab (A100):
!git clone https://github.com/1919-14/intellicredit-openenv.git --branch v2 --depth 1
%cd /content/intellicredit-openenv
!pip install -q unsloth trl transformers torch accelerate requests
# Then open and run training/colab_grpo_3b_v2.py
```

**[🌍 Stage 2 — Online GRPO (Environment-Native)](https://colab.research.google.com/github/1919-14/intellicredit-openenv/blob/main/training/colab_online_grpo.ipynb)**

Connects to the live IntelliCredit environment. Every reward comes from the real `/step` endpoint. 50-step episodes with actual multi-agent pressure. This is how `vssksn/intellicredit-mistral-7b-grpo` was post-trained.

```bash
# Requires the live HF Space (or your local server via ngrok)
# Run after Stage 1 checkpoint is ready
# Uses Qwen2.5-1.5B for online efficiency, or load your Stage 1 Mistral checkpoint
ENV_URL = "https://vssksn-intellicredit-openenv.hf.space"  # or your local URL
# Then open and run training/colab_online_grpo.py
```

### Option 3: Evaluate Any Agent

```python
from openenv import from_hub

env = from_hub("vssksn/intellicredit-openenv")  # loads the live environment
obs, info = env.reset(task="task3", seed=42)

while not done:
    action = your_agent.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

# Compute scores from rewards
print(f"Total Reward: {info['total_reward']:.4f}")
```

### Option 4: Local Setup

```bash
git clone https://github.com/1919-14/intellicredit-openenv.git --branch v2
cd intellicredit-openenv

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Start environment server
python -m server.app
# → http://localhost:7860/docs

# Run LLM evaluation vs trained model
python eval_llm.py --model vssksn/intellicredit-mistral-7b-grpo \
                   --env-url http://localhost:7860 \
                   --out grpo_results.json
```

### Progressive Task Benchmarks

| Task | Difficulty | Key Challenge | Rule-Based Baseline |
|---|---|---|---|
| `task1` | 🟢 Easy | Hard rule compliance on clean profiles | 0.389 |
| `task2` | 🟡 Medium | Forensic alert detection (YELLOW/RED) | 0.325 |
| `task3` | 🔴 Hard | Macro shocks + missing data (-1.0 sentinels) | 0.288 |
| `task4` | 🔥 Expert | Hard-rule violations + repeat applicants | 0.265 |
| `task5` | ⚡ Master | Full: CRAR limits + cascading NPAs + 5 audits | 0.251 |

---

## What We Learned Building This

### Lesson 1: Sparse Rewards Are Harder to Design Than Dense Rewards

It is tempting to give the agent feedback at every step. Dense rewards make training faster. But in a credit environment, dense rewards create a specific failure mode: the agent optimizes individual decisions rather than portfolio health over time.

The delayed maturity event system — loan approved at step 5, maturity check at step 30 — required significant engineering: the WorldState event queue, maturity probability computation, and correct reward attribution to the original approval decision. That engineering difficulty correlates directly with what the environment teaches. The hardest features to build are the most important. If a feature is easy to implement, it probably isn't teaching anything new.

### Lesson 2: The Action Parser Is the Most Underrated Component

Before we built the action parser, the training loop was conceptually complete — and entirely non-functional. Without reliable conversion from free-text LLM output to structured environment actions, nothing downstream works.

The parser handles four cases in strict priority: explicit `submit_decision()` calls, recognized tool call patterns, keyword fallback (scanning for APPROVE/CONDITIONAL/REJECT in free text), and default-to-REJECT when nothing is parseable. The fallback logic alone required more iteration than any other component in the codebase.

The lesson is general: in LLM + environment systems, the interface between model and environment is where most failures occur. Design it first. Test it exhaustively before touching anything else.

### Lesson 3: Multi-Agent Pressure Produces Better Policies

When we tested the agent against a static environment — no borrower reapplication, no regulator audits — it developed a reasonably good policy. When we introduced BorrowerAgent and RegulatorAgent, the policy initially collapsed. Then it recovered to a significantly higher level than the static environment ever achieved.

The adversarial pressure forced generalization. The model could not memorize specific application patterns because the same company would return with different surface numbers. It had to learn underlying risk signals. This is the core insight: an RL agent in a non-adversarial environment learns the training distribution. An RL agent under adversarial pressure learns the underlying structure.

### Lesson 4: Reward Hacking Is Real and Specific — And Requires Specific Defenses

In early training runs, the model discovered that always outputting REJECT scored reasonably well — no NPAs, no concentration violations, no hard rule penalties. The solution was `reward_portfolio`: explicitly penalizing REJECT when capital utilization fell too low. This created genuine tension that could not be resolved by any pure strategy.

The general principle: for every pure strategy (always approve, always reject, always conditional), at least one reward component must penalize it. If any pure strategy is unpenalized, GRPO will find it.

---

## Limitations and What We Are Working On

We believe in honest assessment of what IntelliCredit-X does not yet handle:

**Synthetic Data:** All applications are generated from statistical distributions calibrated on sector profiles. The correlations are realistic but not derived from actual loan performance data. A model trained here would require significant calibration against historical loan books before production deployment.

**Single Regulatory Framework:** The hard rules (RBI mandates, Basel III as implemented in India, MSME classifications) are specific to the Indian credit market. Adapting to EU, US, or Southeast Asian frameworks would require reworking the hard rule system.

**Tool Response Realism:** The investigation tools return structured data derived from the synthetic application. A production system would call real external APIs — CIBIL, MCA21 portal, GST network, court record systems. Those integration complexities and latencies are out of scope for this environment but represent the obvious engineering path forward.

**Evaluation Depth:** Current benchmark scores show aggregate improvement at the episode level. We do not yet have fine-grained analysis of which specific application archetypes the fine-tuned model handles better or worse — which sectors, which fraud patterns, which macro conditions. That granular analysis requires a larger evaluation study.

---

## Resources

| Resource | Link | Description |
|---|---|---|
| 🤗 Live Environment | [HF Space](https://huggingface.co/spaces/vssksn/intellicredit-openenv) | Interactive API + Swagger docs |
| 🤗 Training Dataset | [vssksn/intellicredit-grpo-dataset](https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset) | 2,000 GRPO prompts across 5 task levels |
| 🤗 Fine-Tuned Model | [vssksn/intellicredit-mistral-7b-grpo](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo) | **Post-trained on live environment** (2-stage GRPO) |
| 💻 GitHub Repository | [1919-14/intellicredit-openenv](https://github.com/1919-14/intellicredit-openenv) | Full source code, MIT License |
| 📖 API Docs | [Swagger UI](https://vssksn-intellicredit-openenv.hf.space/docs) | Interactive endpoint documentation |
| 📓 Stage 1 Notebook | [Colab — Offline GRPO](https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing) | Mistral-7B + Unsloth, ~45 min, A100 |
| 🌍 Stage 2 Notebook | [colab_online_grpo.ipynb (Colab)](https://colab.research.google.com/github/1919-14/intellicredit-openenv/blob/main/training/colab_online_grpo.ipynb) | Online GRPO — live env, real rewards |
| 📊 Training Curves | [Figure 1](#what-the-training-curves-tell-us) | GRPO v2 training metrics |
| 📊 Results Chart | [Figure 2](#results-before-vs-after-grpo) | Full evaluation comparison |

---

## Citation

```bibtex
@article{intellicredit2025,
  title     = {IntelliCredit-X: A Multi-Agent Constrained MDP for MSME Credit
               Appraisal with GRPO Fine-Tuning},
  author    = {Narayana, V S S K Sai and Jaiswal, Sujeet},
  year      = {2026},
  note      = {Meta × Hugging Face OpenEnv Hackathon},
  url       = {https://huggingface.co/spaces/vssksn/intellicredit-openenv}
}
```

---

## Acknowledgements

Built for the **Meta × Hugging Face OpenEnv Hackathon 2026** by **V S S K Sai Narayana** and **Sujeet Jaiswal**.

Special thanks to the **OpenEnv framework** for providing the standardized RL environment interface that made this submission possible, to **Unsloth** for making 4-bit QLoRA training on a single A100 fast enough to run the full 3-stage curriculum in 45 minutes, and to the **Indian banking professionals** who validated the problem formulation and confirmed that the hard rules reflect real RBI regulatory requirements.

IntelliCredit-X is 100% open source. Whether you are a researcher studying credit risk and RL, a fintech building automated underwriting pipelines, or a student learning about multi-agent systems — fork it, train it on your data, build your own credit AI.

---

*The credit officer across the table from that MSME founder in Surat is working under time pressure, managing a large application queue, reasoning from fragmented data across disconnected systems. An AI system that has learned to investigate, reason, and comply — trained on millions of simulated decisions with real regulatory consequences — could make that table a more reliable place to sit.*

*That is why we built this. And that is why the environment is open source.*

---
*IntelliCredit-X — Multi-Agent RL Environment for MSME Credit Appraisal*
*V S S K Sai Narayana & Sujeet Jaiswal | Meta × Hugging Face OpenEnv Hackathon 2026*
*MIT License — [GitHub](https://github.com/1919-14/intellicredit-openenv)*
