# 📋 IntelliCredit-X — Project Summary

> *The complete technical story behind building a multi-agent RL environment for MSME credit underwriting and GRPO fine-tuning of Mistral-7B*

**Authors:** V S S K Sai Narayana & Sujeet Jaiswal  
**Event:** Meta × Hugging Face OpenEnv Hackathon 2025  
**Status:** ✅ All 8 Phases Complete — Production Deployed  
**Published:** April 25, 2026 | **License:** MIT

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🤗 Live Environment | [huggingface.co/spaces/vssksn/intellicredit-openenv](https://huggingface.co/spaces/vssksn/intellicredit-openenv) |
| 🤗 GRPO Model | [vssksn/intellicredit-mistral-7b-grpo](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo) |
| 🤗 Training Dataset | [vssksn/intellicredit-grpo-dataset](https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset) |
| 📓 Colab Training Notebook | [Open in Colab](https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing) |
| 💻 GitHub Repository | [1919-14/intellicredit-openenv](https://github.com/1919-14/intellicredit-openenv) |
| 📖 API Docs | [Swagger UI](https://vssksn-intellicredit-openenv.hf.space/docs) |
| 📝 Full Blog Post | [docs/blog.md](./docs/blog.md) |

---

## 🎯 The Problem

The MSME lending sector in India processes **100,000+ loan applications daily**. The traditional solution — hire more senior bankers — doesn't scale:

- A senior officer reviews **~16 applications/day** at 30 minutes each
- Officers cost ₹40–80 lakhs annually
- Human fatigue causes **inconsistent rule application**
- **12–15% annual default rates** due to missed fraud signals buried in data
- No explainable audit trail — decisions driven by "gut feeling"

**Our approach:** Build a reinforcement learning training environment where an AI learns to reason like the best credit officers — gathering evidence, detecting hidden fraud, respecting hard regulatory constraints, and managing portfolio risk across time.

---

## 🏗️ What IntelliCredit-X Is

IntelliCredit-X is a **Constrained Multi-Agent MDP** (Markov Decision Process) built as an OpenEnv-compliant environment. An LLM agent acts as a **Senior Credit Officer** across a 50-step episode representing a full credit committee lifecycle.

**What makes this fundamentally different from a classifier:**

| Challenge | How IntelliCredit-X Models It |
|-----------|-------------------------------|
| Multi-step investigation | Agent calls tools before deciding — not a one-shot classification |
| Delayed consequences | Loan approved at step 5 may default at step 30 — reward arrives 25 steps later |
| Adversarial borrowers | BorrowerAgent improves surface metrics after rejection; hidden PD unchanged |
| Regulatory enforcement | RegulatorAgent audits portfolio every ~10 steps; 3 failures = shutdown |
| Missing data | −1.0 sentinel = masked feature; data absence itself is a risk signal |
| Regulatory hard rules | 6 RBI mandates that auto-reject and penalize regardless of model choice |

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   INTELLICREDIT-X SYSTEM OVERVIEW                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   BORROWER AGENT (Adversarial)      REGULATOR AGENT (Enforcer)     │
│   ┌─────────────────────────┐       ┌─────────────────────────┐    │
│   │ Presents applications   │       │ Audits at ≈10/20/30/    │    │
│   │ Hides real risk behind  │       │ 40/50 steps (±1 jitter) │    │
│   │ improved surface numbers│       │ NPA/CRAR/concentration  │    │
│   │ Reapplies up to 3×      │       │ Shutdown after 3 fails  │    │
│   └────────────┬────────────┘       └────────────┬────────────┘    │
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
│         │       CREDIT OFFICER AGENT (LLM)            │             │
│         │                                             │             │
│         │  Sees 55D observation as a text prompt      │             │
│         │  Calls 3 investigation tools (max 4/step)  │             │
│         │  Writes reasoning + submits decision        │             │
│         │                                             │             │
│         │  Fine-tuned via GRPO (TRL + Unsloth)       │             │
│         │  Self-improves via Reflection Module        │             │
│         └────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 👁️ Observation Space — 55 Dimensions

| Group | Dims | Description |
|-------|------|-------------|
| Application Features | 0–24 | DSCR, Current Ratio, GST gap, DIN score, cheque bounce rate, circular trading flag, ITC mismatch, RONW, collateral… |
| Portfolio State | 25–34 | Capital deployed, NPA rate, CRAR, provisioning coverage, sector concentration flags |
| Macro State | 35–39 | Systemic stress, GDP growth, inflation, credit cycle phase, stressed sector flag |
| Alert State | 40–44 | Running RED/YELLOW alert tallies for episode |
| **Memory Features** *(v2)* | **45–54** | Rolling NPA, approval rate, sector max concentration, macro trend, borrower persistence, audit risk, capital buffer, reflection count, episode progress, world model confidence |

**Key dimension:** `Dim 49: borrower_persistence_score` — `0.0` = first attempt, `0.5` = second, `1.0` = third attempt. A value of 1.0 is the environment's strongest manipulation signal.

---

## 🕹️ Action Space + Tool System

**Actions:** `0 = APPROVE` | `1 = CONDITIONAL` | `2 = REJECT`

Plus optional tool calls before each decision (max 4 per step):

| Tool | Returns | Best Used When |
|------|---------|----------------|
| `get_financial_report(company_id)` | 3yr revenue, EBITDA, auditor remarks, related-party transactions | Borderline financials, need trend data |
| `check_compliance_status(company_id)` | DIN status, NCLT cases, GST filings, CIBIL, prior defaults | RED alert present or low governance score |
| `get_market_intelligence(sector)` | Sector stress, RBI advisory, portfolio exposure, peer NPA rate | Approaching 30% concentration limit |
| `submit_decision(action, reasoning)` | Finalizes step (reasoning ≥ 50 chars mandatory) | After investigation complete |

---

## 🛑 Hard Rules (Non-Negotiable RBI Mandates)

| Rule | Trigger Condition | Outcome |
|------|-------------------|---------|
| HR-01 | DSCR < 1.0 | Auto-REJECT + −2.0 penalty |
| HR-02 | Director DIN < 0.1 (disqualified) | Auto-REJECT + −2.0 penalty |
| HR-03 | RED forensic alert present | Auto-REJECT + −2.0 penalty |
| HR-04 | Cheque bounce rate > 25% | Auto-REJECT + −2.0 penalty |
| HR-05 | GST compliance < 40% | Auto-REJECT + −2.0 penalty |
| HR-06 | Adverse media score > 0.80 | Auto-REJECT + −2.0 penalty |

---

## 📈 Reward System

### Per-Step Components

| Component | Weight | Range | Purpose |
|-----------|--------|-------|---------|
| R1: Decision Correctness | 40% | [−2.0, +1.0] | PD-based: low PD+APPROVE=+1.0; high PD+APPROVE=−2.0 |
| R2: Hard Rule Compliance | 30% | [−2.0, +0.5] | HR+REJECT=+0.5; HR+APPROVE=−2.0 |
| R3: Format Compliance | 15% | [−0.3, +0.3] | `submit_decision()` used=+0.3; parse failure=−0.3 |
| R4: Portfolio Awareness | 15% | [−0.8, +0.3] | NPA>8%+risky approve=−0.5; healthy approve=+0.2 |

### Delayed + Event Rewards

| Event | Reward | When It Fires |
|-------|--------|---------------|
| Loan fully repaid | **+10.0** | T+10 to T+30 after approval |
| Partial default (recovery ≥50%) | **−5.0** | T+10 to T+30 after approval |
| Full default (recovery <50%) | **−15.0 × (1−recovery)** | T+10 to T+30 after approval |
| Clean audit (all checks pass) | **+2.0** | Each audit step |
| Audit violation | **−8.0 per violation** | Each audit step |
| Capital breach (CRAR <12.5%) | **−15.0** | Each audit step |
| 3rd consecutive audit fail | **−50.0 + shutdown** | Terminal |
| Survival bonus (CRAR >12.5%) | **+0.05–+0.10** | Every 10 steps |

### Settlement Score (Step 50)

```
score = 0.30 × portfolio_yield
      + 0.30 × (1 − npa_rate / 0.05)
      + 0.20 × regulatory_compliance
      + 0.20 × capital_utilization

Range: [−1.0, +5.0]   |   Good episode ≥ 3.0
```

---

## 🤖 GRPO Training Pipeline

### Model & Config

| Parameter | Value |
|-----------|-------|
| Base Model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Quantization | 4-bit QLoRA via Unsloth |
| LoRA Rank | 16, targeting q/k/v/o projections |
| Sequence Length | 2048 tokens (strictly enforced) |
| Generations per prompt | 8 |
| Effective batch size | 16 (batch=2, grad_accum=8) |
| KL Beta | 0.001 |
| Hardware | A100 80GB — ~45 minutes total |

### 3-Stage Curriculum

| Stage | Training Data | LR | Temperature | Goal |
|-------|-------------|-----|-------------|------|
| Stage 0 (SFT Warmup) | Mixed tasks | 5e-5 | — | Bootstrap `submit_decision()` format compliance |
| Stage 1 | task1 (Easy) | 5e-6 | 0.9 | Hard rule recognition on clean profiles |
| Stage 2 | task1 + task2 | 5e-6 | 0.9 | Forensic alert detection + tool call initiation |
| Stage 3 | All 5 tasks | 2e-6 | 0.8 | Long-horizon portfolio management under macro shocks |

### Training Dataset

- **2,000 prompts** — 400 per task level (task1–task5)
- Each prompt ~2,400 characters (role + tools + rules + application + portfolio + macro)
- Ground truth metadata: hidden PD, optimal action, hard rules, forensic alerts, CRAR, NPA
- Distribution: **47.2%** hard rules triggered | **28.1%** RED forensic alerts
- Dataset: [vssksn/intellicredit-grpo-dataset](https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset)

### Critical Bugs Fixed During Training

| Bug | Root Cause | Fix Applied |
|-----|-----------|-------------|
| CUDA Index Out of Bounds | Unsloth pads vocab 32768→32832; OOB token IDs caused assertions | Clamp all IDs to `vocab_size−1`; add `valid_mask` |
| Shape mismatch on logits | `full_ids` exceeded 2048 before forward pass; logits truncated | Enforce `full_ids = full_ids[:, :MAX_SEQ_LEN]` before forward |
| Flat KL divergence | `clamp(min=0)` made KL=0 when new policy was more confident | Changed to `abs()` for symmetric KL |
| Zero-LP episodes | Long prompts consumed full context; zero completion tokens | `continue` when `sum(log_probs) == 0` |

---

## 🪞 Self-Improvement Reflection System

The Reflection Module enables **cross-episode learning without weight updates** by injecting structured lessons into the next episode's system prompt.

### How It Works

```
Episode N → Analyze all steps where reward < 0
          → Extract lessons by failure type (6 categories)
          → Store in memory_bank.json (max 20, FIFO eviction)
Episode N+1 → Inject top 5 lessons into system prompt Layer 3
            → Agent makes better decisions without retraining
```

### 6 Lesson Categories

| Trigger | Lesson Format | Severity |
|---------|---------------|----------|
| Hard Rule Violation | `RULE: When [condition], always REJECT` | Critical |
| Delayed Default | `CAUTION: Loans with [pattern] defaulted at step X` | High |
| Audit Failure | `COMPLIANCE: Audit failed due to [metric]` | High |
| Borrower Manipulation | `FRAUD: 3rd-attempt applicant with [pattern] defaulted` | Critical |
| Macro Shock Loss | `MACRO: During [state], be conservative with [sector]` | Medium |
| Portfolio Overexposure | `PORTFOLIO: NPA rate reached X%. Increase rejections.` | High |

**Measured result:** Base model (no fine-tuning) improved average episode score from **0.22 → 0.55** across 30 episodes using only reflection — a **+150% improvement without changing a single weight**.

---

## 📊 Results — GRPO Fine-Tuned vs. Base Model

![Baseline vs Post-Training GRPO](./docs/assets/comparison.png)

| Task | Metric | Base Mistral-7B | GRPO Model | Delta |
|------|--------|----------------|-----------|-------|
| Task 1 (Easy) | Score | 0.900 | **0.955** | **+0.055 ✅** |
| | Accuracy | 80.0% | **86.7%** | **+6.7% ✅** |
| | Total Reward | 2.904 | 3.272 | +0.368 ✅ |
| | Capital Util | 40.0% | **60.0%** | **+20.0% ✅** |
| Task 2 (Medium) | Score | 1.000 | 1.000 | ceiling ✅ |
| | Total Reward | 10.305 | **10.584** | **+0.279 ✅** |
| | Capital Util | 25.0% | **29.2%** | **+4.2% ✅** |
| Task 3 (Hard) | Score | 0.767 | **0.833** | **+0.067 ✅** |
| | Total Reward | 0.215 | **2.491** | **+2.276 ✅ (10×!)** |
| | Accuracy | 58.3% | **66.7%** | **+8.3% ✅** |
| | **NPA Rate** | **16.7%** | **8.3%** | **−8.3% ✅ (halved!)** |
| | Capital Util | 16.7% | **25.0%** | **+8.3% ✅** |

**Zero regressions across all 24 metric-task combinations.**

---

## 🧠 Training Curves

![GRPO Training Curves](./docs/assets/training_curves.png)

| Panel | What It Shows |
|-------|---------------|
| GRPO Loss (red) | Controlled upward drift 0→0.05 — policy meaningfully diverging from base |
| Mean Reward (blue) | Starts −2.0, crosses zero by step 10, stabilizes +0.5 to +1.0 |
| KL Divergence (purple) | Grows to 0.04–0.08 — new behaviors learned, base capabilities preserved |
| `submit_pct` (teal) | Format compliance 0%→40–65% — model acquires the task's vocabulary |

The biggest reward jump happens at the Stage 1→2 transition (when forensic alerts first appear), suggesting forensic pattern recognition was the primary learning bottleneck.

---

## 🗺️ Project Phase Completion

| Phase | Name | Status |
|-------|------|--------|
| Phase 0 | Strategic Alignment & Decisions | ✅ Complete |
| Phase 1 | Environment Upgrade (v1 → v2: 50-step, 55D, multi-agent) | ✅ Complete |
| Phase 2 | Multi-Agent System Design (Borrower + Regulator agents) | ✅ Complete |
| Phase 3 | Tool Calling System (3 tools + parser + anti-hacking) | ✅ Complete |
| Phase 4 | Reward System Redesign (sparse/delayed + 4 components) | ✅ Complete |
| Phase 5 | Self-Improvement & Reflection System | ✅ Complete |
| Phase 6 | GRPO Training Pipeline (Unsloth + TRL, 3-stage curriculum) | ✅ Complete |
| Phase 7 | Evaluation & Proof Generation | ✅ Complete |
| Phase 8 | GRPO Stabilization + Real Model Evaluation | ✅ Complete |

---

## 📁 File Inventory

| File | Role | Phase |
|------|------|-------|
| `server/app.py` | FastAPI server — /reset, /step, /info, /health | P0 |
| `server/intellicredit_env.py` | v2 core: WorldState, 50-step lifecycle, multi-agent | P1–P4 |
| `server/dataset.py` | Application generator (Anchor × Sector × Size × Tier) | P0 |
| `server/reward.py` | R1-R4 reward engine + settlement grader | P4 |
| `server/action_parser.py` | LLM text → tool call / decision parser (6 parse levels) | P3 |
| `server/tool_executor.py` | Read-only tool execution (financial, compliance, market) | P3 |
| `server/agent_loop.py` | Agent orchestrator + prompt injection | P3–P5 |
| `server/reflection.py` | Self-improvement + memory bank | P5 |
| `training/colab_grpo_3b_v2.py` | PRIMARY: Unsloth GRPO training script (A100, ~45 min) | P6/P8 |
| `training/generate_dataset.py` | 2,000-prompt GRPO dataset generator | P6 |
| `training/grpo_rewards.py` | 4 GRPO reward functions (R1-R4) | P6 |
| `training/train_grpo.py` | 3-stage curriculum pipeline | P6 |
| `evaluation/evaluate.py` | Multi-mode evaluation engine | P7 |
| `evaluation/compare.py` | Comparison tables + reward curves | P7 |
| `eval_llm.py` | LLM evaluation via HTTP (base vs trained) | P8 |
| `compare_results.py` | Bar chart comparison generator | P8 |
| `models.py` | Pydantic schemas (55D observation, action) | P1 |
| `inference.py` | LLM inference wrapper | P0 |
| `docs/blog.md` | Full technical blog post (~5,000 words) | P8 |
| `docs/assets/comparison.png` | Baseline vs GRPO results chart | P8 |
| `docs/assets/training_curves.png` | GRPO training curves (Mistral-7B, A100) | P8 |

**Total: ~10,000+ lines of implementation across 21 source files.**

---

## 📚 Citation

```bibtex
@article{intellicredit2025,
  title   = {IntelliCredit-X: A Multi-Agent Constrained MDP for MSME Credit
             Appraisal with GRPO Fine-Tuning},
  author  = {Narayana, V S S K Sai and Jaiswal, Sujeet},
  year    = {2025},
  note    = {Meta × Hugging Face OpenEnv Hackathon},
  url     = {https://huggingface.co/spaces/vssksn/intellicredit-openenv}
}
```

---

*IntelliCredit-X — Built by V S S K Sai Narayana & Sujeet Jaiswal | MIT License*