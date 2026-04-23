---
title: IntelliCredit CreditAppraisal v2
emoji: 🏦
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🏦 IntelliCredit-CreditAppraisal-v2.0

![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blueviolet)
![Version](https://img.shields.io/badge/version-2.0-orange)
![Reinforcement Learning](https://img.shields.io/badge/AI-GRPO%20%2B%20RL-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Live Links:**
- 🚀 **Hugging Face Space**: [vssksn/intellicredit-openenv](https://huggingface.co/spaces/vssksn/intellicredit-openenv)
- 📖 **API Docs (Swagger)**: [Live Swagger UI](https://vssksn-intellicredit-openenv.hf.space/docs)
- 💻 **GitHub Repository**: [1919-14/intellicredit-openenv](https://github.com/1919-14/intellicredit-openenv)
- 📦 **GRPO Training Dataset**: [vssksn/intellicredit-grpo-dataset](https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset)

**IntelliCredit v2** is a **Constrained Multi-Agent MDP** for corporate MSME credit underwriting, built as an [OpenEnv](https://github.com/meta-pytorch/openenv) reinforcement learning environment for the **Meta × Hugging Face OpenEnv Hackathon**.

> **What's new in v2.0**: 50-step episodes (4× longer), 55D observation space (+10D memory), multi-agent simulation (BorrowerAgent + RegulatorAgent), real tool calling (get_financial_report, check_compliance_status, get_market_intelligence), GRPO training pipeline, and a self-improvement reflection system with persistent memory.

---

## 🎯 Core Motivation

The MSME lending sector in India processes millions of credit applications annually. Traditional underwriting is manual, slow, and prone to bias. Key challenges:

1. **Missing Data**: MSMEs often lack formal financial histories — 30–40% of features may be masked.
2. **Hidden Red Flags**: Fraud (circular GST trading, sham director entities, ITC mismatch) is buried in data.
3. **Macro Sensitivity**: Sector-specific shocks collapse loan books rapidly.
4. **Regulatory Constraints**: RBI mandates strict CRAR (>12.5%), NPA rate (<5%), and 6 hard-reject rules.

**Our Goal**: Create an enterprise-grade RL environment where an AI acts as a **Senior Credit Officer** and learns to balance **Yield** vs **Risk** vs **Regulatory Compliance** across a 50-step episode.

---

## ⚙️ How the Environment Works (v2.0)

An agent plays a **50-step Credit Committee Episode**:

```
Step T=1..50:
  1. Environment generates an MSME application (Anchor × Sector × Size × Tier)
  2. Agent sees 55D observation (application + portfolio + macro + memory)
  3. Agent may call up to 4 tools to investigate the application
  4. Agent submits: APPROVE (0) | CONDITIONAL (1) | REJECT (2)
  5. Reward is computed across 4 components (R1-R4)
  6. Approved loans join the Portfolio State
  7. Regulator audits occur at jittered steps (≈10/20/30/40/50)
  8. At episode end: NPA settlement + portfolio grading
```

### Multi-Agent Simulation
- **BorrowerAgent**: Strategically improves surface metrics on repeat applications while keeping true PD constant. Tests whether the agent can detect manipulation.
- **RegulatorAgent**: Triggers audits at jittered steps. Audit violations compound penalties.

---

## 🛑 Regulatory Rules (6 Hard Rules)

| Rule | Condition | Action |
|------|-----------|--------|
| HR-01 | DSCR < 1.0 | Mandatory REJECT |
| HR-02 | Director disqualified (DIN < 0.1) | Mandatory REJECT |
| HR-03 | RED forensic alert present | Mandatory REJECT |
| HR-04 | Cheque bounce rate > 25% | Mandatory REJECT |
| HR-05 | GST compliance < 40% | Mandatory REJECT |
| HR-06 | Severe adverse media (> 0.80) | Mandatory REJECT |

### Portfolio Constraints
| Constraint | Threshold | Consequence |
|------------|-----------|-------------|
| CRAR | > 12.5% | Episode termination if breached |
| NPA Rate | < 5% | Episode termination if breached |
| Sector Concentration | < 30% | -1.0 penalty |
| Single Borrower | < 15% | -0.5 penalty |

---

## 👁️ Observation Space (55D)

The observation space is a **55-dimensional continuous vector**, bounded `[-1.0, 1.0]`.
*(Note: `-1.0` is used as a sentinel for "missing/masked data" — teaching the agent uncertainty.)*

### 1. `application_features` (25-dim)
| Category | Variables |
|:---|:---|
| **Financials** | DSCR, Current Ratio, Debt-to-Equity, EBITDA Margin, Collateral Coverage, RONW |
| **Banking** | OD Utilisation, CC Volatility, Cheque Bounce Rate, Working Capital Cycle |
| **GST / Fraud** | GST CAGR, GST 2A-3B Gap, Related Party Txns, Circular Trading, ITC Mismatch |
| **Governance** | Litigation Count, MCA Charges, Adverse Sentiment, DIN Score, GST Alignment |

### 2. `portfolio_state` (10-dim)
Capital deployed, remaining capital, NPA rate, CRAR, provisioning coverage, sector flags.

### 3. `macro_state` (5-dim)
Systemic stress, stressed sector flag, GDP growth, inflation, credit cycle phase.

### 4. `alert_state` (5-dim)
Running tally of RED/YELLOW alerts seen during the episode.

### 5. `memory_features` (10-dim) *(New in v2)*
Cross-episode learned features: repeat applicant flag, historical PD pattern, borrower manipulation score.

---

## 🕹️ Action Space

**Discrete(3)** — plus optional tool calls before the final decision:

| Action | Decision | Business Consequence |
|:---:|:---|:---|
| **`0`** | **APPROVE** | Full yield (+0.7–+1.0). Absorbs full default risk. |
| **`1`** | **CONDITIONAL** | Partial yield (+0.3–+0.6). Forces covenants, lowers risk. |
| **`2`** | **REJECT** | Zero yield. Eliminates default risk. Required for toxic profiles. |

### 🔧 Tool Calling (New in v2)
Before deciding, the agent may invoke up to 4 tools:

```
get_financial_report("company_name")
  → 3-year revenue trend, EBITDA margin, debt schedule, auditor remarks

check_compliance_status("company_name")
  → MCA filings, GST returns, DIN status, NCLT cases, prior defaults

get_market_intelligence("sector_name")
  → Sector NPA peer rate, RBI advisory, macro headwinds/tailwinds
```

---

## 📈 Reward System (v2.0)

| Component | Weight | Range | Description |
|-----------|--------|-------|-------------|
| R1: Decision Correctness | 40% | [-2.0, +1.0] | PD-based decision quality |
| R2: Hard Rule Compliance | 30% | [-2.0, +0.5] | RBI hard rule adherence |
| R3: Format Compliance | 15% | [-0.3, +0.3] | Output format quality |
| R4: Portfolio Awareness | 15% | [-0.8, +0.3] | Portfolio state sensitivity |

### Anti-Gaming Mechanisms
- **Audit Jitter**: Regulator audits occur at random offsets (±3 steps) — prevents reward timing exploitation
- **Parser Hardening**: Last-decision-wins rule, reasoning length penalty for padding
- **Borrower Manipulation**: BorrowerAgent improves surface metrics without changing true PD

---

## 📊 Baseline Evaluation Results (v2.0)

Tested with a deterministic RuleBasedAgent across 25 episodes (5 per task):

| Task | Difficulty | Avg Score | Avg Accuracy | NPA Rate |
|------|------------|-----------|--------------|----------|
| task1 | Easy | 0.389 | 77.9% | 4.8% |
| task2 | Medium-Forensic | 0.325 | 66.6% | 8.9% |
| task3 | Hard-MacroShock | 0.288 | 81.5% | 20.2% |
| task4 | Expert-HardRules | 0.265 | 85.9% | 26.7% |
| task5 | Master-Full | 0.251 | 77.8% | 6.7% |

### GRPO Training (Expected Post-Training)
| Metric | Baseline | Post-GRPO | Delta |
|--------|----------|-----------|-------|
| Avg Score | 0.30 | ~0.70 | +133% |
| Hard Rule Violations | 20.7% | ~2% | −90% |
| NPA Rate | 13.4% | ~3% | −78% |

---

## 🧠 Self-Improvement System (Phase 5)

The reflection module enables **cross-episode learning without weight updates**:

```
Episode N → Analyze failures → Extract lesson → Store in memory_bank.json
Episode N+1 → Inject top 5 lessons into system prompt → Better decisions
```

- Lessons are severity-sorted (critical → major → minor)
- Up to 20 lessons stored (FIFO eviction)
- Deduplicated by semantic similarity

---

## 🎮 Testing Live on Hugging Face Spaces

Play as the credit officer directly via Swagger UI:

**1. Reset an episode:**
```json
POST /reset
{ "episode_id": "my-session-1", "seed": 42, "task_id": "task2" }
```

**2. Make a decision (simple):**
```json
POST /step
{
  "episode_id": "my-session-1",
  "action": { "decision": 1 },
  "timeout_s": 30
}
```

**3. Make a decision with tool call:**
```json
POST /step
{
  "episode_id": "my-session-1",
  "action": { "raw_llm_output": "check_compliance_status('Bharat Engineering')" },
  "timeout_s": 30
}
```

Repeat up to 50 times to complete the episode. Check `/grade` for your final score.

---

## 💻 Local Setup

```bash
git clone https://github.com/1919-14/intellicredit-openenv.git
cd intellicredit-openenv

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# → http://localhost:7860/docs
```

### Run Evaluation
```bash
# Baseline (rule-based agent)
python evaluation/evaluate.py --mode baseline --episodes 20

# Reflection module (cross-episode learning)
python evaluation/evaluate.py --mode reflection --n-episodes 30

# Generate comparison table + charts
python evaluation/compare.py --mode all
```

### Run GRPO Training
```bash
# Generate training dataset (2000 prompts)
python training/generate_dataset.py

# Train — Stage 1 (easy tasks only, dry run to validate)
python training/train_grpo.py --stage 1 --dry-run

# Train — All 3 stages
python training/train_grpo.py --stage all --model llama3_8b

# Export merged model
python training/train_grpo.py --export --push
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t intellicredit-v2 .

# Run
docker run -p 7860:7860 intellicredit-v2

# With HF token (for LLM inference)
docker run -p 7860:7860 -e HF_TOKEN="your-token" intellicredit-v2
```

### Resource Requirements (Environment Server Only)
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 vCPUs | 4 vCPUs |
| RAM | 2 GB | 4 GB |
| Disk | 1 GB | 2 GB |
| Port | 7860 | 7860 |

> ⚠️ **GRPO Training** requires a separate GPU machine (A100/H100 recommended). The HF Space only runs the environment server — no GPU needed.

---

## 🔄 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token (for LLM inference) | *Required for inference.py* |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | LLM model for inference | `meta-llama/Llama-3.3-70B-Instruct` |

---

## 🏆 Task Descriptions

| Task | Steps | Difficulty | Key Challenge |
|---|:---:|---|---|
| `task1` | 50 | Easy | Clean profiles, no shocks |
| `task2` | 50 | Medium | Forensic alerts (YELLOW/RED) |
| `task3` | 50 | Hard | Macro shocks + missing data |
| `task4` | 50 | Expert | Hard-rule violations + repeat applicants |
| `task5` | 50 | Master | Full: CRAR limits + sector concentration |

---

## 📁 Project Structure

```
intellicredit-openenv/
├── server/
│   ├── app.py              # FastAPI server (reset, step, grade)
│   ├── intellicredit_env.py # v2 environment core (50-step, multi-agent)
│   ├── dataset.py          # Application generator (Anchor × Sector × Size)
│   ├── reward.py           # R1-R4 reward engine + PortfolioState
│   ├── action_parser.py    # LLM output → tool/decision parser
│   ├── tool_executor.py    # Tool execution (financial, compliance, market)
│   ├── agent_loop.py       # Agent orchestrator + prompt injection
│   └── reflection.py       # Self-improvement + memory bank
├── training/
│   ├── generate_dataset.py # 2000-prompt GRPO dataset generator
│   ├── grpo_rewards.py     # 4 GRPO reward functions
│   └── train_grpo.py       # 3-stage curriculum GRPO training
├── evaluation/
│   ├── evaluate.py         # Multi-mode evaluation engine
│   └── compare.py          # Comparison tables + reward curves
├── models.py               # Pydantic schemas (55D obs, action)
├── inference.py            # LLM inference wrapper
├── Dockerfile              # HF Spaces deployment
└── requirements.txt        # Python dependencies
```

---

## 📚 Citation

```bibtex
@article{intellicredit2025,
  title={IntelliCredit v2: A Constrained Multi-Agent MDP for MSME Credit Appraisal with GRPO Fine-Tuning},
  author={V S S K Sai Narayana, Sujeet Jaiswal},
  year={2025},
  note={OpenEnv Hackathon Submission — Meta × Hugging Face}
}
```

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.
