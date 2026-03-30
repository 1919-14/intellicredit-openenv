---
title: IntelliCredit CreditAppraisal v1
emoji: 🏦
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🏦 IntelliCredit-CreditAppraisal-v1

![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blueviolet)
![Reinforcement Learning](https://img.shields.io/badge/AI-Reinforcement_Learning-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Live Links:**
- 🚀 **Hugging Face Space**: [vssksn/intellicredit-openenv](https://huggingface.co/spaces/vssksn/intellicredit-openenv)
- 📖 **API Testing Documentation (Swagger)**: [Live Swagger UI](https://vssksn-intellicredit-openenv.hf.space/docs)
- 💻 **GitHub Repository**: [1919-14/intellicredit-openenv](https://github.com/1919-14/intellicredit-openenv)

**IntelliCredit** is a Constrained Multi-Objective MDP for corporate credit underwriting, built as an [OpenEnv](https://github.com/meta-pytorch/openenv) reinforcement learning environment for the Meta × Hugging Face OpenEnv Hackathon.

---

## 🎯 Core Motivation

The MSME (Micro, Small, and Medium Enterprises) lending sector is the backbone of developing economies like India. However, underwriting these loans is notoriously difficult due to:
1. **Missing Data**: MSMEs often lack formal financial histories.
2. **Hidden Red Flags**: Fraud (e.g., circular GST trading, director litigation) is deeply buried.
3. **Macro Sensitivity**: Vulnerable sectors collapse rapidly during economic shocks.

**Our Goal**: Create an enterprise-grade RL environment where an AI acts as a **Senior Credit Officer**. It must balance **Yield** (approving high-interest loans) against **Risk** (default penalties), while strictly adhering to real-world Banking Regulations like Basel III Capital to Risk-Weighted Assets Ratios (CRAR).

---

## ⚙️ How Our Environment Works

An agent plays out a 12-step **"Credit Committee"** episode. 

1. **Step Generation**: At each timestep (`T = 1..12`), the environment randomly generates an Anchor-based MSME application.
2. **Observation**: The agent sees 45 variables representing the application's financials, the bank's current portfolio status, and global macroeconomic indicators.
3. **Action**: The agent casts a decision: **APPROVE (0)**, **CONDITIONAL (1)**, or **REJECT (2)**.
4. **Reward & Transition**: The environment immediately credits base yield. However, approved loans join the "Portfolio State."
5. **Delayed Consequences**: At $T=7$, an economic shock may trigger. Loans approved at $T=2$ might suddenly default, slapping the agent with a massive delayed `-2.0` NPA penalty in the current timestep.

---

## 🛑 Regulatory Constraints & Hard Rules

Unlike standard purely reward-based environments, IntelliCredit mimics strict financial compliance.

### Soft Constraints (Portfolio Health)
- **CRAR (Capital to Risk-Weighted Assets Ratio)**: The bank must maintain CRAR > 12.5%. If the agent approves too many risky loans and capital runs dry, the bank violates Basel III limits.
- **Single Borrower Limit**: A single sector/tier cannot compose >15% of the total loan book.

### Hard Compliance Rules (Immediate Terminations)
If the agent attempts to `APPROVE` or `CONDITIONAL` a loan that violates anti-money laundering or severe risk rules, the bank's internal compliance engine intercepts the action. The loan is rejected, and the agent is struck with a severe penalty (`-1.0` to `-1.5`).
- **HR-01**: DSCR < 1.0 (Insufficient cash flow)
- **HR-02**: Director Disqualified (DIN score < 0.1)
- **HR-03**: RED Forensic Alert Present
- **HR-04**: Cheque bounce rate > 25%

---

## 👁️ Parameter Explanation (Observation Space)

The observation space is a **45-dimensional continuous vector**, bounded `[-1.0, 1.0]`. 
*(Note: $-1.0$ is heavily utilized as a sentinel value for "Missing/Masked Data", teaching the agent uncertainty).*

### 1. `application_features` (25-dim)
Raw features of the MSME requesting a loan.
| Category | Variables |
|:---|:---|
| **Financials** | DSCR Proxy, Current Ratio, Debt-to-Equity, EBITDA Margin, Collateral Coverage, Return on Net Worth |
| **Banking Behavior** | OD Utilisation, CC Volatility, Cheque Bounce Frequency, Working Capital Cycle |
| **GST / Fraud** | GST Turnover CAGR, GST 2A vs 3B Gap, Related Party Txns, Circular Trading, ITC Mismatch Flag |
| **Governance** | Promoter Litigation Count, MCA Charge Count, Adverse News Sentiment |

### 2. `portfolio_state` (10-dim)
The ongoing health of the bank. (Capital Deployed, Remaining Capital, True NPA Rate, Provisioning Coverage, Real-time CRAR).

### 3. `macro_state` (5-dim)
Simulated global economy variables. (Systemic Stress, Stressed Sector Flag, GDP Growth, Inflation Rate, Credit Cycle).

### 4. `alert_state` (5-dim)
Aggregated, running tally of recent red-flags seen in the current episode.

---

## 🕹️ Action Space

The action space is **Discrete(3)**.

| Action (`int`) | Decision | Business Consequence |
|:---:|:---|:---|
| **`0`** | **APPROVE** | Maximize expected yield (+0.7 to +1.0). Absorbs full default risk. |
| **`1`** | **CONDITIONAL** | Modest expected yield (+0.3 to +0.6). Forces collateral covenants; lowers default risk. |
| **`2`** | **REJECT** | Zero Yield. Eliminates all default risk. Required for toxic/fraudulent profiles. |

---

## 📈 Learning Curve (Baseline PPO Agent)

To prove solvability, we trained an SB3 `PPO` baseline agent for 500,000 timesteps.

**Training Convergence:**
- **At 0 - 50k Steps**: The agent acts randomly. It frequently triggers Hard Rule violations (-1.5 penalties) and quickly bankrupts the portfolio's CRAR constraints. *Average Reward = -1.20*
- **At 200k Steps**: The agent begins recognizing the `-1.0` Sentinel Data indicators for missing documents and learns to `REJECT` applications with `Director Disqualified` flags. *Average Reward = +1.45*
- **At 500k Steps**: The agent fully balances risk vs yield. It correctly utilizes `CONDITIONAL` approvals for medium-risk sectors anticipating the $T=7$ macro-shocks.
- **Final Episode Score**: `~3.57` (Outperforming a randomized baseline by 400%).

---

## 🎮 Testing Live on Hugging Face Spaces

You can manually play as the AI agent directly via our deployed Swagger UI!

1. **Open the API Docs**: Navigate to [https://vssksn-intellicredit-openenv.hf.space/docs](https://vssksn-intellicredit-openenv.hf.space/docs)
2. **Start the Episode (`/reset`)**:
   - Open the `POST /reset` block and click "Try it out".
   - Set a Session ID (Required for state tracking!):
     ```json
     { "episode_id": "my-session-123", "seed": 42 }
     ```
   - Click **Execute**. The response will contain the massive 45-dim application data and a text summary.
3. **Make a Decision (`/step`)**:
   - Open the `POST /step` block and click "Try it out".
   - **Crucial**: Use the identical `episode_id`!
     ```json
     {
       "episode_id": "my-session-123",
       "action": { "decision": 1 },
       "timeout_s": 30
     }
     ```
   - Click **Execute**. The server will return your reward and generate the next company's data. Repeat 12 times to finish the game!

---

## 💻 Local Setup & Installation

### 1. Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

### 2. Installation
```bash
git clone https://github.com/1919-14/intellicredit-openenv.git
cd intellicredit-openenv

# Using uv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Run the Server
```bash
python server/app.py --port 7860
```
Navigate to `http://localhost:7860/docs` to interface visually.

---

## 🤖 LLM Evaluator Baseline

Instead of standard RL, you can plug this environment into any Generative AI model (GPT-4o, LLaMA-3) and evaluate if it natively understands credit dynamics:

```bash
# Uses Hugging Face's Free Serverless API
export HF_TOKEN="your-hf-token"
python inference.py
```

---

## 🏆 Tasks & Scoring

The environment contains 5 progressive tasks. The evaluator grades decisions using a multi-objective formula:

- **50%** — Accuracy (matching optimal algorithm decisions)
- **25%** — Hard Rule Compliance (0% if any mandatory reject is approved)
- **15%** — NPA Management (portfolio default rate)
- **10%** — Capital Utilization

| Task | Steps | Description |
|---|:---:|---|
| `task1` | 5 | Easy — Clean profiles, no macro shocks |
| `task2` | 8 | Medium — Forensic alerts present |
| `task3` | 12 | Hard — Macro shocks and missing data |
| `task4` | 12 | Expert — Regulatory hard-rule violations |
| `task5` | 12 | Master — Full constraints (CRAR, Sector limits) |

---

## 📜 License
MIT License.
