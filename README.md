---
title: IntelliCredit CreditAppraisal v1
emoji: 🏦
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# IntelliCredit-CreditAppraisal-v1

**IntelliCredit** is a Constrained Multi-Objective MDP for corporate credit underwriting, built as an [OpenEnv](https://github.com/meta-pytorch/openenv) reinforcement learning environment for the Meta × Hugging Face OpenEnv Hackathon.

## 📖 Environment Description

An AI agent acts as a Senior Credit Officer at an Indian MSME lending institution, reviewing 5-12 sequential credit applications per episode. Each decision impacts the bank's portfolio health, capital reserves, and regulatory compliance.

**Key Dynamics:**
- **Delayed Rewards**: Approving a risky loan at $T=2$ may yield short-term interest, but default later at $T=7$, resulting in a severe `-2.0` delayed NPA penalty.
- **Macro-Economic Shocks**: The environment simulates stress across 8 sectors. A shock can trigger sector-wide defaults midway through an episode.
- **Strict Compliance**: Hard regulatory rules (e.g., CIBIL defaults, forensic alerts) must trigger an immediate REJECT. Approving these results in severe penalties, while the compliance system automatically blocks the loan from entering the portfolio.
- **Single Borrower & CRAR Constraints**: The agent must maintain a minimum Capital to Risk-Weighted Assets Ratio (CRAR) > 12.5% and avoid overexposing the bank to a single borrower (> 15% of portfolio).

---

## 👁️ Observation Space

The observation space is a **45-dimensional continuous vector** constructed to provide the agent with both localized application data and global portfolio/macro context.

`observation.application_features` (25-dim)
Values are normalized to `[-1.0, 1.0]`. Missing data (e.g., unavailable GST filings) is represented by a strict `-1.0` sentinel value.
- **Financials**: DSCR, Current Ratio, Debt-to-Equity, EBITDA Margin, Collateral Coverage
- **Behavioral**: Cheque Bounce Frequency, OD Utilisation, CC Volatility
- **Compliance**: GST Turnover CAGR, GST 2A vs 3B Gap, Related Party Txns, Circular Trading
- **Governance**: Promoter Litigation Count, Adverse News Sentiment, Succession Risk

`observation.portfolio_state` (10-dim)
- Capital Deployed, Remaining Capital, NPA Rate, Provisioning Coverage, CRAR, Sector Concentration, Largest Single Exposure, Active Loans count, etc.

`observation.macro_state` (5-dim)
- Systemic Stress, Stressed Sector Flag, GDP Growth, Inflation Rate, Credit Cycle

`observation.alert_state` (5-dim)
- Portfolio-wide frequency of recent red flags: [CC Spike, Bounce Surge, GST Miss, Adverse Media, Credit Degradation].

*LLM Agents also receive an `observation.application_summary.text_summary` containing a human-readable prompt with dynamically hidden values where data is missing.*

---

## 🕹️ Action Space

The action space is **Discrete(3)**. At each timestep, the agent must evaluate the current application and make a credit decision.

| Action | Decision | Description |
|:---:|:---|:---|
| **`0`** | **APPROVE** | Fully fund the requested loan amount at standard interest rates. Risks delayed NPA if the applicant defaults. |
| **`1`** | **CONDITIONAL** | Approve the loan but mandate strict terms (e.g., higher interest, extra collateral). Reduces default probability but yields lower base reward. |
| **`2`** | **REJECT** | Decline the loan. Zero risk of default, but consumes 1 timestep without generating yield. Required for hard-rule compliance. |

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/1919-14/intellicredit-openenv.git
cd intellicredit-openenv

# Using uv (Recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Or using pip
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Running the Server Locally
To start the FastAPI / OpenEnv HTTP Interface:

```bash
python server/app.py --port 7860
```
- Navigate to `http://localhost:7860/docs` to test `/reset` and `/step` via Swagger UI.

---

## 🚀 Usage

### Python Native (Gymnasium-style)
You can directly interact with the Python class without HTTP overhead:

```python
from server.intellicredit_env import IntelliCreditEnvironment
from models import IntelliCreditAction

# Initialize environment
env = IntelliCreditEnvironment(task_id="task3")
obs = env.reset(seed=42)

print(obs.application_summary.text_summary)

while not obs.done:
    # Action: 0=APPROVE, 1=CONDITIONAL, 2=REJECT
    action = IntelliCreditAction(decision=1) 
    obs = env.step(action)
    print(f"Step {obs.timestep} | Reward: {obs.reward:.2f}")

print(f"Episode Score: {obs.episode_score:.2f} / 1.0")
```

### LLM Baseline Script
We have included a baseline evaluation script that automatically tests open-source LLMs (default: `meta-llama/Llama-3.3-70B-Instruct`) against the environment tasks.

```bash
# Set your Hugging Face API Token (Free Serverless Inference API is supported)
export HF_TOKEN="your-hf-token-here"

# Run the inference evaluator
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

## License
MIT
