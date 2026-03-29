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

A **Constrained Multi-Objective MDP** for corporate credit underwriting, built as an [OpenEnv](https://github.com/meta-pytorch/openenv) reinforcement learning environment.

An AI agent acts as a Senior Credit Officer at an Indian MSME lending institution, reviewing 5-12 sequential credit applications per episode. Each decision (APPROVE / CONDITIONAL / REJECT) impacts the portfolio's capital reserves, NPA rate, and regulatory compliance.

## Environment Highlights

| Feature | Detail |
|---|---|
| **Observation Space** | 40-dimensional: 25 application + 10 portfolio + 5 macro |
| **Action Space** | Discrete(3): APPROVE, CONDITIONAL, REJECT |
| **Episode Length** | 5-12 steps (varies by task difficulty) |
| **Reward** | Multi-component: correctness + hard-rule compliance + delayed NPA penalties |
| **Constraints** | NPA rate < 5%, capital preservation, sector concentration limits |
| **Delayed Rewards** | Loan approvals at T=2 may default at T=7+ with -2.0 penalty |
| **Macro Shocks** | Interest rate hikes, sector collapses at ~T=7 |

## Tasks

| Task | Difficulty | Steps | Description |
|---|---|---|---|
| `task1` | Easy | 5 | Clean profiles, no macro shocks |
| `task2` | Medium | 8 | Forensic alerts present |
| `task3` | Medium-Hard | 12 | Macro shocks + uncertainty |
| `task4` | Hard | 12 | Regulatory hard-rule violations |
| `task5` | Expert | 12 | Cascading delayed NPAs |

## Architecture

```
ScalarXOpenEnv/
├── server/
│   ├── app.py                    # FastAPI server (OpenEnv spec)
│   ├── intellicredit_env.py      # Core environment (reset/step/state)
│   ├── dataset.py                # Anchor-based synthetic data generator
│   └── reward.py                 # Reward computation + portfolio tracking
├── models.py                     # Pydantic Action/Observation types
├── inference.py                  # LLM baseline agent (OpenAI client)
├── openenv.yaml                  # OpenEnv specification
├── Dockerfile                    # HuggingFace Space deployment
├── requirements.txt              # Python dependencies
└── README.md
```

## Key Design: Anchor-Based Data Generation

Applications are generated using a **Tier × Sector × Size** anchoring system:

1. **Tier (A/B/C/D)**: Hidden credit quality that determines all feature distributions
2. **Sector (8 sectors)**: Industry-specific margins, collateral, macro sensitivity
3. **Size (Micro/Small/Medium/Large)**: Revenue and loan amount scaling

Features are generated in dependency clusters to ensure **semantic coherence** — a company with high DSCR cannot simultaneously have terrible CIBIL and zero net worth. This is critical because **LLM evaluators read the data as text** and would detect incoherent scenarios.

## Hard Rules (Mandatory Rejections)

| Rule | Condition |
|---|---|
| HR-01 | DSCR < 1.0 |
| HR-02 | Director disqualified (DIN score < 0.1) |
| HR-03 | RED forensic alert present |
| HR-04 | Cheque bounce rate > 25% |
| HR-05 | GST filing compliance < 40% |
| HR-06 | Adverse media score > 0.80 |

## Quick Start

```python
from server.intellicredit_env import IntelliCreditEnvironment
from models import IntelliCreditAction

env = IntelliCreditEnvironment(task_id="task1")
obs = env.reset()

while not obs.done:
    action = IntelliCreditAction(decision=1)  # CONDITIONAL
    obs = env.step(action)
    print(f"Step {obs.timestep}: reward={obs.reward:.2f}")

print(f"Final Score: {obs.episode_score:.4f}")
```

## LLM Baseline

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your-token-here"
python inference.py
```

## Scoring (0.0 → 1.0)

- **50%** — Decision accuracy (matching optimal action)
- **25%** — Hard rule compliance (no approved hard-reject cases)
- **15%** — NPA management (low portfolio default rate)
- **10%** — Capital utilization (efficient deployment)

## License

MIT
