"""
FastAPI application for IntelliCredit-CreditAppraisal-v2.0 Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute a credit decision (or tool call)
    - GET /state: Get current environment state
    - GET /info: Get environment metadata
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

v2 changes:
    - 50-step episodes (was 12)
    - 55D observation (45D base + 10D memory features)
    - Multi-agent: BorrowerAgent + RegulatorAgent simulated by env
    - Regulator audits at steps 10/20/30/40/50
    - Tool call support via action_parser
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import IntelliCreditAction, IntelliCreditObservation
    from .intellicredit_env import IntelliCreditEnvironment, TASK_CONFIGS
except ImportError:
    from models import IntelliCreditAction, IntelliCreditObservation
    from server.intellicredit_env import IntelliCreditEnvironment, TASK_CONFIGS

_DESCRIPTION = """
## IntelliCredit Credit Appraisal Environment

**MSME Credit Decision RL Environment** — Meta × Hugging Face OpenEnv Hackathon.

---

### ⚠️ IMPORTANT: Session Workflow

Every HTTP call must carry the **same `episode_id`** to maintain session state:

**Step 1 — Reset:**
```json
POST /reset
{ "episode_id": "my-session-1", "seed": 42 }
```

**Step 2 — Step (repeat up to 50 times):**
```json
POST /step
{ "episode_id": "my-session-1", "action": { "decision": 0 }, "timeout_s": 30 }
```

> **v2**: Regulator audits auto-fire at steps 10, 20, 30, 40, 50 (±1 jitter).
> Rejected borrowers re-apply after a 3–5 step cooling period with improved surface profiles but unchanged hidden PD.

> `decision` values: **0 = APPROVE**, **1 = CONDITIONAL**, **2 = REJECT**

---

### Tasks (v2 — all 50 steps)
| Task | Steps | Description |
|------|-------|-------------|
| task1 | 50 | Easy — Clean profiles, macro shock at step ~22 |
| task2 | 50 | Medium — Forensic alerts present, shock at step ~22 |
| task3 | 50 | Hard — Macro shocks + missing data + repeat applicants |
| task4 | 50 | Expert — Hard-rule violations, tight CRAR windows |
| task5 | 50 | Master — Full constraints, cascading NPAs, 5 regulatory audits |

---

### v2 Highlights

**Observation Space: 55D**
| Dims | Sub-space | Description |
|------|-----------|-------------|
| 0–24 | Application features | 25 financial/forensic ratios per borrower |
| 25–34 | Portfolio state | NPA rate, CRAR, sector concentration, capital |
| 35–39 | Macro state | Stress index, shock flag, GDP, inflation, credit cycle |
| 40–44 | Alert state | Active forensic alert categories |
| 45–54 | Memory features (NEW) | Rolling NPA, approval rate, macro trend, audit risk… |

**6 Hard Rules (auto-REJECT + penalty)**
- HR-01: DSCR < 1.0x
- HR-02: CIBIL < 650
- HR-03: Active RED forensic alert
- HR-04: Sector under active macro shock
- HR-05: GST filing gap > 3 months
- HR-06: Adverse media score > 0.8

**Multi-Agent System**
- 🏦 **Credit Officer** (your agent) — approves/rejects applications
- 📋 **BorrowerAgent** — rejected borrowers reapply with improved surface metrics
- ⚖️ **RegulatorAgent** — audits portfolio at steps 10/20/30/40/50, applies penalties/shutdown

**Reward Components**
- Correctness bonus/penalty vs. optimal decision
- Hard rule penalty: −2.0 (APPROVE) / −1.0 (CONDITIONAL) on HR violation
- Delayed NPA penalty: fires 5–15 steps after approval
- Audit bonus (+2.0 clean) / penalty (−8 NPA, −15 CRAR, −8 sector)
- Survival bonus every 10 steps (if CRAR > 12.5%)
- Reasoning quality bonus (R1-style format reward)
"""

# Create the app with web interface and README integration
app = create_app(
    IntelliCreditEnvironment,
    IntelliCreditAction,
    IntelliCreditObservation,
    env_name="intellicredit_credit_appraisal",
    max_concurrent_envs=4,
)

# Inject rich description into FastAPI app metadata
app.title = "IntelliCredit-CreditAppraisal-v2"
app.description = _DESCRIPTION
app.version = "2.0.0"


from fastapi.responses import HTMLResponse, RedirectResponse

@app.get("/", response_class=HTMLResponse)
def read_root():
    """IntelliCredit project landing page."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>IntelliCredit-X — Multi-Agent RL for MSME Credit Appraisal</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: #0a0e1a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .hero {
      width: 100%;
      max-width: 900px;
      padding: 60px 24px 40px;
      text-align: center;
    }
    .badge-row {
      display: flex; flex-wrap: wrap; gap: 8px;
      justify-content: center; margin-bottom: 32px;
    }
    .badge-row a img { height: 22px; border-radius: 4px; }
    h1 {
      font-size: clamp(1.8rem, 4vw, 2.8rem);
      font-weight: 800;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.2;
      margin-bottom: 16px;
    }
    .subtitle {
      font-size: 1.05rem;
      color: #94a3b8;
      max-width: 640px;
      margin: 0 auto 40px;
      line-height: 1.7;
    }
    .cards {
      width: 100%;
      max-width: 900px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      padding: 0 24px 24px;
    }
    .card {
      background: linear-gradient(145deg, #111827, #1e293b);
      border: 1px solid #1e293b;
      border-radius: 16px;
      padding: 24px;
      text-decoration: none;
      color: inherit;
      transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
      transform: translateY(-4px);
      border-color: #667eea;
      box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    .card-icon { font-size: 2rem; margin-bottom: 12px; display: block; }
    .card-title { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
    .card-desc { font-size: 0.83rem; color: #64748b; line-height: 1.6; }
    .api-btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin: 32px auto;
      padding: 14px 32px;
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: #fff;
      border-radius: 50px;
      text-decoration: none;
      font-weight: 600;
      font-size: 1rem;
      transition: opacity 0.2s ease, transform 0.2s ease;
      box-shadow: 0 4px 24px rgba(102, 126, 234, 0.4);
    }
    .api-btn:hover { opacity: 0.9; transform: scale(1.03); }
    .results-bar {
      width: 100%;
      max-width: 900px;
      background: linear-gradient(145deg, #111827, #1e293b);
      border: 1px solid #1e293b;
      border-radius: 16px;
      padding: 28px 32px;
      margin: 8px 24px 0;
      display: flex;
      flex-wrap: wrap;
      gap: 24px;
      justify-content: space-around;
    }
    .stat { text-align: center; }
    .stat-num {
      font-size: 1.9rem; font-weight: 800;
      background: linear-gradient(135deg, #667eea, #f093fb);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .stat-label { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
    footer {
      margin-top: 48px;
      padding: 24px;
      font-size: 0.8rem;
      color: #334155;
      text-align: center;
    }
    footer a { color: #667eea; text-decoration: none; }
  </style>
</head>
<body>
  <div class="hero">
    <div class="badge-row">
      <a href="https://huggingface.co/spaces/vssksn/intellicredit-openenv" target="_blank">
        <img src="https://img.shields.io/badge/🤗_Space-Live_Demo-blue" alt="HF Space" /></a>
      <a href="https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset" target="_blank">
        <img src="https://img.shields.io/badge/🤗_Dataset-GRPO_Data-green" alt="Dataset" /></a>
      <a href="https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo" target="_blank">
        <img src="https://img.shields.io/badge/🤗_Model-Mistral--7B_GRPO-orange" alt="Model" /></a>
      <a href="https://github.com/1919-14/intellicredit-openenv" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-intellicredit--openenv-black" alt="GitHub" /></a>
    </div>
    <h1>🏦 IntelliCredit-X</h1>
    <p class="subtitle">
      A multi-agent RL environment where Mistral-7B learns to act as a Senior Credit Officer —
      investigating fraud signals, managing a live loan portfolio across 50-step episodes,
      and respecting hard RBI mandates. Trained via 2-stage GRPO on the live environment.
    </p>
    <a class="api-btn" href="/docs">⚡ Open Interactive API (Swagger UI)</a>
  </div>

  <div class="results-bar">
    <div class="stat"><div class="stat-num">55D</div><div class="stat-label">Observation Space</div></div>
    <div class="stat"><div class="stat-num">50</div><div class="stat-label">Steps per Episode</div></div>
    <div class="stat"><div class="stat-num">3</div><div class="stat-label">Agents</div></div>
    <div class="stat"><div class="stat-num">10×</div><div class="stat-label">Reward Improvement (Task 3)</div></div>
    <div class="stat"><div class="stat-num">−8.3%</div><div class="stat-label">NPA Rate After GRPO</div></div>
  </div>

  <div class="cards" style="margin-top:24px">
    <a class="card" href="https://huggingface.co/spaces/vssksn/intellicredit-openenv/blob/main/docs/blog.md" target="_blank">
      <span class="card-icon">📝</span>
      <div class="card-title">Technical Blog</div>
      <div class="card-desc">Architecture, 2-stage GRPO training pipeline, and full benchmark results.</div>
    </a>
    <a class="card" href="https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo" target="_blank">
      <span class="card-icon">🤖</span>
      <div class="card-title">Fine-Tuned Model</div>
      <div class="card-desc">Mistral-7B post-trained on live environment interactions via online GRPO.</div>
    </a>
    <a class="card" href="https://huggingface.co/datasets/vssksn/intellicredit-grpo-dataset" target="_blank">
      <span class="card-icon">📊</span>
      <div class="card-title">Training Dataset</div>
      <div class="card-desc">2,000 GRPO prompts across 5 task levels — published on Hugging Face.</div>
    </a>
    <a class="card" href="https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing" target="_blank">
      <span class="card-icon">🚀</span>
      <div class="card-title">Stage 1 — Offline GRPO</div>
      <div class="card-desc">Mistral-7B + Unsloth on A100, ~45 min. Pre-training on 2,000 curated prompts.</div>
    </a>
    <a class="card" href="https://colab.research.google.com/github/1919-14/intellicredit-openenv/blob/main/training/colab_online_grpo.ipynb" target="_blank">
      <span class="card-icon">🌍</span>
      <div class="card-title">Stage 2 — Online GRPO</div>
      <div class="card-desc">Post-training on this live environment. Every reward from the real /step endpoint.</div>
    </a>
    <a class="card" href="https://github.com/1919-14/intellicredit-openenv" target="_blank">
      <span class="card-icon">💻</span>
      <div class="card-title">GitHub Repository</div>
      <div class="card-desc">Full source code — environment, training, evaluation, MIT License.</div>
    </a>
  </div>

  <footer>
    Built for <strong>Meta × Hugging Face OpenEnv Hackathon 2026</strong> by
    <a href="https://huggingface.co/vssksn" target="_blank">vssksn</a> &amp;
    <a href="https://github.com/1919-14" target="_blank">Sujeet Jaiswal</a> ·
    <a href="/docs">API Docs</a> ·
    <a href="/info">JSON Info</a> ·
    <a href="https://github.com/1919-14/intellicredit-openenv/blob/main/LICENSE">MIT License</a>
  </footer>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/health")
def health_check():
    """Health check for Docker/Kubernetes."""
    return {"status": "healthy"}

# GAP 15: /info endpoint returning environment metadata
@app.get("/info")
def get_info():
    """Return environment metadata for validators and documentation."""
    return {
        "name": "IntelliCredit-CreditAppraisal-v2",
        "version": "2.0.0",
        "observation_dims": 55,
        "observation_sub_spaces": {
            "application_features": 25,   # dims 0-24  (unchanged from v1)
            "portfolio_state": 10,         # dims 25-34 (unchanged from v1)
            "macro_state": 5,              # dims 35-39 (unchanged from v1)
            "alert_state": 5,              # dims 40-44 (unchanged from v1)
            "memory_features": 10,         # dims 45-54 (NEW in v2)
        },
        "memory_feature_labels": {
            "45": "rolling_npa_rate_10step",
            "46": "approval_rate_recent",
            "47": "sector_max_concentration",
            "48": "macro_stress_trend",
            "49": "borrower_persistence_score",
            "50": "audit_risk_score",
            "51": "capital_buffer_ratio",
            "52": "recent_reflection_count",
            "53": "episode_progress",
            "54": "world_model_confidence",
        },
        "action_space": 3,
        "action_labels": {
            "0": "APPROVE",
            "1": "CONDITIONAL_APPROVE",
            "2": "REJECT",
        },
        "tasks": {
            tid: {
                "num_steps": cfg["num_steps"],
                "description": cfg["description"],
            }
            for tid, cfg in TASK_CONFIGS.items()
        },
        "max_steps_per_episode": 50,
        "audit_steps": [10, 20, 30, 40, 50],
        "constraints": {
            "max_npa_rate": 0.05,
            "min_crar": 0.125,
            "max_sector_concentration": 0.30,
            "max_single_borrower_pct": 0.15,
        },
        "hard_rules": [
            "HR-01: DSCR < 1.0",
            "HR-02: Director disqualification (DIN < 0.1)",
            "HR-03: RED forensic alert present",
            "HR-04: Cheque bounce rate > 25%",
            "HR-05: GST compliance < 40%",
            "HR-06: Severe adverse media (sentiment > 0.80)",
        ],
        "multi_agent": {
            "borrower_agent": "Programmatic — reapplies up to 3 times with improved surface profile",
            "regulator_agent": "Programmatic — audits at steps 10/20/30/40/50, can shut down episode",
            "credit_officer": "LLM under training — receives 55D obs + text prompt, calls tools or submits decision",
        },
        "reward_range": [-20.0, 15.0],
        "framework": "OpenEnv (Meta × Hugging Face × PyTorch)",
    }


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)  # main() callable for validation
