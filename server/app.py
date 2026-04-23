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


from fastapi.responses import RedirectResponse

@app.get("/")
def read_root():
    """Redirect root to Swagger UI for Hugging Face Spaces."""
    return RedirectResponse(url="/docs")

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
