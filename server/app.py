"""
FastAPI application for IntelliCredit-CreditAppraisal-v1 Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute a credit decision
    - GET /state: Get current environment state
    - GET /info: Get environment metadata (GAP 15)
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
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


# Create the app with web interface and README integration
app = create_app(
    IntelliCreditEnvironment,
    IntelliCreditAction,
    IntelliCreditObservation,
    env_name="intellicredit_credit_appraisal",
    max_concurrent_envs=4,
)


# GAP 15: /info endpoint returning environment metadata
@app.get("/info")
def get_info():
    """Return environment metadata for validators and documentation."""
    return {
        "name": "IntelliCredit-CreditAppraisal-v1",
        "version": "1.0.0",
        "observation_dims": 45,
        "observation_sub_spaces": {
            "application_features": 25,
            "portfolio_state": 10,
            "macro_state": 5,
            "alert_state": 5,
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
        "max_steps_per_episode": 12,
        "constraints": {
            "max_npa_rate": 0.05,
            "min_crar": 0.125,
            "max_sector_concentration": 0.30,
            "max_single_borrower_pct": 0.15,
        },
        "reward_range": [-5.0, 3.0],
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
    main(port=args.port)
