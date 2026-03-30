"""
IntelliCredit OpenEnv Models
Typed Pydantic models for the OpenEnv spec: Observation, Action, Reward.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, model_validator


class IntelliCreditAction(BaseModel):
    """Agent action for credit decision.

    Accepts all of the following equivalent formats:
        {"decision": 0}             ← canonical format
        {"value": 0}                ← Swagger UI default auto-fill
        {"action": 0}               ← alternative convention
        0                           ← raw integer (JSON primitive)
    """
    decision: int = Field(
        default=0,           # default so missing-field doesn't fail before validator runs
        ge=0, le=2,
        description="Credit decision: 0=APPROVE, 1=CONDITIONAL_APPROVE, 2=REJECT"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional reasoning for the decision (used by LLM agents)"
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_action(cls, data: Any) -> Any:
        """Accept 'value', 'action', or any bare int as fallback for 'decision'."""
        # Raw integer passed directly
        if isinstance(data, int):
            return {"decision": max(0, min(2, data))}

        if isinstance(data, dict):
            # Already has 'decision' — nothing to do
            if "decision" in data:
                return data
            # Swagger default: {"value": 1}
            if "value" in data:
                return {**data, "decision": int(data["value"])}
            # Alternative: {"action": 1}
            if "action" in data:
                return {**data, "decision": int(data["action"])}
            # Last resort: pick first integer value in the dict
            for v in data.values():
                if isinstance(v, (int, float)):
                    return {**data, "decision": max(0, min(2, int(v)))}

        return data

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"decision": 0},                   # canonical (APPROVE)
                {"decision": 1},                   # canonical (CONDITIONAL)
                {"decision": 2},                   # canonical (REJECT)
                {"value": 0},                      # Swagger auto-fill fallback
            ]
        }
    }


class ApplicationSummary(BaseModel):
    """Human-readable summary of the current credit application."""
    company_name: str = ""
    sector: str = ""
    size: str = ""
    text_summary: str = ""


class IntelliCreditObservation(BaseModel):
    """Full observation returned to the agent after each step.
    Total observation dims: 25 (app) + 10 (portfolio) + 5 (macro) + 5 (alerts) = 45
    """

    # 25-dim application feature vector (normalized -1 to 1; -1 = missing)
    application_features: List[float] = Field(
        default_factory=list,
        description="25-dimensional feature vector for current application. Values in [-1,1]; -1 indicates missing data."
    )

    # 10-dim portfolio state vector
    portfolio_state: List[float] = Field(
        default_factory=list,
        description="10-dimensional portfolio state vector"
    )

    # 5-dim macro state vector
    macro_state: List[float] = Field(
        default_factory=list,
        description="5-dimensional macro environment state"
    )

    # 5-dim portfolio alert vector (GAP 1)
    alert_state: List[float] = Field(
        default_factory=list,
        description="5-dimensional portfolio alert vector: [cc_spike, bounce_surge, gst_miss, adverse_media, credit_degradation]"
    )

    # Human-readable summary (for LLM agents)
    application_summary: ApplicationSummary = Field(
        default_factory=ApplicationSummary,
        description="Text summary of the current application"
    )

    # Forensic alerts visible to agent
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Forensic alerts for current application"
    )

    # Episode metadata
    timestep: int = Field(default=0, description="Current timestep in episode")
    done: bool = Field(default=False, description="Whether episode is complete")
    reward: float = Field(default=0.0, description="Reward from last action")

    # Reward breakdown (for transparency)
    reward_components: Dict[str, Any] = Field(
        default_factory=dict,
        description="Breakdown of reward components"
    )

    # Task and grading info
    task_id: str = Field(default="task3", description="Current task identifier")
    episode_score: Optional[float] = Field(
        default=None,
        description="Final episode score (0.0-1.0), only set when done=True"
    )
    score_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed score breakdown, only set when done=True"
    )
