"""
IntelliCredit OpenEnv Models
Typed Pydantic models for the OpenEnv spec: Observation, Action, Reward.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class IntelliCreditAction(BaseModel):
    """Agent action for credit decision."""
    decision: int = Field(
        ...,
        ge=0, le=2,
        description="Credit decision: 0=APPROVE, 1=CONDITIONAL_APPROVE, 2=REJECT"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional reasoning for the decision (used by LLM agents)"
    )


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
