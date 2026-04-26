"""
IntelliCredit OpenEnv Models — v2.0
Typed Pydantic models for the OpenEnv spec: Observation, Action, Reward.

v2 changes:
  - IntelliCreditObservation extended to 55D (+ 10 memory features, dims 45-54)
  - IntelliCreditAction gains is_tool_call flag
  - ApplicationSummary gains repeat-applicant metadata
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, model_validator


class IntelliCreditAction(BaseModel):
    """Agent action for credit decision.

    Accepts all of the following equivalent formats:
        {"decision": 0}             ← canonical format
        {"value": 0}                ← Swagger UI default auto-fill
        {"action": 0}               ← alternative convention
        {"llm_output": "..."}       ← v2 raw LLM output, parsed by env
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
    llm_output: Optional[str] = Field(
        default=None,
        description=(
            "Raw LLM text for v2 agent mode. May contain a tool call such as "
            "get_financial_report(...) or a final submit_decision(...)."
        ),
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Alias for llm_output used by some training clients.",
    )
    parse_type: Optional[str] = Field(
        default=None,
        description="Parser result type supplied by a client-side parser, if available.",
    )
    parse_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Parser confidence supplied by a client-side parser, if available.",
    )
    parse_failure: Optional[bool] = Field(
        default=None,
        description="True if parsing failed and the action was defaulted.",
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Tool name when this action represents a parsed tool call.",
    )
    tool_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool arguments when this action represents a parsed tool call.",
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
                {"llm_output": "get_financial_report('Acme Pvt. Ltd.')"},
                {"llm_output": "submit_decision('REJECT', 'HR-03 red forensic alert present.')"},
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

    v1 dims (UNCHANGED): 25 (app) + 10 (portfolio) + 5 (macro) + 5 (alerts) = 45
    v2 dims (EXTENDED):  45 + 10 (memory features) = 55

    New memory dims 45-54:
      45: rolling_npa_rate_10step
      46: approval_rate_recent
      47: sector_max_concentration
      48: macro_stress_trend
      49: borrower_persistence_score
      50: audit_risk_score
      51: capital_buffer_ratio
      52: recent_reflection_count
      53: episode_progress
      54: world_model_confidence
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

    # ── v2: 10D Memory Features (dims 45-54) ──────────────────────
    memory_features: List[float] = Field(
        default_factory=list,
        description=(
            "10-dimensional memory/context vector (v2). "
            "[rolling_npa_10step, approval_rate_recent, sector_max_concentration, "
            "macro_stress_trend, borrower_persistence_score, audit_risk_score, "
            "capital_buffer_ratio, recent_reflection_count, episode_progress, "
            "world_model_confidence]"
        )
    )

    # ── v2: Multi-agent metadata ───────────────────────────────────
    is_repeat_applicant: bool = Field(
        default=False,
        description="True if current borrower has been rejected before in this episode"
    )
    attempt_number: int = Field(
        default=1,
        description="How many times this borrower has applied (1, 2, or 3)"
    )
    tool_call_count: int = Field(
        default=0,
        description="Number of tool calls made so far in this step (resets each step)"
    )
    regulator_warning_level: int = Field(
        default=0,
        description="Consecutive audit failures (0=clean, 1=warning, 2=penalty, 3=shutdown)"
    )
    audit_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Regulator audit result, populated only on audit steps (10,20,30,40,50)"
    )
    tool_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured tool result returned when a raw LLM tool call is executed."
    )
    tool_result_text: Optional[str] = Field(
        default=None,
        description="Human-readable tool result text for injection into the LLM context."
    )
    last_parse_type: Optional[str] = Field(
        default=None,
        description="Parser type for the last raw LLM action processed by the environment."
    )
    last_tool_name: Optional[str] = Field(
        default=None,
        description="Name of the most recent tool executed by the environment."
    )
