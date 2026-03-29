"""
IntelliCredit Reward & Grading System
=====================================
Multi-component rewards for the constrained MDP environment:
  R_total = R_correctness + R_yield + R_forensic + R_delayed_npa
            + R_constraint + R_efficiency + R_settlement (T=final only)

Portfolio tracking with:
  - 10-dim portfolio state observation
  -  5-dim portfolio alert observation (cc_spike, bounce_surge, gst_miss, adverse_media, credit_degradation)
  - CRAR constraint (>12.5% or episode terminates)
  - Single borrower exposure (>15% → soft penalty)
  - Sector concentration (>30% → soft penalty)
  - NPA rate (>5% → episode terminates)
  - Task-specific grading

Implements GAPs 1, 2, 4, 5, 6, 7, 9, 14 from the implementation plan.
"""

import random
from typing import Dict, List, Any, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# REWARD CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Base rewards for correct decisions
REWARD_CORRECT_APPROVE = 1.0
REWARD_CORRECT_CONDITIONAL = 0.8
REWARD_CORRECT_REJECT = 1.0

# Penalties (reduced for PPO stability — Step 1.3)
PENALTY_WRONG_DECISION = -0.8
PENALTY_HARD_RULE_APPROVE = -1.5
PENALTY_HARD_RULE_CONDITIONAL = -1.0
PENALTY_DELAYED_NPA_APPROVE = -1.5
PENALTY_DELAYED_NPA_CONDITIONAL = -1.0
PENALTY_CONSTRAINT_BREACH = -2.0

# Bonuses
BONUS_FORENSIC_REJECT = 0.3
PENALTY_FORENSIC_APPROVE = -0.5
BONUS_CAPITAL_EFFICIENT = 0.2

# Portfolio constraints
MAX_NPA_RATE = 0.05
MIN_CRAR = 0.125            # GAP 6: Capital adequacy ratio
MAX_SECTOR_CONCENTRATION = 0.30
MAX_SINGLE_BORROWER_PCT = 0.15   # GAP 7: Single borrower exposure
INITIAL_CAPITAL = 100.0     # ₹100 Crores

# Yield parameters (GAP 4)
BASE_YIELD_MIN = 0.02
BASE_YIELD_MAX = 0.08
CONDITIONAL_YIELD_DISCOUNT = 0.85  # 15% monitoring cost

# Settlement weights (GAP 5)
SETTLEMENT_W_YIELD = 0.30
SETTLEMENT_W_NPA = 0.35
SETTLEMENT_W_CAPITAL = 0.15
SETTLEMENT_W_COMPLIANCE = 0.20

# Reward clipping bounds (GAP 14 — tightened for PPO stability)
REWARD_CLIP_MIN = -3.0
REWARD_CLIP_MAX = 2.0


# ═══════════════════════════════════════════════════════════════
# TASK-SPECIFIC GRADER WEIGHTS (GAP 2)
# ═══════════════════════════════════════════════════════════════

TASK_GRADER_WEIGHTS = {
    "task1": {
        "accuracy": 1.0,
    },
    "task2": {
        "accuracy": 0.50,
        "forensic_handling": 0.30,
        "hard_rule_compliance": 0.20,
    },
    "task3": {
        "accuracy": 0.40,
        "missing_data_handling": 0.30,
        "npa_avoidance": 0.20,
        "capital_efficiency": 0.10,
    },
    "task4": {
        "accuracy": 0.30,
        "hard_rule_compliance": 0.25,
        "constraint_satisfaction": 0.25,
        "portfolio_health": 0.20,
    },
    "task5": {
        "accuracy": 0.20,
        "survival_rate": 0.20,
        "constraint_satisfaction": 0.25,
        "npa_management": 0.20,
        "portfolio_health": 0.15,
    },
}


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO STATE
# ═══════════════════════════════════════════════════════════════

class PortfolioState:
    """Tracks the agent's portfolio across an episode."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.total_capital = initial_capital
        self.capital_remaining = initial_capital
        self.capital_deployed = 0.0
        self.total_loans = 0
        self.npa_count = 0
        self.npa_rate = 0.0
        self.sector_exposure: Dict[str, float] = {}
        self.borrower_exposures: List[float] = []  # GAP 7: track individual loan sizes
        self.stressed_accounts: List[Dict] = []
        self.healthy_accounts: List[Dict] = []
        self.pending_npas: List[Dict] = []  # Scheduled future NPAs
        self.cumulative_reward = 0.0
        self.cumulative_yield = 0.0   # GAP 4: track total yield earned
        self.episode_terminated = False
        self.termination_reason = ""
        self.crar = 1.0  # GAP 6: Capital adequacy ratio
        self.steps_survived = 0
        self.total_episode_steps = 12

        # GAP 1: portfolio alert state
        self._recent_cc_spike = False
        self._recent_bounce_surge = False
        self._recent_gst_miss = False
        self._recent_adverse_media = False
        self._recent_credit_degradation = False

    def get_observation(self) -> List[float]:
        """Return 10-dim portfolio observation vector (normalized)."""
        total_exposure = sum(self.sector_exposure.values())
        top_sector_pct = (
            max(self.sector_exposure.values()) / max(total_exposure, 1)
            if self.sector_exposure else 0
        )

        return [
            self.capital_remaining / self.total_capital,             # 0: capital remaining %
            self.capital_deployed / self.total_capital,              # 1: capital deployed %
            self.npa_rate,                                           # 2: NPA rate
            min(len(self.sector_exposure) / 8.0, 1.0),              # 3: sector diversification
            top_sector_pct,                                          # 4: top sector concentration
            min(len(self.stressed_accounts) / 5.0, 1.0),            # 5: stressed count (norm)
            min(len(self.healthy_accounts) / 10.0, 1.0),            # 6: healthy count (norm)
            min(self.total_loans / 12.0, 1.0),                      # 7: total loans (norm)
            max(0.0, min(1.0, self.cumulative_reward / 20.0 + 0.5)),  # 8: cumulative reward (centered)
            max(0.0, min(1.0, self.crar)),                           # 9: CRAR (normalized)
        ]

    def get_alert_observation(self) -> List[float]:
        """Return 5-dim portfolio alert vector (GAP 1)."""
        return [
            1.0 if self._recent_cc_spike else 0.0,
            1.0 if self._recent_bounce_surge else 0.0,
            1.0 if self._recent_gst_miss else 0.0,
            1.0 if self._recent_adverse_media else 0.0,
            1.0 if self._recent_credit_degradation else 0.0,
        ]

    def update_alerts_from_application(self, app_features: Dict, alerts: List[Dict]):
        """Update portfolio-level alerts based on the current application (GAP 1)."""
        # CC utilisation spike in current app → portfolio alert
        self._recent_cc_spike = app_features.get("cc_utilisation_volatility", 0) > 0.6

        # Bounce surge from current application
        self._recent_bounce_surge = app_features.get("cheque_bounce_frequency", 0) > 0.4

        # GST filing miss
        self._recent_gst_miss = app_features.get("revenue_gst_alignment", 1.0) < 0.5

        # Adverse media from current alerts
        self._recent_adverse_media = any(
            a.get("type") == "ADVERSE_MEDIA" for a in alerts
        )

        # Credit degradation: NPA rate rising or stressed accounts growing
        self._recent_credit_degradation = (
            self.npa_rate > 0.03 or len(self.stressed_accounts) >= 2
        )

    def approve_loan(
        self, loan_amount: float, sector: str,
        app_meta: Dict, is_conditional: bool = False,
    ) -> Tuple[float, bool]:
        """
        Process a loan approval. Returns (immediate_reward_modifier, constraint_breached).
        GAP 9: CONDITIONAL reduces loan amount by 20%, reduces NPA prob by 30%.
        GAP 7: Checks single borrower exposure.
        """
        # GAP 9: Conditional approval reduces effective loan amount
        effective_loan = loan_amount * 0.80 if is_conditional else loan_amount

        # Check capital constraint
        if effective_loan > self.capital_remaining:
            return -1.0, False  # Can't approve without capital

        self.capital_remaining -= effective_loan
        self.capital_deployed += effective_loan
        self.total_loans += 1
        self.borrower_exposures.append(effective_loan)  # GAP 7

        # Track sector exposure
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + effective_loan

        # Check sector concentration (soft constraint)
        total_deployed = sum(self.sector_exposure.values())
        sector_pct = self.sector_exposure[sector] / total_deployed if total_deployed > 0 else 0

        reward_mod = 0.0
        if sector_pct > MAX_SECTOR_CONCENTRATION:
            reward_mod -= 0.5  # Over-concentrated (spec says -0.5)

        # GAP 7: Single borrower exposure check
        borrower_pct = effective_loan / self.total_capital
        if borrower_pct > MAX_SINGLE_BORROWER_PCT:
            reward_mod -= 1.0  # Spec says -1.0 penalty

        # Schedule potential NPA (delayed consequence)
        hidden_pd = app_meta.get("hidden_pd", 0.5)

        # GAP 9: Conditional approval reduces NPA probability by 30%
        adjusted_pd = hidden_pd * 0.70 if is_conditional else hidden_pd

        will_default = random.random() < adjusted_pd

        if will_default:
            # Schedule NPA with delay
            tier = app_meta.get("tier", "C")
            if tier == "D":
                delay = random.randint(2, 5)
            elif tier == "C":
                delay = random.randint(3, 6)
            else:
                delay = random.randint(4, 8)

            current_t = app_meta.get("timestep", 1)
            default_t = current_t + delay

            collateral = app_meta.get("collateral_coverage", 1.0)
            if collateral > 1.5:
                recovery_rate = 0.60
            elif collateral > 1.0:
                recovery_rate = 0.40
            elif collateral > 0.5:
                recovery_rate = 0.25
            else:
                recovery_rate = 0.10

            self.pending_npas.append({
                "default_timestep": default_t,
                "loan_amount": effective_loan,
                "recovery_rate": recovery_rate,
                "sector": sector,
                "is_conditional": is_conditional,
                "approval_timestep": current_t,
            })
        else:
            self.healthy_accounts.append({
                "loan_amount": effective_loan,
                "sector": sector,
            })

        # Update CRAR (GAP 6)
        self._update_crar()

        return reward_mod, False

    def _update_crar(self):
        """Compute Capital Adequacy Ratio (GAP 6)."""
        risk_weighted_assets = 0.0
        for exp in self.borrower_exposures:
            risk_weighted_assets += exp * 1.0  # Simplified: 100% risk weight
        for npa_info in self.stressed_accounts:
            risk_weighted_assets += npa_info.get("loan_amount", 0) * 1.5  # 150% for stressed

        if risk_weighted_assets > 0:
            self.crar = self.capital_remaining / risk_weighted_assets
        else:
            self.crar = 1.0  # No assets → fully capitalised

    def process_timestep(self, current_t: int) -> Tuple[float, bool]:
        """
        Process NPA revelations at current timestep.
        Returns (delayed_penalty, episode_terminated).
        GAP 6: Also checks CRAR constraint.
        """
        self.steps_survived = current_t
        delayed_penalty = 0.0

        remaining_npas = []
        for npa in self.pending_npas:
            if npa["default_timestep"] <= current_t:
                # NPA event!
                loss = npa["loan_amount"] * (1 - npa["recovery_rate"])
                self.capital_remaining -= loss
                self.npa_count += 1

                # Update NPA rate
                if self.total_loans > 0:
                    self.npa_rate = self.npa_count / self.total_loans

                # Delayed penalty
                if npa["is_conditional"]:
                    delayed_penalty += PENALTY_DELAYED_NPA_CONDITIONAL
                else:
                    delayed_penalty += PENALTY_DELAYED_NPA_APPROVE

                self.stressed_accounts.append(npa)
            else:
                remaining_npas.append(npa)

        self.pending_npas = remaining_npas

        # Update CRAR after NPA processing
        self._update_crar()

        # Check regulatory constraints
        # NPA rate constraint
        if self.npa_rate > MAX_NPA_RATE and self.total_loans >= 3:
            self.episode_terminated = True
            self.termination_reason = f"NPA rate {self.npa_rate:.1%} exceeds {MAX_NPA_RATE:.0%} limit"
            delayed_penalty += PENALTY_CONSTRAINT_BREACH

        # Capital exhausted
        if self.capital_remaining < 0:
            self.episode_terminated = True
            self.termination_reason = "Capital exhausted"
            delayed_penalty += PENALTY_CONSTRAINT_BREACH

        # GAP 6: CRAR constraint
        if self.crar < MIN_CRAR and self.capital_deployed > 0:
            self.episode_terminated = True
            self.termination_reason = f"CRAR {self.crar:.1%} below {MIN_CRAR:.1%} minimum"
            delayed_penalty += PENALTY_CONSTRAINT_BREACH

        return delayed_penalty, self.episode_terminated


# ═══════════════════════════════════════════════════════════════
# YIELD REWARD (GAP 4)
# ═══════════════════════════════════════════════════════════════

def _compute_yield_reward(
    action: int,
    hidden_pd: float,
    loan_amount: float,
    portfolio: PortfolioState,
) -> float:
    """
    Compute yield reward for an action.
    GAP 4: R_yield = base_yield × risk_premium for APPROVE,
           reduced by 15% for CONDITIONAL, 0 for REJECT.
    """
    if action == 2:  # REJECT
        return 0.0

    # Base yield scales with loan amount (normalised)
    base_yield = BASE_YIELD_MIN + (BASE_YIELD_MAX - BASE_YIELD_MIN) * (loan_amount / 20.0)
    base_yield = min(base_yield, BASE_YIELD_MAX)

    # Risk premium: higher PD → higher yield demanded
    risk_premium = 1.0 + hidden_pd * 0.5

    raw_yield = base_yield * risk_premium

    if action == 1:  # CONDITIONAL — monitoring cost
        raw_yield *= CONDITIONAL_YIELD_DISCOUNT

    portfolio.cumulative_yield += raw_yield
    return raw_yield


# ═══════════════════════════════════════════════════════════════
# SETTLEMENT REWARD (GAP 5) — fires only at episode end
# ═══════════════════════════════════════════════════════════════

def compute_settlement_reward(portfolio: PortfolioState) -> float:
    """
    End-of-episode portfolio evaluation.
    GAP 5: Weighted combination of yield, NPA, capital, compliance.
    """
    # Yield score
    max_yield = INITIAL_CAPITAL * BASE_YIELD_MAX
    yield_score = min(1.0, portfolio.cumulative_yield / max(max_yield, 0.01))

    # NPA avoidance score
    npa_score = max(0.0, 1.0 - portfolio.npa_rate / MAX_NPA_RATE)

    # Capital utilisation score (optimal ~60-70%)
    utilization = portfolio.capital_deployed / portfolio.total_capital if portfolio.total_capital > 0 else 0
    capital_score = min(1.0, utilization / 0.70)

    # Compliance score (binary: did we survive?)
    compliance_score = 0.0 if portfolio.episode_terminated else 1.0

    settlement = (
        SETTLEMENT_W_YIELD * yield_score
        + SETTLEMENT_W_NPA * npa_score
        + SETTLEMENT_W_CAPITAL * capital_score
        + SETTLEMENT_W_COMPLIANCE * compliance_score
    )

    return settlement


# ═══════════════════════════════════════════════════════════════
# REWARD COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_step_reward(
    action: int,
    app_metadata: Dict[str, Any],
    portfolio: PortfolioState,
    is_final_step: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the immediate reward for a step action.
    action: 0=APPROVE, 1=CONDITIONAL, 2=REJECT
    Returns (total_reward, reward_components_dict)
    """
    optimal = app_metadata["optimal_action"]
    hidden_pd = app_metadata["hidden_pd"]
    hard_rules = app_metadata.get("hard_rules_triggered", [])
    alerts = app_metadata.get("alerts", [])
    red_alerts = [a for a in alerts if a["severity"] == "RED"]

    components: Dict[str, float] = {}

    # ─── Component 1: Base Correctness ────────────────────────
    if hard_rules and action != 2:
        # GAP 11: Hard rule violation — heavy penalty
        if action == 0:
            components["hard_rule_penalty"] = PENALTY_HARD_RULE_APPROVE
        else:
            components["hard_rule_penalty"] = PENALTY_HARD_RULE_CONDITIONAL
    elif hard_rules and action == 2:
        components["hard_rule_bonus"] = 1.5  # Correctly rejected a hard-rule case
    elif action == optimal:
        if action == 0:
            components["base_correct"] = REWARD_CORRECT_APPROVE
        elif action == 1:
            components["base_correct"] = REWARD_CORRECT_CONDITIONAL
        else:
            components["base_correct"] = REWARD_CORRECT_REJECT
    else:
        # Partial credit for close decisions
        if optimal == 0 and action == 1:
            components["base_correct"] = 0.3   # Conservative but acceptable
        elif optimal == 1 and action == 0:
            components["base_correct"] = -0.3  # Risky but not terrible
        elif optimal == 1 and action == 2:
            components["base_correct"] = 0.2   # Over-cautious but safe
        elif optimal == 2 and action == 1:
            components["base_correct"] = -0.5  # Should have rejected
        else:
            components["base_correct"] = PENALTY_WRONG_DECISION

    # ─── Component 2: Forensic Alert Alignment ───────────────
    if red_alerts:
        if action == 2:
            components["forensic_bonus"] = BONUS_FORENSIC_REJECT
        elif action == 0:
            components["forensic_penalty"] = PENALTY_FORENSIC_APPROVE

    # ─── Component 3: Yield Reward (GAP 4) ───────────────────
    loan_amount = app_metadata.get("loan_amount_cr", 5.0)
    yield_reward = _compute_yield_reward(action, hidden_pd, loan_amount, portfolio)
    if yield_reward != 0:
        components["yield_reward"] = round(yield_reward, 4)

    # ─── Component 4: Portfolio Impact ────────────────────────
    if action in (0, 1):  # Approve or Conditional
        is_conditional = (action == 1)

        app_meta_for_portfolio = {
            "hidden_pd": hidden_pd,
            "tier": app_metadata.get("tier", "C"),
            "timestep": app_metadata.get("timestep", 1),
            "collateral_coverage": app_metadata.get("collateral_ratio", 1.0),
        }

        portfolio_mod, _ = portfolio.approve_loan(
            loan_amount=loan_amount,
            sector=app_metadata.get("sector", "Manufacturing"),
            app_meta=app_meta_for_portfolio,
            is_conditional=is_conditional,  # GAP 9
        )
        if portfolio_mod != 0:
            components["portfolio_impact"] = portfolio_mod

    # ─── Component 5: Capital Preservation (for rejects) ─────
    if action == 2 and hidden_pd >= 0.40:
        components["capital_preservation"] = 0.1

    # ─── Component 6: Efficiency (capital utilisation) ────────
    if portfolio.total_loans > 0:
        util = portfolio.capital_deployed / portfolio.total_capital
        if util > 0.70:
            components["efficiency_bonus"] = 0.1
        elif util < 0.30 and portfolio.steps_survived > 6:
            components["efficiency_penalty"] = -0.1

    # ─── Component 7: Settlement Reward (GAP 5) ──────────────
    if is_final_step:
        settlement = compute_settlement_reward(portfolio)
        components["settlement_reward"] = round(settlement, 4)

    # ─── Aggregate & Clip (GAP 14) ───────────────────────────
    total = sum(components.values())
    total = max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, total))
    components["total"] = round(total, 4)
    portfolio.cumulative_reward += total

    return total, components


# ═══════════════════════════════════════════════════════════════
# TASK GRADERS (0.0 to 1.0 score) — GAP 2
# ═══════════════════════════════════════════════════════════════

def grade_episode(
    actions: List[int],
    applications: List[Dict[str, Any]],
    portfolio: PortfolioState,
    task_id: str = "task3",
) -> Dict[str, Any]:
    """
    Grade an entire episode using task-specific weights.
    Returns score in [0.0, 1.0].
    """
    if not actions or not applications:
        return {"score": 0.0, "breakdown": {}}

    n = min(len(actions), len(applications))
    weights = TASK_GRADER_WEIGHTS.get(task_id, TASK_GRADER_WEIGHTS["task3"])

    # ─── Compute individual metric components ────────────────
    correct = 0
    forensic_correct = 0
    forensic_total = 0
    hard_rule_correct = 0
    hard_rule_total = 0
    missing_bad_approvals = 0
    missing_total = 0

    for i in range(n):
        action = actions[i]
        meta = applications[i]["metadata"]
        optimal = meta["optimal_action"]
        hard_rules = meta.get("hard_rules_triggered", [])
        alerts = meta.get("alerts", [])
        red_alerts = [a for a in alerts if a.get("severity") == "RED"]
        has_missing = meta.get("has_missing_features", False)

        if action == optimal:
            correct += 1

        # Forensic handling: RED alerts should trigger reject/conditional
        if red_alerts:
            forensic_total += 1
            if action >= 1:  # CONDITIONAL or REJECT
                forensic_correct += 1

        # Hard rule compliance
        if hard_rules:
            hard_rule_total += 1
            if action == 2:
                hard_rule_correct += 1

        # Missing data handling (Task 3)
        if has_missing:
            missing_total += 1
            if action == 0:  # Approved with missing data = bad
                missing_bad_approvals += 1

    # Raw metrics
    accuracy = correct / n if n > 0 else 0.0
    forensic_handling = forensic_correct / forensic_total if forensic_total > 0 else 1.0
    hard_rule_compliance = hard_rule_correct / hard_rule_total if hard_rule_total > 0 else 1.0
    constraint_satisfaction = 0.0 if portfolio.episode_terminated else 1.0
    npa_avoidance = max(0.0, 1.0 - portfolio.npa_rate / MAX_NPA_RATE)
    npa_management = max(0.0, 1.0 - portfolio.npa_rate / 0.10)

    utilization = portfolio.capital_deployed / portfolio.total_capital if portfolio.total_capital > 0 else 0
    capital_efficiency = min(1.0, utilization / 0.70)

    capital_remaining_pct = portfolio.capital_remaining / portfolio.total_capital if portfolio.total_capital > 0 else 0
    portfolio_health = max(0.0, capital_remaining_pct * (1 - portfolio.npa_rate))

    survival_rate = portfolio.steps_survived / portfolio.total_episode_steps

    missing_data_handling = (
        1.0 - (missing_bad_approvals / missing_total)
        if missing_total > 0 else 1.0
    )

    # ─── Build the metric dict ───────────────────────────────
    metrics = {
        "accuracy": accuracy,
        "forensic_handling": forensic_handling,
        "hard_rule_compliance": hard_rule_compliance,
        "constraint_satisfaction": constraint_satisfaction,
        "npa_avoidance": npa_avoidance,
        "npa_management": npa_management,
        "capital_efficiency": capital_efficiency,
        "portfolio_health": portfolio_health,
        "survival_rate": survival_rate,
        "missing_data_handling": missing_data_handling,
    }

    # ─── Apply task-specific weights ─────────────────────────
    score = 0.0
    for metric_name, weight in weights.items():
        score += weight * metrics.get(metric_name, 0.0)

    # Constraint breach = heavy penalty
    if portfolio.episode_terminated:
        score *= 0.5

    final_score = max(0.0, min(1.0, score))

    return {
        "score": round(final_score, 4),
        "breakdown": {
            "accuracy": round(accuracy, 4),
            "correct_decisions": correct,
            "total_decisions": n,
            "hard_rule_compliance": round(hard_rule_compliance, 4),
            "forensic_handling": round(forensic_handling, 4),
            "npa_count": portfolio.npa_count,
            "npa_rate": round(portfolio.npa_rate, 4),
            "crar": round(portfolio.crar, 4),
            "capital_utilization": round(utilization, 4),
            "survival_rate": round(survival_rate, 4),
            "episode_terminated": portfolio.episode_terminated,
            "termination_reason": portfolio.termination_reason,
            "task_weights": weights,
        },
    }
