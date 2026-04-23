"""
IntelliCredit Reward & Grading System — v2.0 (Phase 4 Redesign)
================================================================
Sparse+delayed reward design — forces real planning, not gaming.

FOUR WEIGHTED REWARD FUNCTIONS (per step):
  R1 Decision Correctness  (40%): PD-based ±rewards
  R2 Hard Rule Compliance  (30%): +0.5 bonus / -2.0 penalty
  R3 Format Compliance     (10%): +0.3/+0.1 / -0.3/-0.1
  R4 Portfolio Awareness   (20%): NPA/CRAR/sector/late-game

DELAYED EVENTS (fire T+10 to T+30 after decision):
  Loan repaid       → +10.0  (REWARD_LOAN_REPAID)
  Loan partial      → -5.0   (PENALTY_LOAN_PARTIAL)
  Loan full default → -15.0  (PENALTY_LOAN_DEFAULTED)

SURVIVAL BONUS (every 10 steps if solvent):
  CRAR > 15%          → +0.1
  CRAR 12.5%-15%      → +0.05
  CRAR < 12.5%        → episode terminates

SETTLEMENT (step 50 only): range [-1.0, +5.0]

PER-STEP CLIP: [-5.0, +3.0]

ANTI-HACKING:
  PD score from HIDDEN features agent cannot see directly.
  Tool results READ-ONLY — cannot manipulate env.
  Reasoning quality checked — empty reasoning penalized.
  Delayed NPA — agent cannot see future defaults.
  Audit timing jittered ±1 step — cannot predict exactly.
"""

import random
from typing import Dict, List, Any, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# REWARD CONSTANTS
# ═══════════════════════════════════════════════════════════════

# ── R1: Decision Correctness (40% weight) ────────────────────────────
# Correct action rewards — based on PD model score
REWARD_CORRECT_APPROVE      = 1.0   # PD < 0.25 → APPROVE correct
REWARD_CORRECT_CONDITIONAL  = 0.8   # PD 0.25-0.45 → CONDITIONAL correct
REWARD_CORRECT_REJECT       = 1.0   # PD >= 0.45 → REJECT correct

# Wrong direction penalties
PENALTY_HIGH_PD_APPROVE     = -2.0  # PD >= 0.45 + APPROVE → dangerous
PENALTY_LOW_PD_REJECT       = -0.3  # PD < 0.25 + REJECT → opportunity cost
PENALTY_MED_PD_APPROVE      = -0.8  # PD 0.25-0.45 + APPROVE → risky no conditions
PENALTY_WRONG_DECISION      = -0.8  # generic wrong decision fallback

# ── R2: Hard Rule Compliance (30% weight) ────────────────────────────
BONUS_HARD_RULE_REJECT      =  0.5  # per hard rule correctly rejected (+0.5 each)
PENALTY_HARD_RULE_APPROVE   = -2.0  # violated hard rule → approve (spec: -2.0)
PENALTY_HARD_RULE_CONDITIONAL = -1.0  # conditional when must reject

# ── R3: Format Compliance (10% weight) ───────────────────────────────
BONUS_FORMAT_VALID_SUBMIT   =  0.3  # submit_decision with valid action word
PENALTY_FORMAT_INVALID      = -0.3  # invalid/unclear action
BONUS_FORMAT_CLEAR_KEYWORD  =  0.1  # response is one clear word from {APPROVE,CONDITIONAL,REJECT}
PENALTY_FORMAT_CONFUSING    = -0.1  # extra text confuses parser

# ── R4: Portfolio Awareness (20% weight) ─────────────────────────────
PENALTY_NPA_HIGH_APPROVE    = -0.5  # NPA > 8% + APPROVE → adding risk
BONUS_NPA_HIGH_REJECT       =  0.3  # NPA > 8% + REJECT → conservative
PENALTY_SECTOR_CONC_APPROVE = -0.8  # sector near 30% + APPROVE same sector
PENALTY_CRAR_THIN_APPROVE   = -0.5  # CRAR buffer < 1% above min + APPROVE
BONUS_LATE_GAME_GOOD        =  0.3  # episode_progress > 0.8 + CRAR healthy + good loan

# ── Delayed Maturity Events (sparse, fires T+10 to T+30) ─────────────
REWARD_LOAN_REPAID          =  10.0  # loan fully repaid at maturity
PENALTY_LOAN_PARTIAL        =  -5.0  # loan partially repaid (stressed)
PENALTY_LOAN_DEFAULTED      = -15.0  # loan fully defaulted

# ── Survival Bonus (fires every 10 steps) ────────────────────────────
BONUS_SURVIVAL_CRAR_STRONG  =  0.10   # CRAR > 15%
BONUS_SURVIVAL_CRAR_STRESSED =  0.05  # CRAR 12.5% - 15%

# ── Tool Efficiency (from agent_loop.py) ─────────────────────────────
BONUS_TOOL_HELPED_CORRECT   =  0.20
BONUS_COMPLIANCE_TOOL_RED   =  0.30
BONUS_MARKET_TOOL_CONC      =  0.20
PENALTY_OVER_LIMIT_FORCE    = -0.50
PENALTY_EXTRA_CALL          = -0.10
PENALTY_REDUNDANT_CALL      = -0.10
PENALTY_MALFORMED_CALL      = -0.05
PENALTY_SHORT_REASONING     = -0.20

# ── Forensic alerts ───────────────────────────────────────────────────
BONUS_FORENSIC_REJECT       =  0.3
PENALTY_FORENSIC_APPROVE    = -0.5
BONUS_CAPITAL_EFFICIENT     =  0.2

# ── Legacy / constraint ───────────────────────────────────────────────
PENALTY_DELAYED_NPA_APPROVE     = -1.5
PENALTY_DELAYED_NPA_CONDITIONAL = -1.0
PENALTY_CONSTRAINT_BREACH       = -2.0

# ── Portfolio constraints ─────────────────────────────────────────────
MAX_NPA_RATE             = 0.05
NPA_HIGH_THRESHOLD       = 0.08    # R4: above this → portfolio aware penalties
MIN_CRAR                 = 0.125
CRAR_STRONG_THRESHOLD    = 0.15    # survival bonus threshold
CRAR_THIN_BUFFER         = 0.01    # R4: CRAR within 1% of min → thin buffer
MAX_SECTOR_CONCENTRATION = 0.30
SECTOR_NEAR_LIMIT        = 0.25    # R4: sector at 25%+ → concentration concern
MAX_SINGLE_BORROWER_PCT  = 0.15
INITIAL_CAPITAL          = 100.0

# ── Yield ─────────────────────────────────────────────────────────────
BASE_YIELD_MIN             = 0.02
BASE_YIELD_MAX             = 0.08
CONDITIONAL_YIELD_DISCOUNT = 0.85

# ── Settlement (fires at step 50) ─────────────────────────────────────
# Formula: 0.30×yield + 0.30×(1-npa) + 0.20×compliance + 0.20×capital_util
# Range: [-1.0, +5.0]   (A good episode ends with settlement >= 3.0)
SETTLEMENT_W_YIELD      = 0.30
SETTLEMENT_W_NPA        = 0.30
SETTLEMENT_W_CAPITAL    = 0.20
SETTLEMENT_W_COMPLIANCE = 0.20

# ── Reward clipping ───────────────────────────────────────────────────
# Per-step: [-5.0, +3.0]   Episode total range: [-250, +150]
REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX =  3.0


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
        "accuracy": 0.35,
        "missing_data_handling": 0.25,
        "npa_avoidance": 0.20,
        "capital_efficiency": 0.10,
        "tool_efficiency": 0.10,       # v2: rewards intelligent tool usage
    },
    "task4": {
        "accuracy": 0.25,
        "hard_rule_compliance": 0.25,
        "constraint_satisfaction": 0.25,
        "portfolio_health": 0.15,
        "tool_efficiency": 0.10,       # v2
    },
    "task5": {
        "accuracy": 0.20,
        "survival_rate": 0.20,
        "constraint_satisfaction": 0.20,
        "npa_management": 0.20,
        "portfolio_health": 0.10,
        "tool_efficiency": 0.10,       # v2
    },
}


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO STATE
# ═══════════════════════════════════════════════════════════════

class PortfolioState:
    """Tracks the agent's portfolio across a 50-step episode (v2)."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.total_capital       = initial_capital
        self.capital_remaining   = initial_capital
        self.available_capital   = initial_capital   # v2: used by regulator haircut
        self.capital_deployed    = 0.0
        self.total_loans         = 0
        self.npa_count           = 0
        self.npa_rate            = 0.0
        self.sector_exposure     : Dict[str, float] = {}
        self.borrower_exposures  : List[float] = []
        self.stressed_accounts   : List[Dict] = []
        self.healthy_accounts    : List[Dict] = []
        self.pending_npas        : List[Dict] = []   # scheduled future NPAs
        self.cumulative_reward   = 0.0
        self.cumulative_yield    = 0.0
        self.episode_terminated  = False
        self.termination_reason  = ""
        self.crar                = 1.0
        self.steps_survived      = 0
        self.total_episode_steps = 50     # v2: 50-step episodes

        # v2: maturity event stats
        self.loans_repaid    = 0
        self.loans_defaulted = 0
        self.total_maturity_reward = 0.0

        # v2: tool efficiency tracker (populated by agent_loop, read by grader)
        self.total_tool_calls    = 0
        self.useful_tool_calls   = 0    # calls that preceded a correct decision

        # Alert state (5D)
        self._recent_cc_spike           = False
        self._recent_bounce_surge       = False
        self._recent_gst_miss           = False
        self._recent_adverse_media      = False
        self._recent_credit_degradation = False

    def get_observation(self) -> List[float]:
        """Return 10-dim portfolio observation vector (normalized)."""
        total_exposure = sum(self.sector_exposure.values())
        top_sector_pct = (
            max(self.sector_exposure.values()) / max(total_exposure, 1)
            if self.sector_exposure else 0
        )
        return [
            self.capital_remaining / self.total_capital,               # 0: capital remaining %
            self.capital_deployed / self.total_capital,                # 1: capital deployed %
            self.npa_rate,                                             # 2: NPA rate
            min(len(self.sector_exposure) / 8.0, 1.0),                # 3: sector diversification
            top_sector_pct,                                            # 4: top sector concentration
            min(len(self.stressed_accounts) / 5.0, 1.0),              # 5: stressed count (norm)
            min(len(self.healthy_accounts) / 10.0, 1.0),              # 6: healthy count (norm)
            min(self.total_loans / 50.0, 1.0),                        # 7: total loans (norm, v2=50)
            max(0.0, min(1.0, self.cumulative_reward / 50.0 + 0.5)),  # 8: cum reward (centered)
            max(0.0, min(1.0, self.crar)),                             # 9: CRAR (normalized)
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
        current_t = app_meta.get("timestep", 1)

        if will_default:
            # Schedule NPA with delay
            tier = app_meta.get("tier", "C")
            if tier == "D":
                delay = random.randint(2, 5)
            elif tier == "C":
                delay = random.randint(3, 6)
            else:
                delay = random.randint(4, 8)

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
            # Schedule maturity reward T+10 to T+30 after approval
            maturity_delay = random.randint(10, 30)
            self.healthy_accounts.append({
                "loan_amount"  : effective_loan,
                "sector"       : sector,
                "maturity_step": current_t + maturity_delay,
                "rewarded"     : False,
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
        Process NPA/maturity revelations at current timestep.

        v2: Fires sparse maturity rewards:
            Repaid   → +10.0  (REWARD_LOAN_REPAID)
            Defaulted → -15.0  (PENALTY_LOAN_DEFAULTED, net of recovery)

        Returns (delayed_reward, episode_terminated).
        """
        self.steps_survived = current_t
        delayed_reward = 0.0

        remaining_npas = []
        for npa in self.pending_npas:
            if npa["default_timestep"] <= current_t:
                # ── Maturity event fires ─────────────────────────────────
                loss = npa["loan_amount"] * (1 - npa["recovery_rate"])
                self.capital_remaining -= loss
                self.npa_count += 1

                if self.total_loans > 0:
                    self.npa_rate = self.npa_count / self.total_loans

                # Phase 4: tiered maturity penalty based on recovery rate
                recovery = npa["recovery_rate"]
                if recovery >= 0.50:
                    # Partial recovery — stressed but not catastrophic
                    maturity_reward = PENALTY_LOAN_PARTIAL
                else:
                    # Full default — proportional to actual net loss
                    net_loss_pct = 1.0 - recovery
                    maturity_reward = PENALTY_LOAN_DEFAULTED * net_loss_pct

                delayed_reward += maturity_reward
                self.total_maturity_reward += maturity_reward
                self.loans_defaulted += 1
                self.stressed_accounts.append(npa)
            else:
                remaining_npas.append(npa)

        self.pending_npas = remaining_npas

        # ── Check for healthy loan maturities (repaid loans) ────
        # Healthy loans scheduled to mature: simulate repayment reward
        remaining_healthy = []
        for loan in self.healthy_accounts:
            maturity_step = loan.get("maturity_step", current_t + 1)
            if maturity_step <= current_t and not loan.get("rewarded", False):
                loan["rewarded"] = True
                delayed_reward += REWARD_LOAN_REPAID
                self.total_maturity_reward += REWARD_LOAN_REPAID
                self.loans_repaid += 1
            remaining_healthy.append(loan)
        self.healthy_accounts = remaining_healthy

        # Update CRAR after NPA processing
        self._update_crar()

        # ── Regulatory constraints ───────────────────────────────
        if self.npa_rate > MAX_NPA_RATE and self.total_loans >= 3:
            self.episode_terminated = True
            self.termination_reason = f"NPA rate {self.npa_rate:.1%} exceeds {MAX_NPA_RATE:.0%} limit"
            delayed_reward += PENALTY_CONSTRAINT_BREACH

        if self.capital_remaining < 0:
            self.episode_terminated = True
            self.termination_reason = "Capital exhausted"
            delayed_reward += PENALTY_CONSTRAINT_BREACH

        if self.crar < MIN_CRAR and self.capital_deployed > 0:
            self.episode_terminated = True
            self.termination_reason = f"CRAR {self.crar:.1%} below {MIN_CRAR:.1%} minimum"
            delayed_reward += PENALTY_CONSTRAINT_BREACH

        return delayed_reward, self.episode_terminated


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
    End-of-episode portfolio evaluation — fires at step 50 only.

    Phase 4 formula:
      settlement = 0.30×yield + 0.30×(1-npa) + 0.20×compliance + 0.20×capital_util

    Scaled to range [-1.0, +5.0].
      Perfect episode → +5.0
      Good episode    → +3.0 (target benchmark)
      Average episode → +1.0 - +2.0
      Terminated      → negative
    """
    # Yield score [0, 1]
    max_yield    = INITIAL_CAPITAL * BASE_YIELD_MAX * 2.0
    yield_score  = min(1.0, portfolio.cumulative_yield / max(max_yield, 0.01))

    # NPA avoidance score [0, 1]
    npa_score    = max(0.0, 1.0 - portfolio.npa_rate / MAX_NPA_RATE)

    # Capital utilisation score [0, 1] — optimal 60-80% for 50 steps
    utilization  = portfolio.capital_deployed / portfolio.total_capital if portfolio.total_capital > 0 else 0
    capital_score = min(1.0, utilization / 0.75)

    # Compliance score [0, 1]
    compliance_score = 0.0 if portfolio.episode_terminated else 1.0

    settlement_normalized = (
        SETTLEMENT_W_YIELD      * yield_score
        + SETTLEMENT_W_NPA      * npa_score
        + SETTLEMENT_W_CAPITAL  * capital_score
        + SETTLEMENT_W_COMPLIANCE * compliance_score
    )  # range [0, 1]

    # Scale to [-1.0, +5.0]
    # 0.0 → -1.0  (terrible episode)
    # 0.5 → +2.0  (acceptable)
    # 1.0 → +5.0  (perfect)
    settlement = settlement_normalized * 6.0 - 1.0
    return round(settlement, 4)


# ═══════════════════════════════════════════════════════════════
# R3: FORMAT COMPLIANCE REWARD
# ═══════════════════════════════════════════════════════════════

def compute_format_compliance_reward(
    parse_type      : str,
    parse_confidence: float,
    reasoning_len   : int,
) -> float:
    """
    R3: Format Compliance Reward (10% weight).

    Awards:
      +0.3  submit_decision() called with valid action word
      +0.1  response is one clear word {APPROVE, CONDITIONAL, REJECT}
      -0.3  invalid/unclear action (parse_failure)
      -0.1  extra text that confused parser (low confidence)
    """
    if parse_type == "final_decision" and parse_confidence >= 0.85:
        reward = BONUS_FORMAT_VALID_SUBMIT   # +0.3
        if parse_confidence >= 0.95:
            reward += BONUS_FORMAT_CLEAR_KEYWORD  # +0.1 — very clean
        return round(reward, 4)

    elif parse_type == "fallback_keyword":
        # Keyword found but not proper submit_decision format
        if parse_confidence >= 0.65:
            return BONUS_FORMAT_CLEAR_KEYWORD   # +0.1 — at least unambiguous keyword
        else:
            return PENALTY_FORMAT_CONFUSING     # -0.1 — ambiguous

    elif parse_type in ("default_reject",) or parse_confidence < 0.50:
        return PENALTY_FORMAT_INVALID           # -0.3

    return 0.0


# ═══════════════════════════════════════════════════════════════
# R4: PORTFOLIO AWARENESS REWARD
# ═══════════════════════════════════════════════════════════════

def compute_portfolio_awareness_reward(
    action          : int,
    portfolio       : PortfolioState,
    app_metadata    : Dict[str, Any],
    episode_progress: float,  # 0.0 = step 1, 1.0 = step 50
) -> float:
    """
    R4: Portfolio Awareness Reward (20% weight).

    Penalizes decisions that ignore portfolio state:
      NPA > 8% + APPROVE              → -0.5
      NPA > 8% + REJECT               → +0.3 (good conservative)
      Sector near 30% + APPROVE same  → -0.8
      CRAR buffer thin + APPROVE      → -0.5
      Late game + CRAR healthy + good → +0.3
    """
    reward = 0.0
    sector = app_metadata.get("sector", "")
    hidden_pd = app_metadata.get("hidden_pd", 0.5)

    # ── NPA awareness ─────────────────────────────────────────────
    if portfolio.npa_rate > NPA_HIGH_THRESHOLD:
        if action == 0:   # APPROVE when NPA already dangerously high
            reward += PENALTY_NPA_HIGH_APPROVE   # -0.5
        elif action == 2: # REJECT — conservative when needed
            reward += BONUS_NPA_HIGH_REJECT       # +0.3

    # ── Sector concentration awareness ───────────────────────────
    if sector and action in (0, 1) and portfolio.sector_exposure:
        total_deployed = sum(portfolio.sector_exposure.values())
        if total_deployed > 0:
            current_sector_pct = portfolio.sector_exposure.get(sector, 0.0) / total_deployed
            if current_sector_pct > SECTOR_NEAR_LIMIT:   # already at 25%+
                reward += PENALTY_SECTOR_CONC_APPROVE    # -0.8

    # ── CRAR buffer awareness ─────────────────────────────────────
    if action in (0, 1):
        crar_buffer = portfolio.crar - MIN_CRAR
        if 0 < crar_buffer < CRAR_THIN_BUFFER:
            reward += PENALTY_CRAR_THIN_APPROVE   # -0.5

    # ── Late-game good decision bonus ────────────────────────────
    if episode_progress > 0.80 and portfolio.crar >= CRAR_STRONG_THRESHOLD:
        if action == 0 and hidden_pd < 0.25:
            reward += BONUS_LATE_GAME_GOOD   # +0.3 — late game, healthy, good loan

    return round(reward, 4)


# ═══════════════════════════════════════════════════════════════
# SURVIVAL BONUS
# ═══════════════════════════════════════════════════════════════

def compute_survival_bonus(crar: float) -> float:
    """
    Survival bonus — called every 10 steps if CRAR is above minimum.
    CRAR > 15%         → +0.10
    CRAR 12.5%-15%     →  +0.05 (survived but stressed)
    CRAR < 12.5%       →  0.00  (episode terminates via PortfolioState)
    """
    if crar >= CRAR_STRONG_THRESHOLD:
        return BONUS_SURVIVAL_CRAR_STRONG
    elif crar >= MIN_CRAR:
        return BONUS_SURVIVAL_CRAR_STRESSED
    return 0.0


# ═══════════════════════════════════════════════════════════════
# REWARD COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_step_reward(
    action               : int,
    app_metadata         : Dict[str, Any],
    portfolio            : PortfolioState,
    is_final_step        : bool  = False,
    tool_efficiency_delta: float = 0.0,
    parse_type           : str   = "final_decision",
    parse_confidence     : float = 0.90,
    reasoning_len        : int   = 100,
    episode_progress     : float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-step reward using 4 weighted functions (Phase 4).

      R1 Decision Correctness  (40%): PD-range based ±rewards
      R2 Hard Rule Compliance  (30%): +0.5 per HR / -2.0 violation
      R3 Format Compliance     (10%): +0.3/+0.1 / -0.3/-0.1
      R4 Portfolio Awareness   (20%): NPA/CRAR/sector/late-game

    Plus:
      Yield reward, tool efficiency overlay, settlement (final step).

    Returns (total_reward, reward_components_dict)
    """
    hidden_pd  = app_metadata["hidden_pd"]
    hard_rules = app_metadata.get("hard_rules_triggered", [])
    alerts     = app_metadata.get("alerts", [])
    red_alerts = [a for a in alerts if a["severity"] == "RED"]
    loan_amount = app_metadata.get("loan_amount_cr", 5.0)
    sector      = app_metadata.get("sector", "Manufacturing")

    components: Dict[str, float] = {}

    # ═══ R1: Decision Correctness (40% of signal) ══════════════════
    if hard_rules:
        # Hard rules take priority — R2 handles this, but R1 still fires
        if action == 0:
            components["r1_correctness"] = PENALTY_HIGH_PD_APPROVE    # -2.0
        elif action == 1:
            components["r1_correctness"] = PENALTY_MED_PD_APPROVE     # -0.8
        else:
            components["r1_correctness"] = REWARD_CORRECT_REJECT       # +1.0
    else:
        # PD-range based correctness
        if hidden_pd >= 0.45:    # HIGH risk band — should REJECT
            if action == 2:
                components["r1_correctness"] = REWARD_CORRECT_REJECT   # +1.0
            elif action == 1:
                components["r1_correctness"] = -0.5                    # borderline
            else:
                components["r1_correctness"] = PENALTY_HIGH_PD_APPROVE # -2.0 dangerous
        elif hidden_pd >= 0.25:  # MEDIUM risk band — should CONDITIONAL
            if action == 1:
                components["r1_correctness"] = REWARD_CORRECT_CONDITIONAL  # +0.8
            elif action == 0:
                components["r1_correctness"] = PENALTY_MED_PD_APPROVE      # -0.8
            else:
                components["r1_correctness"] = 0.2   # over-cautious but ok
        else:                    # LOW risk band — should APPROVE
            if action == 0:
                components["r1_correctness"] = REWARD_CORRECT_APPROVE   # +1.0
            elif action == 1:
                components["r1_correctness"] = 0.3                      # conservative ok
            else:
                components["r1_correctness"] = PENALTY_LOW_PD_REJECT    # -0.3 opp cost

    # ═══ R2: Hard Rule Compliance (30% of signal) ══════════════════
    if hard_rules:
        if action == 2:   # Correctly rejected — +0.5 per hard rule (capped at +1.5)
            hr_bonus = min(len(hard_rules) * BONUS_HARD_RULE_REJECT, 1.5)
            components["r2_hr_compliance"] = hr_bonus
        elif action == 0:
            components["r2_hr_compliance"] = PENALTY_HARD_RULE_APPROVE      # -2.0
        else:
            components["r2_hr_compliance"] = PENALTY_HARD_RULE_CONDITIONAL  # -1.0

    # Forensic RED alert alignment
    if red_alerts:
        if action == 2:
            components["r2_forensic"] = BONUS_FORENSIC_REJECT    # +0.3
        elif action == 0:
            components["r2_forensic"] = PENALTY_FORENSIC_APPROVE  # -0.5

    # ═══ R3: Format Compliance (10% of signal) ═════════════════════
    r3 = compute_format_compliance_reward(parse_type, parse_confidence, reasoning_len)
    if r3 != 0.0:
        components["r3_format"] = r3

    # ═══ R4: Portfolio Awareness (20% of signal) ═══════════════════
    r4 = compute_portfolio_awareness_reward(action, portfolio, app_metadata, episode_progress)
    if r4 != 0.0:
        components["r4_portfolio"] = r4

    # ═══ Yield Reward ══════════════════════════════════════════════
    yield_reward = _compute_yield_reward(action, hidden_pd, loan_amount, portfolio)
    if yield_reward != 0:
        components["yield_reward"] = round(yield_reward, 4)

    # ═══ Portfolio Loan Booking ════════════════════════════════════
    if action in (0, 1):
        is_conditional = (action == 1)
        app_meta_for_portfolio = {
            "hidden_pd"         : hidden_pd,
            "tier"              : app_metadata.get("tier", "C"),
            "timestep"          : app_metadata.get("timestep", 1),
            "collateral_coverage": app_metadata.get("collateral_ratio", 1.0),
        }
        portfolio_mod, _ = portfolio.approve_loan(
            loan_amount    = loan_amount,
            sector         = sector,
            app_meta       = app_meta_for_portfolio,
            is_conditional = is_conditional,
        )
        if portfolio_mod != 0:
            components["portfolio_impact"] = portfolio_mod

    # ═══ Capital Preservation (rejects of high-PD) ═════════════════
    if action == 2 and hidden_pd >= 0.40:
        components["capital_preservation"] = 0.1

    # ═══ Tool Efficiency Overlay ═══════════════════════════════════
    if tool_efficiency_delta != 0.0:
        components["tool_efficiency"] = round(tool_efficiency_delta, 4)

    # ═══ Settlement Reward (final step only) ═══════════════════════
    if is_final_step:
        settlement = compute_settlement_reward(portfolio)
        components["settlement_reward"] = round(settlement, 4)

    # ═══ Aggregate & Per-step Clip [-5.0, +3.0] ════════════════════
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

        # v2: tool efficiency — classify a tool call as useful if correct decision followed
        # (tracked via portfolio.usedtool_calls, set by agent_loop)

    # Raw metrics
    accuracy            = correct / n if n > 0 else 0.0
    forensic_handling   = forensic_correct / forensic_total if forensic_total > 0 else 1.0
    hard_rule_compliance= hard_rule_correct / hard_rule_total if hard_rule_total > 0 else 1.0
    constraint_satisfaction = 0.0 if portfolio.episode_terminated else 1.0
    npa_avoidance       = max(0.0, 1.0 - portfolio.npa_rate / MAX_NPA_RATE)
    npa_management      = max(0.0, 1.0 - portfolio.npa_rate / 0.10)

    utilization         = portfolio.capital_deployed / portfolio.total_capital if portfolio.total_capital > 0 else 0
    capital_efficiency  = min(1.0, utilization / 0.70)

    capital_remaining_pct = portfolio.capital_remaining / portfolio.total_capital if portfolio.total_capital > 0 else 0
    portfolio_health    = max(0.0, capital_remaining_pct * (1 - portfolio.npa_rate))

    survival_rate       = portfolio.steps_survived / portfolio.total_episode_steps

    missing_data_handling = (
        1.0 - (missing_bad_approvals / missing_total)
        if missing_total > 0 else 1.0
    )

    # v2: tool efficiency metric
    tool_efficiency = (
        portfolio.useful_tool_calls / max(portfolio.total_tool_calls, 1)
        if portfolio.total_tool_calls > 0 else 0.5  # neutral if no tools used
    )

    # ─── Build the metric dict ───────────────────────────────
    metrics = {
        "accuracy"               : accuracy,
        "forensic_handling"      : forensic_handling,
        "hard_rule_compliance"   : hard_rule_compliance,
        "constraint_satisfaction": constraint_satisfaction,
        "npa_avoidance"          : npa_avoidance,
        "npa_management"         : npa_management,
        "capital_efficiency"     : capital_efficiency,
        "portfolio_health"       : portfolio_health,
        "survival_rate"          : survival_rate,
        "missing_data_handling"  : missing_data_handling,
        "tool_efficiency"        : tool_efficiency,   # v2
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
