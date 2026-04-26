"""
IntelliCredit v2 — World State & Multi-Agent Environment
=========================================================
Upgrades v1 environment to v2.0:

  - 50-step episodes (was 12)
  - 55D observation (45D + 10 memory features)
  - Persistent WorldState across the episode
  - BorrowerAgent: re-application state machine (up to 3 attempts)
  - RegulatorAgent: audits at steps 10/20/30/40/50 with escalation
  - Tool call support (step counter does NOT advance on tool calls)
  - All v1 endpoints/schemas remain backward compatible
"""

from uuid import uuid4
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import random
import math
import json

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import IntelliCreditAction, IntelliCreditObservation, ApplicationSummary
except ImportError:
    from models import IntelliCreditAction, IntelliCreditObservation, ApplicationSummary

try:
    from .dataset import generate_episode, application_to_text, generate_application, SECTORS
    from .reward import (
        PortfolioState, compute_step_reward, grade_episode,
        compute_survival_bonus,
    )
except ImportError:
    from server.dataset import generate_episode, application_to_text, generate_application, SECTORS
    from server.reward import (
        PortfolioState, compute_step_reward, grade_episode,
        compute_survival_bonus,
    )


# ═══════════════════════════════════════════════════════════════
# TASK CONFIGURATIONS  (v2: all tasks now go to 50 steps)
# ═══════════════════════════════════════════════════════════════

TASK_CONFIGS = {
    "task1": {"num_steps": 50, "description": "Easy — Clean profiles, macro shock at step 40"},
    "task2": {"num_steps": 50, "description": "Medium — Forensic alerts, shock at step 30"},
    "task3": {"num_steps": 50, "description": "Hard — Macro shocks + missing data + repeat applicants"},
    "task4": {"num_steps": 50, "description": "Expert — Hard-rule violations, tight CRAR windows"},
    "task5": {"num_steps": 50, "description": "Master — Full constraints, cascading NPAs, 5 audits"},
}

# Base audit steps — jittered ±1 per episode (see WorldState._jitter_audits)
AUDIT_STEPS_BASE = {10, 20, 30, 40, 50}

# Regulator reward constants
AUDIT_BONUS_CLEAN       =  2.0
AUDIT_PENALTY_NPA       = -8.0
AUDIT_PENALTY_CRAR      = -15.0
AUDIT_PENALTY_SECTOR    = -8.0
AUDIT_PENALTY_BORROWER  = -5.0
MATURITY_REWARD_REPAID  =  10.0
MATURITY_PENALTY_DEFAULT = -15.0
CAPITAL_HAIRCUT_PCT     = 0.10   # 10% capital removed on 2nd consecutive audit fail


# ═══════════════════════════════════════════════════════════════
# CREDIT CYCLE ENUM
# ═══════════════════════════════════════════════════════════════

class CreditCyclePhase(str, Enum):
    EXPANSION   = "EXPANSION"
    PEAK        = "PEAK"
    CONTRACTION = "CONTRACTION"
    TROUGH      = "TROUGH"


# ═══════════════════════════════════════════════════════════════
# WORLD STATE
# ═══════════════════════════════════════════════════════════════

class WorldState:
    """
    Persistent macro + sector + event state for an entire 50-step episode.
    Resets with each episode. NOT reset between steps.
    """

    def __init__(self, seed: int = 42):
        rng = random.Random(seed)

        # ── Macro Economy Layer ──────────────────────────────────
        self.interest_rate_trend: float = rng.uniform(0.06, 0.09)
        self.gdp_growth_index: float = rng.uniform(0.55, 0.75)
        self.inflation_index: float = rng.uniform(0.25, 0.40)
        self.credit_cycle_phase: CreditCyclePhase = CreditCyclePhase.EXPANSION
        self.macro_shock_active: bool = False
        self.macro_stress: float = 0.20
        self.macro_stress_history: List[float] = []

        # Shock schedule: fires between step 20–25 (random)
        self.shock_step: int = rng.randint(20, 25)
        self.shock_sector: str = rng.choice(SECTORS)

        # ── Phase 4: Audit step jitter ±1 per episode ───────────
        self.audit_steps: set = self._jitter_audits(rng)

        # ── Sector Health Layer ──────────────────────────────────
        self.sector_exposure: Dict[str, float] = {}        # sector → deployed ₹ Cr
        self.sector_stress_scores: Dict[str, float] = {s: 0.0 for s in SECTORS}
        self.sector_npa_rates: Dict[str, float] = {s: 0.0 for s in SECTORS}

        # ── Capital & Portfolio Layer ────────────────────────────
        self.total_capital_deployed: float = 0.0
        self.available_capital: float = 100.0      # ₹100 Cr baseline
        self.current_crar: float = 1.0
        self.current_npa_rate: float = 0.0
        self.approved_loans_ledger: List[Dict[str, Any]] = []

        # ── Pending Events Layer ─────────────────────────────────
        # Each entry: {loan_id, step_due, pd_score, loan_amount, sector}
        self.pending_maturity_checks: List[Dict[str, Any]] = []

        # ── Borrower History Layer ───────────────────────────────
        # borrower_id → {attempts, last_action, profile, retry_step}
        self.seen_borrowers: Dict[str, Dict[str, Any]] = {}
        self.rejected_borrowers_queue: List[Dict[str, Any]] = []  # ready to reapply
        self.approved_borrowers: List[str] = []

        # ── Regulator Tracking ───────────────────────────────────
        self.consecutive_audit_failures: int = 0
        self.audit_history: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []

        # ── Memory Feature Accumulators ──────────────────────────
        self.recent_decisions: List[int] = []       # last 10 actions
        self.recent_npa_flags: List[float] = []     # last 10 steps npa_rate
        self.reflection_count: int = 0              # lessons stored
        self.tool_calls_total: int = 0              # world_model_confidence proxy

    # ── Macro drift (called every step) ─────────────────────────

    @staticmethod
    def _jitter_audits(rng: random.Random) -> set:
        """
        Phase 4 anti-hacking: jitter each audit step ±1 so agent cannot
        predict exactly when audits fire and game the portfolio.
        Base: {10, 20, 30, 40, 50}, jitter: choice([-1, 0, +1]).
        Steps clamped to [2, 50].
        """
        base = [10, 20, 30, 40, 50]
        jittered = set()
        for s in base:
            j = rng.choice([-1, 0, 1])
            jittered.add(max(2, min(50, s + j)))
        return jittered

    def tick_macro(self, step: int) -> None:
        """Advance macro state for this timestep."""
        # Save history for trend calculation
        self.macro_stress_history.append(self.macro_stress)

        # Macro shock fires at shock_step
        if step == self.shock_step:
            self.macro_shock_active = True
            self.macro_stress = random.uniform(0.65, 0.85)
            self.sector_stress_scores[self.shock_sector] = random.uniform(0.60, 0.90)
            # Spike NPA probability for loans in shocked sector
            self._apply_shock_to_portfolio()

        elif self.macro_shock_active:
            # Gradual buildup after shock
            self.macro_stress = min(0.90, self.macro_stress + random.uniform(0.01, 0.03))
        else:
            # Slow drift before shock
            if step % 5 == 0:
                drift = random.uniform(-0.02, 0.04)
                self.macro_stress = max(0.10, min(0.60, self.macro_stress + drift))

        # Update interest rate trend
        if self.macro_shock_active:
            self.interest_rate_trend = min(0.12, self.interest_rate_trend + 0.005)
        else:
            self.interest_rate_trend += random.uniform(-0.002, 0.003)
            self.interest_rate_trend = max(0.04, min(0.12, self.interest_rate_trend))

        # Update credit cycle
        if step < 15:
            self.credit_cycle_phase = CreditCyclePhase.EXPANSION
        elif step < 25:
            self.credit_cycle_phase = CreditCyclePhase.PEAK
        elif step < 38:
            self.credit_cycle_phase = CreditCyclePhase.CONTRACTION
        else:
            self.credit_cycle_phase = CreditCyclePhase.TROUGH

        # Macro state list for observation (matches original 5D format)
        self.gdp_growth_index = max(0.0, 0.7 - self.macro_stress * 0.5)
        self.inflation_index  = min(1.0, 0.3 + self.macro_stress * 0.4)

    def _apply_shock_to_portfolio(self) -> None:
        """When macro shock fires, increase default prob for pending loans in stressed sector."""
        for entry in self.pending_maturity_checks:
            if entry.get("sector") == self.shock_sector:
                entry["pd_score"] = min(0.99, entry["pd_score"] * 1.40)

    def record_decision(self, action: int, npa_rate: float) -> None:
        self.recent_decisions.append(action)
        self.recent_npa_flags.append(npa_rate)
        if len(self.recent_decisions) > 10:
            self.recent_decisions.pop(0)
            self.recent_npa_flags.pop(0)

    def get_macro_obs(self) -> List[float]:
        """Return 5D macro observation vector (same format as v1)."""
        cycle_val = {
            CreditCyclePhase.EXPANSION: 0.8,
            CreditCyclePhase.PEAK: 0.6,
            CreditCyclePhase.CONTRACTION: 0.3,
            CreditCyclePhase.TROUGH: 0.1,
        }.get(self.credit_cycle_phase, 0.5)
        return [
            self.macro_stress,
            1.0 if self.macro_shock_active else 0.0,
            self.gdp_growth_index,
            self.inflation_index,
            cycle_val,
        ]

    def compute_memory_features(
        self,
        portfolio: PortfolioState,
        current_step: int,
        total_steps: int,
        borrower_persistence: float,
        reflection_count: int,
        tool_calls_total: int,
    ) -> List[float]:
        """Compute 10D memory feature vector (dims 45–54)."""
        # 45: rolling NPA rate over last 10 steps
        rolling_npa = (
            sum(self.recent_npa_flags) / len(self.recent_npa_flags)
            if self.recent_npa_flags else 0.0
        )

        # 46: approval rate over last 10 decisions
        approval_rate = (
            sum(1 for a in self.recent_decisions if a == 0) / len(self.recent_decisions)
            if self.recent_decisions else 0.5
        )

        # 47: highest sector concentration
        total_exposed = sum(portfolio.sector_exposure.values())
        sector_max_conc = (
            max(portfolio.sector_exposure.values()) / max(total_exposed, 1.0)
            if portfolio.sector_exposure else 0.0
        )

        # 48: macro stress trend (positive = worsening, negative = improving)
        if len(self.macro_stress_history) >= 3:
            trend = self.macro_stress_history[-1] - self.macro_stress_history[-3]
        else:
            trend = 0.0
        macro_stress_trend = max(-1.0, min(1.0, trend * 5))   # scale to [-1, 1]

        # 49: borrower persistence score
        bps = max(0.0, min(1.0, borrower_persistence))

        # 50: audit risk score — how close is next audit
        next_audit = min((a for a in AUDIT_STEPS_BASE if a > current_step), default=50)
        steps_to_audit = next_audit - current_step
        audit_risk = max(0.0, 1.0 - steps_to_audit / 10.0)

        # 51: capital buffer ratio — how far above CRAR minimum
        crar_buffer = max(0.0, min(1.0, (portfolio.crar - 0.125) / 0.125))

        # 52: reflection count (normalized)
        refl_norm = min(1.0, reflection_count / 10.0)

        # 53: episode progress
        ep_progress = current_step / max(total_steps, 1)

        # 54: world model confidence (tool calls → better model)
        wm_confidence = min(1.0, tool_calls_total / 20.0)

        return [
            round(rolling_npa, 4),
            round(approval_rate, 4),
            round(sector_max_conc, 4),
            round(macro_stress_trend, 4),
            round(bps, 4),
            round(audit_risk, 4),
            round(crar_buffer, 4),
            round(refl_norm, 4),
            round(ep_progress, 4),
            round(wm_confidence, 4),
        ]


# ═══════════════════════════════════════════════════════════════
# BORROWER AGENT (programmatic, lives inside env)
# ═══════════════════════════════════════════════════════════════

class BorrowerAgent:
    """
    Manages the re-application state machine for rejected borrowers.
    Surface profile improves each attempt; hidden risk stays same or worsens.
    """

    MAX_ATTEMPTS = 3
    COOLING_MIN  = 3
    COOLING_MAX  = 5

    def __init__(self):
        # borrower_id → {attempts, base_app, retry_step, pd_original}
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._pending_retry: List[Dict[str, Any]] = []  # sorted by retry_step

    def register_new(self, borrower_id: str, app: Dict[str, Any], step: int) -> None:
        """Register a first-time borrower application."""
        self._registry[borrower_id] = {
            "attempts": 1,
            "base_app": app,
            "retry_step": None,
            "pd_original": app["metadata"]["hidden_pd"],
        }

    def on_rejected(self, borrower_id: str, current_step: int) -> None:
        """Called when a borrower is rejected. Schedules a retry if under max attempts."""
        rec = self._registry.get(borrower_id)
        if rec is None or rec["attempts"] >= self.MAX_ATTEMPTS:
            return

        cooling = random.randint(self.COOLING_MIN, self.COOLING_MAX)
        retry_step = current_step + cooling
        rec["retry_step"] = retry_step

        improved_app = self._improve_profile(rec["base_app"], rec["attempts"])
        improved_app["metadata"]["borrower_id"]      = borrower_id
        improved_app["metadata"]["is_repeat_applicant"] = True
        improved_app["metadata"]["attempt_number"]   = rec["attempts"] + 1

        self._pending_retry.append({
            "borrower_id": borrower_id,
            "retry_step": retry_step,
            "app": improved_app,
        })

    def pop_ready(self, current_step: int) -> Optional[Dict[str, Any]]:
        """Return most urgent retry-app due at or before current_step, or None."""
        due = [r for r in self._pending_retry if r["retry_step"] <= current_step]
        if not due:
            return None
        # Pick earliest due
        due.sort(key=lambda r: r["retry_step"])
        entry = due[0]
        self._pending_retry.remove(entry)
        # Increment attempt counter
        rec = self._registry.get(entry["borrower_id"])
        if rec:
            rec["attempts"] += 1
        return entry["app"]

    @staticmethod
    def _improve_profile(app: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """
        Return a new application with improved surface metrics.
        Hidden PD is NOT improved (and may worsen on 2nd attempt).
        """
        import copy
        new_app = copy.deepcopy(app)
        f = new_app["features"]
        r = new_app["raw_values"]
        m = new_app["metadata"]

        if attempt == 1:
            # First retry: cosmetic financial ratio improvements
            _boost(f, "dscr_proxy",              0.08)
            _boost(f, "current_ratio",            0.05)
            _reduce(f, "debt_to_equity",          0.10)
            _boost(f, "collateral_coverage_ratio",0.15)
            _boost(r, "dscr",                     0.08)
            _boost(r, "current_ratio",            0.05)
            _boost(r, "collateral_ratio",         0.15)
        else:
            # Second retry: further minor improvements + PD worsens slightly
            _boost(f, "dscr_proxy",              0.05)
            _boost(f, "current_ratio",            0.03)
            _reduce(f, "debt_to_equity",          0.05)
            # Remove lowest-severity forensic flags (cosmetic)
            alerts = m.get("alerts", [])
            m["alerts"] = [a for a in alerts if a.get("severity") == "RED"]
            # Hidden PD actually increases (desperation signal)
            m["hidden_pd"] = min(0.99, m["hidden_pd"] * 1.12)

        return new_app


def _boost(d: Dict, key: str, pct: float) -> None:
    if key in d and isinstance(d[key], (int, float)) and d[key] != -1.0:
        d[key] = min(1.0, d[key] * (1 + pct))

def _reduce(d: Dict, key: str, pct: float) -> None:
    if key in d and isinstance(d[key], (int, float)) and d[key] != -1.0:
        d[key] = max(0.0, d[key] * (1 - pct))


# ═══════════════════════════════════════════════════════════════
# REGULATOR AGENT (programmatic, lives inside env)
# ═══════════════════════════════════════════════════════════════

class RegulatorAgent:
    """Audits portfolio at scheduled steps and applies penalties/bonuses."""

    NPA_WARNING_THRESHOLD    = 0.03
    NPA_VIOLATION_THRESHOLD  = 0.05
    CRAR_WARNING_THRESHOLD   = 0.15
    CRAR_VIOLATION_THRESHOLD = 0.125
    SECTOR_WARNING_THRESHOLD = 0.25
    SECTOR_VIOLATION_THRESHOLD = 0.30
    BORROWER_WARNING_THRESHOLD = 0.12
    BORROWER_VIOLATION_THRESHOLD = 0.15

    def audit(
        self,
        step: int,
        portfolio: PortfolioState,
        world: WorldState,
    ) -> Tuple[float, Dict[str, Any], bool]:
        """
        Run a full regulatory audit.

        Returns:
            (total_reward_delta, audit_report_dict, episode_terminated)
        """
        report: Dict[str, Any] = {
            "step": step,
            "checks": {},
            "violations": [],
            "warnings": [],
            "total_penalty": 0.0,
            "is_clean": True,
        }
        penalty = 0.0
        terminated = False

        # ── Check 1: NPA Rate ────────────────────────────────────
        npa = portfolio.npa_rate
        if npa >= self.NPA_VIOLATION_THRESHOLD:
            penalty += AUDIT_PENALTY_NPA
            report["violations"].append(f"NPA rate {npa:.1%} ≥ 5% limit")
            report["is_clean"] = False
        elif npa >= self.NPA_WARNING_THRESHOLD:
            report["warnings"].append(f"NPA rate {npa:.1%} approaching 5% limit")
        report["checks"]["npa_rate"] = round(npa, 4)

        # ── Check 2: CRAR ────────────────────────────────────────
        crar = portfolio.crar
        if crar < self.CRAR_VIOLATION_THRESHOLD:
            penalty += AUDIT_PENALTY_CRAR
            report["violations"].append(f"CRAR {crar:.1%} < 12.5% minimum")
            report["is_clean"] = False
        elif crar < self.CRAR_WARNING_THRESHOLD:
            report["warnings"].append(f"CRAR {crar:.1%} approaching 12.5% minimum")
        report["checks"]["crar"] = round(crar, 4)

        # ── Check 3: Sector Concentration ───────────────────────
        total_deployed = sum(portfolio.sector_exposure.values())
        max_sector_pct = 0.0
        worst_sector   = ""
        if total_deployed > 0:
            for sector, amt in portfolio.sector_exposure.items():
                pct = amt / total_deployed
                if pct > max_sector_pct:
                    max_sector_pct = pct
                    worst_sector   = sector
        if max_sector_pct > self.SECTOR_VIOLATION_THRESHOLD:
            penalty += AUDIT_PENALTY_SECTOR
            report["violations"].append(
                f"Sector concentration {worst_sector}: {max_sector_pct:.1%} > 30%"
            )
            report["is_clean"] = False
        elif max_sector_pct > self.SECTOR_WARNING_THRESHOLD:
            report["warnings"].append(
                f"Sector {worst_sector} at {max_sector_pct:.1%} approaching 30%"
            )
        report["checks"]["max_sector_concentration"] = round(max_sector_pct, 4)

        # ── Check 4: Single Borrower Limit ───────────────────────
        max_borrower_pct = 0.0
        if portfolio.total_capital > 0 and portfolio.borrower_exposures:
            max_borrower_pct = max(portfolio.borrower_exposures) / portfolio.total_capital
        if max_borrower_pct > self.BORROWER_VIOLATION_THRESHOLD:
            penalty += AUDIT_PENALTY_BORROWER
            report["violations"].append(
                f"Single borrower exposure {max_borrower_pct:.1%} > 15%"
            )
            report["is_clean"] = False
        elif max_borrower_pct > self.BORROWER_WARNING_THRESHOLD:
            report["warnings"].append(
                f"Single borrower at {max_borrower_pct:.1%} approaching 15% limit"
            )
        report["checks"]["max_borrower_exposure"] = round(max_borrower_pct, 4)

        # ── Escalation logic ─────────────────────────────────────
        if report["is_clean"]:
            world.consecutive_audit_failures = 0
            penalty += AUDIT_BONUS_CLEAN
            report["outcome"] = "CLEAN — Audit passed"
        else:
            world.consecutive_audit_failures += 1
            report["outcome"] = f"FAILED — {len(report['violations'])} violation(s)"

            if world.consecutive_audit_failures == 2:
                # Capital haircut
                haircut = portfolio.capital_remaining * CAPITAL_HAIRCUT_PCT
                portfolio.capital_remaining -= haircut
                portfolio.available_capital = portfolio.capital_remaining
                report["capital_haircut"] = round(haircut, 2)
                report["outcome"] += " | Capital haircut applied"

            elif world.consecutive_audit_failures >= 3:
                # Shutdown
                terminated = True
                portfolio.episode_terminated = True
                portfolio.termination_reason = "REGULATORY_SHUTDOWN: 3 consecutive audit failures"
                report["outcome"] += " | REGULATORY SHUTDOWN — Episode terminated"

        report["total_penalty"] = round(penalty, 4)
        report["consecutive_failures"] = world.consecutive_audit_failures
        world.audit_history.append(report)

        return penalty, report, terminated


# ═══════════════════════════════════════════════════════════════
# GLOBAL SESSION STORE
# ═══════════════════════════════════════════════════════════════

_SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def _tool_signature(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Stable signature used to detect duplicate tool calls within one step."""
    return json.dumps(
        {"tool": tool_name, "args": tool_args},
        sort_keys=True,
        default=str,
    )


# ═══════════════════════════════════════════════════════════════
# MAIN ENVIRONMENT — v2.0
# ═══════════════════════════════════════════════════════════════

class IntelliCreditEnvironment(Environment):
    """
    IntelliCredit-CreditAppraisal-v2.0

    50-step multi-agent credit underwriting environment.
    Backward compatible with all v1 OpenEnv endpoints.

    New in v2:
      - 55D observation space (45D base + 10D memory)
      - WorldState persists across the full episode
      - BorrowerAgent: re-application state machine
      - RegulatorAgent: audits at steps 10/20/30/40/50
      - Tool call support (step counter does NOT advance on tool calls)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "task3"):
        self._task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._portfolio: Optional[PortfolioState] = None
        self._world: Optional[WorldState] = None
        self._borrower_agent: Optional[BorrowerAgent] = None
        self._regulator: Optional[RegulatorAgent] = None
        self._applications: List[Dict[str, Any]] = []
        self._current_step = 0
        self._actions_taken: List[int] = []
        self._episode_seed = 42
        self._done = False
        self._total_steps = TASK_CONFIGS.get(task_id, TASK_CONFIGS["task3"])["num_steps"]
        # v2: tool call tracking per step
        self._step_tool_call_count: int = 0
        self._total_tool_calls: int = 0
        self._step_tool_signatures: set = set()
        self._step_redundant_tool_calls: int = 0
        # v2: current app borrower id for rejection tracking
        self._current_borrower_id: Optional[str] = None

    # ── Session Persistence (HTTP stateless bridge) ──────────────

    def _save_to_store(self, episode_id: str) -> None:
        _SESSION_STORE[episode_id] = {
            "task_id":              self._task_id,
            "portfolio":            self._portfolio,
            "world":                self._world,
            "borrower_agent":       self._borrower_agent,
            "regulator":            self._regulator,
            "applications":         self._applications,
            "current_step":         self._current_step,
            "actions_taken":        list(self._actions_taken),
            "episode_seed":         self._episode_seed,
            "done":                 self._done,
            "total_steps":          self._total_steps,
            "step_tool_call_count": self._step_tool_call_count,
            "total_tool_calls":     self._total_tool_calls,
            "step_tool_signatures":  list(self._step_tool_signatures),
            "step_redundant_tool_calls": self._step_redundant_tool_calls,
            "current_borrower_id":  self._current_borrower_id,
        }

    def _restore_from_store(self, episode_id: str) -> bool:
        data = _SESSION_STORE.get(episode_id)
        if data is None:
            return False
        self._task_id              = data["task_id"]
        self._portfolio            = data["portfolio"]
        self._world                = data["world"]
        self._borrower_agent       = data["borrower_agent"]
        self._regulator            = data["regulator"]
        self._applications         = data["applications"]
        self._current_step         = data["current_step"]
        self._actions_taken        = data["actions_taken"]
        self._episode_seed         = data["episode_seed"]
        self._done                 = data["done"]
        self._total_steps          = data["total_steps"]
        self._step_tool_call_count = data.get("step_tool_call_count", 0)
        self._total_tool_calls     = data.get("total_tool_calls", 0)
        self._step_tool_signatures = set(data.get("step_tool_signatures", []))
        self._step_redundant_tool_calls = data.get("step_redundant_tool_calls", 0)
        self._current_borrower_id  = data.get("current_borrower_id")
        self._state = State(episode_id=episode_id, step_count=self._current_step)
        return True

    # ── Reset ────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> IntelliCreditObservation:
        """Reset environment for a new 50-step episode."""
        if task_id:
            self._task_id = task_id

        self._episode_seed = seed if seed is not None else random.randint(1, 100_000)
        resolved_episode_id = episode_id or str(uuid4())
        self._state    = State(episode_id=resolved_episode_id, step_count=0)
        self._portfolio = PortfolioState()
        self._world     = WorldState(seed=self._episode_seed)
        self._borrower_agent = BorrowerAgent()
        self._regulator      = RegulatorAgent()
        self._current_step   = 0
        self._actions_taken  = []
        self._done           = False
        self._total_steps    = TASK_CONFIGS.get(self._task_id, TASK_CONFIGS["task3"])["num_steps"]
        self._portfolio.total_episode_steps = self._total_steps
        self._step_tool_call_count = 0
        self._total_tool_calls     = 0
        self._step_tool_signatures  = set()
        self._step_redundant_tool_calls = 0
        self._current_borrower_id  = None

        # Generate all base applications upfront
        self._applications = generate_episode(
            task_id=self._task_id,
            num_steps=self._total_steps,
            seed=self._episode_seed,
        )

        # Register all generated borrowers in BorrowerAgent
        for i, app in enumerate(self._applications):
            bid = app["metadata"].get("borrower_id", f"B-{i:03d}")
            app["metadata"]["borrower_id"] = bid
            self._borrower_agent.register_new(bid, app, step=i + 1)

        self._save_to_store(resolved_episode_id)
        return self._build_observation(reward=0.0, reward_components={})

    # ── Step ─────────────────────────────────────────────────────

    def step(
        self,
        action: IntelliCreditAction,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> IntelliCreditObservation:
        """
        Execute one credit decision or tool call.

        If the action carries a tool call (parsed upstream), the step counter
        does NOT advance — only a tool result is returned.
        For final decisions (APPROVE/CONDITIONAL/REJECT), the step advances.
        """
        # Restore session state (HTTP framework creates fresh instance each call)
        if episode_id and self._portfolio is None:
            if not self._restore_from_store(episode_id):
                return self._build_observation(
                    reward=0.0,
                    reward_components={"error": "unknown_episode_id"},
                    force_done=True,
                )

        if self._done or self._portfolio is None:
            return self._build_observation(
                reward=0.0,
                reward_components={"error": -1.0},
                force_done=True,
            )

        forced_tool_limit = False
        raw_llm_output = (
            getattr(action, "llm_output", None)
            or getattr(action, "raw_text", None)
        )

        # v2 online-agent mode: parse raw LLM text at the environment boundary.
        # Tool calls return a tool result and do NOT advance macro time or step count.
        if raw_llm_output:
            try:
                from .action_parser import parse_llm_output
                from .tool_executor import execute_tool
            except ImportError:
                from server.action_parser import parse_llm_output
                from server.tool_executor import execute_tool

            parsed = parse_llm_output(raw_llm_output)

            if parsed["parse_type"] == "tool_call":
                if self._step_tool_call_count >= 4:
                    forced_tool_limit = True
                    action.decision = 1
                    action.reasoning = (
                        "[FORCED CONDITIONAL: MAX_TOOL_CALLS_EXCEEDED] "
                        "The agent exceeded the four-tool limit before making a final decision."
                    )
                    action.parse_type = "default_reject"
                    action.parse_confidence = 0.0
                    action.parse_failure = True
                else:
                    tool_name = parsed["tool_name"]
                    tool_args = parsed["tool_args"] or {}
                    signature = _tool_signature(tool_name, tool_args)
                    is_redundant = signature in self._step_tool_signatures
                    self._step_tool_signatures.add(signature)

                    tool_result = execute_tool(tool_name, tool_args, self)
                    self._step_tool_call_count += 1
                    self._total_tool_calls += 1
                    if self._portfolio:
                        self._portfolio.total_tool_calls += 1
                    if is_redundant:
                        self._step_redundant_tool_calls += 1
                        tool_result = dict(tool_result)
                        tool_result["redundant"] = True
                        tool_result["display_text"] = (
                            "[REDUNDANT TOOL CALL: this repeats a tool+args already used "
                            "in the current decision step]\n"
                            + tool_result.get("display_text", "")
                        )
                    else:
                        tool_result = {**tool_result, "redundant": False}

                    if episode_id:
                        self._save_to_store(episode_id)

                    return self._build_observation(
                        reward=0.0,
                        reward_components={
                            "tool_call": 0.0,
                            "tool_call_count": self._step_tool_call_count,
                            "redundant_tool_call": is_redundant,
                        },
                        tool_result=tool_result,
                        last_parse_type="tool_call",
                        last_tool_name=tool_name,
                    )

            else:
                action.decision = parsed["action"]
                action.reasoning = parsed["reasoning"]
                action.parse_type = parsed["parse_type"]
                action.parse_confidence = parsed["parse_confidence"]
                action.parse_failure = parsed["parse_failure"]

        # ── Advance macro state for this step ───────────────────
        self._world.tick_macro(self._current_step + 1)

        # ── Check if a retry borrower should replace current app ─
        retry_app = self._borrower_agent.pop_ready(self._current_step + 1)
        if retry_app is not None:
            # Insert retry as the current application (overrides generated one)
            self._applications[self._current_step] = retry_app

        # ── Fetch current application ────────────────────────────
        original_decision = action.decision
        app = self._applications[self._current_step]
        meta = app["metadata"]
        self._current_borrower_id = meta.get("borrower_id", f"B-{self._current_step:03d}")

        # ── Hard rule override ───────────────────────────────────
        hard_rules = meta.get("hard_rules_triggered", [])
        hard_rule_overridden = False
        if hard_rules and original_decision != 2:
            hard_rule_overridden = True
            effective_decision = 2
        else:
            effective_decision = original_decision

        self._actions_taken.append(original_decision)

        # Attach raw values needed by reward
        meta["loan_amount_cr"]  = app["raw_values"].get("loan_amount_cr", 5.0)
        meta["collateral_ratio"] = app["raw_values"].get("collateral_ratio", 1.0)

        # ── Compute step reward (Phase 4: pass parser metadata) ─────
        is_final = (self._current_step + 1 >= self._total_steps)
        episode_progress = (self._current_step + 1) / self._total_steps

        # Parse metadata from action (set by agent_loop; defaults for direct env usage)
        parse_type       = getattr(action, "parse_type",       "final_decision")
        parse_confidence = getattr(action, "parse_confidence", 0.90)
        reasoning_len    = len(getattr(action, "reasoning",    "") or "")
        if parse_type is None:
            parse_type = "final_decision"
        if parse_confidence is None:
            parse_confidence = 0.90

        tool_efficiency_delta = 0.0
        if self._step_tool_call_count > 0:
            if effective_decision == meta.get("optimal_action"):
                tool_efficiency_delta += 0.20
            elif self._step_tool_call_count >= 3:
                tool_efficiency_delta -= 0.10
            if self._step_redundant_tool_calls:
                tool_efficiency_delta -= 0.10 * self._step_redundant_tool_calls
        if forced_tool_limit:
            tool_efficiency_delta -= 0.50

        reward, components = compute_step_reward(
            action           = effective_decision,
            app_metadata     = meta,
            portfolio        = self._portfolio,
            is_final_step    = is_final,
            tool_efficiency_delta = tool_efficiency_delta,
            parse_type       = parse_type,
            parse_confidence = parse_confidence,
            reasoning_len    = reasoning_len,
            episode_progress = episode_progress,
        )

        # Hard rule override penalty
        if hard_rule_overridden:
            if "hard_rule_bonus" in components:
                reward -= components.pop("hard_rule_bonus")
            from server.reward import PENALTY_HARD_RULE_APPROVE, PENALTY_HARD_RULE_CONDITIONAL
            if original_decision == 0:
                components["hard_rule_penalty"] = PENALTY_HARD_RULE_APPROVE
                reward += PENALTY_HARD_RULE_APPROVE
            else:
                components["hard_rule_penalty"] = PENALTY_HARD_RULE_CONDITIONAL
                reward += PENALTY_HARD_RULE_CONDITIONAL
            components["hard_rule_override"] = True

        if self._step_tool_call_count > 0 and effective_decision == meta.get("optimal_action"):
            self._portfolio.useful_tool_calls += self._step_tool_call_count

        # ── Update portfolio alerts ──────────────────────────────
        self._portfolio.update_alerts_from_application(
            app_features=app["features"],
            alerts=meta.get("alerts", []),
        )

        # ── BorrowerAgent: track rejection ────────────────────────
        if effective_decision == 2:
            self._borrower_agent.on_rejected(self._current_borrower_id, self._current_step + 1)

        # ── Advance step counter ──────────────────────────────────
        self._current_step += 1
        self._state.step_count = self._current_step
        self._step_tool_call_count = 0   # reset per-step tool count
        self._step_tool_signatures = set()
        self._step_redundant_tool_calls = 0

        # ── WorldState: record decision ───────────────────────────
        self._world.record_decision(effective_decision, self._portfolio.npa_rate)

        # ── Process NPA maturity events ───────────────────────────
        npa_penalty, terminated = self._portfolio.process_timestep(self._current_step)
        if npa_penalty != 0:
            reward += npa_penalty
            components["delayed_npa_penalty"] = round(npa_penalty, 4)

        # ── Phase 4: Survival bonus (every 10 steps if solvent) ───
        if self._current_step % 10 == 0 and not terminated:
            survival = compute_survival_bonus(self._portfolio.crar)
            if survival > 0:
                reward += survival
                components["survival_bonus"] = round(survival, 4)

        # ── Regulator Audit (jittered steps from WorldState) ──────
        audit_result = None
        if self._current_step in self._world.audit_steps:
            audit_reward, audit_result, audit_terminated = self._regulator.audit(
                step=self._current_step,
                portfolio=self._portfolio,
                world=self._world,
            )
            reward += audit_reward
            components["audit_reward"] = round(audit_reward, 4)
            if audit_terminated:
                terminated = True

        # ── Episode termination check ─────────────────────────────
        if terminated or self._current_step >= self._total_steps:
            self._done = True

        # ── Persist / clean up ────────────────────────────────────
        if episode_id:
            if self._done:
                _SESSION_STORE.pop(episode_id, None)
            else:
                self._save_to_store(episode_id)

        return self._build_observation(
            reward=reward,
            reward_components=components,
            audit_result=audit_result,
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Observation Builder ───────────────────────────────────────

    def _build_observation(
        self,
        reward: float,
        reward_components: Dict[str, float],
        force_done: bool = False,
        audit_result: Optional[Dict] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        last_parse_type: Optional[str] = None,
        last_tool_name: Optional[str] = None,
    ) -> IntelliCreditObservation:
        """Build full 55D observation (45D legacy + 10D memory features)."""
        done = force_done or self._done

        # Application features (25D) — identical to v1
        is_repeat   = False
        attempt_num = 1
        if not done and self._current_step < len(self._applications):
            app = self._applications[self._current_step]
            features_dict = app["features"]
            feature_keys = [
                "promoter_litigation_count", "mca_charge_count",
                "adverse_news_sentiment",    "promoter_din_score",
                "dscr_proxy",                "bank_od_utilisation_pct",
                "cc_utilisation_volatility", "gst_turnover_cagr",
                "current_ratio",             "debt_to_equity",
                "return_on_net_worth",       "ebitda_margin",
                "collateral_coverage_ratio", "gst_2a_vs_3b_gap_pct",
                "revenue_gst_alignment",     "itc_mismatch_flag",
                "circular_trading_ratio",    "cheque_bounce_frequency",
                "related_party_txn_pct",     "working_capital_cycle_days",
                "factory_operational_flag",  "capacity_utilisation_pct",
                "succession_risk_flag",       "sector_risk_score",
                "management_stability_score",
            ]
            app_features = [round(features_dict.get(k, 0.0), 4) for k in feature_keys]
            text_summary = application_to_text(app)
            meta = app["metadata"]
            alerts = meta.get("alerts", [])
            is_repeat   = meta.get("is_repeat_applicant", False)
            attempt_num = meta.get("attempt_number", 1)
            summary = ApplicationSummary(
                company_name=meta.get("company_name", "Unknown"),
                sector=meta.get("sector", "Unknown"),
                size=meta.get("size", "Unknown"),
                text_summary=text_summary,
            )
        else:
            app_features = [0.0] * 25
            alerts = []
            summary = ApplicationSummary(text_summary="Episode complete. No more applications.")

        # Portfolio obs (10D)
        portfolio_obs = self._portfolio.get_observation() if self._portfolio else [0.0] * 10

        # Alert obs (5D) — same as v1
        alert_obs = self._portfolio.get_alert_observation() if self._portfolio else [0.0] * 5

        # Macro obs (5D) — from WorldState in v2
        macro_obs = self._world.get_macro_obs() if self._world else [0.2, 0.0, 0.5, 0.5, 0.5]

        # v2: 10D memory features
        borrower_persistence = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}.get(
            attempt_num if is_repeat else 1, 0.0
        )
        memory_obs = (
            self._world.compute_memory_features(
                portfolio=self._portfolio,
                current_step=self._current_step,
                total_steps=self._total_steps,
                borrower_persistence=borrower_persistence,
                reflection_count=self._world.reflection_count,
                tool_calls_total=self._total_tool_calls,
            )
            if self._world and self._portfolio
            else [0.0] * 10
        )

        # Grading at episode end
        episode_score   = None
        score_breakdown = None
        if done and self._portfolio:
            grade = grade_episode(
                actions=self._actions_taken,
                applications=self._applications[:len(self._actions_taken)],
                portfolio=self._portfolio,
                task_id=self._task_id,
            )
            episode_score   = grade["score"]
            score_breakdown = grade["breakdown"]

        reg_warning = (
            self._world.consecutive_audit_failures if self._world else 0
        )

        return IntelliCreditObservation(
            # 45D base (v1 — unchanged)
            application_features=app_features,
            portfolio_state=portfolio_obs,
            macro_state=macro_obs,
            alert_state=alert_obs,
            application_summary=summary,
            alerts=alerts,
            timestep=self._current_step,
            done=done,
            reward=round(reward, 4),
            reward_components=reward_components,
            task_id=self._task_id,
            episode_score=episode_score,
            score_breakdown=score_breakdown,
            # 10D memory (v2)
            memory_features=memory_obs,
            # Multi-agent metadata (v2)
            is_repeat_applicant=is_repeat,
            attempt_number=attempt_num,
            tool_call_count=self._step_tool_call_count,
            regulator_warning_level=reg_warning,
            audit_result=audit_result,
            tool_result=tool_result,
            tool_result_text=(
                tool_result.get("display_text") if isinstance(tool_result, dict) else None
            ),
            last_parse_type=last_parse_type,
            last_tool_name=last_tool_name,
        )


# ── Quick smoke test ─────────────────────────────────────────────

if __name__ == "__main__":
    env = IntelliCreditEnvironment(task_id="task1")
    obs = env.reset(seed=42)

    print(f"Reset: step={obs.timestep}, app_dims={len(obs.application_features)}")
    print(f"Portfolio: {obs.portfolio_state[:3]}")
    print(f"Macro:     {obs.macro_state}")
    print(f"Memory:    {obs.memory_features}")
    print(f"Total obs dims: {len(obs.application_features) + len(obs.portfolio_state) + len(obs.macro_state) + len(obs.alert_state) + len(obs.memory_features)}")
    print(f"Summary: {obs.application_summary.company_name}")

    total_reward = 0.0
    for i in range(12):
        action = IntelliCreditAction(decision=random.randint(0, 2))
        obs = env.step(action)
        total_reward += obs.reward
        print(f"Step {obs.timestep:>2}: reward={obs.reward:+.2f} | done={obs.done} | audit={obs.audit_result is not None} | repeat={obs.is_repeat_applicant}")
        if obs.audit_result:
            print(f"    → Audit: {obs.audit_result.get('outcome')}")
        if obs.done:
            print(f"Episode Score: {obs.episode_score}")
            break

    print(f"\nTotal reward earned: {total_reward:.2f}")
