"""
IntelliCredit-CreditAppraisal-v1 Environment
=============================================
A Constrained Multi-Objective MDP for corporate credit underwriting.
12-step sequential episodes with delayed NPA rewards, macro shocks,
and regulatory hard constraints.

Implements the OpenEnv Environment interface.

GAP 1:  45-dim observation (25 app + 10 portfolio + 5 macro + 5 alerts)
GAP 5:  Settlement reward at final timestep
GAP 11: Hard rule override — forces REJECT, applies penalty
GAP 12: Proper seeding for reproducibility
"""

from uuid import uuid4
from typing import Dict, List, Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import IntelliCreditAction, IntelliCreditObservation, ApplicationSummary
except ImportError:
    from models import IntelliCreditAction, IntelliCreditObservation, ApplicationSummary

try:
    from .dataset import generate_episode, application_to_text
    from .reward import PortfolioState, compute_step_reward, grade_episode
except ImportError:
    from server.dataset import generate_episode, application_to_text
    from server.reward import PortfolioState, compute_step_reward, grade_episode

import random


# Task configurations
TASK_CONFIGS = {
    "task1": {"num_steps": 5,  "description": "Easy — Clean profiles, no macro shocks"},
    "task2": {"num_steps": 8,  "description": "Medium — Forensic alerts present"},
    "task3": {"num_steps": 12, "description": "Medium-Hard — Macro shocks + missing data"},
    "task4": {"num_steps": 12, "description": "Hard — Regulatory hard-rule violations"},
    "task5": {"num_steps": 12, "description": "Expert — Cascading delayed NPAs + unpredictable"},
}


class IntelliCreditEnvironment(Environment):
    """
    A sequential credit portfolio management environment.

    The agent acts as a Senior Credit Officer reviewing 5-12 credit applications
    per episode. Each decision (APPROVE/CONDITIONAL/REJECT) affects the portfolio's
    health, capital reserves, and regulatory compliance.

    Key mechanics:
    - 45-dimensional observation (application + portfolio + macro + alerts)
    - Delayed NPA rewards (approve now → default later)
    - Macro-economic shocks mid-episode
    - Hard regulatory constraints that terminate episodes
    - Task-specific grading with weighted scoring formulas
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "task3"):
        self._task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._portfolio: Optional[PortfolioState] = None
        self._applications: List[Dict[str, Any]] = []
        self._current_step = 0
        self._actions_taken: List[int] = []
        self._episode_seed = 42
        self._done = False
        self._total_steps = TASK_CONFIGS.get(task_id, TASK_CONFIGS["task3"])["num_steps"]
        self._macro_state = [0.2, 0.0, 0.5, 0.5, 0.5]  # stress, shock, gdp, inflation, cycle

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None, episode_id: Optional[str] = None) -> IntelliCreditObservation:
        """Reset environment for a new episode.
        GAP 12: Accept seed for reproducibility.
        FIX: Preserve the episode_id provided by the OpenEnv framework so that
             subsequent /step calls are correctly routed to this instance.
        """
        if task_id:
            self._task_id = task_id

        # GAP 12: Use provided seed or generate one
        if seed is not None:
            self._episode_seed = seed
        else:
            self._episode_seed = random.randint(1, 100000)

        # CRITICAL: use the caller-provided episode_id (from the HTTP framework)
        # so session routing works. Fall back to a new UUID only if not given.
        resolved_episode_id = episode_id or str(uuid4())
        self._state = State(episode_id=resolved_episode_id, step_count=0)
        self._portfolio = PortfolioState()
        self._current_step = 0
        self._actions_taken = []
        self._done = False
        self._total_steps = TASK_CONFIGS.get(self._task_id, TASK_CONFIGS["task3"])["num_steps"]
        self._portfolio.total_episode_steps = self._total_steps

        # Generate all applications for this episode
        self._applications = generate_episode(
            task_id=self._task_id,
            num_steps=self._total_steps,
            seed=self._episode_seed,
        )

        # Initialize macro state
        self._macro_state = [0.2, 0.0, 0.5, 0.5, 0.5]

        # Return initial observation
        return self._build_observation(reward=0.0, reward_components={})

    def step(self, action: IntelliCreditAction) -> IntelliCreditObservation:
        """Execute one step: agent makes a credit decision on the current application.
        ISSUE 4: Hard rule override — original action used for penalty, effective action (REJECT) for portfolio.
        """
        if self._done or self._portfolio is None:
            return self._build_observation(
                reward=0.0,
                reward_components={"error": -1.0},
                force_done=True,
            )

        original_decision = action.decision
        app = self._applications[self._current_step]
        meta = app["metadata"]

        # ISSUE 4: Hard rule override
        # The original action is recorded for grading (so the grader knows the agent made a bad call).
        # The effective action is forced to REJECT so the loan never enters the portfolio.
        hard_rules = meta.get("hard_rules_triggered", [])
        hard_rule_overridden = False

        if hard_rules and original_decision != 2:
            hard_rule_overridden = True
            effective_decision = 2  # Bank's compliance system rejects it
        else:
            effective_decision = original_decision

        # Record the ORIGINAL action for grading (tests whether agent learned hard rules)
        self._actions_taken.append(original_decision)

        # Add raw values to metadata for reward computation
        meta["loan_amount_cr"] = app["raw_values"].get("loan_amount_cr", 5.0)
        meta["collateral_ratio"] = app["raw_values"].get("collateral_ratio", 1.0)

        # Check if this is the final step
        is_final = (self._current_step + 1 >= self._total_steps)

        # Compute reward using EFFECTIVE action (portfolio only sees REJECT for HR violations)
        # But correctness is computed against ORIGINAL action vs optimal
        reward, components = compute_step_reward(
            action=effective_decision,  # ISSUE 4: portfolio sees REJECT
            app_metadata=meta,
            portfolio=self._portfolio,
            is_final_step=is_final,
        )

        # ISSUE 4: If overridden, add extra penalty for the bad original decision
        # The base correctness from compute_step_reward used effective_decision=2 vs optimal=2
        # which gives a "correct" bonus. We REMOVE that bonus and ADD the penalty instead.
        if hard_rule_overridden:
            # Undo the incorrect "correct reject" bonus
            if "hard_rule_bonus" in components:
                reward -= components.pop("hard_rule_bonus")
            # Apply penalty for the original bad decision
            from server.reward import PENALTY_HARD_RULE_APPROVE, PENALTY_HARD_RULE_CONDITIONAL
            if original_decision == 0:
                components["hard_rule_penalty"] = PENALTY_HARD_RULE_APPROVE
                reward += PENALTY_HARD_RULE_APPROVE
            else:  # original_decision == 1
                components["hard_rule_penalty"] = PENALTY_HARD_RULE_CONDITIONAL
                reward += PENALTY_HARD_RULE_CONDITIONAL
            components["hard_rule_override"] = True

        # Update portfolio alerts from current application (GAP 1)
        self._portfolio.update_alerts_from_application(
            app_features=app["features"],
            alerts=meta.get("alerts", []),
        )

        # Process delayed NPAs
        self._current_step += 1
        self._state.step_count = self._current_step
        npa_penalty, terminated = self._portfolio.process_timestep(self._current_step)

        if npa_penalty != 0:
            reward += npa_penalty
            components["delayed_npa_penalty"] = npa_penalty

        # Update macro state from application metadata
        macro_stress = meta.get("macro_stress", 0.2)
        stressed_sector = meta.get("sector_under_stress")
        self._macro_state = [
            macro_stress,
            1.0 if stressed_sector else 0.0,
            max(0, 0.7 - macro_stress * 0.5),   # GDP growth inversely related
            min(1, 0.3 + macro_stress * 0.4),   # Inflation increases with stress
            max(0, 0.6 - macro_stress * 0.3),   # Credit cycle contracts
        ]

        # Check if episode is done
        if terminated or self._current_step >= self._total_steps:
            self._done = True

        return self._build_observation(
            reward=reward,
            reward_components=components,
        )

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(
        self,
        reward: float,
        reward_components: Dict[str, float],
        force_done: bool = False,
    ) -> IntelliCreditObservation:
        """Construct the full 45-dim observation for the agent.
        GAP 1: Now includes 5-dim alert sub-space.
        """
        done = force_done or self._done

        # Application features
        if not done and self._current_step < len(self._applications):
            app = self._applications[self._current_step]
            features_dict = app["features"]
            # Convert to ordered list (25-dim)
            feature_keys = [
                "promoter_litigation_count", "mca_charge_count",
                "adverse_news_sentiment", "promoter_din_score",
                "dscr_proxy", "bank_od_utilisation_pct",
                "cc_utilisation_volatility", "gst_turnover_cagr",
                "current_ratio", "debt_to_equity",
                "return_on_net_worth", "ebitda_margin",
                "collateral_coverage_ratio", "gst_2a_vs_3b_gap_pct",
                "revenue_gst_alignment", "itc_mismatch_flag",
                "circular_trading_ratio", "cheque_bounce_frequency",
                "related_party_txn_pct", "working_capital_cycle_days",
                "factory_operational_flag", "capacity_utilisation_pct",
                "succession_risk_flag", "sector_risk_score",
                "management_stability_score",
            ]
            app_features = [round(features_dict.get(k, 0.0), 4) for k in feature_keys]

            # Build text summary
            text_summary = application_to_text(app)
            meta = app["metadata"]
            alerts = meta.get("alerts", [])

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

        # Portfolio state (10-dim)
        portfolio_obs = self._portfolio.get_observation() if self._portfolio else [0.0] * 10

        # GAP 1: Portfolio alerts (5-dim)
        alert_obs = self._portfolio.get_alert_observation() if self._portfolio else [0.0] * 5

        # Grading at end of episode
        episode_score = None
        score_breakdown = None
        if done and self._portfolio:
            grade = grade_episode(
                actions=self._actions_taken,
                applications=self._applications[:len(self._actions_taken)],
                portfolio=self._portfolio,
                task_id=self._task_id,  # GAP 2: pass task_id for weighted grading
            )
            episode_score = grade["score"]
            score_breakdown = grade["breakdown"]

        return IntelliCreditObservation(
            application_features=app_features,
            portfolio_state=portfolio_obs,
            macro_state=self._macro_state,
            alert_state=alert_obs,  # GAP 1: new 5-dim field
            application_summary=summary,
            alerts=alerts,
            timestep=self._current_step,
            done=done,
            reward=round(reward, 4),
            reward_components=reward_components,
            task_id=self._task_id,
            episode_score=episode_score,
            score_breakdown=score_breakdown,
        )


# Quick test
if __name__ == "__main__":
    env = IntelliCreditEnvironment(task_id="task1")
    obs = env.reset(seed=42)
    print(f"Reset: timestep={obs.timestep}, features={len(obs.application_features)}")
    print(f"Portfolio: {obs.portfolio_state[:3]}")
    print(f"Alert state: {obs.alert_state}")
    print(f"Total obs dims: {len(obs.application_features) + len(obs.portfolio_state) + len(obs.macro_state) + len(obs.alert_state)}")
    print(f"Summary: {obs.application_summary.company_name}")

    for i in range(5):
        action = IntelliCreditAction(decision=random.randint(0, 2))
        obs = env.step(action)
        print(f"Step {obs.timestep}: reward={obs.reward:.2f}, done={obs.done}")
        if obs.done:
            print(f"Episode Score: {obs.episode_score}")
            print(f"Breakdown: {obs.score_breakdown}")
            break
