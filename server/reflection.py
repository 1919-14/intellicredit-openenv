"""
IntelliCredit v2 — Self-Improvement & Reflection System (Phase 5)
==================================================================
Cross-episode learning without weight updates.

Components:
  1. EpisodeOutcomeAnalyzer  — post-episode analysis of every negative step
  2. LessonExtractor         — generates structured lessons from failure patterns
  3. MemoryBank              — persistent lesson storage (JSON, max 20 lessons)
  4. ImprovementTracker      — tracks score trends across episodes

Flow:
  Episode ends → Analyzer collects failure data → Extractor generates lessons
  → MemoryBank stores (dedup, severity sort, FIFO) → Next episode prompt
  includes top 5 lessons → LLM makes better decisions → repeat

Anti-gaming:
  - Lessons are read-only in prompt (agent cannot modify memory bank)
  - Max 20 lessons keeps context window manageable
  - Each lesson ≤ 100 chars to prevent prompt bloat
  - Score trend is tracked but NOT visible to agent
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

MAX_LESSONS           = 20       # FIFO eviction when full
MAX_LESSON_CHARS      = 100      # truncate if longer
TOP_LESSONS_IN_PROMPT = 5        # inject top 5 by severity into LLM prompt
SEVERITY_ORDER        = {"critical": 0, "high": 1, "medium": 2, "low": 3}

DEFAULT_MEMORY_PATH   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "memory_bank.json"
)


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class StepFailure:
    """Single negative-reward step from an episode."""
    step           : int
    action         : int              # 0/1/2
    action_label   : str              # APPROVE/CONDITIONAL/REJECT
    hidden_pd      : float
    reward         : float
    reward_components : Dict[str, float]
    hard_rules     : List[str]
    alerts         : List[Dict]
    sector         : str
    loan_amount    : float
    is_repeat      : bool
    attempt_number : int
    portfolio_npa  : float
    portfolio_crar : float
    worst_component: str              # which reward component caused the most loss


@dataclass
class EpisodeSummary:
    """Full episode outcome summary for analysis."""
    episode_number    : int
    total_steps       : int
    steps_survived    : int
    final_score       : float
    final_crar        : float
    final_npa_rate    : float
    total_defaults    : int
    total_approvals   : int
    total_rejects     : int
    audit_outcomes    : List[Dict]
    terminated_early  : bool
    termination_reason: str
    step_failures     : List[StepFailure]
    biggest_loss_step : Optional[StepFailure]
    repeat_app_defaults: int          # approved repeat applicants that defaulted
    cumulative_reward : float
    score_breakdown   : Dict[str, Any]


@dataclass
class Lesson:
    """Single learned lesson."""
    episode     : int
    type        : str                 # hard_rule_violation, delayed_default, audit_failure, etc.
    lesson      : str                 # human-readable lesson text (≤100 chars)
    severity    : str                 # critical, high, medium, low
    reward_lost : float
    seen_count  : int = 1
    timestamp   : float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "episode"    : self.episode,
            "type"       : self.type,
            "lesson"     : self.lesson,
            "severity"   : self.severity,
            "reward_lost": round(self.reward_lost, 2),
            "seen_count" : self.seen_count,
            "timestamp"  : self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Lesson":
        return cls(
            episode     = d["episode"],
            type        = d["type"],
            lesson      = d["lesson"],
            severity    = d["severity"],
            reward_lost = d["reward_lost"],
            seen_count  = d.get("seen_count", 1),
            timestamp   = d.get("timestamp", time.time()),
        )


# ═══════════════════════════════════════════════════════════════
# STEP 5.1: EPISODE OUTCOME ANALYZER
# ═══════════════════════════════════════════════════════════════

class EpisodeOutcomeAnalyzer:
    """
    Post-episode analysis engine.

    After every episode ends, collects data from every step where reward < 0.
    Builds a full EpisodeSummary for the LessonExtractor.
    """

    @staticmethod
    def analyze(
        episode_number : int,
        actions        : List[int],
        applications   : List[Dict[str, Any]],
        portfolio      : Any,           # PortfolioState
        world          : Any,           # WorldState
        step_rewards   : List[float],   # reward per step
        step_components: List[Dict],    # reward_components per step
    ) -> EpisodeSummary:
        """Analyze a completed episode and return structured summary."""

        action_labels = {0: "APPROVE", 1: "CONDITIONAL", 2: "REJECT"}
        n = min(len(actions), len(applications), len(step_rewards))

        # Collect negative-reward steps
        failures: List[StepFailure] = []
        biggest_loss: Optional[StepFailure] = None

        for i in range(n):
            reward = step_rewards[i]
            if reward >= 0:
                continue

            meta = applications[i]["metadata"]
            comps = step_components[i] if i < len(step_components) else {}

            # Find worst component
            worst_comp = ""
            worst_val  = 0.0
            for k, v in comps.items():
                if k == "total":
                    continue
                if isinstance(v, (int, float)) and v < worst_val:
                    worst_val = v
                    worst_comp = k

            sf = StepFailure(
                step            = i + 1,
                action          = actions[i],
                action_label    = action_labels.get(actions[i], "?"),
                hidden_pd       = meta.get("hidden_pd", 0.5),
                reward          = reward,
                reward_components = comps,
                hard_rules      = meta.get("hard_rules_triggered", []),
                alerts          = meta.get("alerts", []),
                sector          = meta.get("sector", "Unknown"),
                loan_amount     = meta.get("loan_amount_cr", 5.0),
                is_repeat       = meta.get("is_repeat_applicant", False),
                attempt_number  = meta.get("attempt_number", 1),
                portfolio_npa   = comps.get("__portfolio_npa", portfolio.npa_rate),
                portfolio_crar  = comps.get("__portfolio_crar", portfolio.crar),
                worst_component = worst_comp,
            )
            failures.append(sf)

            if biggest_loss is None or reward < biggest_loss.reward:
                biggest_loss = sf

        # Count repeat-applicant defaults
        repeat_defaults = 0
        for acc in portfolio.stressed_accounts:
            if acc.get("is_repeat", False):
                repeat_defaults += 1

        return EpisodeSummary(
            episode_number     = episode_number,
            total_steps        = portfolio.total_episode_steps,
            steps_survived     = portfolio.steps_survived,
            final_score        = getattr(portfolio, "episode_score", 0.0),
            final_crar         = portfolio.crar,
            final_npa_rate     = portfolio.npa_rate,
            total_defaults     = portfolio.loans_defaulted,
            total_approvals    = sum(1 for a in actions if a in (0, 1)),
            total_rejects      = sum(1 for a in actions if a == 2),
            audit_outcomes     = world.audit_history if world else [],
            terminated_early   = portfolio.episode_terminated,
            termination_reason = portfolio.termination_reason or "",
            step_failures      = failures,
            biggest_loss_step  = biggest_loss,
            repeat_app_defaults= repeat_defaults,
            cumulative_reward  = portfolio.cumulative_reward,
            score_breakdown    = {},
        )


# ═══════════════════════════════════════════════════════════════
# STEP 5.2: LESSON EXTRACTION LOGIC
# ═══════════════════════════════════════════════════════════════

class LessonExtractor:
    """
    Generates structured lessons from episode failure patterns.

    Trigger types:
      - hard_rule_violation     → RULE: When [condition], always [action]
      - delayed_default         → CAUTION: Loans with [pattern] defaulted
      - audit_failure           → COMPLIANCE: Audit failed due to [metric]
      - borrower_manipulation   → FRAUD RISK: Repeat applicants with [pattern]
      - macro_shock_loss        → MACRO: During [state], be conservative with [sector]
      - portfolio_overexposure  → PORTFOLIO: [metric] was at [value], caused [problem]
    """

    @staticmethod
    def extract(summary: EpisodeSummary) -> List[Lesson]:
        """Extract all applicable lessons from an episode summary."""
        lessons: List[Lesson] = []
        ep = summary.episode_number

        # ── Trigger 1: Hard Rule Violations ───────────────────────
        hr_failures = [f for f in summary.step_failures if f.hard_rules and f.action != 2]
        if hr_failures:
            # Group by rule type
            rule_counts: Dict[str, int] = defaultdict(int)
            for f in hr_failures:
                for rule in f.hard_rules:
                    rule_counts[rule] += 1

            for rule_name, count in rule_counts.items():
                rule_desc = _HARD_RULE_DESCRIPTIONS.get(rule_name, rule_name)
                lesson_text = f"RULE: When {rule_desc}, always REJECT. Cost: -2.0 each."
                lessons.append(Lesson(
                    episode    = ep,
                    type       = "hard_rule_violation",
                    lesson     = _truncate(lesson_text),
                    severity   = "critical",
                    reward_lost= -2.0 * count,
                ))

        # ── Trigger 2: Delayed Default Events ────────────────────
        if summary.total_defaults > 0:
            # Find the sectors with most defaults
            default_sectors: Dict[str, int] = defaultdict(int)
            for f in summary.step_failures:
                if "delayed_npa_penalty" in f.reward_components:
                    default_sectors[f.sector] += 1

            # Also check stressed accounts on portfolio
            for sector, count in default_sectors.items():
                lesson_text = f"CAUTION: {count} loan(s) in {sector} defaulted. Be more cautious."
                lessons.append(Lesson(
                    episode    = ep,
                    type       = "delayed_default",
                    lesson     = _truncate(lesson_text),
                    severity   = "high",
                    reward_lost= -15.0 * count,
                ))

        # ── Trigger 3: Audit Failures ─────────────────────────────
        failed_audits = [a for a in summary.audit_outcomes if not a.get("is_clean", True)]
        for audit in failed_audits:
            violations = audit.get("violations", [])
            for v in violations[:2]:  # max 2 lessons per audit
                lesson_text = f"COMPLIANCE: Audit failed: {v}"
                lessons.append(Lesson(
                    episode    = ep,
                    type       = "audit_failure",
                    lesson     = _truncate(lesson_text),
                    severity   = "high",
                    reward_lost= audit.get("total_penalty", -3.0),
                ))

        # ── Trigger 4: Borrower Manipulation ──────────────────────
        if summary.repeat_app_defaults > 0:
            lesson_text = (
                f"FRAUD RISK: {summary.repeat_app_defaults} repeat applicant(s) "
                f"defaulted after reapplication."
            )
            lessons.append(Lesson(
                episode    = ep,
                type       = "borrower_manipulation",
                lesson     = _truncate(lesson_text),
                severity   = "critical",
                reward_lost= -15.0 * summary.repeat_app_defaults,
            ))

        # ── Trigger 5: Macro Shock Losses ─────────────────────────
        # Steps with high-PD approvals during macro stress
        macro_losses = [
            f for f in summary.step_failures
            if f.action == 0 and f.hidden_pd >= 0.35
            and f.reward < -1.0
        ]
        if len(macro_losses) >= 2:
            sectors = set(f.sector for f in macro_losses)
            sector_str = ", ".join(list(sectors)[:3])
            lesson_text = f"MACRO: High-risk approvals in {sector_str} caused losses. Be conservative."
            lessons.append(Lesson(
                episode    = ep,
                type       = "macro_shock_loss",
                lesson     = _truncate(lesson_text),
                severity   = "medium",
                reward_lost= sum(f.reward for f in macro_losses),
            ))

        # ── Trigger 6: Portfolio Overexposure ─────────────────────
        if summary.final_npa_rate > 0.04:
            lesson_text = f"PORTFOLIO: NPA rate reached {summary.final_npa_rate:.1%}. Reject more marginal loans."
            lessons.append(Lesson(
                episode    = ep,
                type       = "portfolio_overexposure",
                lesson     = _truncate(lesson_text),
                severity   = "high",
                reward_lost= -5.0,
            ))

        if summary.terminated_early:
            reason_short = summary.termination_reason.split(":")[0] if summary.termination_reason else "Unknown"
            lesson_text = f"SURVIVAL: Episode terminated early ({reason_short}). Preserve capital."
            lessons.append(Lesson(
                episode    = ep,
                type       = "early_termination",
                lesson     = _truncate(lesson_text),
                severity   = "critical",
                reward_lost= -10.0,
            ))

        return lessons


# Hard rule descriptions for readable lessons
_HARD_RULE_DESCRIPTIONS = {
    "HR-01": "DSCR < 1.0",
    "HR-02": "Director is disqualified",
    "HR-03": "RED forensic alert present",
    "HR-04": "Cheque bounce rate > 25%",
    "HR-05": "GST compliance < 40%",
    "HR-06": "Severe adverse media score > 0.80",
}

def _truncate(text: str) -> str:
    """Truncate lesson text to MAX_LESSON_CHARS."""
    if len(text) <= MAX_LESSON_CHARS:
        return text
    return text[:MAX_LESSON_CHARS - 3] + "..."


# ═══════════════════════════════════════════════════════════════
# STEP 5.3: MEMORY BANK
# ═══════════════════════════════════════════════════════════════

class MemoryBank:
    """
    Persistent lesson storage with deduplication and FIFO eviction.

    Rules:
      - Maximum 20 lessons (FIFO eviction when full)
      - Deduplicate similar lessons (same type + same trigger → increment count)
      - Sort by severity first (critical > high > medium > low)
      - Sort by recency second (newer lessons first within same severity)
      - Tracks episode score trend for evidence generation
    """

    def __init__(self, path: str = DEFAULT_MEMORY_PATH):
        self._path = path
        self._lessons: List[Lesson] = []
        self._total_episodes: int = 0
        self._score_trend: List[float] = []
        self._load()

    def _load(self) -> None:
        """Load memory bank from disk."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            self._lessons = [Lesson.from_dict(d) for d in data.get("lessons", [])]
            self._total_episodes = data.get("total_episodes", 0)
            self._score_trend = data.get("average_score_trend", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            self._lessons = []

    def _save(self) -> None:
        """Persist memory bank to disk."""
        data = {
            "lessons"            : [l.to_dict() for l in self._lessons],
            "total_episodes"     : self._total_episodes,
            "average_score_trend": self._score_trend,
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def add_lessons(self, new_lessons: List[Lesson], episode_score: float) -> int:
        """
        Add lessons from a completed episode.

        Returns number of net new lessons added (after dedup).
        """
        self._total_episodes += 1
        self._score_trend.append(round(episode_score, 4))

        added = 0
        for lesson in new_lessons:
            # Deduplication: same type + similar text → increment count
            existing = self._find_duplicate(lesson)
            if existing is not None:
                existing.seen_count += 1
                existing.timestamp = time.time()
                existing.reward_lost = min(existing.reward_lost, lesson.reward_lost)
                # Update episode to most recent
                existing.episode = lesson.episode
            else:
                self._lessons.append(lesson)
                added += 1

        # Sort by severity (critical first), then recency (newest first)
        self._lessons.sort(key=lambda l: (
            SEVERITY_ORDER.get(l.severity, 99),
            -l.timestamp,
        ))

        # FIFO eviction: remove oldest lowest-severity lessons if over limit
        while len(self._lessons) > MAX_LESSONS:
            self._lessons.pop()  # remove last (lowest severity, oldest)

        self._save()
        return added

    def _find_duplicate(self, lesson: Lesson) -> Optional[Lesson]:
        """Find existing lesson with same type and similar text."""
        for existing in self._lessons:
            if existing.type != lesson.type:
                continue
            # Same type — check text similarity (simple prefix match)
            if (existing.lesson[:40] == lesson.lesson[:40] or
                existing.lesson == lesson.lesson):
                return existing
        return None

    def get_top_lessons(self, n: int = TOP_LESSONS_IN_PROMPT) -> List[Lesson]:
        """Return top N lessons by severity for prompt injection."""
        return self._lessons[:n]

    def get_lessons_text(self, n: int = TOP_LESSONS_IN_PROMPT) -> str:
        """
        Format top lessons as text block for LLM prompt injection.

        Returns empty string if no lessons yet (episode 1).
        """
        top = self.get_top_lessons(n)
        if not top:
            return ""

        lines = ["═══ PAST LESSONS LEARNED ═══"]
        lines.append("(From previous episodes — apply these to avoid repeating mistakes)")
        lines.append("")

        for i, lesson in enumerate(top, 1):
            severity_icon = {
                "critical": "🔴",
                "high"    : "🟠",
                "medium"  : "🟡",
                "low"     : "🟢",
            }.get(lesson.severity, "⚪")

            count_note = f" (seen {lesson.seen_count}x)" if lesson.seen_count > 1 else ""
            lines.append(f"  {i}. {severity_icon} {lesson.lesson}{count_note}")

        lines.append("")
        return "\n".join(lines)

    def get_score_trend(self) -> List[float]:
        """Return episode score trend for plotting."""
        return list(self._score_trend)

    def get_stats(self) -> Dict[str, Any]:
        """Return memory bank statistics."""
        return {
            "total_lessons"   : len(self._lessons),
            "total_episodes"  : self._total_episodes,
            "score_trend"     : self._score_trend[-10:],
            "severity_counts" : {
                s: sum(1 for l in self._lessons if l.severity == s)
                for s in ["critical", "high", "medium", "low"]
            },
            "type_counts"     : {
                t: sum(1 for l in self._lessons if l.type == t)
                for t in set(l.type for l in self._lessons)
            },
        }

    def clear(self) -> None:
        """Reset memory bank (for fresh evaluation runs)."""
        self._lessons = []
        self._total_episodes = 0
        self._score_trend = []
        self._save()

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def lesson_count(self) -> int:
        return len(self._lessons)


# ═══════════════════════════════════════════════════════════════
# STEP 5.5: IMPROVEMENT TRACKER
# ═══════════════════════════════════════════════════════════════

class ImprovementTracker:
    """
    Tracks and reports on self-improvement across episodes.

    Evidence generation:
      Run 1 (episodes 1-10):  empty memory, baseline performance
      Run 2 (episodes 11-20): lessons from Run 1, expected improvement
      Run 3 (episodes 21-30): refined lessons, approaching trained model
    """

    def __init__(self, memory_bank: MemoryBank):
        self._bank = memory_bank

    def get_improvement_report(self) -> Dict[str, Any]:
        """Generate improvement evidence report."""
        trend = self._bank.get_score_trend()
        n = len(trend)

        if n == 0:
            return {"status": "no_episodes", "message": "No episodes completed yet."}

        # Compute phase averages
        phases: Dict[str, Dict] = {}
        if n >= 10:
            phases["baseline"] = {
                "episodes" : "1-10",
                "avg_score": round(sum(trend[:10]) / 10, 4),
                "scores"   : trend[:10],
            }
        if n >= 20:
            phases["improved"] = {
                "episodes" : "11-20",
                "avg_score": round(sum(trend[10:20]) / 10, 4),
                "scores"   : trend[10:20],
            }
        if n >= 30:
            phases["refined"] = {
                "episodes" : "21-30",
                "avg_score": round(sum(trend[20:30]) / 10, 4),
                "scores"   : trend[20:30],
            }

        # Overall trend
        if n >= 2:
            first_half = trend[:n//2]
            second_half = trend[n//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            improvement = second_avg - first_avg
        else:
            improvement = 0.0

        return {
            "status"           : "ok",
            "total_episodes"   : n,
            "overall_avg"      : round(sum(trend) / n, 4),
            "improvement_delta": round(improvement, 4),
            "improving"        : improvement > 0.02,
            "phases"           : phases,
            "latest_5_scores"  : trend[-5:],
            "memory_stats"     : self._bank.get_stats(),
        }


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: FULL POST-EPISODE PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_post_episode_reflection(
    episode_number  : int,
    actions         : List[int],
    applications    : List[Dict[str, Any]],
    portfolio       : Any,
    world           : Any,
    step_rewards    : List[float],
    step_components : List[Dict],
    episode_score   : float,
    memory_bank     : MemoryBank,
) -> Dict[str, Any]:
    """
    Full post-episode pipeline: analyze → extract → store → report.

    Call this at the end of every episode.

    Returns:
        Dict with analysis summary, lessons added, memory stats.
    """
    # Step 1: Analyze
    summary = EpisodeOutcomeAnalyzer.analyze(
        episode_number  = episode_number,
        actions         = actions,
        applications    = applications,
        portfolio       = portfolio,
        world           = world,
        step_rewards    = step_rewards,
        step_components = step_components,
    )
    summary.final_score = episode_score

    # Step 2: Extract lessons
    lessons = LessonExtractor.extract(summary)

    # Step 3: Store in memory bank
    added = memory_bank.add_lessons(lessons, episode_score)

    # Step 4: Update world state reflection count
    if world:
        world.reflection_count = memory_bank.lesson_count

    return {
        "episode"         : episode_number,
        "episode_score"   : round(episode_score, 4),
        "failures_found"  : len(summary.step_failures),
        "lessons_extracted": len(lessons),
        "lessons_added"   : added,
        "total_lessons"   : memory_bank.lesson_count,
        "biggest_loss"    : {
            "step"   : summary.biggest_loss_step.step,
            "reward" : round(summary.biggest_loss_step.reward, 3),
            "action" : summary.biggest_loss_step.action_label,
            "cause"  : summary.biggest_loss_step.worst_component,
        } if summary.biggest_loss_step else None,
        "score_trend"     : memory_bank.get_score_trend()[-10:],
    }


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from server.intellicredit_env import IntelliCreditEnvironment
    from models import IntelliCreditAction
    import random

    print("=" * 65)
    print("  Phase 5 Reflection System — Self-Test (3 Episodes)")
    print("=" * 65)

    # Use temp path for test
    test_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_memory_bank.json"
    )

    bank = MemoryBank(path=test_path)
    bank.clear()

    for ep in range(1, 4):
        print(f"\n{'─'*60}")
        print(f"  Episode {ep}")
        print(f"{'─'*60}")

        env = IntelliCreditEnvironment(task_id="task3")
        obs = env.reset(seed=ep * 10)

        step_rewards = []
        step_comps   = []
        actions_list = []

        for step in range(50):
            # Simulate random decisions
            decision = random.choice([0, 0, 1, 2])
            action = IntelliCreditAction(decision=decision)
            obs = env.step(action)

            step_rewards.append(obs.reward)
            step_comps.append(dict(obs.reward_components))
            actions_list.append(decision)

            if obs.done:
                break

        score = obs.episode_score or 0.0

        # Run reflection
        result = run_post_episode_reflection(
            episode_number  = ep,
            actions         = actions_list,
            applications    = env._applications[:len(actions_list)],
            portfolio       = env._portfolio,
            world           = env._world,
            step_rewards    = step_rewards,
            step_components = step_comps,
            episode_score   = score,
            memory_bank     = bank,
        )

        print(f"  Score: {result['episode_score']:.3f}")
        print(f"  Failures: {result['failures_found']}")
        print(f"  Lessons extracted: {result['lessons_extracted']}")
        print(f"  Lessons stored: {result['total_lessons']}")
        if result["biggest_loss"]:
            bl = result["biggest_loss"]
            print(f"  Biggest loss: step {bl['step']} | {bl['action']} | {bl['reward']} | {bl['cause']}")

    # Show lessons
    print(f"\n{'─'*60}")
    print("  Memory Bank Contents")
    print(f"{'─'*60}")
    lessons_text = bank.get_lessons_text(n=10)
    print(lessons_text)

    # Improvement report
    tracker = ImprovementTracker(bank)
    report = tracker.get_improvement_report()
    print(f"\n  Score Trend: {report['latest_5_scores']}")
    print(f"  Overall Avg: {report['overall_avg']}")

    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)

    print(f"\n{'='*65}")
    print("  Phase 5 Self-Test Complete ✓")
    print(f"{'='*65}")
