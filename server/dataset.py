"""
IntelliCredit Synthetic Dataset Generator
Steps 2-7: Anchor System, Tier Archetypes, Sector Profiles,
Size Scaling, Feature Dependencies, Forensic Alerts.

Generates coherent, LLM-readable corporate credit applications
anchored to (Tier x Sector x Size) for the OpenEnv RL environment.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any

# ═══════════════════════════════════════════════════════════════
# STEP 2: ANCHOR SYSTEM CONSTANTS
# ═══════════════════════════════════════════════════════════════

TIERS = ["A", "B", "C", "D"]

SECTORS = [
    "Manufacturing", "IT_Services", "Retail_Trading", "Infrastructure",
    "Pharma", "Agriculture", "Financial_Services", "Trading",
]

SECTOR_PROBABILITIES = {
    "Manufacturing": 0.25, "IT_Services": 0.15, "Retail_Trading": 0.15,
    "Infrastructure": 0.10, "Pharma": 0.08, "Agriculture": 0.10,
    "Financial_Services": 0.07, "Trading": 0.10,
}

SIZES = ["Micro", "Small", "Medium", "Large"]
SIZE_PROBABILITIES = {"Micro": 0.40, "Small": 0.30, "Medium": 0.20, "Large": 0.10}

# Tier probabilities per task difficulty
TIER_PROBABILITIES = {
    "task1": {"A": 0.50, "B": 0.30, "C": 0.15, "D": 0.05},
    "task2": {"A": 0.30, "B": 0.30, "C": 0.25, "D": 0.15},
    "task3": {"A": 0.25, "B": 0.30, "C": 0.25, "D": 0.20},
    "task4": {"A": 0.15, "B": 0.25, "C": 0.30, "D": 0.30},
    "task5": {"A": 0.10, "B": 0.20, "C": 0.35, "D": 0.35},
}

# ═══════════════════════════════════════════════════════════════
# STEP 3: TIER ARCHETYPE BASE PARAMETERS
# ═══════════════════════════════════════════════════════════════

TIER_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # (mean, std) for each feature per tier
    "A": {
        "cibil": (790, 25), "litigation": (0.1, 0.3), "adverse_sentiment": (0.15, 0.06),
        "din_score": (0.88, 0.05), "dscr": (1.85, 0.25), "od_util": (25, 10),
        "cc_volatility": (8, 3), "gst_cagr": (22, 6), "current_ratio": (2.10, 0.35),
        "de_ratio": (0.55, 0.18), "ronw": (0.20, 0.05), "ebitda_margin_base": (0.15, 0.03),
        "collateral": (1.80, 0.30), "gst_gap": (2, 1.5), "gst_alignment": (0.95, 0.03),
        "itc_mismatch_prob": (0.03,), "circular_trading": (0.01, 0.008),
        "bounce_freq": (0.02, 0.015), "related_party": (6, 2.5),
        "wc_cycle_base": (40, 10), "factory_op": (0.95, 0.04),
        "capacity_util": (75, 8), "succession_risk_prob": (0.05,),
        "sector_risk_base": (0.25, 0.08), "mgmt_stability": (0.88, 0.05),
        "mca_charges": (0.3, 0.5),
    },
    "B": {
        "cibil": (700, 30), "litigation": (0.6, 0.7), "adverse_sentiment": (0.35, 0.10),
        "din_score": (0.72, 0.10), "dscr": (1.30, 0.20), "od_util": (50, 15),
        "cc_volatility": (14, 5), "gst_cagr": (12, 6), "current_ratio": (1.50, 0.25),
        "de_ratio": (1.20, 0.30), "ronw": (0.13, 0.05), "ebitda_margin_base": (0.12, 0.03),
        "collateral": (1.30, 0.25), "gst_gap": (8, 3), "gst_alignment": (0.82, 0.06),
        "itc_mismatch_prob": (0.12,), "circular_trading": (0.03, 0.02),
        "bounce_freq": (0.08, 0.04), "related_party": (10, 3),
        "wc_cycle_base": (50, 12), "factory_op": (0.85, 0.08),
        "capacity_util": (65, 10), "succession_risk_prob": (0.15,),
        "sector_risk_base": (0.38, 0.10), "mgmt_stability": (0.68, 0.10),
        "mca_charges": (0.8, 0.7),
    },
    "C": {
        "cibil": (600, 28), "litigation": (1.8, 1.2), "adverse_sentiment": (0.55, 0.12),
        "din_score": (0.52, 0.12), "dscr": (1.00, 0.15), "od_util": (72, 12),
        "cc_volatility": (20, 6), "gst_cagr": (3, 5), "current_ratio": (1.10, 0.18),
        "de_ratio": (2.00, 0.40), "ronw": (0.06, 0.04), "ebitda_margin_base": (0.08, 0.03),
        "collateral": (0.95, 0.20), "gst_gap": (18, 5), "gst_alignment": (0.68, 0.10),
        "itc_mismatch_prob": (0.30,), "circular_trading": (0.08, 0.04),
        "bounce_freq": (0.18, 0.06), "related_party": (16, 4),
        "wc_cycle_base": (65, 15), "factory_op": (0.70, 0.12),
        "capacity_util": (52, 12), "succession_risk_prob": (0.30,),
        "sector_risk_base": (0.55, 0.10), "mgmt_stability": (0.50, 0.12),
        "mca_charges": (1.8, 1.0),
    },
    "D": {
        "cibil": (470, 50), "litigation": (3.5, 1.5), "adverse_sentiment": (0.78, 0.10),
        "din_score": (0.25, 0.15), "dscr": (0.65, 0.18), "od_util": (90, 6),
        "cc_volatility": (28, 8), "gst_cagr": (-8, 6), "current_ratio": (0.70, 0.18),
        "de_ratio": (3.80, 0.80), "ronw": (-0.05, 0.08), "ebitda_margin_base": (0.03, 0.04),
        "collateral": (0.55, 0.20), "gst_gap": (35, 10), "gst_alignment": (0.42, 0.12),
        "itc_mismatch_prob": (0.55,), "circular_trading": (0.15, 0.06),
        "bounce_freq": (0.35, 0.10), "related_party": (22, 5),
        "wc_cycle_base": (85, 18), "factory_op": (0.45, 0.18),
        "capacity_util": (35, 12), "succession_risk_prob": (0.55,),
        "sector_risk_base": (0.72, 0.10), "mgmt_stability": (0.30, 0.12),
        "mca_charges": (3.5, 1.5),
    },
}

# ═══════════════════════════════════════════════════════════════
# STEP 4: SECTOR PROFILES
# ═══════════════════════════════════════════════════════════════

SECTOR_PROFILES = {
    "Manufacturing":       {"margin_adj": 0.00, "wc_adj": 0,  "collateral_adj": 0.15, "macro_sensitivity": 0.70, "dscr_shock": 0.15},
    "IT_Services":         {"margin_adj": 0.08, "wc_adj": -15, "collateral_adj": -0.25, "macro_sensitivity": 0.50, "dscr_shock": 0.10},
    "Retail_Trading":      {"margin_adj": -0.05, "wc_adj": -10, "collateral_adj": 0.00, "macro_sensitivity": 0.55, "dscr_shock": 0.10},
    "Infrastructure":      {"margin_adj": 0.02, "wc_adj": 25, "collateral_adj": 0.20, "macro_sensitivity": 0.85, "dscr_shock": 0.20},
    "Pharma":              {"margin_adj": 0.06, "wc_adj": 5,  "collateral_adj": 0.00, "macro_sensitivity": 0.15, "dscr_shock": 0.02},
    "Agriculture":         {"margin_adj": -0.04, "wc_adj": 15, "collateral_adj": 0.00, "macro_sensitivity": 0.60, "dscr_shock": 0.12},
    "Financial_Services":  {"margin_adj": 0.04, "wc_adj": -20, "collateral_adj": -0.10, "macro_sensitivity": 0.75, "dscr_shock": 0.18},
    "Trading":             {"margin_adj": -0.08, "wc_adj": -5, "collateral_adj": -0.10, "macro_sensitivity": 0.45, "dscr_shock": 0.08},
}

# ═══════════════════════════════════════════════════════════════
# STEP 5: SIZE SCALING
# ═══════════════════════════════════════════════════════════════

SIZE_SCALES = {
    "Micro":  {"revenue": (3, 4),   "loan": (1, 3),   "networth": (0.5, 2), "margin_adj": -0.01, "cibil_adj": -5},
    "Small":  {"revenue": (15, 12), "loan": (8, 6),   "networth": (5, 5),   "margin_adj": 0.00,  "cibil_adj": 0},
    "Medium": {"revenue": (80, 50), "loan": (35, 25), "networth": (25, 20), "margin_adj": 0.01,  "cibil_adj": 10},
    "Large":  {"revenue": (300, 80),"loan": (100, 50),"networth": (80, 40), "margin_adj": 0.02,  "cibil_adj": 15},
}


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _sample_normal(mean: float, std: float) -> float:
    return random.gauss(mean, std)


def _sample_poisson(lam: float) -> int:
    return max(0, int(random.gauss(lam, math.sqrt(max(lam, 0.1)))))


def _weighted_choice(options: list, weights: list) -> str:
    return random.choices(options, weights=weights, k=1)[0]


# ═══════════════════════════════════════════════════════════════
# STEP 6-7: CORE GENERATION FUNCTION
# ═══════════════════════════════════════════════════════════════

def generate_application(
    task_id: str = "task3",
    macro_stress: float = 0.3,
    sector_under_stress: Optional[str] = None,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a single coherent credit application.
    Returns dict with 25 normalized features + metadata.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    # ─── ANCHOR SELECTION ─────────────────────────────────────
    tier_probs = TIER_PROBABILITIES.get(task_id, TIER_PROBABILITIES["task3"])

    # Macro stress shifts tier distribution
    if macro_stress > 0.6:
        shift = min(0.10, (macro_stress - 0.6) * 0.25)
        tier_probs = dict(tier_probs)
        tier_probs["A"] = max(0.05, tier_probs["A"] - shift)
        tier_probs["D"] = min(0.50, tier_probs["D"] + shift)
        total = sum(tier_probs.values())
        tier_probs = {k: v / total for k, v in tier_probs.items()}

    tier = _weighted_choice(TIERS, [tier_probs[t] for t in TIERS])
    sector = _weighted_choice(SECTORS, [SECTOR_PROBABILITIES[s] for s in SECTORS])
    size = _weighted_choice(SIZES, [SIZE_PROBABILITIES[s] for s in SIZES])

    tp = TIER_PARAMS[tier]
    sp = SECTOR_PROFILES[sector]
    ss = SIZE_SCALES[size]

    # ─── ROUND 1: PRIMARY FEATURES ───────────────────────────
    cibil_raw = _clamp(
        _sample_normal(tp["cibil"][0] + ss["cibil_adj"], tp["cibil"][1]),
        300, 900
    )

    revenue_raw = max(0.5, _sample_normal(ss["revenue"][0], ss["revenue"][1]))
    loan_raw = max(0.3, _sample_normal(ss["loan"][0], ss["loan"][1]))

    dscr_raw = _clamp(
        _sample_normal(tp["dscr"][0], tp["dscr"][1]),
        0.1, 4.0
    )
    # Macro shock reduces DSCR
    if macro_stress > 0.5:
        dscr_raw -= sp["dscr_shock"] * (macro_stress - 0.5) * 2
        dscr_raw = max(0.1, dscr_raw)
    # Extra shock if this company's sector is under stress
    if sector_under_stress and sector == sector_under_stress:
        dscr_raw -= sp["dscr_shock"] * 0.5
        dscr_raw = max(0.1, dscr_raw)

    # ─── ROUND 2: PROFITABILITY CLUSTER ──────────────────────
    ebitda_margin = _clamp(
        _sample_normal(
            tp["ebitda_margin_base"][0] + sp["margin_adj"] + ss["margin_adj"],
            tp["ebitda_margin_base"][1]
        ),
        -0.10, 0.40
    )
    # Dependency: if DSCR is high, margins tend higher
    if dscr_raw > tp["dscr"][0] + tp["dscr"][1]:
        ebitda_margin += 0.01

    ronw = _clamp(
        _sample_normal(tp["ronw"][0], tp["ronw"][1]),
        -0.30, 0.50
    )
    # Dependency: if margins are strong, RONW is better
    if ebitda_margin > tp["ebitda_margin_base"][0]:
        ronw += 0.02

    gst_cagr_raw = _sample_normal(tp["gst_cagr"][0], tp["gst_cagr"][1])
    if macro_stress > 0.6:
        gst_cagr_raw -= (macro_stress - 0.6) * 10
    gst_cagr_raw = _clamp(gst_cagr_raw, -30, 50)

    # Net worth (correlated with revenue and margins)
    nw_base = max(0.1, _sample_normal(ss["networth"][0], ss["networth"][1]))
    if tier == "D" and random.random() < 0.30:
        nw_base = -abs(nw_base) * 0.3  # 30% chance negative for Tier D

    # ─── ROUND 3: LEVERAGE-LIQUIDITY CLUSTER ─────────────────
    de_ratio = _clamp(
        _sample_normal(tp["de_ratio"][0], tp["de_ratio"][1]),
        0.05, 8.0
    )

    current_ratio = _clamp(
        _sample_normal(tp["current_ratio"][0], tp["current_ratio"][1]),
        0.2, 5.0
    )
    # Dependency: high leverage → lower liquidity
    if de_ratio > tp["de_ratio"][0] + tp["de_ratio"][1] * 0.5:
        current_ratio -= 0.15
        current_ratio = max(0.2, current_ratio)

    od_util = _clamp(
        _sample_normal(tp["od_util"][0], tp["od_util"][1]),
        0, 100
    )
    # Macro stress increases OD usage
    if macro_stress > 0.5:
        od_util += (macro_stress - 0.5) * 15
        od_util = _clamp(od_util, 0, 100)
    # Dependency: high leverage → more OD
    if de_ratio > tp["de_ratio"][0]:
        od_util += 5
        od_util = _clamp(od_util, 0, 100)

    cc_volatility = _clamp(
        _sample_normal(tp["cc_volatility"][0], tp["cc_volatility"][1]),
        0, 50
    )
    # Dependency: high OD → more volatile
    if od_util > 75:
        cc_volatility += 3

    wc_cycle = _clamp(
        _sample_normal(tp["wc_cycle_base"][0] + sp["wc_adj"], tp["wc_cycle_base"][1]),
        5, 180
    )
    if macro_stress > 0.6:
        wc_cycle += (macro_stress - 0.6) * 12

    # ─── ROUND 4: COMPLIANCE CLUSTER ─────────────────────────
    gst_gap = _clamp(
        _sample_normal(tp["gst_gap"][0], tp["gst_gap"][1]),
        0, 80
    )

    gst_alignment = _clamp(
        _sample_normal(tp["gst_alignment"][0], tp["gst_alignment"][1]),
        0.1, 1.0
    )
    # Dependency: large gap → poor alignment
    if gst_gap > 15:
        gst_alignment -= 0.10
        gst_alignment = max(0.1, gst_alignment)

    itc_mismatch = 1 if random.random() < tp["itc_mismatch_prob"][0] else 0
    # Dependency: large gap → MUST be ITC mismatch (Test 5)
    if gst_gap > 15:
        itc_mismatch = 1

    circular_trading = _clamp(
        _sample_normal(tp["circular_trading"][0], tp["circular_trading"][1]),
        0, 0.50
    )

    bounce_freq = _clamp(
        _sample_normal(tp["bounce_freq"][0], tp["bounce_freq"][1]),
        0, 0.80
    )
    # Macro stress → more bounces
    if macro_stress > 0.6:
        bounce_freq += (macro_stress - 0.6) * 0.08
        bounce_freq = _clamp(bounce_freq, 0, 0.80)
    # Dependency: high OD → more bounces
    if od_util > 75:
        bounce_freq += 0.03
        bounce_freq = _clamp(bounce_freq, 0, 0.80)

    related_party = _clamp(
        _sample_normal(tp["related_party"][0], tp["related_party"][1]),
        0, 50
    )

    collateral = _clamp(
        _sample_normal(tp["collateral"][0] + sp["collateral_adj"], tp["collateral"][1]),
        0.1, 4.0
    )

    # ─── ROUND 5: MANAGEMENT / REPUTATION CLUSTER ────────────
    litigation_raw = _sample_poisson(tp["litigation"][0])

    adverse_sentiment = _clamp(
        _sample_normal(tp["adverse_sentiment"][0], tp["adverse_sentiment"][1]),
        0, 1.0
    )
    # Dependency: more litigation → worse sentiment
    if litigation_raw >= 2:
        adverse_sentiment += 0.12
        adverse_sentiment = _clamp(adverse_sentiment, 0, 1.0)

    din_score = _clamp(
        _sample_normal(tp["din_score"][0], tp["din_score"][1]),
        0, 1.0
    )
    # Dependency: litigation → lower DIN
    if litigation_raw >= 2:
        din_score -= 0.10
        din_score = max(0, din_score)

    mca_charges = _sample_poisson(tp["mca_charges"][0])
    if litigation_raw >= 2:
        mca_charges += 1

    mgmt_stability = _clamp(
        _sample_normal(tp["mgmt_stability"][0], tp["mgmt_stability"][1]),
        0, 1.0
    )
    if litigation_raw >= 2:
        mgmt_stability -= 0.12
        mgmt_stability = max(0, mgmt_stability)

    factory_op = _clamp(
        _sample_normal(tp["factory_op"][0], tp["factory_op"][1]),
        0, 1.0
    )
    capacity_util = _clamp(
        _sample_normal(tp["capacity_util"][0], tp["capacity_util"][1]),
        0, 100
    )
    succession_risk = 1 if random.random() < tp["succession_risk_prob"][0] else 0

    sector_risk = _clamp(
        _sample_normal(tp["sector_risk_base"][0], tp["sector_risk_base"][1]),
        0, 1.0
    )
    if sector_under_stress and sector == sector_under_stress:
        sector_risk += 0.15
        sector_risk = _clamp(sector_risk, 0, 1.0)

    # ─── STEP 7: FORENSIC ALERT GENERATION ───────────────────
    alerts = _generate_alerts(
        gst_gap=gst_gap,
        bounce_freq=bounce_freq,
        circular_trading=circular_trading,
        adverse_sentiment=adverse_sentiment,
        litigation_count=litigation_raw,
        od_util=od_util,
        gst_alignment=gst_alignment,
    )

    # ─── NORMALIZE TO [0,1] FOR OBSERVATION SPACE ────────────
    features = {
        "promoter_litigation_count": _clamp(litigation_raw / 10.0, 0, 1),
        "mca_charge_count": _clamp(mca_charges / 10.0, 0, 1),
        "adverse_news_sentiment": adverse_sentiment,
        "promoter_din_score": din_score,
        "dscr_proxy": _clamp(dscr_raw / 4.0, 0, 1),
        "bank_od_utilisation_pct": od_util / 100.0,
        "cc_utilisation_volatility": _clamp(cc_volatility / 50.0, 0, 1),
        "gst_turnover_cagr": _clamp((gst_cagr_raw + 30) / 80.0, 0, 1),
        "current_ratio": _clamp(current_ratio / 5.0, 0, 1),
        "debt_to_equity": _clamp(de_ratio / 8.0, 0, 1),
        "return_on_net_worth": _clamp((ronw + 0.30) / 0.80, 0, 1),
        "ebitda_margin": _clamp((ebitda_margin + 0.10) / 0.50, 0, 1),
        "collateral_coverage_ratio": _clamp(collateral / 4.0, 0, 1),
        "gst_2a_vs_3b_gap_pct": _clamp(gst_gap / 80.0, 0, 1),
        "revenue_gst_alignment": gst_alignment,
        "itc_mismatch_flag": float(itc_mismatch),
        "circular_trading_ratio": _clamp(circular_trading / 0.50, 0, 1),
        "cheque_bounce_frequency": _clamp(bounce_freq / 0.80, 0, 1),
        "related_party_txn_pct": _clamp(related_party / 50.0, 0, 1),
        "working_capital_cycle_days": _clamp(wc_cycle / 180.0, 0, 1),
        "factory_operational_flag": factory_op,
        "capacity_utilisation_pct": capacity_util / 100.0,
        "succession_risk_flag": float(succession_risk),
        "sector_risk_score": sector_risk,
        "management_stability_score": mgmt_stability,
    }

    # Raw (denormalized) values for LLM prompt and reward computation
    raw_values = {
        "cibil_score": round(cibil_raw),
        "dscr": round(dscr_raw, 2),
        "current_ratio": round(current_ratio, 2),
        "debt_to_equity": round(de_ratio, 2),
        "ebitda_margin_pct": round(ebitda_margin * 100, 1),
        "ronw_pct": round(ronw * 100, 1),
        "revenue_cr": round(revenue_raw, 1),
        "loan_amount_cr": round(loan_raw, 1),
        "net_worth_cr": round(nw_base, 1),
        "od_utilisation_pct": round(od_util, 1),
        "cc_volatility_pct": round(cc_volatility, 1),
        "gst_cagr_pct": round(gst_cagr_raw, 1),
        "gst_gap_pct": round(gst_gap, 1),
        "bounce_rate_pct": round(bounce_freq * 100, 1),
        "collateral_ratio": round(collateral, 2),
        "litigation_count": litigation_raw,
        "wc_cycle_days": round(wc_cycle),
        "related_party_pct": round(related_party, 1),
        "circular_trading_pct": round(circular_trading * 100, 1),
    }

    # ─── HIDDEN PD (STEP 8 — Source of Truth) ────────────────
    hidden_pd = _compute_hidden_pd(
        tier=tier,
        dscr=dscr_raw,
        cibil=cibil_raw,
        current_ratio=current_ratio,
        de_ratio=de_ratio,
        macro_stress=macro_stress,
        sector=sector,
        sector_under_stress=sector_under_stress,
        alerts=alerts,
    )

    # ─── GROUND TRUTH (STEP 10) ──────────────────────────────
    hard_rules_triggered = _check_hard_rules(
        dscr=dscr_raw,
        din_score=din_score,
        bounce_freq=bounce_freq,
        gst_alignment=gst_alignment,
        adverse_sentiment=adverse_sentiment,
        alerts=alerts,
    )

    optimal_action = _determine_optimal_action(
        hidden_pd=hidden_pd,
        hard_rules=hard_rules_triggered,
    )

    # ─── GAP 3: MISSING FEATURES MASKING (Task 3) ────────────
    has_missing = False
    if task_id == "task3" and random.random() < 0.35:  # 20-40% probability
        has_missing = True
        features = _apply_missing_mask(features)

    return {
        "features": features,
        "raw_values": raw_values,
        "metadata": {
            "tier": tier,
            "sector": sector,
            "size": size,
            "hidden_pd": round(hidden_pd, 4),
            "optimal_action": optimal_action,
            "hard_rules_triggered": hard_rules_triggered,
            "alerts": alerts,
            "company_name": _generate_company_name(sector, size),
            "has_missing_features": has_missing,
        },
    }


# ═══════════════════════════════════════════════════════════════
# STEP 7: FORENSIC ALERT GENERATION
# ═══════════════════════════════════════════════════════════════

def _generate_alerts(
    gst_gap: float,
    bounce_freq: float,
    circular_trading: float,
    adverse_sentiment: float,
    litigation_count: int,
    od_util: float,
    gst_alignment: float,
) -> List[Dict[str, Any]]:
    alerts = []

    # ITC Mismatch
    if gst_gap > 15:
        severity = "RED" if gst_gap > 35 else "AMBER"
        alerts.append({
            "type": "ITC_MISMATCH",
            "severity": severity,
            "description": f"GST 2A vs 3B gap of {gst_gap:.0f}% detected",
            "impact_score": -15 if severity == "RED" else -8,
        })

    # Cheque Bounce
    if bounce_freq > 0.15:
        severity = "RED" if bounce_freq > 0.35 else "AMBER"
        alerts.append({
            "type": "CHEQUE_BOUNCE_SURGE",
            "severity": severity,
            "description": f"Cheque bounce rate at {bounce_freq*100:.0f}%",
            "impact_score": -12 if severity == "RED" else -5,
        })

    # Circular Trading
    if circular_trading > 0.15:
        alerts.append({
            "type": "CIRCULAR_TRADING",
            "severity": "RED",
            "description": f"Circular trading pattern detected at {circular_trading*100:.1f}%",
            "impact_score": -20,
        })

    # Adverse Media
    if adverse_sentiment > 0.70:
        severity = "RED" if adverse_sentiment > 0.85 else "AMBER"
        alerts.append({
            "type": "ADVERSE_MEDIA",
            "severity": severity,
            "description": f"Adverse media sentiment score: {adverse_sentiment:.2f}",
            "impact_score": -15 if severity == "RED" else -8,
        })

    # Litigation
    if litigation_count >= 3:
        severity = "RED" if litigation_count >= 5 else "AMBER"
        alerts.append({
            "type": "LITIGATION_EXPOSURE",
            "severity": severity,
            "description": f"{litigation_count} active litigation cases",
            "impact_score": -12 if severity == "RED" else -5,
        })

    # OD Overutilization
    if od_util > 85:
        severity = "RED" if od_util > 95 else "AMBER"
        alerts.append({
            "type": "OD_OVERUTILIZATION",
            "severity": severity,
            "description": f"OD utilisation at {od_util:.0f}%",
            "impact_score": -10 if severity == "RED" else -5,
        })

    # Revenue-GST misalignment
    if gst_alignment < 0.40:
        alerts.append({
            "type": "REVENUE_INFLATION",
            "severity": "RED",
            "description": f"Revenue-GST alignment critically low at {gst_alignment:.0%}",
            "impact_score": -15,
        })

    return alerts


# ═══════════════════════════════════════════════════════════════
# STEP 8: HIDDEN PD COMPUTATION
# ═══════════════════════════════════════════════════════════════

BASE_PD = {"A": 0.05, "B": 0.18, "C": 0.38, "D": 0.68}

def _compute_hidden_pd(
    tier: str,
    dscr: float,
    cibil: float,
    current_ratio: float,
    de_ratio: float,
    macro_stress: float,
    sector: str,
    sector_under_stress: Optional[str],
    alerts: List[Dict],
) -> float:
    pd = BASE_PD[tier]

    # Feature adjustments
    if dscr > 1.5: pd -= 0.05
    elif 0.7 <= dscr < 1.0: pd += 0.08
    elif dscr < 0.7: pd += 0.15

    if cibil > 750: pd -= 0.03
    elif 550 <= cibil < 650: pd += 0.05
    elif cibil < 550: pd += 0.12

    if current_ratio > 2.0: pd -= 0.02
    elif 0.8 <= current_ratio < 1.2: pd += 0.04
    elif current_ratio < 0.8: pd += 0.08

    if de_ratio < 1.0: pd -= 0.02
    elif 2.0 <= de_ratio < 3.5: pd += 0.05
    elif de_ratio >= 3.5: pd += 0.10

    # Macro adjustment
    if macro_stress < 0.3: pd -= 0.02
    elif 0.6 <= macro_stress < 0.8: pd += 0.05
    elif macro_stress >= 0.8: pd += 0.10

    # Sector adjustment
    if sector_under_stress and sector == sector_under_stress:
        pd += 0.08
    sp = SECTOR_PROFILES[sector]
    if macro_stress > 0.6 and sp["macro_sensitivity"] < 0.2:
        pd -= 0.03  # Counter-cyclical (e.g. Pharma)

    # GAP 10: Forensic adjustment — strengthened multiplier
    num_red = sum(1 for a in alerts if a["severity"] == "RED")
    num_amber = sum(1 for a in alerts if a["severity"] == "AMBER")
    pd += num_red * 0.08  # Each RED alert significantly raises PD
    pd += num_amber * 0.03  # AMBER alerts moderately raise PD
    # Cumulative effect: multiple alerts compound risk
    if num_red >= 2:
        pd *= 1.15  # 15% multiplicative boost for multiple RED alerts

    # Noise
    pd += random.gauss(0, 0.04)

    return _clamp(pd, 0.01, 0.99)


# ═══════════════════════════════════════════════════════════════
# GAP 3: MISSING FEATURE MASKING
# ═══════════════════════════════════════════════════════════════

_CRITICAL_FEATURES = [
    "dscr_proxy", "bank_od_utilisation_pct", "current_ratio",
    "debt_to_equity", "collateral_coverage_ratio",
]

_MASKABLE_FEATURES = [
    "promoter_litigation_count", "mca_charge_count", "adverse_news_sentiment",
    "promoter_din_score", "dscr_proxy", "bank_od_utilisation_pct",
    "cc_utilisation_volatility", "gst_turnover_cagr", "current_ratio",
    "debt_to_equity", "return_on_net_worth", "ebitda_margin",
    "collateral_coverage_ratio", "gst_2a_vs_3b_gap_pct",
    "revenue_gst_alignment", "sector_risk_score", "management_stability_score",
]


def _apply_missing_mask(features: Dict[str, float]) -> Dict[str, float]:
    """Mask 3-5 random features as -1.0 (sentinel for missing data).
    At most 2 critical features can be masked at once."""
    n_to_mask = random.randint(3, 5)
    candidates = list(_MASKABLE_FEATURES)
    random.shuffle(candidates)

    masked = dict(features)
    critical_masked = 0
    masked_count = 0

    for feat in candidates:
        if masked_count >= n_to_mask:
            break
        if feat in _CRITICAL_FEATURES:
            if critical_masked >= 2:
                continue  # Don't mask more than 2 critical features
            critical_masked += 1
        masked[feat] = -1.0
        masked_count += 1

    return masked


# ═══════════════════════════════════════════════════════════════
# STEP 10: HARD RULES & GROUND TRUTH
# ═══════════════════════════════════════════════════════════════

def _check_hard_rules(
    dscr: float,
    din_score: float,
    bounce_freq: float,
    gst_alignment: float,
    adverse_sentiment: float,
    alerts: List[Dict],
) -> List[str]:
    triggered = []
    if dscr < 1.0:
        triggered.append("HR-01: DSCR below 1.0")
    if din_score < 0.1:
        triggered.append("HR-02: Director disqualification")
    red_alerts = [a for a in alerts if a["severity"] == "RED"]
    if red_alerts:
        triggered.append(f"HR-03: {len(red_alerts)} RED forensic alert(s)")
    if bounce_freq > 0.25:
        triggered.append("HR-04: Cheque bounce rate > 25%")
    if gst_alignment < 0.40:
        triggered.append("HR-05: GST compliance < 40%")
    if adverse_sentiment > 0.80:
        triggered.append("HR-06: Severe adverse media")
    return triggered


def _determine_optimal_action(
    hidden_pd: float,
    hard_rules: List[str],
) -> int:
    """0=APPROVE, 1=CONDITIONAL, 2=REJECT"""
    if hard_rules:
        return 2  # Any hard rule → REJECT
    if hidden_pd < 0.15:
        return 0  # APPROVE
    elif hidden_pd < 0.40:
        return 1  # CONDITIONAL
    else:
        return 2  # REJECT


# ═══════════════════════════════════════════════════════════════
# COMPANY NAME GENERATOR (for LLM readability)
# ═══════════════════════════════════════════════════════════════

_SECTOR_PREFIXES = {
    "Manufacturing": ["Bharat", "Vishwa", "Shakti", "Tata", "Mahindra", "Godrej"],
    "IT_Services": ["Infosys", "Wipro", "TechM", "Zenith", "NexGen", "CloudX"],
    "Retail_Trading": ["Reliance", "BigBazaar", "Metro", "Star", "Value", "DMart"],
    "Infrastructure": ["L&T", "NHAI", "Adani", "Afcons", "Dilip", "Shapoorji"],
    "Pharma": ["Dr. Reddy", "Sun", "Cipla", "Lupin", "Aurobindo", "Glenmark"],
    "Agriculture": ["ITC", "Chambal", "IFFCO", "Tata", "Rallis", "UPL"],
    "Financial_Services": ["Bajaj", "HDFC", "Muthoot", "Shriram", "Manappuram", "Cholamandalam"],
    "Trading": ["Adani", "Cargill", "Olam", "Louis", "Glencore", "Trafigura"],
}

_SECTOR_SUFFIXES = {
    "Manufacturing": ["Industries", "Manufacturing", "Engineering", "Polymers", "Metals"],
    "IT_Services": ["Solutions", "Technologies", "Digital", "Systems", "Infotech"],
    "Retail_Trading": ["Retail", "Trading Co.", "Mart", "Stores", "Enterprises"],
    "Infrastructure": ["Infra", "Projects", "Construction", "Developers", "Builders"],
    "Pharma": ["Pharma", "Lifesciences", "Healthcare", "Biotech", "Therapeutics"],
    "Agriculture": ["Agri", "FarmTech", "Seeds", "Fertilizers", "AgroProducts"],
    "Financial_Services": ["Finance", "Capital", "Investments", "Finserv", "Lending"],
    "Trading": ["Commodities", "Exports", "Global Trade", "Mercantile", "Commerce"],
}


def _generate_company_name(sector: str, size: str) -> str:
    prefix = random.choice(_SECTOR_PREFIXES.get(sector, ["Acme"]))
    suffix = random.choice(_SECTOR_SUFFIXES.get(sector, ["Corp"]))
    tag = random.choice(["Pvt. Ltd.", "Ltd.", "LLP"]) if size != "Micro" else "Pvt. Ltd."
    return f"{prefix} {suffix} {tag}"


# ═══════════════════════════════════════════════════════════════
# EPISODE GENERATOR: 12-step sequential applications
# ═══════════════════════════════════════════════════════════════

def generate_episode(
    task_id: str = "task3",
    num_steps: int = 12,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate a full episode of sequential credit applications.
    Includes macro shock at ~T=7 for tasks 3-5.
    """
    random.seed(seed)
    applications = []

    macro_stress = 0.2  # starts benign
    sector_under_stress = None

    # Pre-select a sector that will be shocked
    shock_sector = random.choice(SECTORS)
    shock_timestep = 7 if task_id in ("task3", "task4", "task5") else num_steps + 1

    for t in range(1, num_steps + 1):
        # Update macro at shock timestep
        if t == shock_timestep:
            macro_stress = random.uniform(0.65, 0.85)
            sector_under_stress = shock_sector

        # Gradual stress buildup after shock
        if t > shock_timestep:
            macro_stress = min(0.90, macro_stress + random.uniform(0.01, 0.04))

        app = generate_application(
            task_id=task_id,
            macro_stress=macro_stress,
            sector_under_stress=sector_under_stress,
        )
        app["metadata"]["timestep"] = t
        app["metadata"]["macro_stress"] = round(macro_stress, 3)
        app["metadata"]["sector_under_stress"] = sector_under_stress
        applications.append(app)

    return applications


# ═══════════════════════════════════════════════════════════════
# STATE-TO-TEXT (for LLM inference)
# ═══════════════════════════════════════════════════════════════

# Map: feature key in features dict → corresponding raw_values key(s)
_FEATURE_TO_RAW = {
    "dscr_proxy": "dscr",
    "current_ratio": "current_ratio",
    "debt_to_equity": "debt_to_equity",
    "ebitda_margin": "ebitda_margin_pct",
    "return_on_net_worth": "ronw_pct",
    "collateral_coverage_ratio": "collateral_ratio",
    "bank_od_utilisation_pct": "od_utilisation_pct",
    "cc_utilisation_volatility": "cc_volatility_pct",
    "cheque_bounce_frequency": "bounce_rate_pct",
    "gst_turnover_cagr": "gst_cagr_pct",
    "gst_2a_vs_3b_gap_pct": "gst_gap_pct",
    "revenue_gst_alignment": None,  # no raw equivalent, shown inline
    "related_party_txn_pct": "related_party_pct",
    "circular_trading_ratio": "circular_trading_pct",
    "promoter_litigation_count": "litigation_count",
    "adverse_news_sentiment": None,
    "promoter_din_score": None,
    "sector_risk_score": None,
    "management_stability_score": None,
}

_UNAVAILABLE = "⬛ DATA UNAVAILABLE"


def _fmt_or_missing(features: Dict, feature_key: str, raw_val: Any, fmt: str = "{}") -> str:
    """Return formatted raw value or DATA UNAVAILABLE if feature is masked."""
    if features.get(feature_key, 0.0) == -1.0:
        return _UNAVAILABLE
    return fmt.format(raw_val)


def application_to_text(app: Dict[str, Any]) -> str:
    """Convert a generated application into an LLM-readable summary.
    ISSUE 2: Respects -1.0 mask — masked features show 'DATA UNAVAILABLE'.
    """
    m = app["metadata"]
    r = app["raw_values"]
    f = app["features"]
    alerts = m.get("alerts", [])

    # Revenue, loan, net worth, CIBIL are NEVER masked (metadata, not observation features)
    lines = [
        f"═══ CREDIT APPRAISAL SUMMARY — {m['company_name']} ═══",
        f"Sector: {m['sector']} | Size: {m['size']}",
        "",
        "── FINANCIAL PROFILE ──",
        f"  Revenue: ₹{r['revenue_cr']:.1f} Cr | Net Worth: ₹{r['net_worth_cr']:.1f} Cr",
        f"  Loan Requested: ₹{r['loan_amount_cr']:.1f} Cr",
        f"  CIBIL Score: {r['cibil_score']}",
        f"  DSCR: {_fmt_or_missing(f, 'dscr_proxy', r['dscr'], '{:.2f}x')} | Current Ratio: {_fmt_or_missing(f, 'current_ratio', r['current_ratio'], '{:.2f}')}",
        f"  Debt-to-Equity: {_fmt_or_missing(f, 'debt_to_equity', r['debt_to_equity'], '{:.2f}')} | EBITDA Margin: {_fmt_or_missing(f, 'ebitda_margin', r['ebitda_margin_pct'], '{:.1f}%')}",
        f"  Return on Net Worth: {_fmt_or_missing(f, 'return_on_net_worth', r['ronw_pct'], '{:.1f}%')}",
        f"  Collateral Coverage: {_fmt_or_missing(f, 'collateral_coverage_ratio', r['collateral_ratio'], '{:.2f}x')}",
        "",
        "── BANKING BEHAVIOR ──",
        f"  OD Utilisation: {_fmt_or_missing(f, 'bank_od_utilisation_pct', r['od_utilisation_pct'], '{:.1f}%')} | CC Volatility: {_fmt_or_missing(f, 'cc_utilisation_volatility', r['cc_volatility_pct'], '{:.1f}%')}",
        f"  Cheque Bounce Rate: {_fmt_or_missing(f, 'cheque_bounce_frequency', r['bounce_rate_pct'], '{:.1f}%')}",
        f"  Working Capital Cycle: {r['wc_cycle_days']} days",
        "",
        "── GST & COMPLIANCE ──",
        f"  GST Turnover CAGR: {_fmt_or_missing(f, 'gst_turnover_cagr', r['gst_cagr_pct'], '{:.1f}%')}",
        f"  GST 2A vs 3B Gap: {_fmt_or_missing(f, 'gst_2a_vs_3b_gap_pct', r['gst_gap_pct'], '{:.1f}%')}",
        f"  Related Party Transactions: {_fmt_or_missing(f, 'related_party_txn_pct', r['related_party_pct'], '{:.1f}%')}",
        f"  Circular Trading: {r['circular_trading_pct']:.1f}%",
        "",
        "── MANAGEMENT ──",
        f"  Promoter Litigation Cases: {_fmt_or_missing(f, 'promoter_litigation_count', r['litigation_count'])}",
    ]

    # Missing data warning
    missing_count = sum(1 for v in f.values() if v == -1.0)
    if missing_count > 0:
        lines.append("")
        lines.append(f"── ⚠️ INCOMPLETE DATA: {missing_count} field(s) unavailable ──")
        lines.append("  Exercise caution — missing data increases appraisal uncertainty.")

    if alerts:
        lines.append("")
        lines.append("── ⚠️ FORENSIC ALERTS ──")
        for alert in alerts:
            icon = "🔴" if alert["severity"] == "RED" else "🟡"
            lines.append(f"  {icon} [{alert['severity']}] {alert['type']}: {alert['description']}")
    else:
        lines.append("")
        lines.append("── ✅ NO FORENSIC ALERTS ──")

    lines.append("")
    lines.append("DECISION REQUIRED: APPROVE (0), CONDITIONAL (1), or REJECT (2)?")

    return "\n".join(lines)

