"""
IntelliCredit v2 — Tool Execution Engine
=========================================
Executes tool calls made by the LLM Credit Officer agent against the live
environment state. All tools are READ-ONLY — they return information only
and never mutate world state.

Supported tools:
    get_financial_report(company_id)     → detailed financial trends
    check_compliance_status(company_id)  → regulatory / governance status
    get_market_intelligence(sector)      → macro + sector context

Anti-hacking guarantees:
    - All tools are stateless reads.
    - Redundancy detection is handled upstream in agent_loop.py.
    - Malformed args return a structured error, still count as 1 call.
"""

import random
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════
# PUBLIC DISPATCHER
# ═══════════════════════════════════════════════════════════════

SUPPORTED_TOOLS = {
    "get_financial_report",
    "check_compliance_status",
    "get_market_intelligence",
}


def execute_tool(tool_name: str, args: Dict[str, Any], env: Any) -> Dict[str, Any]:
    """
    Dispatch a tool call to the correct handler.

    Args:
        tool_name : Lowercase tool identifier.
        args      : Parsed argument dict from action_parser.
        env       : Live IntelliCreditEnvironment instance.

    Returns:
        Structured result dict containing:
            tool_name    : str
            success      : bool
            data         : dict  (rich result — tool-specific)
            display_text : str   (human-readable, injected into LLM context)
            error        : str | None
    """
    tool_name = tool_name.strip().lower()

    if tool_name == "get_financial_report":
        return _get_financial_report(args, env)
    elif tool_name == "check_compliance_status":
        return _check_compliance_status(args, env)
    elif tool_name == "get_market_intelligence":
        return _get_market_intelligence(args, env)
    else:
        return _error_result(tool_name, f"Unknown tool '{tool_name}'. Supported: {sorted(SUPPORTED_TOOLS)}")


# ═══════════════════════════════════════════════════════════════
# TOOL 1 — get_financial_report
# ═══════════════════════════════════════════════════════════════

def _get_financial_report(args: Dict[str, Any], env: Any) -> Dict[str, Any]:
    """
    Return detailed financial history for the current application.

    Extracts and enriches data from env._applications[env._current_step].
    Simulates 3-year historical trends based on the current ratios + tier.
    """
    tool_name = "get_financial_report"
    try:
        step = env._current_step
        if step >= len(env._applications):
            return _error_result(tool_name, "No active application at current step.")

        app = env._applications[step]
        r   = app["raw_values"]
        m   = app["metadata"]
        f   = app["features"]
        tier = m.get("tier", "C")

        # ── Simulate 3-year revenue trend ────────────────────────
        rev_now   = r.get("revenue_cr", 10.0)
        rev_cagr  = r.get("gst_cagr_pct", 5.0) / 100.0
        rev_1yr   = round(rev_now / (1 + rev_cagr), 2)
        rev_2yr   = round(rev_1yr / (1 + rev_cagr), 2)
        revenue_3yr = [rev_2yr, rev_1yr, rev_now]
        revenue_growth_rate = round(rev_cagr * 100, 1)

        # ── Simulate 3-year EBITDA margin ────────────────────────
        em_now  = r.get("ebitda_margin_pct", 8.0)
        # Tier A tends to improve; Tier D deteriorates
        drift_map = {"A": 0.5, "B": 0.2, "C": -0.3, "D": -0.8}
        drift = drift_map.get(tier, 0.0)
        ebitda_3yr = [
            round(em_now - 2 * drift, 1),
            round(em_now - drift, 1),
            round(em_now, 1),
        ]

        # ── Debt schedule ─────────────────────────────────────────
        loan_ask      = r.get("loan_amount_cr", 5.0)
        de_ratio      = r.get("debt_to_equity", 1.5)
        net_worth     = r.get("net_worth_cr", 10.0)
        current_debt  = round(net_worth * de_ratio * 0.4, 2)
        long_term_debt= round(net_worth * de_ratio * 0.6, 2)
        repayment_pct = {"A": 0.20, "B": 0.15, "C": 0.10, "D": 0.05}.get(tier, 0.10)
        annual_repayment = round((current_debt + long_term_debt) * repayment_pct, 2)

        # ── Auditor remarks (based on tier + alerts) ──────────────
        alerts     = m.get("alerts", [])
        red_alerts = [a for a in alerts if a.get("severity") == "RED"]
        if tier == "D" or len(red_alerts) >= 2:
            auditor_remark = (
                "Going concern qualification issued. Auditors noted significant doubt "
                "about the entity's ability to continue as a going concern for the next 12 months."
            )
        elif tier == "C" or len(red_alerts) == 1:
            auditor_remark = (
                "Emphasis of matter on related-party transactions. "
                "Audit report contains modified opinion on inventory valuation."
            )
        else:
            auditor_remark = "Unqualified (clean) audit opinion for all 3 years."

        # ── Cash flow from operations ─────────────────────────────
        dscr = r.get("dscr", 1.0)
        cfo  = round(loan_ask * dscr * 0.25, 2)  # simplified proxy
        related_party_pct = r.get("related_party_pct", 8.0)

        data = {
            "company_name"          : m.get("company_name", "Unknown"),
            "tier"                  : tier,
            "revenue_3yr_cr"        : revenue_3yr,
            "revenue_growth_rate_pct": revenue_growth_rate,
            "ebitda_margin_3yr_pct" : ebitda_3yr,
            "debt_schedule"         : {
                "current_debt_cr"    : current_debt,
                "long_term_debt_cr"  : long_term_debt,
                "annual_repayment_cr": annual_repayment,
            },
            "auditor_remarks"       : auditor_remark,
            "related_party_txn_pct" : related_party_pct,
            "cash_flow_operations_cr": cfo,
            "loan_requested_cr"     : loan_ask,
        }

        display = _format_financial_report(data)
        return {"tool_name": tool_name, "success": True, "data": data, "display_text": display, "error": None}

    except Exception as exc:
        return _error_result(tool_name, str(exc))


def _format_financial_report(d: Dict) -> str:
    rev = d["revenue_3yr_cr"]
    em  = d["ebitda_margin_3yr_pct"]
    ds  = d["debt_schedule"]
    lines = [
        f"═══ FINANCIAL REPORT — {d['company_name']} (Tier {d['tier']}) ═══",
        "── Revenue (₹ Cr) ──",
        f"  FY-2  : ₹{rev[0]}  |  FY-1: ₹{rev[1]}  |  FY0 (current): ₹{rev[2]}",
        f"  Revenue CAGR: {d['revenue_growth_rate_pct']}%",
        "── EBITDA Margin (%) ──",
        f"  FY-2  : {em[0]}%  |  FY-1: {em[1]}%  |  FY0: {em[2]}%",
        "── Debt Schedule ──",
        f"  Current Debt : ₹{ds['current_debt_cr']} Cr",
        f"  Long-term Debt: ₹{ds['long_term_debt_cr']} Cr",
        f"  Annual Repayment: ₹{ds['annual_repayment_cr']} Cr",
        "── Other ──",
        f"  Cash Flow from Ops: ₹{d['cash_flow_operations_cr']} Cr",
        f"  Related Party Txns: {d['related_party_txn_pct']}% of revenue",
        f"  Auditor Remarks: {d['auditor_remarks']}",
        "═══ END FINANCIAL REPORT ═══",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# TOOL 2 — check_compliance_status
# ═══════════════════════════════════════════════════════════════

def _check_compliance_status(args: Dict[str, Any], env: Any) -> Dict[str, Any]:
    """
    Return regulatory compliance and legal standing for the current application.
    """
    tool_name = "check_compliance_status"
    try:
        step = env._current_step
        if step >= len(env._applications):
            return _error_result(tool_name, "No active application at current step.")

        app = env._applications[step]
        r   = app["raw_values"]
        m   = app["metadata"]
        f   = app["features"]
        tier = m.get("tier", "C")
        hard_rules  = m.get("hard_rules_triggered", [])
        alerts      = m.get("alerts", [])
        is_repeat   = m.get("is_repeat_applicant", False)
        attempt_num = m.get("attempt_number", 1)

        # ── MCA filing status ─────────────────────────────────────
        din_score = f.get("promoter_din_score", 0.5)
        mca_filing_current = din_score > 0.3 and tier in ("A", "B")

        # ── GST returns filed ─────────────────────────────────────
        gst_align = f.get("revenue_gst_alignment", 0.8)
        gst_returns_filed = max(0, min(12, int(gst_align * 12)))

        # ── Director DIN status ───────────────────────────────────
        din_active = din_score >= 0.10
        director_din_status = {
            "primary_director": "ACTIVE" if din_active else "DISQUALIFIED",
            "din_score": round(din_score, 3),
        }

        # ── NCLT cases (from litigation count) ───────────────────
        lit_raw  = r.get("litigation_count", 0)
        nclt_cases = max(0, lit_raw - 1)  # not all litigation = NCLT

        # ── ROC charges ──────────────────────────────────────────
        mca_charges = int(f.get("mca_charge_count", 0) * 10)

        # ── CIBIL (already in raw; normalize to 0-1) ──────────────
        cibil_raw   = r.get("cibil_score", 600)
        cibil_norm  = round((cibil_raw - 300) / 600, 3)

        # ── Previous defaults ─────────────────────────────────────
        prev_defaults = 0
        if tier == "D":
            prev_defaults = random.randint(1, 3)
        elif tier == "C":
            prev_defaults = random.randint(0, 1)

        # ── Hard rules triggered ──────────────────────────────────
        data = {
            "company_name"          : m.get("company_name", "Unknown"),
            "is_repeat_applicant"   : is_repeat,
            "attempt_number"        : attempt_num,
            "mca_filing_current"    : mca_filing_current,
            "gst_returns_filed"     : gst_returns_filed,
            "director_din_status"   : director_din_status,
            "nclt_cases"            : nclt_cases,
            "roc_charges"           : mca_charges,
            "cibil_score"           : cibil_norm,
            "previous_loan_defaults": prev_defaults,
            "hard_rules_triggered"  : hard_rules,
            "forensic_alerts"       : alerts,
        }

        display = _format_compliance_report(data)
        return {"tool_name": tool_name, "success": True, "data": data, "display_text": display, "error": None}

    except Exception as exc:
        return _error_result(tool_name, str(exc))


def _format_compliance_report(d: Dict) -> str:
    din = d["director_din_status"]
    line_repeat = (
        f"  ⚠️  REPEAT APPLICANT — Attempt #{d['attempt_number']} (rejected before)"
        if d["is_repeat_applicant"] else
        "  First-time applicant."
    )
    lines = [
        f"═══ COMPLIANCE STATUS — {d['company_name']} ═══",
        line_repeat,
        "── Registrations ──",
        f"  MCA Filing Current    : {'✅ YES' if d['mca_filing_current'] else '❌ NO'}",
        f"  GST Returns Filed     : {d['gst_returns_filed']}/12",
        f"  Director DIN          : {din['primary_director']} (score={din['din_score']})",
        "── Legal Exposure ──",
        f"  NCLT Cases            : {d['nclt_cases']}",
        f"  ROC Charges           : {d['roc_charges']}",
        f"  Previous Loan Defaults: {d['previous_loan_defaults']}",
        f"  CIBIL Score           : {d['cibil_score']:.3f} (0=bad, 1=excellent)",
    ]
    if d["hard_rules_triggered"]:
        lines.append("── 🚫 HARD RULES TRIGGERED (MANDATORY REJECT) ──")
        for r in d["hard_rules_triggered"]:
            lines.append(f"  • {r}")
    else:
        lines.append("  ✅ No hard rules triggered.")
    if d["forensic_alerts"]:
        lines.append("── Forensic Alerts ──")
        for a in d["forensic_alerts"]:
            icon = "🔴" if a.get("severity") == "RED" else "🟡"
            lines.append(f"  {icon} [{a.get('severity')}] {a.get('type')}: {a.get('description', '')}")
    else:
        lines.append("  ✅ No forensic alerts.")
    lines.append("═══ END COMPLIANCE ═══")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# TOOL 3 — get_market_intelligence
# ═══════════════════════════════════════════════════════════════

def _get_market_intelligence(args: Dict[str, Any], env: Any) -> Dict[str, Any]:
    """
    Return macro-economic context and sector-level intelligence.
    """
    tool_name = "get_market_intelligence"
    try:
        sector_query = (
            args.get("sector", "") or args.get("arg0", "") or ""
        ).strip().strip("'\"")

        world     = env._world
        portfolio = env._portfolio
        step      = env._current_step

        # ── Macro state ───────────────────────────────────────────
        macro = world.get_macro_obs() if world else [0.2, 0.0, 0.5, 0.5, 0.5]
        macro_stress    = macro[0]
        shock_active    = macro[1] > 0.5
        gdp_growth      = macro[2]
        inflation       = macro[3]
        credit_cycle    = world.credit_cycle_phase.value if world else "EXPANSION"

        macro_level = (
            "HIGH STRESS ⚠️" if macro_stress > 0.6
            else ("MODERATE" if macro_stress > 0.35 else "STABLE ✅")
        )

        # ── RBI advisory (triggered by high macro stress) ─────────
        rbi_sector_advisory = macro_stress > 0.65 or shock_active

        # ── Sector exposure from portfolio ────────────────────────
        total_deployed = sum(portfolio.sector_exposure.values()) if portfolio else 1.0
        sector_exposure_pct = 0.0
        matched_sector = ""
        if portfolio and sector_query:
            for sec, amt in portfolio.sector_exposure.items():
                if sector_query.lower() in sec.lower():
                    sector_exposure_pct = amt / max(total_deployed, 1.0) * 100
                    matched_sector = sec
                    break

        # ── Sector-specific stress from WorldState ────────────────
        sector_stress = 0.0
        if world and sector_query:
            for sec, stress in world.sector_stress_scores.items():
                if sector_query.lower() in sec.lower():
                    sector_stress = stress
                    break

        # ── Headwinds / tailwinds (derived from macro) ────────────
        headwinds: List[str] = []
        tailwinds: List[str] = []
        if shock_active:
            headwinds.append("Active macro shock compressing margins across sectors")
        if macro_stress > 0.5:
            headwinds.append("Rising interest rates increasing borrower cost of capital")
        if inflation > 0.6:
            headwinds.append("High inflation eroding real revenue growth")
        if gdp_growth > 0.6:
            tailwinds.append("Strong GDP growth supporting demand recovery")
        if credit_cycle == "EXPANSION":
            tailwinds.append("Credit cycle in expansion phase — favourable lending conditions")
        elif credit_cycle == "TROUGH":
            headwinds.append("Credit cycle at trough — defaults expected to rise")

        # Add sector-specific context
        _sector_headwinds = {
            "Infrastructure": ["Long project cycles increase liquidity risk"],
            "Real_Estate":    ["Unsold inventory pressures cash flow"],
            "Agriculture":    ["Seasonal cash flow creates repayment bunching"],
            "Financial_Services": ["Systemic risk correlation with macro shocks"],
        }
        _sector_tailwinds = {
            "Pharma":       ["Counter-cyclical demand provides stability"],
            "IT_Services":  ["Dollar-denominated revenues provide FX hedge"],
        }
        for key, hw in _sector_headwinds.items():
            if sector_query.lower() in key.lower():
                headwinds.extend(hw)
        for key, tw in _sector_tailwinds.items():
            if sector_query.lower() in key.lower():
                tailwinds.extend(tw)

        # ── Peer NPA rate estimate ─────────────────────────────────
        base_npa = {"A": 0.01, "B": 0.03, "C": 0.06, "D": 0.12}
        peer_npa = round(0.035 + macro_stress * 0.03 + sector_stress * 0.04, 4)

        # ── Macro correlation ─────────────────────────────────────
        _macro_corr = {
            "Manufacturing": 0.70, "Infrastructure": 0.85,
            "IT_Services": 0.50, "Retail_Trading": 0.55,
            "Agriculture": 0.60, "Pharma": 0.15,
            "Financial_Services": 0.75, "Trading": 0.45,
        }
        corr = 0.60
        for key, val in _macro_corr.items():
            if sector_query.lower() in key.lower():
                corr = val
                break

        # ── Regulatory changes ────────────────────────────────────
        recent_reg = (
            "RBI Circular: Enhanced provisioning norms for restructured accounts in stressed sectors."
            if rbi_sector_advisory else
            "No recent material regulatory changes for this sector."
        )

        # ── Concentration warning ─────────────────────────────────
        concentration_warning = None
        if sector_exposure_pct > 25:
            concentration_warning = (
                f"⚠️  Portfolio already has {sector_exposure_pct:.1f}% in "
                f"{matched_sector or sector_query} — approaching 30% limit!"
            )

        data = {
            "sector_queried"               : sector_query or "General",
            "macro_stress_level"           : macro_level,
            "macro_stress_raw"             : round(macro_stress, 3),
            "macro_shock_active"           : shock_active,
            "gdp_growth_index"             : round(gdp_growth, 3),
            "inflation_index"              : round(inflation, 3),
            "credit_cycle_phase"           : credit_cycle,
            "rbi_sector_advisory"          : rbi_sector_advisory,
            "portfolio_exposure_pct"       : round(sector_exposure_pct, 2),
            "sector_stress_score"          : round(sector_stress, 3),
            "headwinds"                    : headwinds,
            "tailwinds"                    : tailwinds,
            "recent_regulatory_changes"    : recent_reg,
            "peer_npa_rate"                : peer_npa,
            "correlation_to_macro_shock"   : corr,
            "concentration_warning"        : concentration_warning,
            "portfolio_npa_rate"           : round(portfolio.npa_rate, 4) if portfolio else 0.0,
            "portfolio_crar"               : round(portfolio.crar, 4) if portfolio else 1.0,
            "steps_to_next_audit"          : min((a for a in {10,20,30,40,50} if a > step), default=50) - step,
        }

        display = _format_market_intelligence(data)
        return {"tool_name": tool_name, "success": True, "data": data, "display_text": display, "error": None}

    except Exception as exc:
        return _error_result(tool_name, str(exc))


def _format_market_intelligence(d: Dict) -> str:
    lines = [
        f"═══ MARKET INTELLIGENCE — Sector: {d['sector_queried']} ═══",
        "── Macro Environment ──",
        f"  Stress Level       : {d['macro_stress_level']} ({d['macro_stress_raw']})",
        f"  Macro Shock Active : {'YES ⚠️' if d['macro_shock_active'] else 'No'}",
        f"  GDP Growth Index   : {d['gdp_growth_index']}",
        f"  Inflation Index    : {d['inflation_index']}",
        f"  Credit Cycle       : {d['credit_cycle_phase']}",
        f"  RBI Sector Advisory: {'⚠️  YES — Exercise caution' if d['rbi_sector_advisory'] else '✅ None'}",
        "── Sector Analytics ──",
        f"  Portfolio Exposure : {d['portfolio_exposure_pct']}%",
        f"  Sector Stress Score: {d['sector_stress_score']}",
        f"  Peer NPA Rate      : {d['peer_npa_rate']:.2%}",
        f"  Macro Correlation  : {d['correlation_to_macro_shock']}",
    ]
    if d["headwinds"]:
        lines.append("  Headwinds:")
        for h in d["headwinds"]:
            lines.append(f"    ⬇ {h}")
    if d["tailwinds"]:
        lines.append("  Tailwinds:")
        for t in d["tailwinds"]:
            lines.append(f"    ⬆ {t}")
    lines.append(f"  Reg Changes: {d['recent_regulatory_changes']}")
    if d["concentration_warning"]:
        lines.append(f"  {d['concentration_warning']}")
    lines.append("── Portfolio Health ──")
    lines.append(f"  NPA Rate      : {d['portfolio_npa_rate']:.2%}")
    lines.append(f"  CRAR          : {d['portfolio_crar']:.2%}")
    lines.append(f"  Steps to Audit: {d['steps_to_next_audit']}")
    lines.append("═══ END INTELLIGENCE ═══")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _error_result(tool_name: str, message: str) -> Dict[str, Any]:
    display = f"[TOOL ERROR: {tool_name}] {message}"
    return {
        "tool_name"   : tool_name,
        "success"     : False,
        "data"        : {},
        "display_text": display,
        "error"       : message,
    }


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from server.intellicredit_env import IntelliCreditEnvironment

    env = IntelliCreditEnvironment(task_id="task3")
    env.reset(seed=42)

    print("=" * 65)
    print("  tool_executor.py Self-Test")
    print("=" * 65)

    for tool, args in [
        ("get_financial_report",   {"company_id": "Bharat Manufacturing Pvt. Ltd."}),
        ("check_compliance_status",{"company_id": "Bharat Manufacturing Pvt. Ltd."}),
        ("get_market_intelligence",{"sector": "Manufacturing"}),
        ("unknown_tool",           {}),
    ]:
        result = execute_tool(tool, args, env)
        status = "✅" if result["success"] else "❌"
        print(f"\n{status}  {tool}()")
        print(result["display_text"][:400])
        print("...")

    print("\n" + "=" * 65)
    print("  Self-test complete.")
    print("=" * 65)
