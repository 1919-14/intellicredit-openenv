"""
IntelliCredit v2 — LLM Action Parser & Tool Execution Flow
===========================================================
Parses free-form LLM output into structured actions for the environment.

Supports four parse paths (in strict priority order):
    1. Tool call detection  → execute tool, do NOT advance step counter
    2. submit_decision()    → extract decision + reasoning, advance step
    3. Fallback keyword     → APPROVE / CONDITIONAL / REJECT word scan
    4. Default fallback     → safe REJECT, parse_failure=True

parse_llm_output() return schema:
    {
        "parse_type"       : "tool_call" | "final_decision" | "fallback_keyword" | "default_reject",
        "tool_name"        : str | None,
        "tool_args"        : dict | None,
        "action"           : int,   # 0=APPROVE, 1=CONDITIONAL, 2=REJECT
        "reasoning"        : str,
        "parse_confidence" : float, # 0.0–1.0
        "parse_failure"    : bool
    }
"""

import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular imports; env passed at runtime

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

ACTION_MAP: Dict[str, int] = {
    "APPROVE": 0,
    "APPROVED": 0,
    "CONDITIONAL": 1,
    "CONDITIONAL_APPROVE": 1,
    "CONDITIONAL APPROVE": 1,
    "REJECT": 2,
    "REJECTED": 2,
    "DECLINE": 2,
    "DENY": 2,
}

SUPPORTED_TOOLS: List[str] = [
    "get_financial_report",
    "check_compliance_status",
    "get_market_intelligence",
    "submit_decision",
]

# Regex patterns (compiled once for performance)

# Matches: tool_name(arg1, arg2) or tool_name('arg1', arg2=val)
_RE_TOOL_CALL = re.compile(
    r"\b((?:get_financial_report|check_compliance_status|get_market_intelligence))\s*\(([^)]*)\)",
    re.IGNORECASE,
)

# Matches: submit_decision('ACTION', 'reasoning text') with various quote styles
_RE_SUBMIT_DECISION = re.compile(
    r"submit_decision\s*\(\s*['\"]?([A-Z_]+)['\"]?\s*,\s*['\"]?(.*?)['\"]?\s*\)",
    re.IGNORECASE | re.DOTALL,
)

# Matches standalone decision keywords (whole-word boundary)
_RE_KEYWORD = re.compile(
    r"\b(APPROVE(?:D)?|CONDITIONAL(?:_APPROVE)?|REJECT(?:ED)?|DECLINE|DENY)\b",
    re.IGNORECASE,
)

# Matches quoted string arguments in tool calls
_RE_QUOTED_ARG = re.compile(r"['\"]([^'\"]*)['\"]")


# ═══════════════════════════════════════════════════════════════
# ARGUMENT PARSING HELPERS
# ═══════════════════════════════════════════════════════════════

def _parse_tool_args(raw_args: str, tool_name: str) -> Dict[str, Any]:
    """
    Parse raw argument string from a tool call into a dict.
    Handles: positional strings, key=value pairs, bare words.
    Falls back to {"raw": raw_args} if fully malformed.
    """
    raw_args = raw_args.strip()
    if not raw_args:
        return {}

    result: Dict[str, Any] = {}

    # -- Handle key=value pairs first --------------------------------
    kv_pattern = re.compile(r"(\w+)\s*=\s*['\"]?([^,'\"]+)['\"]?")
    kv_matches = kv_pattern.findall(raw_args)
    if kv_matches:
        for k, v in kv_matches:
            result[k.strip()] = v.strip()
        return result

    # -- Handle positional quoted strings ----------------------------
    quoted = _RE_QUOTED_ARG.findall(raw_args)
    if quoted:
        # First positional arg is always "company_id" or "sector"
        param_names = _get_param_names(tool_name)
        for i, val in enumerate(quoted):
            key = param_names[i] if i < len(param_names) else f"arg{i}"
            result[key] = val.strip()
        return result

    # -- Bare word (no quotes) ---------------------------------------
    bare = raw_args.strip().strip("'\"")
    if bare:
        param_names = _get_param_names(tool_name)
        result[param_names[0] if param_names else "arg0"] = bare
        return result

    return {"raw": raw_args}


def _get_param_names(tool_name: str) -> List[str]:
    """Return expected positional parameter names per tool."""
    mapping = {
        "get_financial_report": ["company_id"],
        "check_compliance_status": ["company_id"],
        "get_market_intelligence": ["sector"],
    }
    return mapping.get(tool_name.lower(), ["arg0"])


# ═══════════════════════════════════════════════════════════════
# CORE PARSER
# ═══════════════════════════════════════════════════════════════

def parse_llm_output(text: str) -> Dict[str, Any]:
    """
    Parse free-form LLM output text into a structured action dict.

    Priority order:
        1. Tool call (get_financial_report, check_compliance_status, get_market_intelligence)
        2. submit_decision(ACTION, reasoning)
        3. Standalone keyword fallback (APPROVE / CONDITIONAL / REJECT)
        4. Safe default → REJECT with parse_failure=True

    Args:
        text: Raw LLM generation string.

    Returns:
        Dict with keys: parse_type, tool_name, tool_args, action,
                        reasoning, parse_confidence, parse_failure.
    """
    if not text or not text.strip():
        return _default_reject("Empty LLM output received.")

    text_clean = text.strip()

    # ── STEP 1: Tool call detection ──────────────────────────────
    tool_matches = list(_RE_TOOL_CALL.finditer(text_clean))
    if tool_matches:
        # Use FIRST tool call found (LLM calls one tool at a time)
        m = tool_matches[0]
        tool_name = m.group(1).lower()
        raw_args = m.group(2)
        tool_args = _parse_tool_args(raw_args, tool_name)
        return {
            "parse_type": "tool_call",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "action": 2,            # No decision yet; env does NOT advance step
            "reasoning": f"Calling tool: {tool_name}({raw_args.strip()})",
            "parse_confidence": 0.95,
            "parse_failure": False,
        }

    # ── STEP 2: submit_decision() detection ─────────────────────
    submit_matches = list(_RE_SUBMIT_DECISION.finditer(text_clean))
    if submit_matches:
        # ANTI-HACKING: Use LAST occurrence if multiple submit_decision calls
        m = submit_matches[-1]
        raw_action = m.group(1).upper().strip()
        reasoning = m.group(2).strip() if m.group(2) else ""
        action = _normalize_action(raw_action)

        # Penalize empty reasoning
        confidence = 0.90 if reasoning else 0.65
        if not reasoning:
            reasoning = "[WARNING: No reasoning provided — penalty flag set]"

        return {
            "parse_type": "final_decision",
            "tool_name": None,
            "tool_args": None,
            "action": action,
            "reasoning": reasoning,
            "parse_confidence": confidence,
            "parse_failure": False,
        }

    # ── STEP 3: Fallback keyword scan ───────────────────────────
    keyword_matches = list(_RE_KEYWORD.finditer(text_clean))
    if keyword_matches:
        # Use LAST keyword found (anti-hacking: final stance wins)
        kw = keyword_matches[-1].group(1).upper()
        action = ACTION_MAP.get(kw, 2)

        # Extract surrounding text as reasoning context
        match_end = keyword_matches[-1].end()
        reasoning_hint = text_clean[:match_end].strip()[-200:]  # last 200 chars

        return {
            "parse_type": "fallback_keyword",
            "tool_name": None,
            "tool_args": None,
            "action": action,
            "reasoning": f"[KEYWORD FALLBACK] Detected '{kw}' in: ...{reasoning_hint}",
            "parse_confidence": 0.55,
            "parse_failure": False,
        }

    # ── STEP 4: Default safe fallback ───────────────────────────
    return _default_reject(
        f"Parser could not extract any action from output (len={len(text_clean)}). "
        f"Preview: '{text_clean[:120]}'"
    )


def _normalize_action(raw: str) -> int:
    """Map a raw action string to int. Invalid → 2 (REJECT, safe default)."""
    return ACTION_MAP.get(raw, ACTION_MAP.get(raw.replace("_", " "), 2))


def _default_reject(reason: str) -> Dict[str, Any]:
    """Return a safe REJECT result with parse_failure=True."""
    return {
        "parse_type": "default_reject",
        "tool_name": None,
        "tool_args": None,
        "action": 2,
        "reasoning": f"[PARSE FAILURE — defaulting to REJECT] {reason}",
        "parse_confidence": 0.0,
        "parse_failure": True,
    }


# ═══════════════════════════════════════════════════════════════
# TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, tool_args: Dict[str, Any], env: Any) -> str:
    """
    Execute a tool call against the live environment state.

    Returns a string that is injected back into the LLM's context
    as a "tool result" message.

    Args:
        tool_name: Lowercase tool name (e.g. "get_financial_report").
        tool_args: Parsed argument dict.
        env: The IntelliCreditEnvironment instance (has _applications, _portfolio, etc.)

    Returns:
        Human-readable tool result string.
    """
    tool_name = tool_name.lower().strip()

    if tool_name == "get_financial_report":
        return _tool_get_financial_report(tool_args, env)
    elif tool_name == "check_compliance_status":
        return _tool_check_compliance_status(tool_args, env)
    elif tool_name == "get_market_intelligence":
        return _tool_get_market_intelligence(tool_args, env)
    else:
        return f"[TOOL ERROR] Unknown tool: '{tool_name}'. Supported: {SUPPORTED_TOOLS}"


def _tool_get_financial_report(args: Dict[str, Any], env: Any) -> str:
    """Return detailed financial metrics for the current application."""
    try:
        step = env._current_step
        if step >= len(env._applications):
            return "[TOOL RESULT: get_financial_report] No active application."

        app = env._applications[step]
        r = app["raw_values"]
        m = app["metadata"]

        lines = [
            "═══ FINANCIAL REPORT ═══",
            f"Company : {m.get('company_name', 'Unknown')}",
            f"Sector  : {m.get('sector', 'Unknown')} | Size: {m.get('size', 'Unknown')}",
            f"Tier    : {m.get('tier', 'Unknown')}",
            "─── Key Financials ───",
            f"  Revenue       : ₹{r.get('revenue_cr', 'N/A')} Cr",
            f"  Net Worth     : ₹{r.get('net_worth_cr', 'N/A')} Cr",
            f"  Loan Ask      : ₹{r.get('loan_amount_cr', 'N/A')} Cr",
            f"  CIBIL Score   : {r.get('cibil_score', 'N/A')}",
            f"  DSCR          : {r.get('dscr', 'N/A')}x",
            f"  Current Ratio : {r.get('current_ratio', 'N/A')}",
            f"  D/E Ratio     : {r.get('debt_to_equity', 'N/A')}",
            f"  EBITDA Margin : {r.get('ebitda_margin_pct', 'N/A')}%",
            f"  RONW          : {r.get('ronw_pct', 'N/A')}%",
            f"  Collateral    : {r.get('collateral_ratio', 'N/A')}x",
            "─── Banking Behaviour ───",
            f"  OD Utilisation: {r.get('od_utilisation_pct', 'N/A')}%",
            f"  CC Volatility : {r.get('cc_volatility_pct', 'N/A')}%",
            f"  Cheque Bounce : {r.get('bounce_rate_pct', 'N/A')}%",
            f"  WC Cycle      : {r.get('wc_cycle_days', 'N/A')} days",
            "═══ END REPORT ═══",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"[TOOL ERROR: get_financial_report] {exc}"


def _tool_check_compliance_status(args: Dict[str, Any], env: Any) -> str:
    """Return hard rule status and forensic alerts for the current application."""
    try:
        step = env._current_step
        if step >= len(env._applications):
            return "[TOOL RESULT: check_compliance_status] No active application."

        app = env._applications[step]
        m = app["metadata"]
        hard_rules: List[str] = m.get("hard_rules_triggered", [])
        alerts: List[Dict] = m.get("alerts", [])
        is_repeat = m.get("is_repeat_applicant", False)
        attempt_num = m.get("attempt_number", 1)

        lines = ["═══ COMPLIANCE STATUS ═══"]

        if is_repeat:
            lines.append(f"⚠️  REPEAT APPLICANT — Attempt #{attempt_num} (was rejected before)")

        if hard_rules:
            lines.append("🚫 HARD RULES TRIGGERED (MANDATORY REJECT):")
            for rule in hard_rules:
                lines.append(f"   • {rule}")
        else:
            lines.append("✅ No hard rules triggered.")

        if alerts:
            lines.append("── Forensic Alerts ──")
            for alert in alerts:
                icon = "🔴" if alert.get("severity") == "RED" else "🟡"
                lines.append(
                    f"  {icon} [{alert.get('severity')}] {alert.get('type')}: "
                    f"{alert.get('description', '')}"
                )
        else:
            lines.append("✅ No forensic alerts.")

        lines.append("═══ END COMPLIANCE ═══")
        return "\n".join(lines)
    except Exception as exc:
        return f"[TOOL ERROR: check_compliance_status] {exc}"


def _tool_get_market_intelligence(args: Dict[str, Any], env: Any) -> str:
    """Return macro-economic and sector-specific stress data."""
    try:
        sector_query = args.get("sector", "").strip()
        macro = env._macro_state  # [stress, shock, gdp, inflation, cycle]
        portfolio = env._portfolio

        macro_label = "HIGH STRESS" if macro[0] > 0.6 else ("MODERATE" if macro[0] > 0.35 else "STABLE")
        shock_active = "YES ⚠️" if macro[1] > 0.5 else "No"

        lines = [
            "═══ MARKET INTELLIGENCE ═══",
            f"Macro Stress Level : {macro[0]:.2f} ({macro_label})",
            f"Macro Shock Active : {shock_active}",
            f"GDP Growth Index   : {macro[2]:.2f}",
            f"Inflation Index    : {macro[3]:.2f}",
            f"Credit Cycle Phase : {macro[4]:.2f}",
        ]

        if portfolio:
            lines.append("── Portfolio Snapshot ──")
            lines.append(f"  NPA Rate  : {portfolio.npa_rate:.1%}")
            lines.append(f"  CRAR      : {portfolio.crar:.1%}")
            cap_deployed_pct = (
                portfolio.capital_deployed / portfolio.total_capital * 100
                if portfolio.total_capital > 0 else 0
            )
            lines.append(f"  Capital Deployed : {cap_deployed_pct:.1f}%")
            lines.append(f"  Capital Remaining: ₹{portfolio.capital_remaining:.1f} Cr")

            if sector_query and portfolio.sector_exposure:
                sector_pct = 0.0
                for k, v in portfolio.sector_exposure.items():
                    if sector_query.lower() in k.lower():
                        total = sum(portfolio.sector_exposure.values())
                        sector_pct = v / total * 100 if total > 0 else 0.0
                        break
                lines.append(f"── Sector: {sector_query} ──")
                lines.append(f"  Current Exposure : {sector_pct:.1f}%")
                if sector_pct > 25:
                    lines.append("  ⚠️  Approaching sector concentration limit (30%)")
                elif sector_pct > 30:
                    lines.append("  🚫 SECTOR LIMIT BREACHED — further approvals penalised")

        lines.append("═══ END INTELLIGENCE ═══")
        return "\n".join(lines)
    except Exception as exc:
        return f"[TOOL ERROR: get_market_intelligence] {exc}"


# ═══════════════════════════════════════════════════════════════
# HIGH-LEVEL STEP PROCESSOR
# ═══════════════════════════════════════════════════════════════

def process_llm_step(llm_output: str, env: Any) -> Dict[str, Any]:
    """
    Top-level entry point for the RL training loop.

    Parses LLM output, executes tools or submits decisions.

    Rule:
        - Tool call → execute tool, return tool_result. Step counter NOT advanced.
        - Final decision → call env.step(), return reward + next observation.

    Args:
        llm_output: Raw text generated by the LLM.
        env: Live IntelliCreditEnvironment instance.

    Returns:
        Dict with keys:
            step_advanced (bool),
            parse_result (dict from parse_llm_output),
            tool_result (str | None),
            env_response (IntelliCreditObservation | None),
            reward (float),
    """
    parse_result = parse_llm_output(llm_output)

    # ── Tool call branch (step does NOT advance) ─────────────────
    if parse_result["parse_type"] == "tool_call":
        tool_result = execute_tool(
            tool_name=parse_result["tool_name"],
            tool_args=parse_result["tool_args"] or {},
            env=env,
        )
        # Increment the env's world_model_confidence (dim 54 of memory features)
        if hasattr(env, "_tool_call_count"):
            env._tool_call_count = getattr(env, "_tool_call_count", 0) + 1
        return {
            "step_advanced": False,
            "parse_result": parse_result,
            "tool_result": tool_result,
            "env_response": None,
            "reward": 0.0,
        }

    # ── Final decision branch (step DOES advance) ────────────────
    from models import IntelliCreditAction  # local import to avoid circular deps

    action_int = parse_result["action"]
    reasoning = parse_result["reasoning"]

    action_obj = IntelliCreditAction(
        decision=action_int,
        reasoning=reasoning,
    )
    # Pass episode_id if tracked by env
    episode_id = getattr(env, "_state", None)
    episode_id_str = episode_id.episode_id if episode_id else None

    obs = env.step(action=action_obj, episode_id=episode_id_str)

    return {
        "step_advanced": True,
        "parse_result": parse_result,
        "tool_result": None,
        "env_response": obs,
        "reward": obs.reward,
    }


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        # (description, input_text, expected_parse_type, expected_action)
        (
            "Tool call — get_financial_report",
            "Let me check the details first. get_financial_report('TechM Solutions Ltd')",
            "tool_call", 2,
        ),
        (
            "Tool call — check_compliance_status",
            "I want to verify compliance. check_compliance_status(company_id='Bharat Manufacturing')",
            "tool_call", 2,
        ),
        (
            "Tool call — get_market_intelligence",
            "Macro context needed. get_market_intelligence('IT_Services')",
            "tool_call", 2,
        ),
        (
            "submit_decision — APPROVE",
            "Based on strong DSCR and clean compliance. submit_decision('APPROVE', 'DSCR 1.85x, no alerts, Tier A borrower')",
            "final_decision", 0,
        ),
        (
            "submit_decision — REJECT",
            "Multiple red flags found. submit_decision('REJECT', 'HR-01 triggered: DSCR < 1.0')",
            "final_decision", 2,
        ),
        (
            "submit_decision — CONDITIONAL",
            "Acceptable with conditions. submit_decision('CONDITIONAL', 'Require quarterly cash flow statements')",
            "final_decision", 1,
        ),
        (
            "Fallback keyword — APPROVE",
            "After careful review, APPROVE this loan application.",
            "fallback_keyword", 0,
        ),
        (
            "Fallback keyword — REJECT",
            "Risk is too high. I recommend REJECT.",
            "fallback_keyword", 2,
        ),
        (
            "Anti-hacking — last decision wins",
            "Initially APPROVE but after reviewing forensic data REJECT.",
            "fallback_keyword", 2,
        ),
        (
            "Default fallback — empty",
            "",
            "default_reject", 2,
        ),
        (
            "Default fallback — unrecognized",
            "This is very risky and I am unsure what to do here.",
            "default_reject", 2,
        ),
        (
            "No reasoning — penalty flag",
            "submit_decision('APPROVE', '')",
            "final_decision", 0,
        ),
    ]

    print("=" * 70)
    print("  IntelliCredit v2 — action_parser.py Self-Test")
    print("=" * 70)

    passed = 0
    failed = 0
    for desc, text, exp_type, exp_action in test_cases:
        result = parse_llm_output(text)
        ok_type = result["parse_type"] == exp_type
        ok_action = result["action"] == exp_action
        status = "✅ PASS" if (ok_type and ok_action) else "❌ FAIL"
        if ok_type and ok_action:
            passed += 1
        else:
            failed += 1
        print(f"\n{status}  {desc}")
        if not ok_type:
            print(f"       parse_type: got={result['parse_type']!r}  want={exp_type!r}")
        if not ok_action:
            print(f"       action:     got={result['action']}         want={exp_action}")
        print(f"       confidence: {result['parse_confidence']:.2f} | failure: {result['parse_failure']}")
        print(f"       reasoning:  {result['reasoning'][:80]!r}")

    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed / {len(test_cases)} total")
    print("=" * 70)
