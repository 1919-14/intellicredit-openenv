"""
FastAPI application for IntelliCredit-CreditAppraisal-v2.0 Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute a credit decision (or tool call)
    - GET /state: Get current environment state
    - GET /info: Get environment metadata
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

v2 changes:
    - 50-step episodes (was 12)
    - 55D observation (45D base + 10D memory features)
    - Multi-agent: BorrowerAgent + RegulatorAgent simulated by env
    - Regulator audits at steps 10/20/30/40/50
    - Tool call support via action_parser
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import IntelliCreditAction, IntelliCreditObservation
    from .intellicredit_env import IntelliCreditEnvironment, TASK_CONFIGS
except ImportError:
    from models import IntelliCreditAction, IntelliCreditObservation
    from server.intellicredit_env import IntelliCreditEnvironment, TASK_CONFIGS

_DESCRIPTION = """
## IntelliCredit Credit Appraisal Environment

**MSME Credit Decision RL Environment** — Meta × Hugging Face OpenEnv Hackathon.

---

### ⚠️ IMPORTANT: Session Workflow

Every HTTP call must carry the **same `episode_id`** to maintain session state:

**Step 1 — Reset:**
```json
POST /reset
{ "episode_id": "my-session-1", "seed": 42 }
```

**Step 2 — Step (repeat up to 50 times):**
```json
POST /step
{ "episode_id": "my-session-1", "action": { "decision": 0 }, "timeout_s": 30 }
```

> **v2**: Regulator audits auto-fire at steps 10, 20, 30, 40, 50 (±1 jitter).
> Rejected borrowers re-apply after a 3–5 step cooling period with improved surface profiles but unchanged hidden PD.

> `decision` values: **0 = APPROVE**, **1 = CONDITIONAL**, **2 = REJECT**

---

### Tasks (v2 — all 50 steps)
| Task | Steps | Description |
|------|-------|-------------|
| task1 | 50 | Easy — Clean profiles, macro shock at step ~22 |
| task2 | 50 | Medium — Forensic alerts present, shock at step ~22 |
| task3 | 50 | Hard — Macro shocks + missing data + repeat applicants |
| task4 | 50 | Expert — Hard-rule violations, tight CRAR windows |
| task5 | 50 | Master — Full constraints, cascading NPAs, 5 regulatory audits |

---

### v2 Highlights

**Observation Space: 55D**
| Dims | Sub-space | Description |
|------|-----------|-------------|
| 0–24 | Application features | 25 financial/forensic ratios per borrower |
| 25–34 | Portfolio state | NPA rate, CRAR, sector concentration, capital |
| 35–39 | Macro state | Stress index, shock flag, GDP, inflation, credit cycle |
| 40–44 | Alert state | Active forensic alert categories |
| 45–54 | Memory features (NEW) | Rolling NPA, approval rate, macro trend, audit risk… |

**6 Hard Rules (auto-REJECT + penalty)**
- HR-01: DSCR < 1.0x
- HR-02: CIBIL < 650
- HR-03: Active RED forensic alert
- HR-04: Sector under active macro shock
- HR-05: GST filing gap > 3 months
- HR-06: Adverse media score > 0.8

**Multi-Agent System**
- 🏦 **Credit Officer** (your agent) — approves/rejects applications
- 📋 **BorrowerAgent** — rejected borrowers reapply with improved surface metrics
- ⚖️ **RegulatorAgent** — audits portfolio at steps 10/20/30/40/50, applies penalties/shutdown

**Reward Components**
- Correctness bonus/penalty vs. optimal decision
- Hard rule penalty: −2.0 (APPROVE) / −1.0 (CONDITIONAL) on HR violation
- Delayed NPA penalty: fires 5–15 steps after approval
- Audit bonus (+2.0 clean) / penalty (−8 NPA, −15 CRAR, −8 sector)
- Survival bonus every 10 steps (if CRAR > 12.5%)
- Reasoning quality bonus (R1-style format reward)
"""

# Create the app with web interface and README integration
app = create_app(
    IntelliCreditEnvironment,
    IntelliCreditAction,
    IntelliCreditObservation,
    env_name="intellicredit_credit_appraisal",
    max_concurrent_envs=4,
)

# Inject rich description into FastAPI app metadata
app.title = "IntelliCredit-CreditAppraisal-v2"
app.description = _DESCRIPTION
app.version = "2.0.0"


from fastapi.responses import HTMLResponse, RedirectResponse

@app.get("/", response_class=HTMLResponse)
def read_root():
    """IntelliCredit project landing page."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <meta name="description" content="IntelliCredit-X: Multi-Agent RL environment for MSME credit appraisal using GRPO fine-tuned Mistral-7B."/>
  <title>IntelliCredit-X — AI Credit Officer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    :root{--bg:#040d12;--surface:#071a24;--surface2:#0c2333;--border:#0e3048;--accent:#0ea5e9;--accent2:#10b981;--accent3:#f59e0b;--text:#e2e8f0;--muted:#64748b;--dim:#1e3a4f;}
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    html{scroll-behavior:smooth;}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
    nav{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;padding:14px 32px;background:rgba(4,13,18,0.88);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);}
    .nav-brand{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1rem;color:var(--accent);text-decoration:none;}
    .nav-brand span{color:var(--text);}
    .nav-links{display:flex;gap:6px;}
    .nav-links a{padding:7px 14px;border-radius:8px;font-size:0.82rem;font-weight:500;color:var(--muted);text-decoration:none;transition:color 0.2s,background 0.2s;}
    .nav-links a:hover{color:var(--text);background:var(--dim);}
    .nav-cta{padding:8px 18px!important;background:var(--accent)!important;color:#fff!important;border-radius:8px!important;}
    .hero{position:relative;display:flex;flex-direction:column;align-items:center;text-align:center;padding:80px 24px 60px;overflow:hidden;}
    .glow1{position:absolute;width:600px;height:600px;background:radial-gradient(circle,rgba(14,165,233,0.11) 0%,transparent 70%);top:-100px;left:50%;transform:translateX(-50%);pointer-events:none;}
    .glow2{position:absolute;width:400px;height:400px;background:radial-gradient(circle,rgba(16,185,129,0.07) 0%,transparent 70%);bottom:0;right:10%;pointer-events:none;}
    .pill{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;background:rgba(14,165,233,0.1);border:1px solid rgba(14,165,233,0.3);border-radius:50px;font-size:0.78rem;font-weight:600;color:var(--accent);margin-bottom:24px;}
    .dot{width:6px;height:6px;border-radius:50%;background:var(--accent2);animation:pulse 2s infinite;}
    @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.35;}}
    h1{font-size:clamp(2.2rem,5vw,3.6rem);font-weight:900;line-height:1.1;letter-spacing:-0.03em;margin-bottom:20px;}
    .l1{display:block;color:var(--text);}
    .l2{display:block;background:linear-gradient(90deg,var(--accent) 0%,var(--accent2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
    .hero-sub{max-width:600px;margin:0 auto 36px;font-size:1.05rem;color:var(--muted);line-height:1.8;}
    .hero-sub strong{color:var(--text);}
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin-bottom:36px;}
    .btn-p{display:inline-flex;align-items:center;gap:8px;padding:13px 28px;background:var(--accent);color:#fff;border-radius:10px;text-decoration:none;font-weight:700;font-size:0.95rem;transition:transform 0.15s,opacity 0.15s;box-shadow:0 0 32px rgba(14,165,233,0.3);}
    .btn-p:hover{transform:translateY(-2px);opacity:0.9;}
    .btn-g{display:inline-flex;align-items:center;gap:8px;padding:13px 28px;background:transparent;border:1px solid var(--border);color:var(--text);border-radius:10px;text-decoration:none;font-weight:600;font-size:0.95rem;transition:border-color 0.2s,background 0.2s;}
    .btn-g:hover{border-color:var(--accent);background:var(--dim);}
    .badges{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;}
    .badges a img{height:20px;border-radius:4px;}
    .stats-section{padding:0 24px 48px;max-width:1000px;margin:0 auto;}
    .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--border);border-radius:16px;overflow:hidden;border:1px solid var(--border);}
    .sc{background:var(--surface);padding:28px 20px;text-align:center;transition:background 0.2s;}
    .sc:hover{background:var(--surface2);}
    .sn{font-size:2rem;font-weight:900;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
    .sl{font-size:0.73rem;color:var(--muted);margin-top:4px;font-weight:500;text-transform:uppercase;letter-spacing:0.04em;}
    section.w{max-width:1000px;margin:0 auto;padding:0 24px 64px;}
    .stag{display:inline-block;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--accent2);background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);padding:4px 10px;border-radius:50px;margin-bottom:12px;}
    .stitle{font-size:clamp(1.4rem,3vw,1.9rem);font-weight:800;margin-bottom:8px;letter-spacing:-0.02em;}
    .ssub{color:var(--muted);font-size:0.95rem;margin-bottom:32px;line-height:1.7;}
    .steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px;}
    .step{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:24px;transition:border-color 0.2s,transform 0.2s;}
    .step:hover{border-color:var(--accent);transform:translateY(-3px);}
    .stepn{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:var(--accent);opacity:0.25;line-height:1;margin-bottom:12px;}
    .stept{font-size:0.95rem;font-weight:700;margin-bottom:8px;}
    .stepd{font-size:0.82rem;color:var(--muted);line-height:1.6;}
    .agents{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;}
    .ac{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:24px;display:flex;gap:16px;transition:border-color 0.2s;}
    .ac:hover{border-color:var(--accent);}
    .ai{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;}
    .ai.b{background:rgba(14,165,233,0.15);}.ai.g{background:rgba(16,185,129,0.15);}.ai.a{background:rgba(245,158,11,0.15);}
    .an{font-weight:700;margin-bottom:4px;}.ar{font-size:0.78rem;color:var(--accent);font-weight:600;margin-bottom:8px;}.ad{font-size:0.82rem;color:var(--muted);line-height:1.5;}
    .rtable{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;}
    .rtable table{width:100%;border-collapse:collapse;font-size:0.84rem;}
    .rtable th{background:var(--surface2);padding:12px 16px;text-align:left;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);}
    .rtable td{padding:12px 16px;border-top:1px solid var(--border);}
    .rtable tr:hover td{background:var(--surface2);}
    .up{color:var(--accent2);font-weight:700;}
    .te{background:rgba(16,185,129,0.15);color:var(--accent2);padding:2px 8px;border-radius:50px;font-size:0.72rem;font-weight:700;}
    .tm{background:rgba(14,165,233,0.15);color:var(--accent);padding:2px 8px;border-radius:50px;font-size:0.72rem;font-weight:700;}
    .th{background:rgba(245,158,11,0.15);color:var(--accent3);padding:2px 8px;border-radius:50px;font-size:0.72rem;font-weight:700;}
    .cb{background:#020810;border:1px solid var(--border);border-radius:12px;overflow:hidden;}
    .ch{padding:10px 16px;background:var(--surface);border-bottom:1px solid var(--border);font-size:0.75rem;color:var(--muted);display:flex;align-items:center;gap:8px;}
    .dots{display:flex;gap:5px;}.dots span{width:11px;height:11px;border-radius:50%;}
    .d1{background:#ff5f56;}.d2{background:#ffbd2e;}.d3{background:#27c93f;}
    .cb pre{font-family:'JetBrains Mono',monospace;font-size:0.78rem;line-height:1.7;padding:20px;overflow-x:auto;color:#a8c7d4;}
    .kw{color:#7dd3fc;}.str{color:#86efac;}.cm{color:#374151;}.num{color:var(--accent3);}
    .resources{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}
    .rc{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:22px;text-decoration:none;color:inherit;display:flex;gap:14px;align-items:flex-start;transition:border-color 0.2s,transform 0.2s;}
    .rc:hover{border-color:var(--accent2);transform:translateY(-3px);}
    .ri{width:42px;height:42px;border-radius:10px;background:var(--dim);display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;}
    .rt{font-size:0.9rem;font-weight:700;margin-bottom:4px;}.rd{font-size:0.78rem;color:var(--muted);line-height:1.5;}
    .ra{margin-left:auto;color:var(--muted);font-size:0.9rem;flex-shrink:0;margin-top:2px;}
    footer{border-top:1px solid var(--border);padding:32px 24px;text-align:center;font-size:0.8rem;color:var(--muted);}
    footer a{color:var(--accent);text-decoration:none;}
    footer a:hover{text-decoration:underline;}

    .img-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden;margin-top:8px;}
    .img-label{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--border);background:var(--surface2);}
    .fig-tag{font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:50px;background:rgba(14,165,233,0.15);color:var(--accent);letter-spacing:0.05em;}
    .fig-title{font-size:0.85rem;font-weight:600;color:var(--muted);}
    .img-wrap{padding:16px;background:#020810;text-align:center;}
    .img-wrap img{max-width:100%;height:auto;border-radius:8px;display:block;margin:0 auto;}
    .img-caption{padding:20px;border-top:1px solid var(--border);}
    .caption-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;}
    .citem{display:flex;flex-direction:column;gap:4px;}
    .cnum{font-size:0.8rem;font-weight:700;}
    .cdesc{font-size:0.78rem;color:var(--muted);line-height:1.5;}
    @media(max-width:640px){nav{padding:12px 16px;}.nav-links{display:none;}.hero{padding:48px 16px 36px;}}
  </style>
</head>
<body>
<nav>
  <a class="nav-brand" href="/">🏦 <span>IntelliCredit-X</span></a>
  <div class="nav-links">
    <a href="https://huggingface.co/spaces/vssksn/intellicredit-openenv/blob/main/docs/blog.md" target="_blank">Blog</a>
    <a href="https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo" target="_blank">Model</a>
    <a href="https://github.com/1919-14/intellicredit-openenv" target="_blank">GitHub</a>
    <a href="/docs" class="nav-cta">API Docs ↗</a>
  </div>
</nav>
<div class="hero">
  <div class="glow1"></div><div class="glow2"></div>
  <div class="pill"><span class="dot"></span> Meta × Hugging Face OpenEnv Hackathon 2026</div>
  <h1><span class="l1">Teaching an LLM to Think</span><span class="l2">Like a Credit Officer</span></h1>
  <p class="hero-sub">
    <strong>IntelliCredit-X</strong> is a multi-agent RL environment where Mistral-7B learns to investigate fraud signals,
    manage a live loan portfolio across <strong>50-step episodes</strong>, and make RBI-compliant credit decisions —
    trained via <strong>2-stage GRPO</strong> directly on the live environment.
  </p>
  <div class="hero-actions">
    <a class="btn-p" href="/docs">⚡ Try the Live API</a>
    <a class="btn-g" href="https://huggingface.co/spaces/vssksn/intellicredit-openenv/blob/main/docs/blog.md" target="_blank">📖 Read the Blog</a>
    <a class="btn-g" href="https://github.com/1919-14/intellicredit-openenv" target="_blank">⭐ GitHub</a>
  </div>
  <div class="badges">
    <a href="https://huggingface.co/spaces/vssksn/intellicredit-openenv" target="_blank"><img src="https://img.shields.io/badge/🤗_Space-Live_Demo-0ea5e9?style=flat" alt="HF Space"/></a>
    <a href="https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo" target="_blank"><img src="https://img.shields.io/badge/🤗_Model-Mistral--7B_GRPO-10b981?style=flat" alt="Model"/></a>
    <a href="https://huggingface.co/datasets/vssksn/intellicredit-grpo-v2" target="_blank"><img src="https://img.shields.io/badge/🤗_Dataset-GRPO_v2-f59e0b?style=flat" alt="Dataset"/></a>
    <a href="https://github.com/1919-14/intellicredit-openenv" target="_blank"><img src="https://img.shields.io/badge/GitHub-MIT_License-475569?style=flat" alt="GitHub"/></a>
  </div>
</div>
<div class="stats-section">
  <div class="stats-grid">
    <div class="sc"><div class="sn">55D</div><div class="sl">Observation Space</div></div>
    <div class="sc"><div class="sn">50</div><div class="sl">Steps / Episode</div></div>
    <div class="sc"><div class="sn">3</div><div class="sl">Agents</div></div>
    <div class="sc"><div class="sn">10x</div><div class="sl">Reward Gain Task 3</div></div>
    <div class="sc"><div class="sn">-8.3%</div><div class="sl">NPA After GRPO</div></div>
    <div class="sc"><div class="sn">0</div><div class="sl">Regressions</div></div>
  </div>
</div>
<section class="w">
  <div class="stag">How It Works</div>
  <div class="stitle">Credit AI in 4 Steps</div>
  <p class="ssub">Each episode simulates a full credit committee lifecycle. The agent must gather evidence, reason through risk, and submit compliant decisions under real regulatory pressure.</p>
  <div class="steps">
    <div class="step"><div class="stepn">01</div><div class="stept">Receive Application</div><div class="stepd">Agent receives a 55D observation — financials, forensic alerts, portfolio state, macro conditions, and memory of past decisions.</div></div>
    <div class="step"><div class="stepn">02</div><div class="stept">Investigate with Tools</div><div class="stepd">Calls up to 4 tools per step: <code style="color:var(--accent);font-family:monospace">get_financial_report()</code>, <code style="color:var(--accent);font-family:monospace">check_compliance()</code>, <code style="color:var(--accent);font-family:monospace">get_market_intel()</code>.</div></div>
    <div class="step"><div class="stepn">03</div><div class="stept">Submit Decision</div><div class="stepd">Calls <code style="color:var(--accent2);font-family:monospace">submit_decision(action, reasoning)</code> with ≥50-char reasoning. 6 hard rules auto-override violations.</div></div>
    <div class="step"><div class="stepn">04</div><div class="stept">Face Consequences</div><div class="stepd">Loans mature T+10–30 steps later. Regulator audits fire at steps ~10/20/30/40/50. 3 failures = shutdown (−50 reward).</div></div>
  </div>
</section>
<section class="w">
  <div class="stag">Multi-Agent System</div>
  <div class="stitle">Three Agents, One Environment</div>
  <p class="ssub">The environment simulates the full credit ecosystem — not just individual decisions.</p>
  <div class="agents">
    <div class="ac"><div class="ai b">🏦</div><div><div class="an">Credit Officer (LLM)</div><div class="ar">Your Agent Under Training</div><div class="ad">Mistral-7B fine-tuned via GRPO. Receives 55D obs, calls investigation tools, submits APPROVE / CONDITIONAL / REJECT with written reasoning.</div></div></div>
    <div class="ac"><div class="ai g">📋</div><div><div class="an">Borrower Agent</div><div class="ar">Adversarial Pressure</div><div class="ad">Rejected borrowers reapply up to 3x with improved surface metrics but unchanged hidden PD — forcing the agent to learn true risk signals.</div></div></div>
    <div class="ac"><div class="ai a">⚖️</div><div><div class="an">Regulator Agent</div><div class="ar">Compliance Enforcer</div><div class="ad">Audits portfolio at steps ≈10/20/30/40/50 (±1 jitter). Checks NPA rate, CRAR, sector concentration. Episode shutdown on 3 consecutive fails.</div></div></div>
  </div>
</section>
<section class="w">
  <div class="stag">Benchmark Results</div>
  <div class="stitle">Before vs. After GRPO</div>
  <p class="ssub">Evaluated across 3 task difficulties. Zero regressions — every metric improved or held steady.</p>
  <div class="rtable"><table>
    <tr><th>Task</th><th>Difficulty</th><th>Metric</th><th>Base Mistral-7B</th><th>GRPO Model</th><th>Delta</th></tr>
    <tr><td>Task 1</td><td><span class="te">Easy</span></td><td>Accuracy</td><td>80.0%</td><td>86.7%</td><td class="up">+6.7% ✓</td></tr>
    <tr><td>Task 1</td><td><span class="te">Easy</span></td><td>Capital Utilization</td><td>40.0%</td><td>60.0%</td><td class="up">+20.0% ✓</td></tr>
    <tr><td>Task 2</td><td><span class="tm">Medium</span></td><td>Total Reward</td><td>10.305</td><td>10.584</td><td class="up">+0.279 ✓</td></tr>
    <tr><td>Task 3</td><td><span class="th">Hard</span></td><td>Total Reward</td><td>0.215</td><td>2.491</td><td class="up">+10x ✓</td></tr>
    <tr><td>Task 3</td><td><span class="th">Hard</span></td><td>NPA Rate</td><td>16.7%</td><td>8.3%</td><td class="up">-8.3% ✓</td></tr>
  </table></div>
</section>

<section class="w">
  <div class="stag">Training Curves</div>
  <div class="stitle">What the Training Curves Tell Us</div>
  <p class="ssub">
    Four panels reveal the full story of what the model learned and when — across three curriculum stages (dashed lines mark transitions).
    <strong style="color:var(--text)">Mean reward climbs from −2.0 to +1.0</strong>, format compliance rises from 0% to 65%,
    and KL divergence stays safely below 0.12, confirming the model changed without forgetting language capabilities.
  </p>
  <div class="img-card">
    <div class="img-label">
      <span class="fig-tag">Figure 1</span>
      <span class="fig-title">GRPO v2 Training Curves — 3-Stage Curriculum</span>
    </div>
    <div class="img-wrap">
      <img src="https://github.com/user-attachments/assets/d225eb30-db76-4edb-bbc2-b429c6222095"
           alt="IntelliCredit GRPO Training Curves" loading="lazy"/>
    </div>
    <div class="img-caption">
      <div class="caption-grid">
        <div class="citem"><span class="cnum" style="color:#f87171">GRPO Loss</span><span class="cdesc">Starts near zero, climbs to 0.02–0.05 — healthy policy divergence from reference.</span></div>
        <div class="citem"><span class="cnum" style="color:#60a5fa">Mean Reward</span><span class="cdesc">−2.0 → 0 at Stage 1 end → stable +0.5–1.0. Stage 3 dip then re-stabilises.</span></div>
        <div class="citem"><span class="cnum" style="color:#c084fc">KL Divergence</span><span class="cdesc">Grows 0→0.08, stays below 0.12 threshold — genuine learning, no catastrophic forgetting.</span></div>
        <div class="citem"><span class="cnum" style="color:#2dd4bf">submit_pct</span><span class="cdesc">Format compliance: 0% → 40–65%. The model acquired the vocabulary of the task.</span></div>
      </div>
    </div>
  </div>
</section>

<section class="w">
  <div class="stag">Evaluation</div>
  <div class="stitle">Before vs. After GRPO — Full Comparison</div>
  <p class="ssub">
    Per-task, per-metric comparison of base Mistral-7B (blue) vs. GRPO-trained IntelliCredit model (green).
    <strong style="color:var(--accent2)">Zero regressions across all 24 metric-task combinations.</strong>
    The hardest task (Task 3) shows the most dramatic improvement — NPA rate cut in half, total reward up 10×.
  </p>
  <div class="img-card">
    <div class="img-label">
      <span class="fig-tag">Figure 2</span>
      <span class="fig-title">Base Mistral-7B vs. GRPO IntelliCredit — All Tasks</span>
    </div>
    <div class="img-wrap">
      <img src="https://github.com/user-attachments/assets/53dfddff-d17c-4b63-8d22-a763f25c2bd7"
           alt="IntelliCredit GRPO Results Comparison" loading="lazy"/>
    </div>
    <div class="img-caption">
      <div class="caption-grid">
        <div class="citem"><span class="cnum" style="color:var(--accent2)">Task 1 (Easy)</span><span class="cdesc">Accuracy +6.7%, capital utilization +20%. The GRPO model deploys more capital into correctly identified safe loans.</span></div>
        <div class="citem"><span class="cnum" style="color:var(--accent)">Task 2 (Medium)</span><span class="cdesc">Both models hit perfect Task Score (1.000). GRPO squeezes +0.28 extra reward from better capital efficiency.</span></div>
        <div class="citem"><span class="cnum" style="color:var(--accent3)">Task 3 (Hard)</span><span class="cdesc">Total reward 0.215 → 2.491 (+10×). NPA 16.7% → 8.3% (halved). True portfolio-level risk management learned.</span></div>
        <div class="citem"><span class="cnum" style="color:var(--text)">Key Insight</span><span class="cdesc">Model learned that surface improvement + behavioural red flags = escalating risk. It calls tools; base model doesn't.</span></div>
      </div>
    </div>
  </div>
</section>

<section class="w">
  <div class="stag">Quick Start</div>
  <div class="stitle">Start an Episode in 2 Calls</div>
  <p class="ssub">The environment is live and accepts HTTP from any client — no install required.</p>
  <div class="cb">
    <div class="ch"><div class="dots"><span class="d1"></span><span class="d2"></span><span class="d3"></span></div>bash — curl</div>
    <pre><span class="cm"># Step 1: Reset (start a new episode)</span>
<span class="kw">curl</span> -X POST https://vssksn-intellicredit-openenv.hf.space/reset \
  -H <span class="str">"Content-Type: application/json"</span> \
  -d <span class="str">'{"episode_id":"demo-1","seed":42,"task_id":"task3"}'</span>

<span class="cm"># Step 2: Submit a decision  (0=APPROVE 1=CONDITIONAL 2=REJECT)</span>
<span class="kw">curl</span> -X POST https://vssksn-intellicredit-openenv.hf.space/step \
  -H <span class="str">"Content-Type: application/json"</span> \
  -d <span class="str">'{"episode_id":"demo-1","action":{"decision":2}}'</span></pre>
  </div>
</section>
<section class="w">
  <div class="stag">Resources</div>
  <div class="stitle">Everything Open Source</div>
  <p class="ssub">All artefacts published on Hugging Face and GitHub under MIT License.</p>
  <div class="resources">
    <a class="rc" href="https://huggingface.co/spaces/vssksn/intellicredit-openenv/blob/main/docs/blog.md" target="_blank"><div class="ri">📝</div><div><div class="rt">Technical Blog</div><div class="rd">Architecture, 2-stage GRPO, training curves, full results.</div></div><span class="ra">↗</span></a>
    <a class="rc" href="https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo" target="_blank"><div class="ri">🤖</div><div><div class="rt">Fine-Tuned Model</div><div class="rd">Mistral-7B post-trained on live environment via online GRPO.</div></div><span class="ra">↗</span></a>
    <a class="rc" href="https://huggingface.co/datasets/vssksn/intellicredit-grpo-v2" target="_blank"><div class="ri">📊</div><div><div class="rt">Training Dataset</div><div class="rd">2,000 GRPO prompts across 5 task levels — intellicredit-grpo-v2.</div></div><span class="ra">↗</span></a>
    <a class="rc" href="https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing" target="_blank"><div class="ri">🚀</div><div><div class="rt">Stage 1 — Offline GRPO</div><div class="rd">Mistral-7B + Unsloth, A100, ~45 min. Pre-train on 2,000 prompts.</div></div><span class="ra">↗</span></a>
    <a class="rc" href="https://colab.research.google.com/github/1919-14/intellicredit-openenv/blob/main/training/colab_online_grpo.ipynb" target="_blank"><div class="ri">🌍</div><div><div class="rt">Stage 2 — Online GRPO</div><div class="rd">Post-train on this live env. Real rewards from /step endpoint.</div></div><span class="ra">↗</span></a>
    <a class="rc" href="https://github.com/1919-14/intellicredit-openenv" target="_blank"><div class="ri">💻</div><div><div class="rt">GitHub Repository</div><div class="rd">Full source — env, training scripts, evaluation. MIT License.</div></div><span class="ra">↗</span></a>
    <a class="rc" href="/docs" target="_blank"><div class="ri">⚡</div><div><div class="rt">Swagger UI</div><div class="rd">Interactive API — run /reset and /step right in the browser.</div></div><span class="ra">↗</span></a>
    <a class="rc" href="/info" target="_blank"><div class="ri">🔍</div><div><div class="rt">Environment Info JSON</div><div class="rd">Full metadata — observation dims, action space, tasks, constraints.</div></div><span class="ra">↗</span></a>
  </div>
</section>
<footer>
  Built for <strong>Meta x Hugging Face OpenEnv Hackathon 2026</strong> by
  <a href="https://huggingface.co/vssksn" target="_blank">V S S K Sai Narayana</a> &amp;
  <a href="https://github.com/1919-14" target="_blank">Sujeet Jaiswal</a>
  &nbsp;&middot;&nbsp;<a href="/docs">API Docs</a>
  &nbsp;&middot;&nbsp;<a href="/info">JSON Info</a>
  &nbsp;&middot;&nbsp;<a href="https://github.com/1919-14/intellicredit-openenv/blob/main/LICENSE">MIT License</a>
</footer>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/health")
def health_check():
    """Health check for Docker/Kubernetes."""
    return {"status": "healthy"}

# GAP 15: /info endpoint returning environment metadata
@app.get("/info")
def get_info():
    """Return environment metadata for validators and documentation."""
    return {
        "name": "IntelliCredit-CreditAppraisal-v2",
        "version": "2.0.0",
        "observation_dims": 55,
        "observation_sub_spaces": {
            "application_features": 25,   # dims 0-24  (unchanged from v1)
            "portfolio_state": 10,         # dims 25-34 (unchanged from v1)
            "macro_state": 5,              # dims 35-39 (unchanged from v1)
            "alert_state": 5,              # dims 40-44 (unchanged from v1)
            "memory_features": 10,         # dims 45-54 (NEW in v2)
        },
        "memory_feature_labels": {
            "45": "rolling_npa_rate_10step",
            "46": "approval_rate_recent",
            "47": "sector_max_concentration",
            "48": "macro_stress_trend",
            "49": "borrower_persistence_score",
            "50": "audit_risk_score",
            "51": "capital_buffer_ratio",
            "52": "recent_reflection_count",
            "53": "episode_progress",
            "54": "world_model_confidence",
        },
        "action_space": 3,
        "action_labels": {
            "0": "APPROVE",
            "1": "CONDITIONAL_APPROVE",
            "2": "REJECT",
        },
        "tasks": {
            tid: {
                "num_steps": cfg["num_steps"],
                "description": cfg["description"],
            }
            for tid, cfg in TASK_CONFIGS.items()
        },
        "max_steps_per_episode": 50,
        "audit_steps": [10, 20, 30, 40, 50],
        "constraints": {
            "max_npa_rate": 0.05,
            "min_crar": 0.125,
            "max_sector_concentration": 0.30,
            "max_single_borrower_pct": 0.15,
        },
        "hard_rules": [
            "HR-01: DSCR < 1.0",
            "HR-02: Director disqualification (DIN < 0.1)",
            "HR-03: RED forensic alert present",
            "HR-04: Cheque bounce rate > 25%",
            "HR-05: GST compliance < 40%",
            "HR-06: Severe adverse media (sentiment > 0.80)",
        ],
        "multi_agent": {
            "borrower_agent": "Programmatic — reapplies up to 3 times with improved surface profile",
            "regulator_agent": "Programmatic — audits at steps 10/20/30/40/50, can shut down episode",
            "credit_officer": "LLM under training — receives 55D obs + text prompt, calls tools or submits decision",
        },
        "reward_range": [-20.0, 15.0],
        "framework": "OpenEnv (Meta × Hugging Face × PyTorch)",
    }


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)  # main() callable for validation
