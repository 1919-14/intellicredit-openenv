# 🏦 IntelliCredit-X: The Story of Teaching an AI to Catch What Humans Miss

*By **V S S K Sai Narayana** & **Sujeet Jaiswal** — Meta × Hugging Face OpenEnv Hackathon 2026*

> 📖 **This is the quick-read mini-blog.** For the full 5,000-word deep technical writeup, see [the complete blog](./blog.md).

---

## The Loan That Should Never Have Been Approved

A credit manager at a mid-sized Indian NBFC sits across the table from an MSME founder who runs a textile unit in Surat. Revenue growing at 22% CAGR. GST filings look clean. Collateral is offered. Everything checks out on the surface. The loan gets approved.

**Eight months later, that loan becomes a Non-Performing Asset.**

A forensic audit reveals the ugly truth: the revenue growth was inflated through circular trading between three related-party entities. The GST numbers — when cross-checked against actual bank statements — showed a 23% mismatch. The director had two undisclosed NCLT cases. *Every signal was there, buried in data that would have taken three days to manually cross-reference.*

This story plays out across India **thousands of times a day.** Over 100,000 MSME loan applications are processed daily. A senior credit officer reviews just 16 per day. That is 0.016% coverage by human experts. The rest? Junior staff, rule engines, and gut feeling. Default rates run at 12–15% annually as a direct consequence.

**We asked ourselves a simple question:** *What if an AI could learn to think like the best credit officers — not just classify applications, but actually investigate them?*

---

## What We Built

**IntelliCredit-X** is not a loan classifier. It is a full training ground — an OpenEnv-compliant reinforcement learning environment — where an LLM learns to act as a **Senior Credit Officer** at an Indian lending institution.

Here is what makes it genuinely different:

🔍 **The AI investigates before deciding.** Just like a real credit officer pulls records from multiple systems, our agent calls investigation tools — financial reports, compliance databases, market intelligence — before making a decision. It does not look at a spreadsheet and output APPROVE or REJECT. It *reasons*.

⏳ **Consequences are delayed.** A loan approved at step 5 may default at step 30. The agent must learn to think ahead — not just optimize for the next application.

🎭 **Other agents fight back.** A rejected borrower? They do not disappear. They return with better-looking numbers while the underlying risk is unchanged or worse. A regulator audits the portfolio every 10 steps and *shuts you down* if you fail three times.

⚖️ **The rules are non-negotiable.** Six hard RBI mandates are enforced automatically. Violate them and you get penalized regardless of what the model wants to do. These are the same rules real Indian banks must follow.

---

## How We Trained the AI

We used a **2-stage GRPO (Group Relative Policy Optimization)** approach to fine-tune **Mistral-7B**:

**Stage 1 — Classroom Training (Offline GRPO):** We created a curated dataset of 2,000 credit scenarios and trained the model to recognize hard rules, detect forensic red flags, and use the proper tool-call format. ~45 minutes on an A100. Think of it as sending the AI to credit officer school.

**Stage 2 — On-the-Job Training (Online GRPO):** We plugged the model directly into the live IntelliCredit environment. Every single reward signal came from the *actual* environment — real 50-step episodes, real adversarial borrowers, real regulatory audits. This is where the AI learned to truly operate under pressure.

The model learned through trial and error: sample multiple possible decisions, see which ones get rewarded, shift behavior toward the winners.

---

## The Proof: It Actually Works

### Training Curves — The AI Learning in Real-Time

<img width="1600" height="1142" alt="IntelliCredit GRPO Training Curves" src="https://github.com/user-attachments/assets/dd6b90ad-60d5-432e-9e1b-47cbcdab183e" />

*Watch the blue "Mean Reward" line: it starts at −2.0 (the AI is flailing, violating every rule) and climbs past zero into positive territory. The teal "submit_pct" line shows the AI learning to speak the right language — from 0% format compliance to 40–65%. This is an AI acquiring a new skill in real time.*

### Results — Base Model vs. GRPO-Trained Model

<img width="1600" height="884" alt="IntelliCredit Base vs GRPO Comparison" src="https://github.com/user-attachments/assets/1e4e4a05-9327-45c4-aed7-239ab8d74bbc" />

*Every single green bar is taller than its blue counterpart. Zero regressions across all 24 metric-task combinations.*

Here are the numbers that matter most:

| What Changed | Before (Base Mistral-7B) | After (GRPO-Trained) | Impact |
|---|---|---|---|
| **NPA Rate on Hard Task** | 16.7% | **8.3%** | **Halved! ✅** |
| **Total Reward on Hard Task** | 0.215 | **2.491** | **10× improvement! ✅** |
| **Accuracy on Easy Task** | 80.0% | **86.7%** | **+6.7% ✅** |
| **Capital Utilization** | 40.0% | **60.0%** | **+20% ✅** (deploying more capital into good loans) |

The most telling result? On Task 3 — the hardest scenario with macro shocks, missing data, and repeat adversarial applicants — the base model barely broke even (total reward: 0.215). The GRPO-trained model scored **2.491**. That is a 10× improvement. The AI learned to navigate the storm.

---

## The Moment It Clicked

Here is a real trace that shows the difference training makes:

**The application:** A real estate company, ₹8.5Cr loan. DSCR just above threshold. Circular trading alert active. **This is the borrower's third attempt** — they were rejected twice before and came back with better-looking numbers.

**What the base Mistral-7B did:** Looked at the surface numbers, saw they looked reasonable, and approved. That loan defaulted. Reward: **−2.4**.

**What the GRPO-trained model did:**
1. Called `get_financial_report()` → Found 41% related-party transactions and a "Going Concern" audit qualification
2. Called `check_compliance_status()` → Found 2 new NCLT cases added since the last application
3. Called `get_market_intelligence()` → Sector stress at 0.68 with active RBI advisory
4. **Rejected** with detailed reasoning citing the manipulation signal, the litigation, and the sector risk

Reward: **+1.8**. The trained model did not just make a better decision — it *investigated like a credit officer would* before making that decision.

---

## Try It Yourself

The entire system is open source and live right now:

| Resource | Link |
|---|---|
| 🤗 **Live Environment** | [huggingface.co/spaces/vssksn/intellicredit-openenv](https://huggingface.co/spaces/vssksn/intellicredit-openenv) |
| 🤗 **Trained Model** | [vssksn/intellicredit-mistral-7b-grpo](https://huggingface.co/vssksn/intellicredit-mistral-7b-grpo) |
| 🤗 **Training Dataset** | [vssksn/intellicredit-grpo-v2](https://huggingface.co/datasets/vssksn/intellicredit-grpo-v2) |
| 📓 **Training Notebook** | [Open in Colab](https://colab.research.google.com/drive/1HhVu1JezKoT32zfHIEfAFersxRrwZSYu?usp=sharing) |
| 💻 **GitHub** | [1919-14/intellicredit-openenv](https://github.com/1919-14/intellicredit-openenv) |
| 📖 **Full Technical Blog** | [docs/blog.md](./blog.md) |
| 📖 **API Docs** | [Swagger UI](https://vssksn-intellicredit-openenv.hf.space/docs) |

---

## Why This Matters

The credit officer in Surat is not going away. But right now, they are drowning — 16 applications a day, 30 minutes each, fragmented data across disconnected systems, and the constant pressure to approve. 

An AI that has learned to investigate, reason, and comply — trained on thousands of simulated decisions with real regulatory consequences — does not replace that officer. **It gives them superpowers.** It catches the circular trading they would have missed at 4 PM on a Friday. It remembers the borrower who was rejected twice before. It keeps the portfolio healthy when a macro shock hits and every instinct says "keep lending."

That is why we built IntelliCredit-X. And that is why every line of code is open source.

---

*Built for the Meta × Hugging Face OpenEnv Hackathon 2026*  
*V S S K Sai Narayana & Sujeet Jaiswal*  
*MIT License — [GitHub](https://github.com/1919-14/intellicredit-openenv)*
