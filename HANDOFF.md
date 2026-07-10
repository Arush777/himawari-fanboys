# HANDOFF — everything tried, results, and how to push past 0.87

*Last updated 2026-07-11. Companion to the critical note at the top of the README.
Written so anyone on the team can resume without any chat context.*

## Current submitted state

| What | Where |
| --- | --- |
| Graded code | this repo, `main` @ `54bcc43` — specialist + best-of-2 + frame-grounded selection ("the 0.87 pipeline") |
| Graded image | `ghcr.io/novicecoderinfinity/silver-octo-guacamole:latest`, digest `94109ac…`, public, key baked in, harness-verified |
| Form materials | `submission/SUBMISSION.md` (short + long descriptions, copy-paste ready) and `submission/presentation.pptx` |
| Backup of pre-revert main | branch `backup/portfolio-runtime-0.86` |
| Next candidate pipeline | branch `candidate/facts-critique-v2` (see below) |

## Leaderboard history — every submission and its score

| Score | Pipeline | Verdict |
| ---: | --- | --- |
| 0.80 | Opus, describe-then-style | Opus wrote *worse* styled captions than Sonnet here |
| 0.83 | Sonnet, describe-then-style | previous champion |
| **0.87** | Sonnet, per-style specialists + best-of-2 + frame-grounded selection | **current champion, resubmitted** |
| 0.86 | portfolio runtime (`811a3f0`) | reverted; preserved on the backup branch |

Also A/B-tested for generation, none beat Sonnet: Haiku 4.5, Fireworks Qwen3p7-plus,
Kimi hybrid, Gemini video-native probes.

## Things tried since 0.87, with results

### 1. Facts-grounding + critique/repair stack (branch `candidate/facts-critique-v2`)

On top of the 0.87 pipeline: the describe call also extracts 5–10 checkable **facts**
(ordered by visual prominence); specialists must anchor on them; after selection a
**critique** call scores each caption on the judge's own rubric and a **repair** call
rewrites anything below threshold 5 — with **verify-before-accept** (the rewrite is
re-critiqued and kept only if it scores ≥ the original, so repair can't make things
worse). Plus a **deterministic regex guard** that forces `humorous_non_tech` into repair
when it leaks tech vocabulary (an LLM critique had passed a caption that the leaderboard
judge scored tone_fit 2 for exactly this), and frames at 1024px instead of 768px.

**Results:**
- Absolute LLM-judge score (Sonnet judge, 3 clips): **0.933 vs 0.908** for the 0.87
  image's output — looks like a win, but…
- judge2 pairwise duel vs the 0.87 pipeline (12 GCS clips, 3-judge ensemble, both
  orders, probe-validated, 0 failed calls): **11 wins / 11 losses / 26 ties — a dead
  heat.** Does not clear the 0.55 ship gate. Verdict: hold until it wins on a bigger
  task set. Absolute self-scores and pairwise duels genuinely disagree at this margin.

### 2. Portfolio runtime → 0.86 (real leaderboard data point)

It passed a local A/B gate but lost a point on the hidden leaderboard. Combined with
the dead-heat duel above, the lesson is the same from two directions: **near 0.87 our
local evals cannot reliably resolve differences smaller than ~±0.01–0.02.** Don't spend
a submission on anything that doesn't win locally by a clear margin.

### 3. What the judged duels actually punish (read this before writing prompts)

Mining the decisive judge2 verdicts and every low judged score:

- **Unverifiable flourishes lose duels.** "Dew-covered grass", "early evening",
  background-tree colours, object counts — judges mark these `not_visible`/`unverifiable`
  and the caption loses to a plainer, fully-checkable one. Formal style is a solved
  problem (5/5 everywhere); *all* losses are humour/sarcasm captions over-claiming
  peripheral details.
- **Hard style constraints need hard code.** The `humorous_non_tech` "no technology"
  ban was violated once on the leaderboard (tone_fit 2) and once more in later testing
  with a context-dependent leak ("that mouse hasn't moved once" — the computer mouse).
  LLM self-critique missed both. A regex guard catches the first class; the second
  needs a context-aware rule (treat ambiguous words as tech when the scene contains
  computers) or a blanket "never mention any device" instruction for that style.
- **Judge noise is real.** judge2's own repeatability probe failed once on a 2-point
  median wobble and passed on rerun. Single-run absolute scores on 3 clips move ±0.08.

## Operational gotchas (each of these cost us hours)

1. **Never trust shell-exported API keys.** A stale `ANTHROPIC_API_KEY` in the
   environment silently shadowed the valid `.env` key (`load_dotenv` doesn't override
   by default) and produced misleading 401s. `config.py` on the candidate branch uses
   `load_dotenv(override=True)`; keep that behaviour.
2. **A dead/shadowed key during judging turns every duel into a tie** — always check
   the judge output for zero failed calls before believing a win rate.
3. **GHCR permissions:** repo collaborators cannot create packages under the owner's
   namespace — that's why the image lives under `novicecoderinfinity` for now. The
   package must be **public** or the harness can't pull it. The baked key is therefore
   exposed: **rotate it after grading.**
4. **Apple-silicon Macs need `--platform linux/amd64`** for both build and local runs
   of the graded image; harness-style verification = fresh pull, **no** `-e` env vars,
   mounted `/input` + `/output`, expect all captions non-empty.

## Recommended path to beat 0.87 (in order)

1. **Grow the eval set first.** 12 clips / 48 comparisons cannot detect a +0.01–0.02
   improvement. Extend `eval_input/` to 30+ varied public clips before judging anything.
2. **Attack verifiability, not wit.** The single biggest observed loss category.
   Cheapest version: add "no claims about background colours, counts, or fleeting
   details" to the specialist + selection prompts of the 0.87 pipeline and duel that
   alone — it's most of the candidate branch's value at a fraction of the complexity
   and latency.
3. **Gate every ship on the judge2 protocol:** probes pass → duel on the full set →
   ship only at ≥ 0.55 win rate with zero failed judge calls. The 0.86 submission is
   what skipping the margin costs.
4. **Merge `candidate/facts-critique-v2` only if it clears that bar** on the bigger
   set. It's already reliability-hardened (every stage falls back gracefully) and adds
   ~2 calls/clip, still comfortably inside the 10-minute budget.
5. **When a winner ships:** rebuild the image from `main` with the key via
   `--build-arg`, push (ideally to `ghcr.io/arush777/video-captioner:latest` so it
   lives with this repo), verify harness-style, make it public, resubmit, and update
   `submission/` if the described pipeline changed.
