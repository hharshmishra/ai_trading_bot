# v3.8 — Edge-First Emission + SMS + Learning Repair II
(plan: ~/.claude/plans/tender-wondering-garden.md · evidence: docs/v38-emission-redesign.md)

STATUS 2026-08-14: implementation COMPLETE, 444 tests green.

- [x] 21-day audit (31,728 rows; grading live-verified 20/20 via ccxt)
- [x] Phase 1 — meta train/serve skew + `emitted` leakage fix (candidate_trigger/
      candidate_action on every row); TB grades the SENT direction; trust
      advantage baseline + nightly decay; news lowercase; reset clears marker
- [x] Phase 2 — Smart Money Structure port (sms / sms_bos / sms_choch +
      trend matrix metrics), full-series==window parity locked
- [x] Phase 3 — SMS backtest, pre-registered rule → **SMS_EMIT=false (shadow)**
      (39,368 events landed base+0.2–1.5pts; report: logs/backtest/sms-v38/)
- [x] Phase 4 — evidence ledger + should_emit_signal_v3 + seed script +
      preflight check + message context (track record / model p / structure)
- [x] Phase 5 — docs/v38-emission-redesign.md, README, .env.example, lessons
- [x] Phase 6 — full suite green, pushed to origin/main

## Review

Root causes shipped against: (1) the gate stack suppressed its only proven
signal — NWE crossings 40–50% outside calm vol — down to ~1.5 emissions/day
via flags decided on n≤17; (2) meta_p was corrupted by train/serve skew +
the `emitted` leakage feature; (3) emitted rows were graded on the brain
final instead of the direction actually sent; (4) symmetric trust rewards
were negative-sum at real base rates (derivatives pinned at −10).

v3.8 = evidence ledger (rate ≥0.38 ∧ Wilson LB ≥0.30, n≥25, per
source|tf|regime|vol cohort; probation for new sources; suppressed candidates
still graded so cohorts self-heal), skew-free meta features, sent-direction
grading, advantage-baseline trust with nightly decay, SMS in shadow.

USER-side next (docs/v38-emission-redesign.md §Deploy): manual Oracle deploy
(archive → pull → --policies-only reset → seed_ledger → .env block →
preflight → systemd unit install), rotate the leaked Bot D token, day-20
re-audit ~Sep 3 (protocol in the doc).
