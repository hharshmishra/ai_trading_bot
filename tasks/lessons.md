# Lessons (self-improvement log)

## 2026-07-03 — accuracy upgrade v2

1. **Unit tests green ≠ integrated.** All agent/brain/gate unit tests passed while
   `cycle.py` still passed the legacy 3-agent tuple — the derivatives voter was
   never called in the real path. An offline forced cycle (flags on, LLM mocked,
   broadcast captured, tmp DB) exposed it in minutes. Rule: run the forced-cycle
   integration check after wiring any new component into the cycle.

2. **Never bare-`pip install` into this venv.** `pip install scikit-learn` pulled
   numpy 2.4.6 and broke the pandas 2.0.3 / vendored pandas_ta ABI. Always install
   with explicit pins (`scikit-learn==1.3.2 scipy==1.11.4`) and re-verify
   `numpy==1.25.0` + `import pandas_ta` after any dependency change. preflight
   now asserts this.

3. **Shared ccxt sync clients are not thread-safe.** The cycle fans out via
   asyncio.to_thread; concurrent first-call `load_markets()` on one client races
   inside ccxt with the error swallowed downstream. Pattern: double-checked
   locking for lazy init + a fetch lock around call bursts.

4. **Backtest cache must handle range EXTENSION both directions.** The CSV cache
   originally only topped up forward; asking for an earlier --start silently
   truncated history to the cached window. Backfill-and-prepend fixed it.

5. **Don't trust `pkill` exit alone** — verify with `pgrep` after; the first kill
   left a worker alive.

6. **Check the report's own meta before drawing conclusions.** The first "gate-v2"
   A/B run silently ran the v1 gate — the env flag was set but `--gate v2` was
   omitted from the CLI. The report header (`gate: v1`) exposed it. Every run's
   meta must be read back against the intended configuration. (The accident was
   kept as `trigger-ablation/` — it cleanly isolates the trigger-set effect.)

7. **Lesson-1 pattern recurred in NewsAgent.** Unit tests exercised `NewsRL()`
   (10-dim default) while the live agent ctor pinned `NewsRL(n_features=5)` —
   the 5→10 migration machinery was dead in production and `_pad()` silently
   truncated the event dims. Guard: a wiring test now asserts the AGENT's
   bandit width equals `N_FEATURES`. Rule: when a module constant drives a
   shape/roster, cover the live ctor path, not just the class.

8. **Tests mutated real logs/ artifacts.** Any test constructing an agent
   without monkeypatching its POLICY_PATH rewrote logs/*.json (twice caused
   dirty-tree noise; post-migration it would have rewritten the live news
   policy). Durable fix: `tests/conftest.py` autouse fixture redirects every
   artifact path (policies + nightly outputs) to tmp_path. Rule: the moment an
   artifact path is added to config, add it to that fixture.

9. **A flag-forked function must be selected in ONE place.** Auto grading chose
   `reward_for_v2` when TB_GRADING_ENABLED, but the manual-feedback and
   correction paths hardcoded legacy `reward_for` at five call sites — every
   human grade computed wrong deltas (confirming a timeout auto-grade swung
   the policy by -2.5 instead of 0). Fix: `active_reward_fn()` is the only
   place that picks the map; every caller goes through it. Rule: when adding
   a v2 variant behind a flag, grep every call site of the v1 function and
   route them through one selector. ADDENDUM (v3.2): the brain layer was a
   missed call site — _apply_feedback_to_brain had its own inlined ±1/−4.
   The grep must cover REIMPLEMENTATIONS of the map, not just calls to it.

10. **"Offline" tests must be MADE offline, not assumed offline.** The suite
    loaded the real .env (load_dotenv at agent import), so MACRO flags were
    live and phase1's equivalence test silently fetched REAL CoinGecko
    dominance + reused the REAL bitreinforcex.db via the get_store()
    singleton — green in the morning, red in the afternoon because the actual
    market moved. Fix in conftest: per-test _NoNet transport for the macro
    modules, TTL-cache resets, a tmp-store singleton, and an llm_client
    reset. Rule: every module-level cache, singleton, or env-driven flag
    needs an entry in the hermeticity fixture the day it is born.

11. **Fix at the chokepoint, not the symptom.** load_dotenv() was patched into
    telegram_app only; 5 script entry points stayed env-broken (backtests
    silently ignored UNIVERSE_ADD + gate flags). config.py is the module every
    entry point imports before reading a flag — loading .env THERE (guarded by
    BITREINFORCEX_NO_DOTENV for tests) fixed all of them at once. When a bug is
    "this file didn't load env/config", ask what shared thing every caller
    already touches and put the fix there.

12. **Amount-fingerprint keys must match the comparison, not a reconstruction.**
    USDT order matching compared full-amount millis, but uniqueness was checked
    on `fingerprint % 1 * 1000` — a *reconstruction* that folded half-integer
    base prices (2.5/3.5/4.5) into the wrong namespace, minting duplicate
    amounts. Key any dedup/uniqueness check on the SAME quantity the downstream
    match uses (here: round(total*1000)), never on a derived slice of it.

13. **Sync HTTP in an async handler blocks the whole loop.** The membership
    payment code called requests directly inside PTB async handlers/loops that
    share the single event loop with signal broadcasting — a slow rail froze
    everything for up to the timeout. Every blocking call in an async context
    must go through asyncio.to_thread (the rest of the codebase already did).

14. **RL step size must be anchored to a fixed reference point, not a moving
    mean.** v3.2 normalized indicator steps at |r|=2, but wins are always +1 and
    the worst loss is -4, so wins came out ~5x weaker than wrongs and weights
    ratcheted down. Anchor the win step to the +1 case and the loss step to the
    -4 case (both = historical constants); scale only the in-between losses.

15. **Normalize trust by absolute quality, never by distance from the worst
    agent.** v3.6's shift-normalize (`w = (score - min)/sum`) crowned whoever
    was least-punished — in prod that was the DISABLED sentiment voter (score
    frozen at +1.5 because its conf was always 0), while indicator fell to
    weight 2.3e-06 in 24h. Corollaries: (a) an agent that never plays must
    never hold weight mass — exclude inactive agents from decide-time
    normalization; (b) unbounded score accumulators make recovery impossible —
    clamp them; (c) an asymmetric reward map (−4 wrong / +1 correct) applied
    to TRUST bankrupts every voter that dares to vote at realistic base rates
    and teaches the ensemble to abstain — trust needs a symmetric
    direction-quality signal even when the bandits keep the asymmetric map.

16. **A flag missing from the live .env is a silent no-op, not an error.**
    SENTIMENT_ENABLED entered .env.example the same day prod deployed, the
    live .env predated it, and the 5th voter did nothing for 19 days with no
    warning anywhere. After adding a flag, diff `.env.example` keys against
    every deployed `.env` (preflight now the right home for that check).
    **Fixed in v3.7:** `preflight._env_parity()` diffs `.env.example` key names
    against the live environment and WARNs per missing key — presence, not
    truthiness (`T2_EXTRA_VOTES=` is legitimately empty), names only, never
    values. Run preflight after every `.env` edit.

17. **A verdict from n=17 is an anecdote, not evidence — and it WILL get
    refuted.** v3.7 hard-disabled 1h mixed-regime NWE on 2/17; 21 days later
    the same cohort measured 40.6% on n=96 (the system's best signal,
    suppressed by its own flag). Same story for the 4h NWE ban (12.5% tiny-n
    → 43.8% @ n=16 and climbing). Fix shipped in v3.8: per-source hand flags
    replaced by the evidence ledger (cohort rate + Wilson LB + sample floor),
    and any future threshold decision needs the pre-registered-rule treatment
    the SMS backtest got (rule written into the plan BEFORE the run).

18. **Never let a mid-pipeline value into a persisted feature.** cycle passed
    `emitted`/`trigger_source` into the meta vector BEFORE the final gates
    ran; training rows carried post-gate values. The single-code-path
    featurizer everyone trusted couldn't help — the SKEW WAS IN THE INPUT
    DICT. Rule: every feature must be persisted on every row exactly as
    served (candidate_trigger pattern), and gate outcomes (emitted) are
    labels-adjacent, never features.

19. **Grade the thing the customer received.** Emitted type-1 signals carried
    the trigger's direction while barriers + labels used the brain's internal
    final (ALGOUSDT: sent SELL, graded BUY/tp). Any pipeline where "what we
    sent" and "what we scored" can diverge needs a column that records the
    sent artifact itself (candidate_action) and a grader anchored on it.
