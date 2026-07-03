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
