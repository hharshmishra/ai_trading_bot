"""SKU catalog — the single source of truth for products, durations and prices.

Prices mirror docs/subscription-deck.html (anchor 1 USDT ≈ ₹88, USDT rounded up
for volatility + TRC-20 fees). Products: 'signals' (customer channel access)
and 'pro' (control-bot access) — bundles simply carry both.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple


class SKU(NamedTuple):
    code: str
    products: Tuple[str, ...]      # ('signals',) | ('pro',) | ('signals', 'pro')
    days: int
    inr: int
    usdt: float
    label: str


SKUS = {
    "SIG-7":  SKU("SIG-7",  ("signals",),       7,   199,  2.5, "Signals · 7 days"),
    "SIG-15": SKU("SIG-15", ("signals",),       15,  349,  4.5, "Signals · 15 days"),
    "SIG-30": SKU("SIG-30", ("signals",),       30,  599,  7.0, "Signals · 30 days"),
    "PRO-15": SKU("PRO-15", ("pro",),           15,  299,  3.5, "Pro · 15 days"),
    "PRO-30": SKU("PRO-30", ("pro",),           30,  499,  6.0, "Pro · 30 days"),
    "BUN-30": SKU("BUN-30", ("signals", "pro"), 30,  899, 10.0, "Bundle · 30 days"),
    "FND-90": SKU("FND-90", ("signals", "pro"), 90, 2199, 25.0, "Founders · 90 days"),
}

PRODUCTS = ("signals", "pro")

# Display order — the ONE place the SKU sequence lives. plans_text, the
# storefront keyboard and the renewal keyboard all consume this, so adding a
# SKU can never leave the description and the buy buttons disagreeing.
DISPLAY_ORDER = ("SIG-7", "SIG-15", "SIG-30", "PRO-15", "PRO-30", "BUN-30", "FND-90")


def ordered():
    """SKUs in display order."""
    return [SKUS[c] for c in DISPLAY_ORDER]


def plans_text() -> str:
    """Customer-facing /plans body (HTML), built from the catalog so a price
    change is a one-line edit here."""
    lines = ["⚡ <b>BitReinforceX — AI Signal System</b>", "",
             "Four AI agents vote on 48 pairs, every hour, at real candle "
             "closes. Every signal carries entry, target and stop — and every "
             "signal is graded against what the market actually did.", "",
             "📡 <b>SIGNALS</b> — private channel"]
    for c in ("SIG-7", "SIG-15", "SIG-30"):
        s = SKUS[c]
        lines.append(f" · {s.days} days — ₹{s.inr} ({s.usdt:g} USDT)")
    lines += ["", "🤖 <b>PRO</b> — talk to the agents "
              "(/news /indicator /research /context /regime /derivs)"]
    for c in ("PRO-15", "PRO-30"):
        s = SKUS[c]
        lines.append(f" · {s.days} days — ₹{s.inr} ({s.usdt:g} USDT)")
    b, f = SKUS["BUN-30"], SKUS["FND-90"]
    lines += ["",
              f"💎 <b>BUNDLE</b> — both, {b.days} days — ₹{b.inr} ({b.usdt:g} USDT)",
              f"🏆 <b>FOUNDERS</b> — both, {f.days} days — ₹{f.inr} ({f.usdt:g} USDT)",
              "",
              "Pay by UPI or USDT (TRC-20). Access is instant and automatic.",
              "<i>Educational market analysis, not financial advice.</i>"]
    return "\n".join(lines)
