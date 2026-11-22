"""Hackathon API spend guard for SIDERIUS runs.

SIDERIUS has no run-wide spending limit ("model routing is not a spending
limit"). This module adds one: it sums every token_usage.jsonl under the
project's ledger roots, prices it with config/prices.json, and compares the
result with config/budget.json.

Use in a caller, before EVERY capability invocation:

    from spend_guard import check_budget
    check_budget()            # raises BudgetExceeded at stop_fraction of any budget

CLI (from anywhere):

    python project/scripts/spend_guard.py [--root EXTRA_DIR ...]
    exit 0 = under budget, 3 = a provider reached its stop threshold,
    4 = a budgeted provider used a model with no verified price (cannot enforce).

Conservative by construction: every prompt token is charged at the input
rate (the ledger has no cache split). Provider consoles' own spend limits are
the hard backstop; set them too.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "config"


class BudgetExceeded(RuntimeError):
    pass


class UnpricedModel(RuntimeError):
    pass


def _load(name: str) -> dict:
    return json.loads((CONFIG / name).read_text())


def tally(extra_roots: list[Path] | None = None) -> dict:
    budget = _load("budget.json")
    prices = _load("prices.json")
    roots = [PROJECT / r for r in budget.get("ledger_roots", ["results"])] + list(extra_roots or [])
    usage: dict = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0}))
    files = []
    for root in roots:
        if root.is_dir():
            files += sorted(root.rglob("token_usage.jsonl"))
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            t = row.get("tokens") or {}
            u = usage[row.get("provider") or "unknown"][row.get("served_model") or row.get("model") or "unknown"]
            u["calls"] += 1
            u["prompt"] += int(t.get("prompt") or 0)
            u["completion"] += int(t.get("completion") or 0)
    out = {"ledgers": [str(f) for f in files], "providers": {}}
    for provider, models in usage.items():
        spent, unpriced = 0.0, []
        for model, u in models.items():
            p = prices.get(provider, {}).get(model) or {}
            if p.get("input") is None or p.get("output") is None:
                unpriced.append(model)
                u["usd"] = None
                continue
            u["usd"] = u["prompt"] / 1e6 * p["input"] + u["completion"] / 1e6 * p["output"]
            spent += u["usd"]
        cap = budget["budgets_usd"].get(provider)
        out["providers"][provider] = {
            "models": {m: dict(v) for m, v in models.items()},
            "spent_usd": spent, "budget_usd": cap,
            "stop_at_usd": cap * budget["stop_fraction"] if cap is not None else None,
            "unpriced_models": unpriced,
        }
    for provider, cap in budget["budgets_usd"].items():
        out["providers"].setdefault(provider, {"models": {}, "spent_usd": 0.0, "budget_usd": cap,
                                               "stop_at_usd": cap * budget["stop_fraction"], "unpriced_models": []})
    # Spend made before this project (outside its ledger roots) that the team chose to count.
    for provider, carry in budget.get("carried_over_usd", {}).items():
        p = out["providers"].setdefault(provider, {"models": {}, "spent_usd": 0.0, "budget_usd": budget["budgets_usd"].get(provider),
                                                   "stop_at_usd": None, "unpriced_models": []})
        p["carried_over_usd"] = float(carry["usd"])
        p["spent_usd"] += float(carry["usd"])
    return out


def check_budget(extra_roots: list[Path] | None = None) -> dict:
    """Raise before a capability call if any budgeted provider is at its stop threshold."""
    t = tally(extra_roots)
    for provider, p in t["providers"].items():
        if p["budget_usd"] is None:
            continue
        if p["unpriced_models"]:
            raise UnpricedModel(f"{provider}: no verified price for {p['unpriced_models']} in config/prices.json; "
                                "cannot enforce the budget")
        if p["spent_usd"] >= p["stop_at_usd"]:
            raise BudgetExceeded(f"{provider}: ${p['spent_usd']:.2f} spent >= stop threshold ${p['stop_at_usd']:.2f} "
                                 f"(budget ${p['budget_usd']:.2f})")
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, action="append", default=[], help="extra directory to scan for ledgers")
    a = ap.parse_args()
    t = tally(a.root)
    print(f"{len(t['ledgers'])} ledger file(s)")
    status = 0
    for provider, p in sorted(t["providers"].items()):
        cap = f"${p['budget_usd']:.2f} (stop at ${p['stop_at_usd']:.2f})" if p["budget_usd"] is not None else "no budget (tracked only)"
        carry = f" (incl. ${p['carried_over_usd']:.4f} carried over)" if p.get("carried_over_usd") else ""
        print(f"{provider}: spent ${p['spent_usd']:.4f}{carry} of {cap}")
        for m, u in p["models"].items():
            usd = "UNPRICED" if u["usd"] is None else f"${u['usd']:.4f}"
            print(f"   {m}: {u['calls']} calls, {u['prompt']:,} in / {u['completion']:,} out tokens, {usd}")
        if p["budget_usd"] is not None:
            if p["unpriced_models"]:
                print(f"   CANNOT ENFORCE: unpriced {p['unpriced_models']}")
                status = max(status, 4)
            elif p["spent_usd"] >= p["stop_at_usd"]:
                print("   STOP: threshold reached")
                status = max(status, 3)
    return status


if __name__ == "__main__":
    sys.exit(main())
