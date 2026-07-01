"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
CACHE_WRITE_COST = 1.0
CACHE_READ_DISCOUNT = 0.10
REASONING_CAP_REPORT_PCT = 0.10
REASONING_CAP_ACTION_PCT = 0.05


def _cache_key(row: dict) -> tuple[str, str, str]:
    return (row["team"], row.get("project") or "untagged", row["route_tier"])


def _cache_policy(rows: list[dict]) -> tuple[dict, dict]:
    """Decide which shared prefixes have enough reads to justify caching."""
    stats = defaultdict(lambda: {"reads": 0, "cached_tokens": 0})
    for r in rows:
        cached = int(num(r["cached_input_tokens"]))
        if cached <= 0:
            continue
        key = _cache_key(r)
        stats[key]["reads"] += 1
        stats[key]["cached_tokens"] += cached

    decisions = {
        key: pricing.cache_is_worth_it(
            value["reads"], write_cost=CACHE_WRITE_COST, read_discount=CACHE_READ_DISCOUNT
        )
        for key, value in stats.items()
    }
    group_count = len(stats)
    avg_reads = sum(v["reads"] for v in stats.values()) / group_count if group_count else 0.0
    summary = {
        "break_even_reads": round(pricing.cache_break_even_reads(CACHE_WRITE_COST, CACHE_READ_DISCOUNT), 2),
        "avg_cache_reads": round(avg_reads, 1),
        "groups_total": group_count,
        "groups_worth_it": sum(1 for enabled in decisions.values() if enabled),
        "cached_tokens_evaluated": sum(v["cached_tokens"] for v in stats.values()),
    }
    return decisions, summary


def _empty_reasoning_bucket() -> dict:
    return {"requests": 0, "tokens": 0, "cost": 0.0, "wh": 0.0}


def _cap_savings(
    avoidable_cost: float,
    avoidable_wh: float,
    reasoning_requests: int,
    total_requests: int,
    cap_pct: float,
) -> dict:
    target = int(total_requests * cap_pct)
    if reasoning_requests <= 0 or reasoning_requests <= target:
        fraction = 0.0
    else:
        fraction = (reasoning_requests - target) / reasoning_requests
    return {
        "target_pct": round(cap_pct * 100, 1),
        "target_requests": target,
        "savings_daily": round(avoidable_cost * fraction, 2),
        "wh_savings_daily": round(avoidable_wh * fraction, 1),
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache_decisions, cache_summary = _cache_policy(rows)

    base_cost = opt_cost = 0.0
    total_tokens = 0
    reasoning = {False: _empty_reasoning_bucket(), True: _empty_reasoning_bucket()}
    avoidable_reasoning_cost = 0.0
    avoidable_reasoning_wh = 0.0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        total_tokens += inp + out

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # OPTIMIZED: cascade (route_tier), prompt caching after break-even, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cached_applied = cached if cache_decisions.get(_cache_key(r), False) else 0
        req_cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached_applied, batch=is_batch)
        opt_cost += req_cost

        tokens = inp + out
        req_wh = sustainability.wh_per_query(tokens, is_reasoning=is_reasoning)
        bucket = reasoning[is_reasoning]
        bucket["requests"] += 1
        bucket["tokens"] += tokens
        bucket["cost"] += req_cost
        bucket["wh"] += req_wh

        if is_reasoning:
            normal_out = max(1, round(out / 6.0))
            normal_cost = pricing.request_cost(inp, normal_out, pin, pout, cached_in=cached_applied, batch=is_batch)
            normal_wh = sustainability.wh_per_query(inp + normal_out, is_reasoning=False)
            avoidable_reasoning_cost += max(0.0, req_cost - normal_cost)
            avoidable_reasoning_wh += max(0.0, req_wh - normal_wh)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    total_requests = len(rows)
    reasoning_requests = reasoning[True]["requests"]
    total_wh = reasoning[False]["wh"] + reasoning[True]["wh"]
    reasoning_budget = {
        "requests": reasoning_requests,
        "traffic_pct": round(reasoning_requests / total_requests * 100, 1) if total_requests else 0.0,
        "cost_daily": round(reasoning[True]["cost"], 2),
        "cost_pct": round(reasoning[True]["cost"] / opt_cost * 100, 1) if opt_cost else 0.0,
        "wh_daily": round(reasoning[True]["wh"], 1),
        "wh_pct": round(reasoning[True]["wh"] / total_wh * 100, 1) if total_wh else 0.0,
        "cap_10pct": _cap_savings(
            avoidable_reasoning_cost, avoidable_reasoning_wh, reasoning_requests, total_requests, REASONING_CAP_REPORT_PCT
        ),
        "cap_5pct": _cap_savings(
            avoidable_reasoning_cost, avoidable_reasoning_wh, reasoning_requests, total_requests, REASONING_CAP_ACTION_PCT
        ),
        "routing_rule": "Use reasoning only for eval or user tasks with high complexity; cap default traffic at 5%.",
    }

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(
            "cache economics: "
            f"break-even={cache_summary['break_even_reads']} reads, "
            f"observed avg={cache_summary['avg_cache_reads']} reads/prefix, "
            f"enabled={cache_summary['groups_worth_it']}/{cache_summary['groups_total']} groups"
        )
        print(
            "reasoning budget: "
            f"{reasoning_budget['traffic_pct']}% traffic -> "
            f"{reasoning_budget['cost_pct']}% cost, {reasoning_budget['wh_pct']}% Wh"
        )
        print(
            "reasoning cap: "
            f"10% saves ${reasoning_budget['cap_10pct']['savings_daily']}/day; "
            f"5% saves ${reasoning_budget['cap_5pct']['savings_daily']}/day and "
            f"{reasoning_budget['cap_5pct']['wh_savings_daily']} Wh/day"
        )

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_economics": cache_summary,
        "reasoning_budget": reasoning_budget,
    }


if __name__ == "__main__":
    run()
