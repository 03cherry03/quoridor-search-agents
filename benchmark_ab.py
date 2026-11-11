"""Correctness + complexity comparison on IDENTICAL board states.

Run: python benchmark_ab.py --cases 12 --output results/ab_verification.csv
This is distinct from the win-rate tournament (experiment.py).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiment import random_opening
from quoridor_ai import Agent, Options
from quoridor_engine import Rules, start_state, winner


def compare(cases: int = 12, seed: int = 42, root_k: int = 8,
            margin: int = 2) -> list[dict]:
    out = []
    for i in range(cases):
        # Identical state for both agents, but diversify pawn/wall layouts.
        state = (start_state() if i == 0 else
                 random_opening(seed + i, num_plies=2 * (i % 5 + 1)))
        if winner(state) is not None:
            continue
        snapshot = state
        opts = dict(root_top_k=root_k, wall_margin=margin, reply_top_k=None)
        b = Agent(Options(mode="minimax2", **opts)).choose(state)
        c = Agent(Options(mode="minimax2_ab", **opts)).choose(state)
        if b.action != c.action or b.stats.search_value != c.stats.search_value:
            raise AssertionError(
                f"Different answer at case {i}: plain={b.action}/{b.stats.search_value}; "
                f"AB={c.action}/{c.stats.search_value}")
        if state != snapshot:
            raise AssertionError("State was mutated")
        Rules().apply(state, c.action)  # verify action actually legal
        out.append({
            "case": i,
            "opening_plies": 0 if i == 0 else 2 * (i % 5 + 1),
            "action_equal": True,
            "score_equal": True,
            "action": str(b.action),
            "value": b.stats.search_value,
            "plain_ms": round(b.stats.elapsed_ms, 4),
            "ab_ms": round(c.stats.elapsed_ms, 4),
            "plain_nodes": b.stats.nodes,
            "ab_nodes": c.stats.nodes,
            "plain_bfs": b.stats.bfs_calls,
            "ab_bfs": c.stats.bfs_calls,
            "plain_eval": b.stats.evaluations,
            "ab_eval": c.stats.evaluations,
            "cutoffs": c.stats.cutoffs,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-k", type=int, default=8)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=Path("results/ab_verification.csv"))
    args = parser.parse_args()
    rows = compare(args.cases, args.seed, args.root_k, args.margin)
    if not rows:
        raise ValueError("No states to test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)}/{len(rows)} boards: EXACT same action and value.")
    for key in ("ms", "nodes", "bfs", "eval"):
        b = sum(r["plain_" + key] for r in rows)
        c = sum(r["ab_" + key] for r in rows)
        reduction = 100 * (1 - c / b) if b else 0
        print(f"  {key}: plain {b:.1f} vs alpha-beta {c:.1f}; "
              f"reduction {reduction:.1f}%")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
