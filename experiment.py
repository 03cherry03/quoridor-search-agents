"""Paired AI-vs-AI tournament for Quoridor course project.

Examples:
  python experiment.py --a minimax2 --b minimax2_ab --pairs 4
  python experiment.py --a greedy --b minimax2_ab --pairs 10 --max-plies 120
  python experiment.py --a minimax2_ab --b minimax3_ab --pairs 2 --reply-k 4

The max-ply cap is an EXPERIMENTAL draw rule (not an official Quoridor rule).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from quoridor_ai import Agent, Options, MODES
from quoridor_engine import Action, Rules, State, start_state, winner


@dataclass
class MatchResult:
    game_id: int
    opening_seed: int
    ai_side0: str
    ai_side1: str
    winning_side: Optional[int]
    total_plies: int
    avg_ms: float
    total_bfs: int
    total_evaluations: int
    total_nodes: int
    timeouts: int

    @property
    def winning_ai(self) -> str:
        if self.winning_side is None:
            return "draw"
        return self.ai_side0 if self.winning_side == 0 else self.ai_side1

    def to_row(self) -> dict:
        return {
            "game_id": self.game_id,
            "opening_seed": self.opening_seed,
            "ai_side0": self.ai_side0,
            "ai_side1": self.ai_side1,
            "winner_side": self.winning_side if self.winning_side is not None else "draw",
            "winner_ai": self.winning_ai,
            "total_plies": self.total_plies,
            "avg_move_ms": round(self.avg_ms, 3),
            "total_bfs_calls": self.total_bfs,
            "total_evaluations": self.total_evaluations,
            "total_search_nodes": self.total_nodes,
            "timeouts": self.timeouts,
        }


def random_opening(seed: int, num_plies: int = 4) -> State:
    """Diversify starting positions; the two games in a pair share one opening."""
    rng = random.Random(seed)
    s = start_state()
    rules = Rules()
    for _ in range(num_plies):
        if winner(s) is not None:
            break
        actions = rules.legal_actions(s, wall_margin=2)
        pawns = [a for a in actions if a.kind == "pawn"]
        walls = [a for a in actions if a.kind == "wall"]
        pool = walls if walls and rng.random() < 0.35 else pawns
        s = rules.apply(s, rng.choice(pool))
    return s


def play(state: State, side0: Agent, side1: Agent,
         game_id: int, seed: int, max_plies: int,
         name0: str, name1: str) -> tuple[MatchResult, list[dict]]:
    agents = [side0, side1]
    referee = Rules()
    log: list[dict] = []
    game_start_turn = state.turn
    for ply in range(max_plies):
        if winner(state) is not None:
            break
        side = state.turn
        choice = agents[side].choose(state)
        nxt = referee.apply(state, choice.action)  # Validates chosen action.
        st = choice.stats
        log.append({
            "game_id": game_id,
            "opening_seed": seed,
            "ply": ply + 1,
            "side": side,
            "agent": name0 if side == 0 else name1,
            "action": str(choice.action),
            "wall_margin": agents[side].options.wall_margin,
            "root_top_k": agents[side].options.root_top_k,
            "reply_top_k": agents[side].options.reply_top_k,
            "search_depth": st.completed_depth,
            "value": st.search_value,
            "timed_out": int(st.timed_out),
            "decision_ms": round(st.elapsed_ms, 3),
            "nodes": st.nodes,
            "leaves": st.leaves,
            "evaluations": st.evaluations,
            "bfs_calls": st.bfs_calls,
            "wall_checks": st.wall_checks,
            "cutoffs": st.cutoffs,
        })
        state = nxt
    w = winner(state)
    match = MatchResult(
        game_id=game_id,
        opening_seed=seed,
        ai_side0=name0,
        ai_side1=name1,
        winning_side=w,
        total_plies=len(log),
        avg_ms=sum(x["decision_ms"] for x in log) / max(1, len(log)),
        total_bfs=sum(x["bfs_calls"] for x in log),
        total_evaluations=sum(x["evaluations"] for x in log),
        total_nodes=sum(x["nodes"] for x in log),
        timeouts=sum(x["timed_out"] for x in log),
    )
    return match, log


def run_tournament(a: Options, b: Options, pairs: int = 2,
                   seed: int = 42, opening_plies: int = 4,
                   max_plies: int = 100):
    if pairs < 1:
        raise ValueError("pairs must be positive")
    results: list[MatchResult] = []
    moves = []
    for pair in range(pairs):
        opening_seed = seed + pair
        opening = random_opening(opening_seed, opening_plies)
        # Keep the opening state FIXED while swapping the two algorithms' sides.
        for swapped in (False, True):
            first, second = (a, b) if not swapped else (b, a)
            game_id = pair * 2 + 1 + int(swapped)
            match, log = play(opening, Agent(first), Agent(second), game_id,
                              opening_seed, max_plies, first.mode, second.mode)
            results.append(match)
            moves.extend(log)
            print(f"game {game_id:03d}: {first.mode} (side0) vs {second.mode} "
                  f"(side1) => {match.winning_ai}; plies={match.total_plies}, "
                  f"avg={match.avg_ms:.1f}ms")
    return results, moves


def write_csv(path: Path, data: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def summary(results: list[MatchResult], rows: list[dict], names: tuple[str, str]) -> str:
    win_counts = Counter(r.winning_ai for r in results)
    out = [f"Total: {len(results)} games; draws: {win_counts['draw']}"]
    for name in dict.fromkeys(names):
        wins = win_counts[name]
        draws = win_counts["draw"]
        n = len(results)
        # Score rate counts an experimental draw as 0.5.
        rate = (wins + 0.5 * draws) / n if n else 0
        rows_for = [r for r in rows if r["agent"] == name]
        avg_ms = sum(r["decision_ms"] for r in rows_for) / max(1, len(rows_for))
        avg_bfs = sum(r["bfs_calls"] for r in rows_for) / max(1, len(rows_for))
        avg_nodes = sum(r["nodes"] for r in rows_for) / max(1, len(rows_for))
        out.append(f"{name}: W {wins}, score rate {rate:.1%}, "
                   f"mean move {avg_ms:.1f} ms, mean BFS {avg_bfs:.1f}, "
                   f"mean nodes {avg_nodes:.1f}")
    return "\n".join(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", choices=tuple(MODES), default="minimax2")
    p.add_argument("--b", choices=tuple(MODES), default="minimax2_ab")
    p.add_argument("--pairs", type=int, default=2, help="number of paired openings")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--max-plies", type=int, default=80)
    p.add_argument("--root-k", type=int, default=8)
    p.add_argument("--reply-k", type=int, default=None,
                   help="optional intermediate beam size; same for A and B")
    p.add_argument("--margin", type=int, default=2,
                   help="wall anchors within Chebyshev distance of either pawn")
    p.add_argument("--all-walls", action="store_true",
                   help="consider every wall anchor (slow)")
    p.add_argument("--time-limit", type=float, default=None,
                   help="soft per-move time budget in seconds (e.g. 0.2)")
    p.add_argument("--iterative", action="store_true",
                   help="search depth 1..D and keep last completed depth")
    p.add_argument("--output", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.opening_plies < 0 or args.max_plies < 1:
        raise ValueError("opening-plies must be >=0 and max-plies >0")
    common = dict(root_top_k=args.root_k, reply_top_k=args.reply_k,
                  wall_margin=None if args.all_walls else args.margin,
                  time_limit_s=args.time_limit,
                  iterative_deepening=args.iterative)
    a = Options(mode=args.a, **common)
    b = Options(mode=args.b, **common)
    results, moves = run_tournament(a, b, args.pairs, args.seed,
                                    args.opening_plies, args.max_plies)
    write_csv(args.output / "games.csv", [r.to_row() for r in results])
    write_csv(args.output / "moves.csv", moves)
    report = summary(results, moves, (args.a, args.b))
    (args.output / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nCSV saved in {args.output.resolve()}")


if __name__ == "__main__":
    main()
