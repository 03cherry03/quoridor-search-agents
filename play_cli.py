"""Simple terminal play mode. No external dependencies or server needed.

Run: python play_cli.py --ai minimax2_ab --human-side 0
     python play_cli.py --ai minimax3_ab --reply-k 4

Commands: p ROW COL | w ROW COL h | w ROW COL v | quit
Rows and columns start at 0.
"""
from __future__ import annotations

import argparse

from quoridor_ai import Agent, Options, MODES
from quoridor_engine import Action, Rules, State, _bit, start_state, winner


def pretty_board(state: State) -> str:
    grid = [[" " for _ in range(17)] for _ in range(17)]
    for r in range(9):
        for c in range(9):
            grid[2*r][2*c] = "."
    for r in range(8):
        for c in range(8):
            if state.horizontal & _bit(r, c):
                for x in (2*c, 2*c+1, 2*c+2):
                    grid[2*r+1][x] = "="
            if state.vertical & _bit(r, c):
                for y in (2*r, 2*r+1, 2*r+2):
                    grid[y][2*c+1] = "|"
    for side, (r, c) in enumerate(state.pawns):
        grid[2*r][2*c] = "A" if side == 0 else "B"
    return "\n".join("".join(row) for row in grid)


def parse_action(text: str) -> Action:
    tokens = text.strip().lower().split()
    if len(tokens) == 3 and tokens[0] in ("p", "pawn", "move"):
        return Action("pawn", int(tokens[1]), int(tokens[2]))
    if len(tokens) == 4 and tokens[0] in ("w", "wall"):
        return Action("wall", int(tokens[1]), int(tokens[2]), tokens[3])
    raise ValueError("Use: p ROW COL  or  w ROW COL h/v  or  quit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", choices=tuple(MODES), default="minimax2_ab")
    parser.add_argument("--human-side", type=int, choices=(0, 1), default=0)
    parser.add_argument("--reply-k", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=None)
    args = parser.parse_args()
    ai = Agent(Options(mode=args.ai, root_top_k=8, wall_margin=2,
                       reply_top_k=args.reply_k if args.ai == "minimax3_ab" else None,
                       time_limit_s=args.time_limit,
                       iterative_deepening=bool(args.time_limit)))
    rules = Rules()
    s = start_state()
    print("Side A (0) moves from row 0 to row 8; Side B (1) moves to row 0.")
    print("Examples: p 1 4  |  w 2 4 h  |  quit")
    for turn_number in range(201):
        print("\n" + pretty_board(s))
        print(f"ply={turn_number}, turn={'A' if s.turn == 0 else 'B'}, walls={s.remaining}")
        if winner(s) is not None:
            print(f"Winner: {'A' if winner(s) == 0 else 'B'}")
            return
        if s.turn == args.human_side:
            raw = input("Your action> ").strip()
            if raw.lower() in ("quit", "q", "exit"):
                return
            try:
                action = parse_action(raw)
                s = rules.apply(s, action)
            except (ValueError, IndexError) as e:
                print(f"Invalid action: {e}")
        else:
            choice = ai.choose(s)
            print(f"AI: {choice.action} | depth={choice.stats.completed_depth}, "
                  f"ms={choice.stats.elapsed_ms:.2f}, "
                  f"nodes={choice.stats.nodes}, cutoff={choice.stats.cutoffs}")
            s = rules.apply(s, choice.action)
    print("Stopped at 200 plies (demo limit).")


if __name__ == "__main__":
    main()
