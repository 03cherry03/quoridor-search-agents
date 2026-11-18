"""Play human vs Quoridor AI in a terminal; 0-based coordinates.

Examples:
  python play_cli.py --ai minimax2_ab
  python play_cli.py --ai minimax3_ab --time-limit 0.2 --reply-k 4
"""
from __future__ import annotations

import argparse

from quoridor_ai import Agent, Options, MODES
from quoridor_engine import Action, Rules, State, _bit, start_state, winner


def pretty_board(state: State) -> str:
    grid = [[' ' for _ in range(17)] for _ in range(17)]
    for r in range(9):
        for c in range(9):
            grid[2*r][2*c] = '.'
    for r in range(8):
        for c in range(8):
            if state.horizontal & _bit(r, c):
                for x in (2*c, 2*c+1, 2*c+2):
                    grid[2*r+1][x] = '='
            if state.vertical & _bit(r, c):
                for y in (2*r, 2*r+1, 2*r+2):
                    grid[y][2*c+1] = '|'
    for side, (r, c) in enumerate(state.pawns):
        grid[2*r][2*c] = 'A' if side == 0 else 'B'
    return '\n'.join(''.join(row) for row in grid)


def parse_action(value: str) -> Action:
    pieces = value.strip().lower().split()
    if len(pieces) == 3 and pieces[0] in ('p', 'pawn', 'move'):
        return Action('pawn', int(pieces[1]), int(pieces[2]))
    if len(pieces) == 4 and pieces[0] in ('w', 'wall'):
        return Action('wall', int(pieces[1]), int(pieces[2]), pieces[3])
    raise ValueError('Use p ROW COL or w ROW COL h/v or quit')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ai', choices=tuple(MODES), default='minimax2_ab')
    p.add_argument('--human-side', choices=(0, 1), type=int, default=0)
    p.add_argument('--root-k', type=int, default=8)
    p.add_argument('--reply-k', type=int, default=4)
    p.add_argument('--margin', type=int, default=2)
    p.add_argument('--time-limit', type=float, default=None)
    args = p.parse_args()
    ai = Agent(Options(mode=args.ai, root_top_k=args.root_k,
                       reply_top_k=args.reply_k, wall_margin=args.margin,
                       time_limit_s=args.time_limit,
                       iterative_deepening=args.time_limit is not None))
    rules, state = Rules(), start_state()
    print('Player A (0) moves from row 0 to row 8; Player B (1) to row 0.')
    print('Moves: p 1 4   or   w 2 4 h   or   quit')
    for ply in range(201):
        print('\n' + pretty_board(state))
        print(f'Ply {ply}, turn {state.turn}, remaining walls {state.remaining}')
        if winner(state) is not None:
            print(f'Winner: {winner(state)}')
            return
        if state.turn == args.human_side:
            raw = input('Your action> ').strip()
            if raw.lower() in ('quit', 'q', 'exit'):
                return
            try:
                state = rules.apply(state, parse_action(raw))
            except ValueError as e:
                print(f'Illegal action: {e}')
        else:
            result = ai.choose(state)
            st = result.stats
            print(f'AI plays {result.action} | completed depth={st.completed_depth}, '
                  f'time={st.elapsed_ms:.1f} ms, nodes={st.nodes}')
            state = rules.apply(state, result.action)
    print('Stopped after 200 ply demo cap.')


if __name__ == '__main__':
    main()
