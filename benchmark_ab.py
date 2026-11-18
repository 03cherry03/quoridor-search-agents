"""Fixed-depth no-timeout Minimax vs Alpha-Beta comparisons (depths 2 AND 3).

Same positions, SAME score and beam settings. Verify exact action AND value.
Run separately from a time-limited round-robin league. Default --cases 6.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment import random_opening, write_csv
from quoridor_ai import Agent, Options
from quoridor_engine import Rules, start_state, winner


def compare(cases: int = 6, seed: int = 42, root_k: int = 8,
            margin: int | None = 2, reply_k: int | None = 4,
            depths: tuple[int, ...] = (2, 3)) -> list[dict]:
    if cases < 1 or not set(depths) <= {2, 3}:
        raise ValueError('cases >= 1; supported depths 2 and 3')
    output = []
    for depth in depths:
        for case in range(cases):
            ply = 0 if case == 0 else 2 * (case % 5 + 1)
            state = start_state() if case == 0 else random_opening(seed + case, ply)
            if winner(state) is not None:
                continue
            snapshot = state
            common = dict(root_top_k=root_k, reply_top_k=reply_k,
                          wall_margin=margin, time_limit_s=None,
                          iterative_deepening=False)
            plain = Agent(Options(mode=f'minimax{depth}', **common)).choose(state)
            ab = Agent(Options(mode=f'minimax{depth}_ab', **common)).choose(state)
            if plain.stats.completed_depth != depth or ab.stats.completed_depth != depth:
                raise AssertionError('A fixed-depth benchmark must reach the target')
            if plain.action != ab.action or plain.stats.search_value != ab.stats.search_value:
                raise AssertionError(f'Depth {depth}, case {case}: '
                                     f'plain {plain.action}/{plain.stats.search_value}, '
                                     f'AB {ab.action}/{ab.stats.search_value}')
            if snapshot != state:
                raise AssertionError('Search mutated source state')
            Rules().apply(state, ab.action)
            if ab.stats.nodes > plain.stats.nodes:
                raise AssertionError('Alpha-beta visited more search-tree nodes')
            row = {
                'depth': depth, 'case': case, 'opening_plies': ply,
                'same_action': True, 'same_score': True,
                'action': str(plain.action), 'value': plain.stats.search_value,
                'plain_ms': round(plain.stats.elapsed_ms, 4),
                'ab_ms': round(ab.stats.elapsed_ms, 4),
                'plain_nodes': plain.stats.nodes, 'ab_nodes': ab.stats.nodes,
                'plain_bfs': plain.stats.bfs_calls, 'ab_bfs': ab.stats.bfs_calls,
                'plain_evaluations': plain.stats.evaluations,
                'ab_evaluations': ab.stats.evaluations,
                'ab_cutoffs': ab.stats.cutoffs,
                'root_k': root_k, 'reply_k': reply_k,
                'wall_margin': margin, 'time_limit_s': 'none',
            }
            output.append(row)
            print(f'depth {depth} case {case+1}/{cases}: identical move/value; '
                  f'plain {plain.stats.elapsed_ms:.1f}ms vs AB '
                  f'{ab.stats.elapsed_ms:.1f}ms', flush=True)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--root-k', type=int, default=8)
    p.add_argument('--reply-k', type=int, default=4)
    p.add_argument('--margin', type=int, default=2)
    p.add_argument('--depths', type=int, nargs='+', default=[2, 3])
    p.add_argument('--output', type=Path,
                   default=Path('results/fixed_depth/ab_verification.csv'))
    a = p.parse_args()
    rows = compare(a.cases, a.seed, a.root_k, a.margin,
                   a.reply_k, tuple(a.depths))
    if not rows:
        raise ValueError('No cases were tested')
    write_csv(a.output, rows)
    for depth in a.depths:
        rs = [r for r in rows if r['depth'] == depth]
        print(f'\nDepth {depth}: verified {len(rs)} identical decisions')
        for quantity in ('ms', 'nodes', 'bfs', 'evaluations'):
            plain = sum(r['plain_' + quantity] for r in rs)
            ab = sum(r['ab_' + quantity] for r in rs)
            reduction = (1 - ab / plain) * 100 if plain else 0
            print(f'  {quantity:12s}: plain {plain:,.1f} / AB {ab:,.1f} '
                  f'({reduction:+.1f}% fewer)')
    print(f'CSV saved: {a.output.resolve()}')


if __name__ == '__main__':
    main()
