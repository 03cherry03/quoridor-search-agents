"""Pilot study: measure actual completed depths at common time budgets.

Example:
  python pilot_budget.py --cases 20 --budgets 0.2 0.5 1.0 --output results/pilot

Only a time budget meeting --threshold for BOTH depth-3 methods is recommended.
If none qualifies, the budget remains UNDECIDED: adjust k or test longer budgets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from experiment import random_opening, write_csv
from quoridor_ai import Agent, MODES, Options
from quoridor_engine import start_state, winner

DEFAULT_AGENTS = ('minimax2', 'minimax3', 'minimax2_ab', 'minimax3_ab')


def pick_budget(summary_rows: list[dict], budgets: list[float],
                required: tuple[str, ...] = ('minimax3', 'minimax3_ab'),
                threshold: float = 0.75) -> float | None:
    """Smallest tested common budget at which ALL depth-3 modes qualify."""
    for budget in sorted(budgets):
        if all(any(r['agent'] == agent and r['budget_s'] == budget
                   and r['full_depth_rate'] >= threshold
                   for r in summary_rows) for agent in required):
            return budget
    return None


def run(cases: int, budgets: list[float], seed: int = 42,
        root_k: int = 8, reply_k: int | None = 4,
        margin: int | None = 2, agents: tuple[str, ...] = DEFAULT_AGENTS,
        threshold: float = .75):
    if cases < 1 or any(b <= 0 for b in budgets):
        raise ValueError('cases >= 1 and all budgets > 0 required')
    if any(x not in MODES or x in ('greedy', 'mcts') for x in agents):
        raise ValueError('Pilot is for Minimax-based agents')
    observations = []
    grouped = defaultdict(list)
    for i in range(cases):
        ply = 0 if i == 0 else 2 * (1 + i % 7)
        state = start_state() if i == 0 else random_opening(seed + i, ply)
        if winner(state) is not None:
            continue
        for budget in budgets:
            for mode in agents:
                opts = Options(mode=mode, root_top_k=root_k,
                               reply_top_k=reply_k, wall_margin=margin,
                               time_limit_s=budget, iterative_deepening=True)
                result = Agent(opts).choose(state)
                goal_depth = MODES[mode][0]
                item = {
                    'case': i, 'opening_seed': seed + i,
                    'opening_plies': ply, 'agent': mode,
                    'budget_s': budget,
                    'target_depth': goal_depth,
                    'completed_depth': result.stats.completed_depth,
                    'full_depth': int(result.stats.completed_depth == goal_depth),
                    'fell_back': int(result.stats.completed_depth == 0),
                    'timed_out': int(result.stats.timed_out),
                    'elapsed_ms': round(result.stats.elapsed_ms, 4),
                    'nodes': result.stats.nodes,
                    'bfs_calls': result.stats.bfs_calls,
                }
                observations.append(item)
                grouped[(mode, budget)].append(item)
                print(f'pilot {i+1}/{cases}: {mode:14s} budget={budget:.2f}s '
                      f'depth={result.stats.completed_depth}/{goal_depth} '
                      f'time={result.stats.elapsed_ms:.1f}ms', flush=True)
    summaries = []
    for (mode, budget), items in sorted(grouped.items()):
        n = len(items)
        summaries.append({
            'agent': mode, 'budget_s': budget, 'cases': n,
            'target_depth': MODES[mode][0],
            'mean_completed_depth': round(sum(x['completed_depth'] for x in items) / n, 4),
            'full_depth_rate': round(sum(x['full_depth'] for x in items) / n, 5),
            'fallback_rate': round(sum(x['fell_back'] for x in items) / n, 5),
            'timed_search_rate': round(sum(x['timed_out'] for x in items) / n, 5),
            'mean_ms': round(sum(x['elapsed_ms'] for x in items) / n, 4),
        })
    recommendation = pick_budget(summaries, budgets, threshold=threshold)
    selection = {
        'recommended_budget_s': recommendation,
        'recommended_hard_limit_s': (round(recommendation + .05, 5)
                                     if recommendation is not None else None),
        'status': 'qualified' if recommendation is not None else 'no_tested_budget_qualified',
        'minimum_full_depth_rate': threshold,
        'required_modes': ['minimax3', 'minimax3_ab'],
        'candidate_budgets_s': budgets,
        'cases': cases, 'seed': seed,
        'root_k': root_k, 'reply_k': reply_k, 'wall_margin': margin,
        'method': 'Choose the smallest common budget where both depth-3 agents finish'
                  ' full depth on at least the required fraction of sampled states.',
        'note': 'This is a pilot recommendation, not proof the full league will reach depth 3.'
    }
    return observations, summaries, selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=int, default=12)
    p.add_argument('--budgets', type=float, nargs='+', default=[.2, .5, 1.0])
    p.add_argument('--threshold', type=float, default=.75)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--root-k', type=int, default=8)
    p.add_argument('--reply-k', type=int, default=4)
    p.add_argument('--margin', type=int, default=2)
    p.add_argument('--output', type=Path, default=Path('results/pilot'))
    args = p.parse_args()
    if not 0 < args.threshold <= 1:
        p.error('--threshold must lie in (0,1]')
    observations, summaries, selection = run(
        args.cases, sorted(set(args.budgets)), args.seed,
        args.root_k, args.reply_k, args.margin, threshold=args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'depth_observations.csv', observations)
    write_csv(args.output / 'depth_summary.csv', summaries)
    (args.output / 'budget_choice.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\nPilot summary (full_depth_rate):')
    for r in summaries:
        print(f'  {r["agent"]:15s} {r["budget_s"]:.2f}s: '
              f'{r["full_depth_rate"]:.1%} complete, mean depth '
              f'{r["mean_completed_depth"]:.2f}')
    if selection['recommended_budget_s'] is None:
        print('\nNo tested budget met the depth-3 criterion: try --budgets '
              '0.5 1.0 2.0 or a common smaller --root-k/--reply-k.')
    else:
        print(f'\nSuggested league budget: {selection["recommended_budget_s"]:.2f}s '
              f'(hard forfeit: {selection["recommended_hard_limit_s"]:.2f}s)')
    print(f'Saved: {args.output.resolve()}')


if __name__ == '__main__':
    main()
