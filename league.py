"""Paired round-robin Quoridor tournament (2..5; optional experimental #6 MCTS).

Official course-project score (from supplied PDF):
  Score = 0.7 * WR + 0.3 * (1 + average_margin)/2
  WR = (wins + 0.5 * draws) / games
  Goal-win margin = +/-min(1, loser's remaining BFS distance/8)
  Time-forfeit margin = +/-1; max-ply experimental draw margin = 0.

Run short smoke: python league.py --pairs 1 --max-plies 12
Run fuller:      python league.py --pairs 10 --max-plies 200
Run with pilot:  python league.py --pilot-config results/pilot/budget_choice.json
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import random
import statistics

from experiment import performance_score, run_tournament, write_csv
from quoridor_ai import MODES, Options

LEAGUE_DEFAULT = ('minimax2', 'minimax3', 'minimax2_ab', 'minimax3_ab')


def bootstrap_wr_ci(results, agent: str, samples: int = 1000,
                    seed: int = 2026) -> tuple[float | None, float | None]:
    """Cluster-resample opening seeds; preserve paired/color-swapped games.

    This is a descriptive paired bootstrap. It is not an independence claim.
    With only one distinct opening there is insufficient variation for a CI.
    """
    by_seed = {}
    for r in results:
        if agent in (r.ai_side0, r.ai_side1):
            by_seed.setdefault(r.opening_seed, []).append(r.points_for(agent))
    clusters = list(by_seed.values())
    if len(clusters) < 2 or samples < 1:
        return None, None
    rng = random.Random(seed + sum(ord(ch) for ch in agent))
    vals = []
    for _ in range(samples):
        sampled = [clusters[rng.randrange(len(clusters))]
                   for _ in range(len(clusters))]
        v = [x for group in sampled for x in group]
        vals.append(sum(v) / len(v))
    vals.sort()
    return (vals[int((len(vals) - 1) * .025)],
            vals[int((len(vals) - 1) * .975)])


def league_summary(results, moves: list[dict], agents: list[str],
                   bootstrap_samples: int = 1000) -> tuple[list[dict], list[dict]]:
    standings = []
    head_to_head = []
    for name in agents:
        p = performance_score(results, name)
        own_moves = [m for m in moves if m['agent'] == name]
        n = len(own_moves)
        full_depth_moves = ([m for m in own_moves
                             if m['target_depth'] != 0] if name != 'mcts' else [])
        ci_lo, ci_hi = bootstrap_wr_ci(results, name, bootstrap_samples)
        row = {
            'agent': name, **p,
            'score_w60': .6 * p['win_rate'] + .4 * p['margin_score'],
            'score_w80': .8 * p['win_rate'] + .2 * p['margin_score'],
            'wr_ci95_low': '' if ci_lo is None else round(ci_lo, 5),
            'wr_ci95_high': '' if ci_hi is None else round(ci_hi, 5),
            'mean_move_ms': (sum(m['decision_ms'] for m in own_moves) / n if n else 0),
            'mean_search_nodes': (sum(m['nodes'] for m in own_moves) / n if n else 0),
            'mean_bfs': (sum(m['bfs_calls'] for m in own_moves) / n if n else 0),
            'mean_completed_depth': (sum(m['search_depth'] for m in full_depth_moves)
                                     / len(full_depth_moves) if full_depth_moves else ''),
            'full_depth_rate': (sum(m['search_depth'] == m['target_depth']
                                    for m in full_depth_moves) / len(full_depth_moves)
                                if full_depth_moves else ''),
            'search_budget_hit_rate': (sum(m['timed_out'] for m in own_moves) / n
                                       if n else 0),
            'hard_limit_excess_rate': (sum(m['hard_limit_exceeded'] for m in own_moves) / n
                                       if n else 0),
            'mean_mcts_iterations': (sum(m['mcts_iterations'] for m in own_moves) / n
                                     if name == 'mcts' and n else ''),
        }
        standings.append(row)
    standings.sort(key=lambda r: (-r['score'], -r['win_rate'], r['agent']))
    for rank, r in enumerate(standings, start=1):
        r['rank'] = rank
    for score_key, rank_key in [('score_w60', 'rank_w60'),
                                ('score_w80', 'rank_w80')]:
        for index, r in enumerate(sorted(standings,
                                         key=lambda x: (-x[score_key], -x['win_rate'], x['agent'])), 1):
            r[rank_key] = index
    for a, b in combinations(agents, 2):
        subset = [r for r in results if set((r.ai_side0, r.ai_side1)) == {a, b}]
        scores_a = performance_score(subset, a)
        scores_b = performance_score(subset, b)
        if abs(scores_a['score'] + scores_b['score'] - 1.0) > 1e-9:
            raise AssertionError('Pairwise score must be complementary')
        head_to_head.append({
            'agent_a': a, 'agent_b': b, 'games': len(subset),
            'a_wins': scores_a['wins'], 'b_wins': scores_b['wins'],
            'draws': scores_a['draws'],
            'score_a_vs_b': scores_a['score'], 'score_b_vs_a': scores_b['score'],
            'win_rate_a': scores_a['win_rate'],
            'margin_mean_a': scores_a['average_margin'],
        })
    return standings, head_to_head


def run_league(agents: list[str], pairs: int, seed: int, opening_plies: int,
               max_plies: int, time_limit: float, hard_limit: float,
               root_k: int, reply_k: int | None, margin: int | None,
               bootstrap_samples: int = 1000):
    if len(agents) < 2 or len(set(agents)) != len(agents):
        raise ValueError('At least two unique agents required')
    if pairs < 1 or time_limit <= 0 or hard_limit < time_limit:
        raise ValueError('Invalid pairs/time-limit/hard-limit')
    common = dict(root_top_k=root_k, reply_top_k=reply_k,
                  wall_margin=margin, time_limit_s=time_limit,
                  iterative_deepening=True)
    games = []
    moves = []
    for index, (a, b) in enumerate(combinations(agents, 2)):
        print(f'\n[{index+1}/{len(agents)*(len(agents)-1)//2}] '
              f'{a} vs {b}', flush=True)
        results, rows = run_tournament(Options(mode=a, **common),
                                       Options(mode=b, **common),
                                       pairs=pairs, seed=seed,
                                       opening_plies=opening_plies,
                                       max_plies=max_plies,
                                       hard_limit_s=hard_limit,
                                       game_offset=2*pairs*index,
                                       quiet=False)
        games.extend(results)
        moves.extend(rows)
    standings, head_to_head = league_summary(games, moves, agents,
                                             bootstrap_samples)
    return games, moves, standings, head_to_head


def write_report(path: Path, standings: list[dict], pairs: int,
                 agents: list[str], budget_s: float, hard_s: float,
                 max_plies: int, root_k: int, reply_k: int | None,
                 margin: int | None):
    lines = [
        '# Quoridor league results', '',
        f'Participants: {", ".join(agents)}. Paired openings per matchup: {pairs}.',
        f'Search budget: {budget_s}s; measured-time forfeit: >{hard_s}s;',
        f'experimental max plies: {max_plies}.',
        f'Common beam: root top-k={root_k}, reply top-k={reply_k}, wall margin={margin}.',
        '',
        '| Rank | Agent | Games | W/L/D | WR | Margin score | Score (0.7/0.3) | Mean depth | Depth reached | Mean time (ms) |',
        '|---:|---|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in standings:
        avg_depth = r['mean_completed_depth']
        depth_rate = r['full_depth_rate']
        lines.append(f'| {r["rank"]} | {r["agent"]} | {r["games"]} | '
                     f'{r["wins"]}/{r["losses"]}/{r["draws"]} | '
                     f'{r["win_rate"]:.3f} | {r["margin_score"]:.3f} | '
                     f'{r["score"]:.3f} | '
                     f'{avg_depth:.2f} | {depth_rate:.1%} | '
                     f'{r["mean_move_ms"]:.1f} |'
                     if isinstance(avg_depth, float) else
                     f'| {r["rank"]} | {r["agent"]} | {r["games"]} | '
                     f'{r["wins"]}/{r["losses"]}/{r["draws"]} | '
                     f'{r["win_rate"]:.3f} | {r["margin_score"]:.3f} | '
                     f'{r["score"]:.3f} | n/a | n/a | '
                     f'{r["mean_move_ms"]:.1f} |')
    lines += [
        '',
        'Notes: score = 0.7 * (wins+0.5*draws)/games + '
        '0.3 * (1+mean signed margin)/2.',
        'Goal margin is +/-min(1, loser remaining BFS distance/8); '
        'forfeit +/-1; max-plies draw 0.',
        'This is a course experiment; max-plies draw and time-forfeit are '
        'experimental rules, not official Quoridor rules.',
        'Depth-3 and depth-2 may still make the SAME decisions even when depth-3 '
        'is completed. The pilot measures completion, not an assured difference.',
        'Alpha-beta alone does not guarantee higher win rate at fixed depth.',
        'MCTS, if included, is an experimental heuristic-guided UCT '
        '(no trained neural network). Its search depth is not comparable '
        'to Minimax ply depth.',
        'The per-move time limit is a cooperative Python check. It cannot '
        'preempt arbitrary system pauses; actual elapsed time > hard limit is '
        'scored as a forfeit.',
        'The standings CSV includes 0.6/0.4 and 0.8/0.2 weighted ranking '
        'sensitivity checks: score_w60, rank_w60, score_w80 and rank_w80.',
        'The 95% WR interval, when present in CSV, is a descriptive bootstrap '
        'over paired openings and is not a strong claim of statistical significance.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agents', nargs='+', choices=tuple(MODES),
                   default=list(LEAGUE_DEFAULT))
    p.add_argument('--include-mcts', action='store_true',
                   help='Append experimental 6th UCT agent')
    p.add_argument('--pairs', type=int, default=2,
                   help='Per pair matchup: 2*PAIRS games, swapping player sides')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--opening-plies', type=int, default=4)
    p.add_argument('--max-plies', type=int, default=200)
    p.add_argument('--time-limit', type=float, default=.2,
                   help='Common iterative search budget (s)')
    p.add_argument('--hard-limit', type=float, default=.25,
                   help='Measured decision time forfeit threshold (s)')
    p.add_argument('--pilot-config', type=Path, default=None,
                   help='Use recommended budget from pilot_budget.py JSON')
    p.add_argument('--root-k', type=int, default=8)
    p.add_argument('--reply-k', type=int, default=4)
    p.add_argument('--margin', type=int, default=2)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--output', type=Path, default=Path('results/league'))
    args = p.parse_args()
    agents = list(args.agents)
    if args.include_mcts and 'mcts' not in agents:
        agents.append('mcts')
    if args.pilot_config:
        config = json.loads(args.pilot_config.read_text(encoding='utf-8'))
        suggested = config.get('recommended_budget_s')
        if suggested is None:
            p.error('Pilot found NO eligible common time budget. Try more budgets '
                    'or change beam settings, then rerun pilot.')
        for field, cli in [('root_k', args.root_k), ('reply_k', args.reply_k),
                           ('wall_margin', args.margin)]:
            if config[field] != cli:
                p.error(f'Pilot setting {field}={config[field]} but league uses '
                        f'{cli}. Use identical beam settings.')
        args.time_limit = float(suggested)
        args.hard_limit = float(config['recommended_hard_limit_s'])
        print(f'Using pilot-selected common budget {args.time_limit}s; '
              f'hard limit {args.hard_limit}s')
    if args.max_plies < 1 or args.opening_plies < 0:
        p.error('max-plies must be positive, opening-plies nonnegative')
    games, moves, standings, head = run_league(
        agents, args.pairs, args.seed, args.opening_plies,
        args.max_plies, args.time_limit, args.hard_limit,
        args.root_k, args.reply_k, args.margin, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'games.csv', [g.to_row() for g in games])
    write_csv(args.output/'moves.csv', moves)
    write_csv(args.output/'standings.csv', standings)
    write_csv(args.output/'head_to_head.csv', head)
    config = {
        'agents': agents, 'pairs_per_matchup': args.pairs, 'seed': args.seed,
        'opening_plies': args.opening_plies, 'max_plies': args.max_plies,
        'search_budget_s': args.time_limit, 'measured_forfeit_s': args.hard_limit,
        'root_k': args.root_k, 'reply_k': args.reply_k, 'wall_margin': args.margin,
        'weighted_score': '0.7 * WR + 0.3 * (1 + mean_margin)/2',
        'pilot_config': str(args.pilot_config) if args.pilot_config else None,
    }
    (args.output/'config.json').write_text(json.dumps(config, ensure_ascii=False,
                                                       indent=2)+'\n', encoding='utf-8')
    write_report(args.output/'summary.md', standings, args.pairs, agents,
                 args.time_limit, args.hard_limit, args.max_plies,
                 args.root_k, args.reply_k, args.margin)
    print('\nFINAL STANDINGS (provisional until enough paired games):')
    for row in standings:
        print(f'{row["rank"]}. {row["agent"]}: score={row["score"]:.3f}, '
              f'WR={row["win_rate"]:.1%}, '
              f'mean margin={row["average_margin"]:+.3f}')
    print(f'\nSaved 4 CSV + config + summary to {args.output.resolve()}')


if __name__ == '__main__':
    main()
