"""Refereed AI vs AI games with paired openings and PDF-defined margins.

The wall/pawn game engine is local: this does NOT change the team's web server.
Game-level 0.25 s limit is a *measured forfeit rule*, not an OS preemption.
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from quoridor_ai import Agent, Options, MODES
from quoridor_engine import Rules, State, start_state, winner


@dataclass
class MatchResult:
    game_id: int
    opening_seed: int
    ai_side0: str
    ai_side1: str
    winning_side: Optional[int]
    total_plies: int
    end_state: State
    finish_reason: str  # 'goal', 'max_plies_draw', 'time_forfeit'
    fault_side: Optional[int]
    avg_ms: float
    total_bfs: int
    total_evaluations: int
    total_nodes: int
    timed_search_moves: int
    matchup: str = ''

    @property
    def winning_ai(self) -> str:
        if self.winning_side is None:
            return 'draw'
        return self.ai_side0 if self.winning_side == 0 else self.ai_side1

    @property
    def margin_side0(self) -> float:
        """Team PDF: winner margin +loser's remaining BFS distance/8.

        Time win/loss: +1/-1. Max-plies draw: 0. The other side's margin
        is exactly the negative. Clipping is intentional for distances >8.
        """
        if self.winning_side is None:
            return 0.0
        if self.finish_reason == 'time_forfeit':
            return 1.0 if self.winning_side == 0 else -1.0
        rules = Rules()
        if self.winning_side == 0:
            return min(1.0, rules.shortest_path_len(self.end_state, 1) / 8.0)
        return -min(1.0, rules.shortest_path_len(self.end_state, 0) / 8.0)

    def margin_for(self, name: str) -> float:
        if name == self.ai_side0:
            return self.margin_side0
        if name == self.ai_side1:
            return -self.margin_side0
        raise ValueError(f'{name} did not play this game')

    def points_for(self, name: str) -> float:
        return 0.5 if self.winning_side is None else (1.0 if self.winning_ai == name else 0.0)

    def to_row(self) -> dict:
        return {
            'game_id': self.game_id, 'matchup': self.matchup,
            'opening_seed': self.opening_seed,
            'ai_side0': self.ai_side0, 'ai_side1': self.ai_side1,
            'winner_side': 'draw' if self.winning_side is None else self.winning_side,
            'winner_ai': self.winning_ai, 'finish_reason': self.finish_reason,
            'fault_side': '' if self.fault_side is None else self.fault_side,
            'margin_side0': self.margin_side0,
            'margin_side1': -self.margin_side0,
            'total_plies': self.total_plies,
            'avg_move_ms': round(self.avg_ms, 4),
            'total_bfs_calls': self.total_bfs,
            'total_evaluations': self.total_evaluations,
            'total_search_nodes': self.total_nodes,
            'timed_search_moves': self.timed_search_moves,
        }


def performance_score(results: list[MatchResult], agent: str) -> dict:
    """PDF metric: 0.7*WR + 0.3*(mean margin + 1)/2.

    Win=1, draw=0.5, loss=0. Margin is symmetric for each game.
    This is NOT the search evaluation used to choose an action.
    """
    games = [r for r in results if agent in (r.ai_side0, r.ai_side1)]
    if not games:
        raise ValueError(f'No games for {agent}')
    wins = sum(r.winning_ai == agent for r in games)
    losses = sum(r.winning_ai not in ('draw', agent) for r in games)
    draws = len(games) - wins - losses
    wr = (wins + .5 * draws) / len(games)
    avg_margin = sum(r.margin_for(agent) for r in games) / len(games)
    smargin = (1.0 + avg_margin) / 2.0
    score = 0.7 * wr + 0.3 * smargin
    return dict(games=len(games), wins=wins, losses=losses, draws=draws,
                win_rate=wr, average_margin=avg_margin,
                margin_score=smargin, score=score,
                forfeit_wins=sum(r.finish_reason == 'time_forfeit' and
                                 r.winning_ai == agent for r in games),
                forfeit_losses=sum(r.finish_reason == 'time_forfeit' and
                                   r.winning_ai != agent for r in games))


def random_opening(seed: int, num_plies: int = 4) -> State:
    """Generate one reproducible legal opening; it is reused with sides swapped."""
    if num_plies < 0:
        raise ValueError('num_plies must be nonnegative')
    rng = random.Random(seed)
    s = start_state()
    rules = Rules()
    for _ in range(num_plies):
        if winner(s) is not None:
            break
        actions = rules.legal_actions(s, wall_margin=2)
        pawns = [a for a in actions if a.kind == 'pawn']
        walls = [a for a in actions if a.kind == 'wall']
        pool = walls if walls and rng.random() < .35 else pawns
        s = rules.apply(s, rng.choice(pool))
    return s


def play(state: State, side0: Agent, side1: Agent,
         game_id: int, seed: int, max_plies: int,
         name0: str, name1: str, hard_limit_s: Optional[float] = None,
         matchup: str = '') -> tuple[MatchResult, list[dict]]:
    """Play one game; exceeding hard_limit_s causes immediate time forfeit."""
    if max_plies <= 0:
        raise ValueError('max_plies must be positive')
    if hard_limit_s is not None and hard_limit_s <= 0:
        raise ValueError('hard_limit_s must be positive')
    agents = [side0, side1]
    names = [name0, name1]
    referee = Rules()
    rows: list[dict] = []
    outcome, reason, fault = winner(state), 'goal', None
    for ply in range(max_plies):
        if winner(state) is not None:
            break
        side = state.turn
        choice = agents[side].choose(state)
        st = choice.stats
        time_forfeit = (hard_limit_s is not None and
                        st.elapsed_ms > hard_limit_s * 1000)
        next_state = state if time_forfeit else referee.apply(state, choice.action)
        rows.append({
            'game_id': game_id, 'matchup': matchup,
            'opening_seed': seed, 'ply': ply + 1,
            'side': side, 'agent': names[side], 'action': str(choice.action),
            'applied': int(not time_forfeit), 'hard_limit_exceeded': int(time_forfeit),
            'decision_ms': round(st.elapsed_ms, 4),
            'search_depth': st.completed_depth, 'target_depth': MODES[names[side]][0],
            'timed_out': int(st.timed_out), 'value': st.search_value,
            'nodes': st.nodes, 'leaves': st.leaves,
            'evaluations': st.evaluations, 'bfs_calls': st.bfs_calls,
            'wall_checks': st.wall_checks, 'cutoffs': st.cutoffs,
            'mcts_iterations': st.mcts_iterations,
            'root_top_k': agents[side].options.root_top_k,
            'reply_top_k': agents[side].options.reply_top_k,
            'wall_margin': agents[side].options.wall_margin,
        })
        if time_forfeit:
            outcome, reason, fault = 1 - side, 'time_forfeit', side
            break
        state = next_state
        outcome = winner(state)
        if outcome is not None:
            reason = 'goal'
            break
    else:
        outcome, reason = None, 'max_plies_draw'
    if outcome is None and reason == 'goal':
        reason = 'max_plies_draw'
    completed_moves = sum(row['applied'] for row in rows)
    result = MatchResult(
        game_id=game_id, opening_seed=seed, ai_side0=name0, ai_side1=name1,
        winning_side=outcome, total_plies=completed_moves,
        end_state=state, finish_reason=reason, fault_side=fault,
        avg_ms=sum(row['decision_ms'] for row in rows) / max(1, len(rows)),
        total_bfs=sum(row['bfs_calls'] for row in rows),
        total_evaluations=sum(row['evaluations'] for row in rows),
        total_nodes=sum(row['nodes'] for row in rows),
        timed_search_moves=sum(row['timed_out'] for row in rows), matchup=matchup)
    return result, rows


def run_tournament(a: Options, b: Options, pairs: int = 2,
                   seed: int = 42, opening_plies: int = 4,
                   max_plies: int = 200, hard_limit_s: Optional[float] = None,
                   game_offset: int = 0, quiet: bool = False):
    if pairs < 1:
        raise ValueError('pairs must be positive')
    results, move_rows = [], []
    matchup = f'{a.mode}_vs_{b.mode}'
    for pair in range(pairs):
        opening_seed = seed + pair
        opening = random_opening(opening_seed, opening_plies)
        for swapped in (False, True):
            first, second = (a, b) if not swapped else (b, a)
            game_id = game_offset + 2 * pair + 1 + int(swapped)
            result, moves = play(opening, Agent(first), Agent(second), game_id,
                                 opening_seed, max_plies, first.mode, second.mode,
                                 hard_limit_s=hard_limit_s, matchup=matchup)
            results.append(result)
            move_rows.extend(moves)
            if not quiet:
                print(f'{matchup} game {game_id:03d}: winner {result.winning_ai}; '
                      f'{result.finish_reason}; plies={result.total_plies}; '
                      f'avg={result.avg_ms:.1f}ms', flush=True)
    return results, move_rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(results: list[MatchResult], moves: list[dict], names: tuple[str, ...]) -> str:
    out = [f'Total games={len(results)}, draws={sum(r.winning_side is None for r in results)}']
    for name in dict.fromkeys(names):
        p = performance_score(results, name)
        own_moves = [r for r in moves if r['agent'] == name]
        avg_ms = sum(r['decision_ms'] for r in own_moves) / max(1, len(own_moves))
        avg_depth = sum(r['search_depth'] for r in own_moves) / max(1, len(own_moves))
        out.append(f'{name}: W/L/D={p["wins"]}/{p["losses"]}/{p["draws"]}; '
                   f'WR={p["win_rate"]:.3f}; mean_margin={p["average_margin"]:.3f}; '
                   f'Score={p["score"]:.3f}; mean_ms={avg_ms:.1f}; '
                   f'mean_completed_depth={avg_depth:.2f}')
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a', choices=tuple(MODES), default='minimax2')
    p.add_argument('--b', choices=tuple(MODES), default='minimax2_ab')
    p.add_argument('--pairs', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--opening-plies', type=int, default=4)
    p.add_argument('--max-plies', type=int, default=200)
    p.add_argument('--root-k', type=int, default=8)
    p.add_argument('--reply-k', type=int, default=4)
    p.add_argument('--margin', type=int, default=2)
    p.add_argument('--all-walls', action='store_true')
    p.add_argument('--time-limit', type=float, default=None,
                   help='search budget (s); enables iterative deepening')
    p.add_argument('--hard-limit', type=float, default=None,
                   help='per-move actual-time forfeit threshold (s)')
    p.add_argument('--output', type=Path, default=Path('results/pair'))
    args = p.parse_args()
    if args.a == args.b:
        p.error('Two distinct agent modes are required')
    if args.time_limit is not None and args.hard_limit is not None and args.hard_limit < args.time_limit:
        p.error('hard-limit must be >= time-limit')
    common = dict(root_top_k=args.root_k, reply_top_k=args.reply_k,
                  wall_margin=None if args.all_walls else args.margin,
                  time_limit_s=args.time_limit,
                  iterative_deepening=args.time_limit is not None)
    results, moves = run_tournament(Options(args.a, **common),
                                    Options(args.b, **common),
                                    args.pairs, args.seed, args.opening_plies,
                                    args.max_plies, args.hard_limit)
    write_csv(args.output / 'games.csv', [r.to_row() for r in results])
    write_csv(args.output / 'moves.csv', moves)
    report = summary(results, moves, (args.a, args.b))
    (args.output / 'summary.txt').write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
