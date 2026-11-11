"""Course-project agents: greedy, minimax and alpha-beta minimax.

Search depth counts individual turns (plies).
The BFS distance heuristic deliberately ignores pawn occupancy/jumps.
Root top-k filtering is a BEAM APPROXIMATION, not alpha-beta pruning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Optional

from quoridor_engine import Action, Rules, State, apply_legal, legal_pawn_moves, winner

MODES = {
    "greedy": (1, False),
    "minimax2": (2, False),
    "minimax2_ab": (2, True),
    "minimax3_ab": (3, True),
}
WIN_SCORE = 10000.0


class SearchTimeout(Exception):
    pass


@dataclass(frozen=True)
class Options:
    mode: str = "minimax2_ab"
    root_top_k: Optional[int] = 8
    reply_top_k: Optional[int] = None
    wall_margin: Optional[int] = 2
    time_limit_s: Optional[float] = None
    iterative_deepening: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {tuple(MODES)}")
        for value in (self.root_top_k, self.reply_top_k):
            if value is not None and value < 1:
                raise ValueError("top_k values must be positive or None")
        if self.wall_margin is not None and self.wall_margin < 0:
            raise ValueError("wall_margin must be positive or None")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive or None")


@dataclass
class SearchStats:
    nodes: int = 0                 # Search tree nodes entered (excluding root)
    leaves: int = 0                # Terminal or depth-zero search nodes
    evaluations: int = 0          # All calls of the heuristic (ranking included)
    cutoffs: int = 0              # Alpha-beta cutoff events
    ordered_actions: int = 0      # Candidate simulations for ranking
    bfs_calls: int = 0
    wall_checks: int = 0
    elapsed_ms: float = 0.0
    completed_depth: int = 0
    timed_out: bool = False
    search_value: Optional[float] = None

    def as_dict(self) -> dict:
        return vars(self).copy()


@dataclass
class Choice:
    action: Action
    stats: SearchStats = field(default_factory=SearchStats)


class Agent:
    def __init__(self, options: Options = Options()):
        self.options = options
        self.rules = Rules()
        self.stats = SearchStats()
        self._root = 0
        self._deadline = math.inf

    def _check(self):
        if time.perf_counter() >= self._deadline:
            raise SearchTimeout

    def _score(self, state: State) -> float:
        self.stats.evaluations += 1
        result = winner(state)
        if result is not None:
            return WIN_SCORE if result == self._root else -WIN_SCORE
        my_dist = self.rules.shortest_path_len(state, self._root, self._check)
        op_dist = self.rules.shortest_path_len(state, 1 - self._root, self._check)
        w0 = state.remaining[self._root]
        w1 = state.remaining[1 - self._root]
        return 10.0 * (op_dist - my_dist) + 0.5 * (w0 - w1)

    def _rank(self, state: State, actions: list[Action], limit: Optional[int],
              maximizing: bool) -> list[Action]:
        """Evaluate candidates for beam selection. Stable, deterministic ties."""
        if limit is None:
            return actions
        annotated = []
        for index, action in enumerate(actions):
            self._check()
            value = self._score(apply_legal(state, action))
            self.stats.ordered_actions += 1
            annotated.append(((-value if maximizing else value), index, action))
        annotated.sort()
        return [action for _, _, action in annotated[:limit]]

    def _search(self, state: State, remaining_depth: int, alpha: float,
                beta: float, use_ab: bool) -> float:
        self._check()
        self.stats.nodes += 1
        if remaining_depth == 0 or winner(state) is not None:
            self.stats.leaves += 1
            return self._score(state)

        maximizing = state.turn == self._root
        if self.options.reply_top_k is not None and remaining_depth > 1:
            # Beam-select only at intermediate nodes. At the final ply, lazily
            # iterate moves so alpha-beta can skip wall-legality BFS entirely.
            actions = self.rules.legal_actions(state, self.options.wall_margin, self._check)
            actions = self._rank(state, actions, self.options.reply_top_k, maximizing)
        else:
            # A lazy iterator is important: alpha-beta can skip expensive BFS
            # wall-legality checks for all remaining candidate actions.
            actions = self.rules.iter_legal_actions(state, self.options.wall_margin,
                                                    self._check)
        had_any = False
        if maximizing:
            value = -math.inf
            for action in actions:
                had_any = True
                self._check()
                child = apply_legal(state, action)
                value = max(value, self._search(child, remaining_depth - 1,
                                               alpha, beta, use_ab))
                alpha = max(alpha, value)
                if use_ab and alpha >= beta:
                    self.stats.cutoffs += 1
                    break
            return value if had_any else self._score(state)
        value = math.inf
        for action in actions:
            had_any = True
            self._check()
            child = apply_legal(state, action)
            value = min(value, self._search(child, remaining_depth - 1,
                                           alpha, beta, use_ab))
            beta = min(beta, value)
            if use_ab and beta <= alpha:
                self.stats.cutoffs += 1
                break
        return value if had_any else self._score(state)

    def _root_search(self, state: State, actions: list[Action], depth: int,
                     use_ab: bool) -> tuple[Action, float]:
        """Run one full-depth search. A timeout discards this partial result."""
        best = actions[0]
        best_value = -math.inf
        alpha = -math.inf
        for action in actions:
            self._check()
            child = apply_legal(state, action)
            value = self._search(child, depth - 1, alpha, math.inf, use_ab)
            if value > best_value:
                best_value, best = value, action
            alpha = max(alpha, best_value)
            # Win discovered; no candidate can improve upon WIN_SCORE.
            if best_value >= WIN_SCORE:
                break
        return best, best_value

    def choose(self, state: State) -> Choice:
        if winner(state) is not None:
            raise ValueError("Cannot select a move for a finished game")
        self.rules.clear_cache()
        self.stats = SearchStats()
        self._root = state.turn
        started = time.perf_counter()
        self._deadline = (started + self.options.time_limit_s
                          if self.options.time_limit_s else math.inf)

        # Always compute a legal, cheap fallback BEFORE starting timed search.
        pawns = legal_pawn_moves(state)
        if not pawns:
            raise ValueError("No legal pawn moves")
        # Side 0 moves toward row 8; side 1 toward row 0.
        fallback = min(pawns, key=lambda a: (abs(a.row - (8 if self._root == 0 else 0)),
                                            abs(a.col - 4), a.row, a.col))
        best = fallback
        try:
            if self.options.mode == "greedy":
                actions = self.rules.legal_actions(state, self.options.wall_margin, self._check)
                ranked = self._rank(state, actions, 1, True)
                # _rank(..., 1) determines the one-ply maximum.
                best = ranked[0]
                self.stats.completed_depth = 1
                self.stats.search_value = self._score(apply_legal(state, best))
            else:
                depth, pruning = MODES[self.options.mode]
                actions = self.rules.legal_actions(state, self.options.wall_margin, self._check)
                # Root beam approximation matches the team's original idea.
                actions = self._rank(state, actions, self.options.root_top_k, True)
                depths = (range(1, depth + 1) if self.options.iterative_deepening
                          else (depth,))
                for d in depths:
                    candidate, val = self._root_search(state, actions, d, pruning)
                    best = candidate
                    self.stats.search_value = val
                    self.stats.completed_depth = d
        except SearchTimeout:
            self.stats.timed_out = True
        finally:
            self.stats.elapsed_ms = (time.perf_counter() - started) * 1000
            self.stats.bfs_calls = self.rules.bfs_calls
            self.stats.wall_checks = self.rules.wall_checks
        return Choice(best, self.stats)


def select_move(state: State, mode: str = "minimax2_ab", **kwargs) -> Choice:
    """Convenience API: select_move(state, mode='minimax2_ab', wall_margin=2)."""
    return Agent(Options(mode=mode, **kwargs)).choose(state)
