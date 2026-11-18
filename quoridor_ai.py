"""Quoridor agents for a controlled algorithms comparison.

All minimax variants share the SAME root-perspective evaluation, legal moves,
wall scope and beam settings. Beam limits are approximations, NOT alpha-beta.
A depth is a ply (one player's action). Time-bounded agents use iterative
 deepening and accept only FULLY completed iterations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Optional

from quoridor_engine import Action, Rules, State, apply_legal, legal_pawn_moves, winner

# (Maximum depth in plies, alpha-beta enabled). MCTS is an optional sixth agent.
MODES = {
    "greedy": (1, False),
    "minimax2": (2, False),
    "minimax3": (3, False),
    "minimax2_ab": (2, True),
    "minimax3_ab": (3, True),
    "mcts": (0, False),
}
WIN_SCORE = 10000.0


class SearchTimeout(Exception):
    """Internal signal: the current search depth is INCOMPLETE."""


@dataclass(frozen=True)
class Options:
    mode: str = "minimax2_ab"
    root_top_k: Optional[int] = 8
    reply_top_k: Optional[int] = 4
    wall_margin: Optional[int] = 2
    time_limit_s: Optional[float] = None
    iterative_deepening: bool = False
    wall_weight: float = 0.5
    seed: int = 42
    mcts_iterations: int = 128
    mcts_rollout_plies: int = 4
    mcts_exploration: float = 1.4

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {tuple(MODES)}")
        for k in (self.root_top_k, self.reply_top_k):
            if k is not None and k < 1:
                raise ValueError("top-k must be positive or None")
        if self.wall_margin is not None and self.wall_margin < 0:
            raise ValueError("wall_margin must be nonnegative or None")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive or None")
        if self.mcts_iterations < 1 or self.mcts_rollout_plies < 0:
            raise ValueError("Invalid MCTS settings")
        if self.mcts_exploration < 0:
            raise ValueError("MCTS exploration constant must be nonnegative")


@dataclass
class SearchStats:
    nodes: int = 0
    leaves: int = 0
    evaluations: int = 0
    cutoffs: int = 0
    ordered_actions: int = 0
    bfs_calls: int = 0
    wall_checks: int = 0
    elapsed_ms: float = 0.0
    completed_depth: int = 0
    timed_out: bool = False
    search_value: Optional[float] = None
    mcts_iterations: int = 0

    def as_dict(self) -> dict:
        return vars(self).copy()


@dataclass
class Choice:
    action: Action
    stats: SearchStats = field(default_factory=SearchStats)


@dataclass
class _MCNode:
    state: State
    parent: Optional['_MCNode'] = None
    action: Optional[Action] = None
    untried: Optional[list[Action]] = None
    children: list['_MCNode'] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0


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

    def _score(self, state: State, k: int = 0) -> float:
        """Exactly the team's PDF heuristic (from root player's perspective).

        Terminal +/- (10000 + remaining depth); otherwise:
        D_opp - D_me + wall_weight * (W_me - W_opp) + tau,
        where tau is +0.5 for the root's turn, -0.5 for opponent's turn.
        BFS ignores the other pawn, by design.
        """
        self.stats.evaluations += 1
        outcome = winner(state)
        if outcome is not None:
            return (WIN_SCORE + k) if outcome == self._root else -(WIN_SCORE + k)
        my_dist = self.rules.shortest_path_len(state, self._root, self._check)
        opp_dist = self.rules.shortest_path_len(state, 1 - self._root, self._check)
        wall_diff = state.remaining[self._root] - state.remaining[1 - self._root]
        tau = 0.5 if state.turn == self._root else -0.5
        return (opp_dist - my_dist) + self.options.wall_weight * wall_diff + tau

    def _rank(self, state: State, actions: list[Action], limit: Optional[int],
              maximizing: bool) -> list[Action]:
        """Stable one-ply greedy ordering, optionally selecting top k actions.

        At every node use the same beam setting in plain and alpha-beta modes.
        NOTE: ranking all candidates costs time/BFS even if later pruned.
        """
        if limit is None:
            return actions
        annotated = []
        for index, action in enumerate(actions):
            self._check()
            value = self._score(apply_legal(state, action), 0)
            self.stats.ordered_actions += 1
            annotated.append(((-value if maximizing else value), index, action))
        annotated.sort()
        return [a for _, _, a in annotated[:limit]]

    def _search(self, state: State, remaining_depth: int, alpha: float,
                beta: float, use_ab: bool) -> float:
        self._check()
        self.stats.nodes += 1
        if remaining_depth == 0 or winner(state) is not None:
            self.stats.leaves += 1
            return self._score(state, remaining_depth)
        maximizing = state.turn == self._root
        if self.options.reply_top_k is None:
            # Generate lazily: AB can avoid expensive wall-legality checking.
            actions = self.rules.iter_legal_actions(state, self.options.wall_margin,
                                                    self._check)
        else:
            all_actions = self.rules.legal_actions(state, self.options.wall_margin,
                                                   self._check)
            actions = self._rank(state, all_actions, self.options.reply_top_k,
                                 maximizing)
        had_any = False
        if maximizing:
            value = -math.inf
            for action in actions:
                had_any = True
                self._check()
                value = max(value, self._search(apply_legal(state, action),
                                                remaining_depth - 1, alpha, beta,
                                                use_ab))
                if use_ab:
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        self.stats.cutoffs += 1
                        break
            return value if had_any else self._score(state, remaining_depth)
        value = math.inf
        for action in actions:
            had_any = True
            self._check()
            value = min(value, self._search(apply_legal(state, action),
                                            remaining_depth - 1, alpha, beta,
                                            use_ab))
            if use_ab:
                beta = min(beta, value)
                if beta <= alpha:
                    self.stats.cutoffs += 1
                    break
        return value if had_any else self._score(state, remaining_depth)

    def _root_search(self, state: State, actions: list[Action], depth: int,
                     use_ab: bool) -> tuple[Action, float]:
        """Only a *fully completed* return may replace the prior depth."""
        best, best_value = actions[0], -math.inf
        alpha = -math.inf
        for action in actions:
            self._check()
            value = self._search(apply_legal(state, action), depth - 1,
                                 alpha, math.inf, use_ab)
            if value > best_value:
                best, best_value = action, value
            if use_ab:
                alpha = max(alpha, best_value)
        return best, best_value

    def _mcts_rollout(self, state: State, rng: random.Random) -> float:
        """Short, stochastic playout; heuristic evaluates truncated endings.

        This is *lightweight heuristic MCTS*, not AlphaZero/learned MCTS.
        Use a pawn-biased playout plus sampled legal wall actions to stay fast.
        """
        s = state
        for _ in range(self.options.mcts_rollout_plies):
            self._check()
            if winner(s) is not None:
                break
            actions = legal_pawn_moves(s)
            if s.remaining[s.turn] and rng.random() < 0.30:
                # Sample local walls, checking each actual game's legal path.
                for _ in range(3):
                    pawn = s.pawns[rng.randrange(2)]
                    r = max(0, min(7, pawn[0] + rng.randint(-2, 1)))
                    c = max(0, min(7, pawn[1] + rng.randint(-2, 1)))
                    o = rng.choice(('h', 'v'))
                    if self.rules.can_place_wall(s, r, c, o, check=self._check):
                        actions.append(Action('wall', r, c, o))
                        break
            s = apply_legal(s, rng.choice(actions))
        final_winner = winner(s)
        if final_winner is not None:
            return 1.0 if final_winner == self._root else -1.0
        return math.tanh(self._score(s) / 4.0)

    def _mcts_actions(self, state: State, root: bool) -> list[Action]:
        all_actions = self.rules.legal_actions(state, self.options.wall_margin,
                                               self._check)
        cap = self.options.root_top_k if root else self.options.reply_top_k
        return self._rank(state, all_actions, cap, state.turn == self._root)

    def _mcts(self, state: State, fallback: Action) -> Action:
        """Adversarial UCT: root maximizes reward; opponent minimizes it."""
        rng = random.Random(self.options.seed + state.horizontal * 7
                            + state.vertical * 31 + state.pawns[0][0] * 97
                            + state.pawns[1][1] * 113 + state.turn)
        root = _MCNode(state)
        # Save a legal action before any potentially costly expansion.
        best = fallback
        for _ in range(self.options.mcts_iterations):
            try:
                self._check()
                node = root
                path = [node]
                # Select / expand a single new child.
                while winner(node.state) is None:
                    self._check()
                    if node.untried is None:
                        node.untried = self._mcts_actions(node.state, node is root)
                    if node.untried:
                        index = rng.randrange(len(node.untried))
                        action = node.untried.pop(index)
                        node = _MCNode(apply_legal(node.state, action), node, action)
                        path[-1].children.append(node)
                        path.append(node)
                        self.stats.nodes += 1
                        break
                    if not node.children:
                        break
                    sign = 1 if node.state.turn == self._root else -1
                    log_n = math.log(max(2, node.visits))
                    node = max(node.children, key=lambda child:
                               sign * (child.total_value / child.visits)
                               + self.options.mcts_exploration
                               * math.sqrt(log_n / child.visits))
                    path.append(node)
                reward = self._mcts_rollout(node.state, rng)
                self.stats.leaves += 1
                for visited in path:
                    visited.visits += 1
                    visited.total_value += reward
                self.stats.mcts_iterations += 1
                if root.children:
                    # Selected by most visits, then highest root-perspective value.
                    chosen = max(root.children, key=lambda c:
                                 (c.visits, c.total_value / c.visits))
                    best = chosen.action
                    self.stats.search_value = chosen.total_value / chosen.visits
            except SearchTimeout:
                self.stats.timed_out = True
                break
        return best

    def choose(self, state: State) -> Choice:
        if winner(state) is not None:
            raise ValueError("Cannot select a move after game over")
        self.rules.clear_cache()
        self.stats = SearchStats()
        self._root = state.turn
        started = time.perf_counter()
        self._deadline = (started + self.options.time_limit_s
                          if self.options.time_limit_s is not None else math.inf)

        pawns = legal_pawn_moves(state)
        if not pawns:
            raise ValueError("No legal pawn moves")
        goal = 8 if self._root == 0 else 0
        fallback = min(pawns, key=lambda a:
                       (abs(a.row - goal), abs(a.col - 4), a.row, a.col))
        best = fallback
        try:
            if self.options.mode == "mcts":
                best = self._mcts(state, fallback)
            elif self.options.mode == "greedy":
                actions = self.rules.legal_actions(state, self.options.wall_margin,
                                                   self._check)
                best = self._rank(state, actions, 1, True)[0]
                self.stats.completed_depth = 1
                self.stats.search_value = self._score(apply_legal(state, best))
            else:
                max_depth, pruning = MODES[self.options.mode]
                actions = self.rules.legal_actions(state, self.options.wall_margin,
                                                   self._check)
                actions = self._rank(state, actions, self.options.root_top_k,
                                     maximizing=True)
                # Always iterative under time budget, even without the flag.
                iterative = (self.options.iterative_deepening
                             or self.options.time_limit_s is not None)
                depths = range(1, max_depth + 1) if iterative else (max_depth,)
                for depth in depths:
                    candidate, val = self._root_search(state, actions, depth, pruning)
                    best = candidate
                    self.stats.search_value = val
                    self.stats.completed_depth = depth
        except SearchTimeout:
            self.stats.timed_out = True
        finally:
            self.stats.elapsed_ms = (time.perf_counter() - started) * 1000
            self.stats.bfs_calls = self.rules.bfs_calls
            self.stats.wall_checks = self.rules.wall_checks
        return Choice(best, self.stats)


def select_move(state: State, mode: str = "minimax2_ab", **kwargs) -> Choice:
    return Agent(Options(mode=mode, **kwargs)).choose(state)
