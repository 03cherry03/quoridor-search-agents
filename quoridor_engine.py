"""Quoridor rules for the AI course comparison project.

9x9 board, 10 walls/player. All states are immutable. Wall locations are
8x8 anchors; H blocks vertical travel, V blocks horizontal travel.
No external Python dependencies.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

N = 9
ANCHORS = 8
GOAL_ROWS = (8, 0)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True, slots=True)
class Action:
    kind: str                 # "pawn" or "wall"
    row: int
    col: int
    orientation: str = ""    # "h" or "v" for a wall

    def __post_init__(self):
        if self.kind not in ("pawn", "wall"):
            raise ValueError("Action.kind must be 'pawn' or 'wall'")
        if self.kind == "wall" and self.orientation not in ("h", "v"):
            raise ValueError("Wall orientation must be 'h' or 'v'")

    def as_dict(self) -> dict:
        if self.kind == "pawn":
            return {"type": "move", "row": self.row, "col": self.col}
        return {"type": "wall", "row": self.row,
                "col": self.col, "orientation": self.orientation}

    def __str__(self) -> str:
        return (f"pawn({self.row},{self.col})" if self.kind == "pawn"
                else f"wall({self.row},{self.col},{self.orientation})")


@dataclass(frozen=True, slots=True)
class State:
    pawns: tuple[tuple[int, int], tuple[int, int]] = ((0, 4), (8, 4))
    horizontal: int = 0            # bits 0..63: 8*r + c
    vertical: int = 0
    remaining: tuple[int, int] = (10, 10)
    turn: int = 0

    def __post_init__(self):
        if self.turn not in (0, 1):
            raise ValueError("turn must be 0 or 1")
        if len(self.pawns) != 2 or any(not (0 <= r < N and 0 <= c < N)
                                      for r, c in self.pawns):
            raise ValueError("pawns must be two valid board coordinates")
        if self.pawns[0] == self.pawns[1]:
            raise ValueError("pawns cannot occupy the same square")
        if len(self.remaining) != 2 or any(not (0 <= w <= 10) for w in self.remaining):
            raise ValueError("remaining wall counts must be between 0 and 10")
        if self.horizontal < 0 or self.vertical < 0 or self.horizontal >= 1 << 64 or self.vertical >= 1 << 64:
            raise ValueError("wall bitmasks must be nonnegative 64-bit masks")


def start_state() -> State:
    return State()


def winner(state: State) -> Optional[int]:
    if state.pawns[0][0] == GOAL_ROWS[0]:
        return 0
    if state.pawns[1][0] == GOAL_ROWS[1]:
        return 1
    return None


def _bit(row: int, col: int) -> int:
    return 1 << (ANCHORS * row + col)


def blocked(state: State, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether a wall blocks two adjacent squares (not pawn occupancy)."""
    r, c = a
    rr, cc = b
    if abs(rr - r) + abs(cc - c) != 1:
        raise ValueError("blocked() expects adjacent squares")
    if rr != r:
        row = min(r, rr)
        return bool((c < ANCHORS and state.horizontal & _bit(row, c)) or
                    (c > 0 and state.horizontal & _bit(row, c - 1)))
    col = min(c, cc)
    return bool((r < ANCHORS and state.vertical & _bit(r, col)) or
                (r > 0 and state.vertical & _bit(r - 1, col)))


def neighbors(state: State, cell: tuple[int, int]):
    r, c = cell
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < N and 0 <= cc < N and not blocked(state, cell, (rr, cc)):
            yield rr, cc


def legal_pawn_moves(state: State, player: Optional[int] = None) -> list[Action]:
    """Includes standard moves, straight jumps and forced diagonal jumps."""
    if winner(state) is not None:
        return []
    me = state.turn if player is None else player
    at = state.pawns[me]
    other = state.pawns[1 - me]
    result = set()
    for dr, dc in DIRS:
        nr, nc = at[0] + dr, at[1] + dc
        neighbor = (nr, nc)
        if not (0 <= nr < N and 0 <= nc < N):
            continue
        if blocked(state, at, neighbor):
            continue
        if neighbor != other:
            result.add(neighbor)
            continue
        # Opponent is adjacent. Jump straight if possible, otherwise diagonal.
        back = (other[0] + dr, other[1] + dc)
        if (0 <= back[0] < N and 0 <= back[1] < N
                and not blocked(state, other, back)):
            result.add(back)
        else:
            sides = ((0, -1), (0, 1)) if dr else ((-1, 0), (1, 0))
            for sr, sc in sides:
                dest = (other[0] + sr, other[1] + sc)
                if (0 <= dest[0] < N and 0 <= dest[1] < N
                        and not blocked(state, other, dest)):
                    result.add(dest)
    return [Action("pawn", r, c) for r, c in sorted(result)]


def shape_fits(state: State, row: int, col: int, orientation: str) -> bool:
    """Cheap wall overlap and crossing test; DOES NOT test paths."""
    if not (0 <= row < ANCHORS and 0 <= col < ANCHORS):
        return False
    if orientation == "h":
        if state.vertical & _bit(row, col):
            return False
        for cc in (col - 1, col, col + 1):
            if 0 <= cc < ANCHORS and state.horizontal & _bit(row, cc):
                return False
        return True
    if orientation == "v":
        if state.horizontal & _bit(row, col):
            return False
        for rr in (row - 1, row, row + 1):
            if 0 <= rr < ANCHORS and state.vertical & _bit(rr, col):
                return False
        return True
    return False


def _install_unchecked(state: State, row: int, col: int, orientation: str,
                       player: int) -> State:
    rem = list(state.remaining)
    rem[player] -= 1
    return State(
        pawns=state.pawns,
        horizontal=state.horizontal | (_bit(row, col) if orientation == "h" else 0),
        vertical=state.vertical | (_bit(row, col) if orientation == "v" else 0),
        remaining=tuple(rem),
        turn=1 - player,
    )


class Rules:
    """Rule checks, BFS, and per-instance BFS memoization/statistics."""

    def __init__(self):
        self._dist_cache: dict[tuple, int] = {}
        self.bfs_calls = 0
        self.wall_checks = 0

    def clear_cache(self):
        self._dist_cache.clear()
        self.bfs_calls = 0
        self.wall_checks = 0

    def shortest_path_len(self, state: State, player: int,
                          check: Optional[Callable[[], None]] = None) -> int:
        """BFS shortest edge distance to player's goal; ignores other pawn."""
        pos = state.pawns[player]
        key = (pos, GOAL_ROWS[player], state.horizontal, state.vertical)
        if key in self._dist_cache:
            return self._dist_cache[key]
        self.bfs_calls += 1
        queue = deque([(pos, 0)])
        seen = {pos}
        while queue:
            if check is not None:
                check()
            cell, steps = queue.popleft()
            if cell[0] == GOAL_ROWS[player]:
                self._dist_cache[key] = steps
                return steps
            for nxt in neighbors(state, cell):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, steps + 1))
        self._dist_cache[key] = 10000
        return 10000

    def can_place_wall(self, state: State, row: int, col: int,
                       orientation: str, player: Optional[int] = None,
                       check: Optional[Callable[[], None]] = None) -> bool:
        side = state.turn if player is None else player
        if winner(state) is not None or state.remaining[side] == 0:
            return False
        if not shape_fits(state, row, col, orientation):
            return False
        self.wall_checks += 1
        trial = _install_unchecked(state, row, col, orientation, side)
        return (self.shortest_path_len(trial, 0, check) < 10000 and
                self.shortest_path_len(trial, 1, check) < 10000)

    def iter_legal_actions(self, state: State, wall_margin: Optional[int] = 2,
                           check: Optional[Callable[[], None]] = None):
        """LAZY legal moves: alpha-beta can stop BEFORE BFS wall checks."""
        if winner(state) is not None:
            return
        yield from legal_pawn_moves(state)
        if state.remaining[state.turn] == 0:
            return
        for r in range(ANCHORS):
            for c in range(ANCHORS):
                if check is not None:
                    check()
                if wall_margin is not None and not any(
                    max(abs(r - pr), abs(c - pc)) <= wall_margin
                    for pr, pc in state.pawns
                ):
                    continue
                for orientation in ("h", "v"):
                    if self.can_place_wall(state, r, c, orientation,
                                           check=check):
                        yield Action("wall", r, c, orientation)

    def legal_actions(self, state: State, wall_margin: Optional[int] = 2,
                      check: Optional[Callable[[], None]] = None) -> list[Action]:
        return list(self.iter_legal_actions(state, wall_margin, check))

    def apply(self, state: State, action: Action) -> State:
        """Validated public transition; safe for referee/game loop."""
        if winner(state) is not None:
            raise ValueError("Game has ended")
        if action.kind == "pawn":
            if action not in legal_pawn_moves(state):
                raise ValueError(f"Illegal pawn move: {action}")
        elif not self.can_place_wall(state, action.row, action.col,
                                     action.orientation):
            raise ValueError(f"Illegal wall placement: {action}")
        return apply_legal(state, action)


def apply_legal(state: State, action: Action) -> State:
    """Fast transition for actions already checked by Rules/legal_pawn_moves."""
    if action.kind == "wall":
        return _install_unchecked(state, action.row, action.col,
                                  action.orientation, state.turn)
    ps = list(state.pawns)
    ps[state.turn] = (action.row, action.col)
    return State(pawns=tuple(ps), horizontal=state.horizontal,
                 vertical=state.vertical, remaining=state.remaining,
                 turn=1 - state.turn)


def state_to_dict(state: State) -> dict:
    return {
        "pawns": [list(x) for x in state.pawns],
        "horizontal_walls": [[r, c] for r in range(8) for c in range(8)
                             if state.horizontal & _bit(r, c)],
        "vertical_walls": [[r, c] for r in range(8) for c in range(8)
                           if state.vertical & _bit(r, c)],
        "remaining": list(state.remaining),
        "turn": state.turn,
    }


def state_from_dict(value: dict) -> State:
    h = 0
    v = 0
    for r, c in value.get("horizontal_walls", []):
        if not (0 <= r < 8 and 0 <= c < 8):
            raise ValueError("horizontal wall anchor out of range")
        h |= _bit(r, c)
    for r, c in value.get("vertical_walls", []):
        if not (0 <= r < 8 and 0 <= c < 8):
            raise ValueError("vertical wall anchor out of range")
        v |= _bit(r, c)
    return State(
        pawns=tuple(tuple(x) for x in value["pawns"]),
        horizontal=h,
        vertical=v,
        remaining=tuple(value["remaining"]),
        turn=int(value["turn"]),
    )
