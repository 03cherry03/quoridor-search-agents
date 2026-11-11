"""Integration boundary for an existing web game's server.

This file is RUNNABLE on its own, but is NOT a verified drop-in replacement
for Game_hub/quoridor/server/ai.py. The actual game's state fields, coordinate
system, orientation labels and response protocol must be checked before use.

Expected canonical JSON-compatible state:
{
  "pawns": [[0,4], [8,4]],
  "horizontal_walls": [[row,col], ...],
  "vertical_walls": [[row,col], ...],
  "remaining": [10,10],
  "turn": 0
}
"""
import json
import sys
from quoridor_ai import Agent, Options
from quoridor_engine import state_from_dict, state_to_dict, start_state


def choose_project_ai(state_payload: dict, mode: str = "minimax2_ab") -> dict:
    """Convert provided state to project engine; return action + diagnostics."""
    state = state_from_dict(state_payload)
    opts = Options(mode=mode, root_top_k=8, wall_margin=2,
                   reply_top_k=4 if mode == "minimax3_ab" else None)
    choice = Agent(opts).choose(state)
    return {"action": choice.action.as_dict(), "stats": choice.stats.as_dict()}


if __name__ == "__main__":
    # Demo with initial state. Can optionally accept a JSON file path.
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as file:
            payload = json.load(file)
    else:
        payload = state_to_dict(start_state())
    mode = sys.argv[2] if len(sys.argv) > 2 else "minimax2_ab"
    print(json.dumps(choose_project_ai(payload, mode), ensure_ascii=False, indent=2))
