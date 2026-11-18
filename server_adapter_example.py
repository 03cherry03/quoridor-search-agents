"""Demonstration JSON-state -> action adapter for a hypothetical web server.

NOT a verified drop-in replacement for Game_hub/quoridor/server/ai.py.
Verify coordinates, wall orientations, current turn, game rules, and API shape
against the actual existing server before integrating/deploying anything.
"""
from __future__ import annotations

import json
import sys
from quoridor_ai import Agent, Options
from quoridor_engine import state_from_dict, state_to_dict, start_state


def choose_project_ai(state_payload: dict, mode: str = 'minimax2_ab',
                      time_limit_s: float | None = .2) -> dict:
    state = state_from_dict(state_payload)
    agent = Agent(Options(mode=mode, root_top_k=8, reply_top_k=4,
                          wall_margin=2, time_limit_s=time_limit_s,
                          iterative_deepening=time_limit_s is not None))
    selected = agent.choose(state)
    return {'action': selected.action.as_dict(),
            'stats': selected.stats.as_dict()}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = state_to_dict(start_state())
    mode = sys.argv[2] if len(sys.argv) > 2 else 'minimax2_ab'
    print(json.dumps(choose_project_ai(data, mode), ensure_ascii=False, indent=2))
