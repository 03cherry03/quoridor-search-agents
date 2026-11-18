"""Evaluation-function, league-score, timeout and depth-control tests."""
import unittest

from experiment import MatchResult, performance_score, play
from league import bootstrap_wr_ci, league_summary
from pilot_budget import pick_budget
from quoridor_ai import Agent, Choice, Options, SearchStats
from quoridor_engine import Action, Rules, State, start_state


class TestPDFEvaluation(unittest.TestCase):
    def test_nonterminal_score_and_turn_bonus(self):
        agent = Agent(Options(mode='greedy', wall_weight=0.5))
        agent._root = 0
        self.assertEqual(agent._score(start_state()), 0.5)
        reversed_turn = State(turn=1)
        self.assertEqual(agent._score(reversed_turn), -0.5)
        self.assertEqual(agent._score(State(pawns=((4, 4), (8, 4)))), 4.5)

    def test_remaining_wall_weight(self):
        agent = Agent(Options(mode='greedy', wall_weight=.5))
        agent._root = 0
        self.assertEqual(agent._score(State(remaining=(9, 7))), 1.5)

    def test_terminal_speed_preference(self):
        agent = Agent(Options(mode='greedy'))
        agent._root = 0
        won = State(pawns=((8, 3), (8, 4)), turn=1)
        lost = State(pawns=((0, 4), (0, 3)), turn=0)
        self.assertEqual(agent._score(won, 2), 10002)
        self.assertEqual(agent._score(lost, 2), -10002)

    def test_performance_scoring_and_complementarity(self):
        winstate = State(pawns=((8, 2), (6, 4)), turn=1)
        win = MatchResult(1, 43, 'minimax2', 'minimax2_ab', 0, 12,
                          winstate, 'goal', None, 0, 0, 0, 0, 0)
        draw = MatchResult(2, 43, 'minimax2_ab', 'minimax2', None, 50,
                           start_state(), 'max_plies_draw', None, 0, 0, 0, 0, 0)
        self.assertEqual(win.margin_side0, .75)
        self.assertEqual(draw.margin_side0, 0)
        a = performance_score([win, draw], 'minimax2')
        b = performance_score([win, draw], 'minimax2_ab')
        self.assertAlmostEqual(a['score'], .73125)
        self.assertAlmostEqual(a['score'] + b['score'], 1.0)
        self.assertAlmostEqual(a['win_rate'], .75)
        standings, pairwise = league_summary([win, draw], [], ['minimax2', 'minimax2_ab'])
        self.assertEqual(len(standings), 2)
        self.assertAlmostEqual(pairwise[0]['score_a_vs_b'] + pairwise[0]['score_b_vs_a'], 1)
        self.assertEqual(bootstrap_wr_ci([win, draw], 'minimax2'), (None, None))

    def test_forfeit_counts_as_loss_and_margin_negative_one(self):
        class SlowAgent:
            def __init__(self, mode, ms):
                self.options = Options(mode=mode)
                self.ms = ms

            def choose(self, state):
                action = Rules().legal_actions(state, wall_margin=0)[0]
                return Choice(action, SearchStats(elapsed_ms=self.ms))

        s = start_state()
        match, log = play(s, SlowAgent('minimax2', 260),
                          SlowAgent('minimax2_ab', 20), 1, 8, 20,
                          'minimax2', 'minimax2_ab', hard_limit_s=.25)
        self.assertEqual(match.winning_side, 1)
        self.assertEqual(match.finish_reason, 'time_forfeit')
        self.assertEqual(match.margin_for('minimax2'), -1.0)
        self.assertEqual(match.margin_for('minimax2_ab'), 1.0)
        self.assertEqual(match.total_plies, 0)
        self.assertEqual(log[0]['applied'], 0)
        self.assertEqual(log[0]['hard_limit_exceeded'], 1)

    def test_budget_recommendation_is_conservative(self):
        summary = [
            {'agent': 'minimax3', 'budget_s': .2, 'full_depth_rate': .4},
            {'agent': 'minimax3_ab', 'budget_s': .2, 'full_depth_rate': .9},
            {'agent': 'minimax3', 'budget_s': .5, 'full_depth_rate': .8},
            {'agent': 'minimax3_ab', 'budget_s': .5, 'full_depth_rate': .9},
        ]
        self.assertEqual(pick_budget(summary, [.2, .5], threshold=.75), .5)
        self.assertIsNone(pick_budget(summary, [.2], threshold=.75))


class TestNewModes(unittest.TestCase):
    def test_depth3_minimax_matches_alpha_beta(self):
        state = start_state()
        opts = dict(root_top_k=4, reply_top_k=2, wall_margin=1)
        plain = Agent(Options(mode='minimax3', **opts)).choose(state)
        ab = Agent(Options(mode='minimax3_ab', **opts)).choose(state)
        self.assertEqual(plain.stats.completed_depth, 3)
        self.assertEqual(ab.stats.completed_depth, 3)
        self.assertEqual(plain.action, ab.action)
        self.assertEqual(plain.stats.search_value, ab.stats.search_value)
        self.assertLessEqual(ab.stats.nodes, plain.stats.nodes)

    def test_time_limited_mcts_is_legal(self):
        s = start_state()
        choice = Agent(Options(mode='mcts', time_limit_s=.02,
                               root_top_k=4, reply_top_k=2,
                               wall_margin=1)).choose(s)
        self.assertIn(choice.action, Rules().legal_actions(s))
        self.assertLess(choice.stats.elapsed_ms, 250)

    def test_timed_deepening_only_accepts_complete_depth(self):
        choice = Agent(Options(mode='minimax3', time_limit_s=.01,
                               root_top_k=4, reply_top_k=2,
                               wall_margin=1)).choose(start_state())
        self.assertIn(choice.stats.completed_depth, (0, 1, 2, 3))
        self.assertIn(choice.action, Rules().legal_actions(start_state()))


if __name__ == '__main__':
    unittest.main()
