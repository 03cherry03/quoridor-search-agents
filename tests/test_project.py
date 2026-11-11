"""Run: python -m unittest discover -s tests -v"""
import random
import unittest

from quoridor_ai import Agent, Options
from quoridor_engine import (
    Action, Rules, State, _bit, apply_legal, blocked, legal_pawn_moves,
    shape_fits, start_state, state_from_dict, state_to_dict, winner,
)


class TestQuoridorRules(unittest.TestCase):
    def setUp(self):
        self.rules = Rules()

    def test_initial_state(self):
        s = start_state()
        self.assertEqual(s.pawns, ((0, 4), (8, 4)))
        self.assertEqual(s.remaining, (10, 10))
        self.assertEqual(winner(s), None)
        self.assertEqual(self.rules.shortest_path_len(s, 0), 8)
        self.assertEqual(self.rules.shortest_path_len(s, 1), 8)
        self.assertEqual({(a.row, a.col) for a in legal_pawn_moves(s)},
                         {(1, 4), (0, 3), (0, 5)})

    def test_wall_blocks_two_edges(self):
        s = self.rules.apply(start_state(), Action("wall", 4, 3, "h"))
        self.assertTrue(blocked(s, (4, 3), (5, 3)))
        self.assertTrue(blocked(s, (4, 4), (5, 4)))
        self.assertFalse(blocked(s, (4, 5), (5, 5)))
        self.assertEqual(s.remaining, (9, 10))
        self.assertEqual(s.turn, 1)

    def test_wall_overlap_and_crossing_rejected(self):
        s = self.rules.apply(start_state(), Action("wall", 4, 3, "h"))
        self.assertFalse(shape_fits(s, 4, 3, "v"))
        self.assertFalse(shape_fits(s, 4, 2, "h"))
        self.assertFalse(shape_fits(s, 4, 4, "h"))
        self.assertTrue(shape_fits(s, 4, 5, "h"))
        self.assertFalse(self.rules.can_place_wall(s, 4, 3, "v"))

    def test_jump_and_diagonal(self):
        s = State(pawns=((4, 4), (5, 4)), turn=0)
        moves = {(a.row, a.col) for a in legal_pawn_moves(s)}
        self.assertIn((6, 4), moves)
        self.assertNotIn((5, 3), moves)
        blocked_jump = State(pawns=((4, 4), (5, 4)), horizontal=_bit(5, 4), turn=0)
        moves = {(a.row, a.col) for a in legal_pawn_moves(blocked_jump)}
        self.assertNotIn((6, 4), moves)
        self.assertIn((5, 3), moves)
        self.assertIn((5, 5), moves)

    def test_illegal_pawn_cannot_jump_across_wall(self):
        s = State(horizontal=_bit(0, 4))
        with self.assertRaises(ValueError):
            self.rules.apply(s, Action("pawn", 1, 4))

    def test_serialization_round_trip(self):
        s = start_state()
        for action in [Action("wall", 4, 3, "h"), Action("wall", 2, 6, "v")]:
            s = self.rules.apply(s, action)
        self.assertEqual(s, state_from_dict(state_to_dict(s)))

    def test_referee_legal_actions(self):
        s = start_state()
        rng = random.Random(7)
        for _ in range(10):
            actions = self.rules.legal_actions(s)
            self.assertTrue(actions)
            action = rng.choice(actions)
            s = self.rules.apply(s, action)
            if winner(s) is not None:
                break


class TestSearch(unittest.TestCase):
    def test_immediate_win(self):
        s = State(pawns=((7, 4), (8, 5)), turn=0)
        for mode in ("greedy", "minimax2", "minimax2_ab"):
            decision = Agent(Options(mode=mode, root_top_k=5,
                                     wall_margin=0)).choose(s)
            self.assertEqual(decision.action, Action("pawn", 8, 4))

    def test_minimax_same_value_as_alpha_beta(self):
        rules = Rules()
        s = start_state()
        rng = random.Random(134)
        samples = [s]
        for _ in range(5):
            opts = rules.legal_actions(s, wall_margin=1)
            s = rules.apply(s, rng.choice(opts))
            if winner(s) is None:
                samples.append(s)
        for position in samples:
            baseline = Agent(Options(mode="minimax2", root_top_k=6, wall_margin=1)).choose(position)
            pruned = Agent(Options(mode="minimax2_ab", root_top_k=6, wall_margin=1)).choose(position)
            self.assertEqual(baseline.action, pruned.action)
            self.assertEqual(baseline.stats.search_value, pruned.stats.search_value)
            self.assertFalse(pruned.stats.timed_out)
            self.assertLessEqual(pruned.stats.leaves, baseline.stats.leaves)

    def test_no_mutation(self):
        s = start_state()
        old = state_to_dict(s)
        Agent(Options(mode="minimax2_ab")).choose(s)
        self.assertEqual(state_to_dict(s), old)

    def test_timeout_returns_legal_fallback(self):
        s = start_state()
        answer = Agent(Options(mode="minimax3_ab", root_top_k=8,
                               wall_margin=None, time_limit_s=0.0001,
                               iterative_deepening=True)).choose(s)
        self.assertIn(answer.action, Rules().legal_actions(s))
        self.assertTrue(answer.stats.timed_out)

    def test_depth_three_with_beam(self):
        s = start_state()
        choice = Agent(Options(mode="minimax3_ab", root_top_k=4,
                               reply_top_k=3, wall_margin=1)).choose(s)
        self.assertEqual(choice.stats.completed_depth, 3)
        self.assertIn(choice.action, Rules().legal_actions(s))


if __name__ == "__main__":
    unittest.main()
