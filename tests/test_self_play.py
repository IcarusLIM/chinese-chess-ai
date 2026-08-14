import unittest

import numpy as np

from game import Game
from mcts import InferenceCache, MCTS, search_many
from move_index import get_move_index
from trainer import SelfPlayWorker, build_replay_sample_weights


class UniformModel:
    """快速、确定性的测试模型，同时记录实际推理 batch。"""

    def __init__(self):
        self.num_moves = get_move_index().num_moves
        self.batch_sizes = []

    def predict_batch(self, board_tensors):
        batch_size = len(board_tensors)
        self.batch_sizes.append(batch_size)
        policies = np.full(
            (batch_size, self.num_moves), 1.0 / self.num_moves, dtype=np.float32
        )
        values = np.zeros(batch_size, dtype=np.float32)
        return policies, values


class BatchedSelfPlayTests(unittest.TestCase):
    def test_search_many_keeps_games_unchanged_and_returns_legal_moves(self):
        np.random.seed(7)
        model = UniformModel()
        cache = InferenceCache(1000)
        games = [Game() for _ in range(4)]
        hashes_before = [game.current_hash for game in games]
        searches = [
            MCTS(
                model,
                num_simulations=12,
                inference_batch_size=8,
                inference_cache=cache,
                add_root_noise=True,
            )
            for _ in games
        ]

        results = search_many(
            [(mcts, game, 1.0) for mcts, game in zip(searches, games)],
            inference_batch_size=16,
        )

        move_index = get_move_index()
        for game, original_hash, (move_idx, policy) in zip(
            games, hashes_before, results
        ):
            self.assertEqual(game.current_hash, original_hash)
            self.assertIn(move_index.index_to_move(move_idx), [
                (fr, fc, tr, tc)
                for (fr, fc), (tr, tc) in game.get_legal_moves()
            ])
            self.assertAlmostEqual(sum(policy), 1.0, places=6)

        self.assertGreater(max(model.batch_sizes), 1)

    def test_play_multiple_games_advances_games_together(self):
        np.random.seed(11)
        model = UniformModel()
        worker = SelfPlayWorker(
            model,
            num_simulations=4,
            inference_batch_size=8,
            inference_cache_size=1000,
            max_moves=4,
        )

        samples = worker.play_multiple_games(3)

        self.assertEqual(len(samples), 12)
        self.assertGreater(max(model.batch_sizes), 1)

    def test_recent_replay_weights_have_requested_probability_mass(self):
        weights = build_replay_sample_weights(20000, 5000, 0.5)
        normalized = weights / weights.sum()
        self.assertAlmostEqual(float(normalized[-5000:].sum()), 0.5, places=7)
        self.assertAlmostEqual(float(normalized[:-5000].sum()), 0.5, places=7)

    def test_parallel_actors_use_centralized_inference(self):
        model = UniformModel()
        worker = SelfPlayWorker(
            model,
            num_simulations=2,
            inference_batch_size=2,
            inference_cache_size=100,
            max_moves=2,
            num_workers=2,
            inference_server_batch_size=8,
        )

        samples = worker.play_multiple_games(2)

        self.assertEqual(len(samples), 4)
        self.assertTrue(model.batch_sizes)


if __name__ == '__main__':
    unittest.main()
