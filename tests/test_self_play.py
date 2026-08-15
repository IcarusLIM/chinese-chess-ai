import unittest

import numpy as np
import torch
import torch.nn as nn

from constants import B_KING, EMPTY, R_KING, R_ROOK
from game import Game
from mcts import InferenceCache, MCTS, search_many
from move_index import get_move_index
from encoding import NUM_INPUT_PLANES, canonical_move_index, mirror_index_map
from network import PolicyValueNet
from trainer import (
    GameDataset,
    ModelEvaluator,
    SelfPlayWorker,
    Trainer,
    build_replay_sample_weights,
    compact_training_sample,
    game_outcome,
    resolve_self_play_workers,
)


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


class TinyTrainingModel(nn.Module):
    """用于验证 AMP 溢出恢复路径的最小双头模型。"""

    def __init__(self, device):
        super().__init__()
        self.policy_bias = nn.Parameter(torch.zeros(2, device=device))
        self.value_bias = nn.Parameter(torch.zeros(3, device=device))

    def forward(self, states):
        batch_size = states.shape[0]
        return (
            self.policy_bias.expand(batch_size, -1),
            self.value_bias.expand(batch_size, -1),
        )


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

    def test_compact_samples_keep_fractional_rule_planes(self):
        game = Game()
        game.halfmove_clock = 30
        board = game.get_board_tensor()
        policy = np.zeros(get_move_index().num_moves, dtype=np.float32)
        legal_mask = np.zeros_like(policy, dtype=np.bool_)
        policy[0] = 1.0
        legal_mask[0] = True
        sample = compact_training_sample(board, policy, legal_mask, 0)

        state, _, _, value = GameDataset([sample], augment=False)[0]

        self.assertEqual(tuple(state.shape), (NUM_INPUT_PLANES, 10, 9))
        self.assertAlmostEqual(float(state[17, 0, 0]), 0.25, places=3)
        self.assertEqual(value.item(), 0)

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

    def test_auto_worker_count_leaves_two_logical_cpus_free(self):
        import unittest.mock

        with unittest.mock.patch('trainer.os.cpu_count', return_value=16):
            self.assertEqual(resolve_self_play_workers(0, 25), 8)
            self.assertEqual(resolve_self_play_workers(0, 8), 8)
        self.assertEqual(resolve_self_play_workers(6, 4), 4)

    def test_all_draw_evaluation_does_not_promote_candidate(self):
        evaluator = ModelEvaluator(num_games=2)
        evaluator._play_evaluation_game = lambda *args, **kwargs: (
            'draw', 'repetition'
        )

        result = evaluator.evaluate(object(), object())

        self.assertEqual(result['score_rate'], 0.5)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['termination_counts'], {'repetition': 2})

    def test_perpetual_checker_loses_repetition(self):
        game = Game()
        game.board = [[EMPTY for _ in range(9)] for _ in range(10)]
        game.board[0][3] = R_KING
        game.board[9][5] = B_KING
        game.board[8][4] = R_ROOK
        game.red_to_move = True
        game.move_history = []
        game.halfmove_clock = 0
        game.fullmove_number = 1
        game.current_hash = game._compute_hash()
        game.hash_history = [game.current_hash]

        cycle = [
            ((8, 4), (8, 5)),
            ((9, 5), (9, 4)),
            ((8, 5), (8, 4)),
            ((9, 4), (9, 5)),
        ]
        for _ in range(2):
            for move in cycle:
                game.make_move(*move)

        self.assertEqual(game.get_repetition_result(), 'black_wins')
        self.assertEqual(game.is_game_over(), (True, 'black_wins'))

    def test_canonical_black_move_rotates_180_degrees(self):
        move_index = get_move_index()
        absolute = (0, 1, 2, 2)
        canonical = (9, 7, 7, 6)
        self.assertEqual(
            canonical_move_index(absolute, red_to_move=False),
            move_index.move_to_index(canonical),
        )

    def test_initial_position_is_side_canonical(self):
        red_view = Game().get_board_tensor()
        black_game = Game()
        black_game.red_to_move = False
        black_game.current_hash = black_game._compute_hash()
        black_game.hash_history = [black_game.current_hash]

        self.assertTrue(np.array_equal(red_view, black_game.get_board_tensor()))

    def test_mirror_move_map_is_an_involution(self):
        mapping = mirror_index_map()
        self.assertTrue(np.array_equal(mapping[mapping], np.arange(len(mapping))))

    def test_default_network_is_compact_and_outputs_wdl(self):
        model = PolicyValueNet()
        policy, wdl = model(torch.zeros(2, NUM_INPUT_PLANES, 10, 9))
        self.assertEqual(policy.shape, (2, get_move_index().num_moves))
        self.assertEqual(wdl.shape, (2, 3))
        self.assertLess(model.count_parameters(), 4_000_000)

    def test_wdl_training_step_is_finite(self):
        model = TinyTrainingModel(torch.device('cpu'))
        trainer = Trainer(model, total_training_iterations=2)
        batch = (
            torch.zeros(3, NUM_INPUT_PLANES, 10, 9),
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
            torch.ones(3, 2, dtype=torch.bool),
            torch.tensor([-1, 0, 1], dtype=torch.long),
        )

        metrics = trainer.train_iteration([batch], iteration=1)

        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertEqual(metrics['successful_batches'], 1)

    def test_move_limit_uses_material_adjudication(self):
        game = Game()
        game.board[9][0] = EMPTY  # 红方多一车，超过 5% 阈值

        self.assertEqual(
            game_outcome(
                game, hit_move_limit=True,
                material_adjudication_threshold=0.05,
            ),
            ('red_wins', 'material_adjudication'),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA AMP")
    def test_amp_nonfinite_gradient_is_skipped_and_scale_recovers(self):
        import unittest.mock

        device = torch.device('cuda')
        model = TinyTrainingModel(device)
        trainer = Trainer(model, total_training_iterations=10)
        states = torch.zeros((2, NUM_INPUT_PLANES, 10, 9))
        policies = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        legal_masks = torch.ones((2, 2), dtype=torch.bool)
        values = torch.zeros(2, dtype=torch.long)
        batches = [
            (states, policies, legal_masks, values),
            (states, policies, legal_masks, values),
        ]
        original_clip = torch.nn.utils.clip_grad_norm_
        calls = 0

        def overflow_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return torch.tensor(float('inf'), device=device)
            return original_clip(*args, **kwargs)

        initial_scale = trainer.scaler.get_scale()
        with unittest.mock.patch(
            'trainer.torch.nn.utils.clip_grad_norm_', side_effect=overflow_once
        ):
            metrics = trainer.train_iteration(batches, iteration=1)

        self.assertEqual(metrics['skipped_batches'], 1)
        self.assertEqual(metrics['successful_batches'], 1)
        self.assertLess(metrics['amp_scale'], initial_scale)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
