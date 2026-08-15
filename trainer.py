"""
自对弈训练模块
==============
实现 AlphaZero 风格的自对弈训练流程。

训练流程（循环迭代）：
  1. 自对弈生成训练数据
     - 两个当前模型对弈
     - 使用 MCTS 搜索选择走法
     - 记录每步的 (局面, 策略, 结果)
  2. 训练神经网络
     - 使用自对弈数据训练
     - 损失 = 策略损失 + 价值损失 + L2正则化
  3. 评估候选模型
     - 候选模型 vs 当前基准模型对弈
     - 固定多开局、交换红黑进行成对评估
     - 候选模型达到门槛时更新 best，latest 始终继续训练
  4. 重复

关键超参数：
  - 每次自对弈生成多少局
  - 每局最多走多少步
  - 新样本的目标重放比例
  - 学习率和调度策略

硬件优化：
  - RTX 5070 (12GB VRAM) 优化配置
  - 混合精度训练 (AMP)
  - 批量推理加速 MCTS
"""

import os
import json
import time
import random
import pickle
import multiprocessing as mp
import queue
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Tuple, Dict, Optional
from collections import deque

from game import Game
from network import PolicyValueNet, create_model, get_default_device, read_checkpoint
from mcts import InferenceCache, MCTS, MCTSEvaluator, search_many
from move_index import get_move_index
from encoding import legal_mask_to_canonical, mirror_index_map, policy_to_canonical


def resolve_self_play_workers(requested: int, num_games: int) -> int:
    """解析 actor 数量；0 表示按逻辑核自动预留 2 个核。"""
    if requested > 0:
        return min(requested, max(1, num_games))
    logical_cpus = os.cpu_count() or 1
    return min(8, max(1, num_games), max(1, logical_cpus - 2))


def game_outcome(game: Game, hit_move_limit: bool = False,
                 material_adjudication_threshold: float = None) -> Tuple[str, str]:
    """返回对局结果和可用于日志的终局原因。"""
    is_over, result = game.is_game_over()
    if is_over:
        if result == 'draw':
            reason = game.get_draw_reason() or 'draw'
            if (
                reason == 'no_progress'
                and material_adjudication_threshold is not None
                and material_adjudication_threshold > 0
            ):
                material_result = adjudicate_material(
                    game, material_adjudication_threshold
                )
                if material_result is not None:
                    return material_result, 'material_adjudication'
            return result, reason
        if game.get_repetition_result() == result:
            return result, 'perpetual_check'
        return result, 'decisive'
    if hit_move_limit:
        if (
            material_adjudication_threshold is not None
            and material_adjudication_threshold > 0
        ):
            material_result = adjudicate_material(
                game, material_adjudication_threshold
            )
            if material_result is not None:
                return material_result, 'material_adjudication'
        return 'draw', 'max_moves'
    return 'draw', 'incomplete'


def adjudicate_material(game: Game, threshold: float) -> Optional[str]:
    """以红方视角的子力差裁定超时/无进展对局。"""
    current_side_balance = game.evaluate_material_normalized()
    red_balance = (
        current_side_balance if game.red_to_move else -current_side_balance
    )
    if red_balance >= threshold:
        return 'red_wins'
    if red_balance <= -threshold:
        return 'black_wins'
    return None


class GameDataset(Dataset):
    """
    棋局数据集
    
    样本使用紧凑格式：float16 状态、稀疏 float16 策略、bit-pack 合法掩码、int8 结果。
    读取时展开为训练需要的 dense float32 Tensor。
    """
    
    def __init__(self, data: List[Tuple], augment: bool = True):
        """
        Args:
            data: 紧凑五元组样本列表
        """
        self.data = data
        self.num_moves = get_move_index().num_moves
        self.augment = augment
        self.mirror_map = mirror_index_map()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board_tensor, policy_indices, policy_values, packed_legal_mask, value = self.data[idx]
        policy = np.zeros(self.num_moves, dtype=np.float32)
        policy[np.asarray(policy_indices, dtype=np.int64)] = np.asarray(
            policy_values, dtype=np.float32
        )
        policy_sum = float(policy.sum())
        if policy_sum > 0:
            policy /= policy_sum
        legal_mask = np.unpackbits(
            np.asarray(packed_legal_mask, dtype=np.uint8),
            count=self.num_moves,
            bitorder='little',
        ).astype(np.bool_)
        board = np.asarray(board_tensor, dtype=np.float32)
        if self.augment and np.random.random() < 0.5:
            board = board[:, :, ::-1].copy()
            mirrored_policy = np.zeros_like(policy)
            mirrored_policy[self.mirror_map] = policy
            policy = mirrored_policy
            mirrored_legal_mask = np.zeros_like(legal_mask)
            mirrored_legal_mask[self.mirror_map] = legal_mask
            legal_mask = mirrored_legal_mask
        return (
            torch.from_numpy(board),
            torch.from_numpy(policy),
            torch.from_numpy(legal_mask),
            torch.tensor(value, dtype=torch.long),
        )


def build_replay_sample_weights(
        dataset_size: int, recent_window_size: int,
        recent_sample_fraction: float) -> torch.Tensor:
    """构造近期/历史分层采样权重。"""
    if dataset_size <= 0:
        raise ValueError("dataset_size 必须大于 0")
    if not 0.0 <= recent_sample_fraction <= 1.0:
        raise ValueError("recent_sample_fraction 必须在 0 到 1 之间")
    recent_count = min(max(1, recent_window_size), dataset_size)
    old_count = dataset_size - recent_count
    weights = torch.ones(dataset_size, dtype=torch.double)
    if old_count > 0:
        weights[:old_count] = (1.0 - recent_sample_fraction) / old_count
        weights[old_count:] = recent_sample_fraction / recent_count

    return weights


def compact_training_sample(board_tensor, policy, legal_mask, value) -> Tuple:
    """将自对弈样本压缩为适合长期 replay buffer 保存的格式。"""
    board = np.asarray(board_tensor, dtype=np.float16)
    policy_array = np.asarray(policy, dtype=np.float32)
    policy_indices = np.flatnonzero(policy_array > 0).astype(np.uint16)
    policy_values = policy_array[policy_indices].astype(np.float16)
    packed_legal_mask = np.packbits(
        np.asarray(legal_mask, dtype=np.bool_), bitorder='little'
    )
    compact_value = np.int8(round(float(value)))
    return board, policy_indices, policy_values, packed_legal_mask, compact_value


class SelfPlayWorker:
    """
    自对弈工作器
    
    负责执行自对弈并收集训练数据。
    使用 MCTS 搜索选择走法，记录每步的训练样本。
    """
    
    def __init__(self,
                 model: PolicyValueNet,
                 num_simulations: int = 400,
                 inference_batch_size: int = 64,
                 inference_cache_size: int = 10000,
                 max_moves: int = 200,
                 temperature_threshold: int = 30,
                 num_workers: int = 1,
                 inference_server_batch_size: int = 256,
                 inference_server_wait_ms: float = 5.0,
                 material_adjudication_threshold: float = 0.05):
        """
        Args:
            model: 当前模型
            num_simulations: MCTS 模拟次数
            inference_batch_size: MCTS GPU 叶节点推理批大小
            inference_cache_size: 当前模型权重的 LRU 局面缓存容量
            max_moves: 每局最大走子数（超过判和）
            temperature_threshold: 走子数超过此值后温度降为 0.01
            num_workers: 自对弈 CPU actor 进程数；模型只保留在主进程
            inference_server_batch_size: 集中式模型推理的最大合并 batch
            inference_server_wait_ms: 首个请求后等待其他 actor 合批的毫秒数
        """
        self.model = model
        self.num_simulations = num_simulations
        self.inference_batch_size = inference_batch_size
        self.inference_cache = InferenceCache(inference_cache_size)
        self.max_moves = max_moves
        self.temperature_threshold = temperature_threshold
        self.num_workers = max(1, num_workers)
        self.inference_server_batch_size = max(
            self.inference_batch_size, inference_server_batch_size
        )
        self.inference_server_wait_ms = max(0.0, inference_server_wait_ms)
        self.material_adjudication_threshold = max(
            0.0, material_adjudication_threshold
        )
        self.move_index = get_move_index()
        self.last_inference_calls = 0
        self.last_inference_positions = 0
        self.last_game_summary = None
    
    def play_one_game(self) -> List[Tuple]:
        """
        执行一局自对弈，收集训练数据。
        
        Returns:
            训练样本列表，每个元素为 (board_tensor, policy, legal_mask, value)
            其中 value 是最终游戏结果（从当前走棋方视角）
        """
        game = Game()
        mcts = MCTS(self.model, num_simulations=self.num_simulations,
                    add_root_noise=True,
                    inference_batch_size=self.inference_batch_size,
                    inference_cache=self.inference_cache)
        
        # 存储每一步的数据
        states = []    # 棋盘状态
        policies = []  # MCTS 策略
        legal_masks = []  # 真实合法走法掩码
        
        move_count = 0
        
        while move_count < self.max_moves:
            # 根据走子数决定温度
            # 前 temperature_threshold 步使用温度 1.0（鼓励探索）
            # 之后使用温度 0.01（接近贪心）
            if move_count < self.temperature_threshold:
                temperature = 1.0
            else:
                temperature = 0.01
            
            # MCTS 搜索
            move_idx, policy = mcts.search(game, temperature)
            
            if move_idx < 0:
                break  # 没有合法走法（游戏结束）
            
            # 记录训练数据
            legal_mask = legal_mask_to_canonical(
                mcts.last_legal_indices, game.red_to_move
            )
            states.append(game.get_board_tensor())
            policies.append(policy_to_canonical(policy, game.red_to_move))
            legal_masks.append(legal_mask)
            
            # 执行走子
            move = self.move_index.index_to_move(move_idx)
            if move is None:
                break
            fr, fc, tr, tc = move
            game.make_move((fr, fc), (tr, tc), validate=False)
            move_count += 1
        
        # 确定游戏结果
        result, reason = game_outcome(
            game, move_count >= self.max_moves,
            self.material_adjudication_threshold,
        )
        if result == 'red_wins':
            final_value = 1.0
        elif result == 'black_wins':
            final_value = -1.0
        else:
            final_value = 0.0  # 和棋或超时
        
        # 构建训练样本
        # 价值从当前走棋方的视角：红方走棋时为 final_value，黑方走棋时翻转
        training_data = []
        for i, (state, policy, legal_mask) in enumerate(zip(states, policies, legal_masks)):
            # 第 i 步时的走棋方（红方先走，交替进行）
            if i % 2 == 0:
                value = final_value  # 红方走棋
            else:
                value = -final_value  # 黑方走棋
            training_data.append(
                compact_training_sample(state, policy, legal_mask, value)
            )

        self.last_inference_calls = mcts.inference_calls
        self.last_inference_positions = mcts.inference_positions
        self.last_game_summary = (result, reason, move_count)
        return training_data
    
    def play_multiple_games(self, num_games: int) -> List[Tuple]:
        """
        执行多局自对弈，收集训练数据。
        
        Args:
            num_games: 对弈局数
            
        Returns:
            所有训练样本的列表
        """
        if self.num_workers > 1 and num_games > 1:
            return self._play_multiple_games_parallel(num_games)

        # 模型每轮训练后会更新，因此每轮自对弈前清空上一组权重的缓存。
        self.inference_cache.clear()
        if num_games <= 0:
            return []

        # 同时推进多盘棋，使一个 GPU batch 能包含不同棋局的叶节点。
        slots = []
        for _ in range(num_games):
            slots.append({
                'game': Game(),
                'mcts': MCTS(
                    self.model,
                    num_simulations=self.num_simulations,
                    add_root_noise=True,
                    inference_batch_size=self.inference_batch_size,
                    inference_cache=self.inference_cache,
                ),
                'states': [],
                'policies': [],
                'legal_masks': [],
                'move_count': 0,
                'finished': False,
            })

        completed = 0
        next_progress = 3
        while completed < num_games:
            active_slots = [slot for slot in slots if not slot['finished']]
            requests = []
            for slot in active_slots:
                temperature = (
                    1.0 if slot['move_count'] < self.temperature_threshold else 0.01
                )
                requests.append((slot['mcts'], slot['game'], temperature))

            search_results = search_many(
                requests, inference_batch_size=self.inference_batch_size
            )
            for slot, (move_idx, policy) in zip(active_slots, search_results):
                if move_idx < 0:
                    slot['finished'] = True
                    completed += 1
                    continue

                legal_mask = legal_mask_to_canonical(
                    slot['mcts'].last_legal_indices,
                    slot['game'].red_to_move,
                )
                slot['states'].append(slot['game'].get_board_tensor())
                slot['policies'].append(policy_to_canonical(
                    policy, slot['game'].red_to_move
                ))
                slot['legal_masks'].append(legal_mask)

                move = self.move_index.index_to_move(move_idx)
                if move is None:
                    slot['finished'] = True
                    completed += 1
                    continue
                fr, fc, tr, tc = move
                slot['game'].make_move((fr, fc), (tr, tc), validate=False)
                slot['move_count'] += 1
                if slot['move_count'] >= self.max_moves:
                    slot['finished'] = True
                    completed += 1

            while completed >= next_progress:
                sample_count = sum(len(slot['states']) for slot in slots)
                print(
                    f"  自对弈进度: {min(next_progress, num_games)}/{num_games}，"
                    f"累计样本数: {sample_count}"
                )
                next_progress += 3

        all_data = []
        termination_counts = {}
        for slot in slots:
            result, reason = game_outcome(
                slot['game'], slot['move_count'] >= self.max_moves,
                self.material_adjudication_threshold,
            )
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
            if result == 'red_wins':
                final_value = 1.0
            elif result == 'black_wins':
                final_value = -1.0
            else:
                final_value = 0.0
            for index, (state, policy, legal_mask) in enumerate(zip(
                slot['states'], slot['policies'], slot['legal_masks']
            )):
                value = final_value if index % 2 == 0 else -final_value
                all_data.append(
                    compact_training_sample(state, policy, legal_mask, value)
                )

        total_inference_calls = sum(slot['mcts'].inference_calls for slot in slots)
        total_inference_positions = sum(
            slot['mcts'].inference_positions for slot in slots
        )
        average_batch = (
            total_inference_positions / total_inference_calls if total_inference_calls else 0.0
        )
        print(
            f"  MCTS 网络推理: {total_inference_calls} 批 / "
            f"{total_inference_positions} 个局面，平均批大小: {average_batch:.1f}"
        )
        print(
            f"  推理缓存: {self.inference_cache.hits} 命中 / "
            f"{self.inference_cache.misses} 未命中，当前 {len(self.inference_cache)} 项"
        )
        print(f"  终局统计: {termination_counts}")
        return all_data

    def _play_multiple_games_parallel(self, num_games: int) -> List[Tuple]:
        """多 CPU actor 生成搜索树，主进程集中执行神经网络推理。"""
        worker_count = min(self.num_workers, num_games)
        context = mp.get_context('spawn')
        task_queue = context.Queue()
        request_queue = context.Queue(maxsize=worker_count * 2)
        result_queue = context.Queue()
        response_queues = [context.Queue() for _ in range(worker_count)]

        config = {
            'num_simulations': self.num_simulations,
            'inference_batch_size': self.inference_batch_size,
            'inference_cache_size': self.inference_cache.max_size,
            'max_moves': self.max_moves,
            'temperature_threshold': self.temperature_threshold,
            'material_adjudication_threshold': self.material_adjudication_threshold,
        }
        processes = []
        for worker_id in range(worker_count):
            process = context.Process(
                target=_self_play_actor,
                args=(
                    worker_id, config, task_queue, request_queue,
                    response_queues[worker_id], result_queue,
                ),
            )
            process.start()
            processes.append(process)

        for game_id in range(num_games):
            task_queue.put(game_id)
        for _ in range(worker_count):
            task_queue.put(None)

        game_results = {}
        game_summaries = {}
        done_workers = 0
        inference_calls = 0
        inference_positions = 0
        cache_hits = cache_misses = cache_items = 0
        try:
            while len(game_results) < num_games or done_workers < worker_count:
                requests = []
                try:
                    requests.append(request_queue.get(timeout=0.05))
                    deadline = (
                        time.monotonic() + self.inference_server_wait_ms / 1000.0
                    )
                    total_positions = len(requests[0][2])
                    while total_positions < self.inference_server_batch_size:
                        timeout = deadline - time.monotonic()
                        if timeout <= 0:
                            break
                        try:
                            item = request_queue.get(timeout=timeout)
                        except queue.Empty:
                            break
                        requests.append(item)
                        total_positions += len(item[2])
                except queue.Empty:
                    pass

                if requests:
                    state_batches = []
                    slices = []
                    combined_size = 0
                    for worker_id, request_id, states in requests:
                        start = combined_size
                        state_batches.append(states)
                        combined_size += len(states)
                        slices.append((worker_id, request_id, start, len(states)))
                    combined_states = np.concatenate(state_batches, axis=0)
                    policies, values = self.model.predict_batch(combined_states)
                    inference_calls += 1
                    inference_positions += len(combined_states)
                    for worker_id, request_id, start, length in slices:
                        response_queues[worker_id].put((
                            request_id,
                            policies[start:start + length].astype(
                                np.float16, copy=False
                            ),
                            values[start:start + length].astype(
                                np.float16, copy=False
                            ),
                        ))

                while True:
                    try:
                        message = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    kind = message[0]
                    if kind == 'game':
                        _, game_id, data, summary = message
                        game_results[game_id] = data
                        game_summaries[game_id] = summary
                        completed = len(game_results)
                        if completed % 3 == 0 or completed == num_games:
                            sample_count = sum(len(item) for item in game_results.values())
                            print(
                                f"  自对弈进度: {completed}/{num_games}，"
                                f"累计样本数: {sample_count}"
                            )
                    elif kind == 'done':
                        _, _, hits, misses, items = message
                        done_workers += 1
                        cache_hits += hits
                        cache_misses += misses
                        cache_items += items
                    elif kind == 'error':
                        raise RuntimeError(
                            f"自对弈 actor {message[1]} 异常:\n{message[2]}"
                        )
        finally:
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()

        average_batch = (
            inference_positions / inference_calls if inference_calls else 0.0
        )
        print(
            f"  集中式网络推理: {inference_calls} 批 / "
            f"{inference_positions} 个局面，平均批大小: {average_batch:.1f}"
        )
        print(
            f"  Actor 推理缓存: {cache_hits} 命中 / {cache_misses} 未命中，"
            f"共 {cache_items} 项"
        )
        termination_counts = {}
        for _, reason, _ in game_summaries.values():
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
        print(f"  终局统计: {termination_counts}")
        return [sample for game_id in range(num_games) for sample in game_results[game_id]]


class _InferenceClientModel:
    """自对弈 actor 中的模型代理；实际模型始终位于主进程/GPU。"""

    def __init__(self, worker_id: int, request_queue, response_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.request_id = 0

    def predict_batch(self, board_tensors):
        request_id = self.request_id
        self.request_id += 1
        states = np.asarray(board_tensors, dtype=np.float32)
        self.request_queue.put((self.worker_id, request_id, states))
        response_id, policies, values = self.response_queue.get()
        if response_id != request_id:
            raise RuntimeError(
                f"推理响应乱序: 期望 {request_id}，实际 {response_id}"
            )
        return policies, values


def _self_play_actor(worker_id: int, config: dict, task_queue, request_queue,
                     response_queue, result_queue):
    """子进程入口：只执行 Python 棋局与树搜索，不持有 GPU 模型。"""
    try:
        proxy_model = _InferenceClientModel(worker_id, request_queue, response_queue)
        worker = SelfPlayWorker(proxy_model, num_workers=1, **config)
        while True:
            game_id = task_queue.get()
            if game_id is None:
                break
            data = worker.play_one_game()
            result_queue.put(('game', game_id, data, worker.last_game_summary))
        result_queue.put((
            'done', worker_id, worker.inference_cache.hits,
            worker.inference_cache.misses, len(worker.inference_cache),
        ))
    except Exception:
        result_queue.put(('error', worker_id, traceback.format_exc()))


class Trainer:
    """
    模型训练器
    
    负责：
    1. 使用自对弈数据训练神经网络
    2. 管理学习率调度
    3. 保存和加载模型检查点
    4. 记录训练日志
    """
    
    def __init__(self,
                 model: PolicyValueNet,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-4,
                 total_training_iterations: int = 100,
                 lr_min: float = 1e-5):
        """
        Args:
            model: 策略-价值网络
            learning_rate: 初始学习率
            weight_decay: L2 正则化系数
            total_training_iterations: 总训练迭代数（用于余弦退火）
            lr_min: 最低学习率
        """
        self.model = model
        self.learning_rate = learning_rate
        self.total_training_iterations = total_training_iterations
        self.use_amp = next(model.parameters()).device.type == 'cuda'

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 学习率调度器：余弦退火，平滑衰减无需追踪全局 epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_training_iterations, eta_min=lr_min
        )

        # 混合精度训练（RTX 5070 优化）
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # 训练统计
        self.training_history = []
    
    def train_iteration(self, dataloader: DataLoader, iteration: int) -> Dict[str, float]:
        """
        完成一轮网络训练。
        
        Args:
            dataloader: 训练数据加载器
            iteration: 当前训练迭代编号
            
        Returns:
            训练指标字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        consecutive_overflows = 0

        device = next(self.model.parameters()).device
        for states, target_policies, legal_masks, target_values in dataloader:
            # 移动到计算设备
            states = states.to(device, non_blocking=True)
            target_policies = target_policies.to(device, non_blocking=True)
            legal_masks = legal_masks.to(device, non_blocking=True)
            target_values = target_values.to(device, non_blocking=True)
            
            # channels_last 内存格式
            states = states.to(memory_format=torch.channels_last)
            
            # 清除梯度
            self.optimizer.zero_grad(set_to_none=True)
            
            # 混合精度前向传播（仅 CUDA 启用）
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                policy_logits, wdl_logits = self.model(states)

                # 只屏蔽真正非法的走法；合法但 MCTS 未访问的走法仍参与归一化。
                valid_rows = legal_masks.any(dim=1, keepdim=True)
                effective_masks = torch.where(
                    valid_rows, legal_masks, torch.ones_like(legal_masks)
                )
                # softmax/交叉熵固定使用 FP32。大动作空间下 FP16 的极端
                # logits 虽可能得到有限 loss，反向梯度仍更容易溢出。
                masked_logits = policy_logits.float().masked_fill(
                    ~effective_masks, -1e9
                )

                policy_loss = -torch.mean(
                    torch.sum(target_policies * F.log_softmax(masked_logits, dim=1), dim=1)
                )

                # 标签 -1/0/+1 分别映射到 loss/draw/win。
                value_loss = F.cross_entropy(
                    wdl_logits.float(), target_values + 1
                )

                # 总损失
                loss = policy_loss + value_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"训练损失出现 NaN/Inf: policy={policy_loss.item()}, "
                    f"value={value_loss.item()}"
                )
            
            # 混合精度反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0, error_if_nonfinite=False
                )
                if not torch.isfinite(grad_norm):
                    # total norm 溢出不一定意味着 unscale_ 已发现单个 Inf，
                    # 因此这里明确不调用 optimizer.step，并主动降低 scale。
                    # 旧实现直接抛异常，GradScaler 没有机会恢复。
                    scale_before = self.scaler.get_scale()
                    self.scaler.update(new_scale=max(
                        1.0,
                        scale_before * self.scaler.get_backoff_factor(),
                    ))
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped_batches += 1
                    consecutive_overflows += 1
                    if (
                        skipped_batches <= 3
                        or (skipped_batches & (skipped_batches - 1)) == 0
                    ):
                        print(
                            f"  [AMP] 第 {num_batches + skipped_batches} 个 batch "
                            f"梯度溢出，已跳过；scale {scale_before:.0f} -> "
                            f"{self.scaler.get_scale():.0f}"
                        )
                    if consecutive_overflows >= 8:
                        raise FloatingPointError(
                            "连续 8 个 batch 梯度溢出；已停止训练以保护模型。"
                            f"当前 AMP scale={self.scaler.get_scale():.0f}，"
                            f"学习率={self.optimizer.param_groups[0]['lr']:.3g}"
                        )
                    continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
                consecutive_overflows = 0
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0, error_if_nonfinite=True
                )
                self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        if num_batches == 0:
            raise FloatingPointError("本轮没有任何成功的参数更新，拒绝推进学习率")

        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / max(num_batches, 1)
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        
        metrics = {
            'iteration': iteration,
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'lr': self.scheduler.get_last_lr()[0],
            'successful_batches': num_batches,
            'skipped_batches': skipped_batches,
            'amp_scale': self.scaler.get_scale() if self.use_amp else 1.0,
        }
        
        self.training_history.append(metrics)
        
        return metrics
    
    def save_checkpoint(self, path: str, iteration: int):
        """
        保存模型检查点。
        
        Args:
            path: 保存路径
            iteration: 当前训练迭代编号
        """
        self._ensure_model_finite("保存检查点")
        checkpoint = {
            'iteration': iteration,
            'model_config': {
                'num_blocks': self.model.num_blocks,
                'channels': self.model.channels,
                'policy_channels': self.model.policy_channels,
            },
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'training_history': self.training_history,
        }
        
        temp_path = f"{path}.tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
        print(f"[训练] 检查点已保存: {path}")

    def save_model(self, path: str, iteration: int, state_dict=None):
        """保存不含优化器的对弈/部署模型。"""
        self._ensure_model_finite("保存模型")
        checkpoint = {
            'iteration': iteration,
            'model_config': {
                'num_blocks': self.model.num_blocks,
                'channels': self.model.channels,
                'policy_channels': self.model.policy_channels,
            },
            'model_state_dict': (
                self.model.state_dict() if state_dict is None else state_dict
            ),
        }
        temp_path = f"{path}.tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
    
    def load_checkpoint(self, path: str) -> int:
        """
        加载模型检查点。

        Args:
            path: 检查点文件路径

        Returns:
            检查点对应的训练迭代编号
        """
        device = next(self.model.parameters()).device
        checkpoint = read_checkpoint(path, str(device))
        return self.restore_checkpoint(checkpoint, path)

    def restore_checkpoint(self, checkpoint: Dict, source: str) -> int:
        """从已读取的检查点恢复模型、优化器和训练状态。"""

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._ensure_model_finite("加载检查点")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # 允许续训时延长总迭代数，同时保留已完成的调度进度。
        self.scheduler.T_max = self.total_training_iterations
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.training_history = checkpoint.get('training_history', [])

        iteration = checkpoint['iteration']
        print(f"[训练] 检查点已加载: {source} (iteration {iteration})")
        return iteration

    def _ensure_model_finite(self, operation: str):
        """阻止包含 NaN/Inf 的模型继续训练或写入检查点。"""
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"{operation}失败：参数 {name} 包含 NaN/Inf")


class ModelEvaluator:
    """
    模型评估器
    
    使用固定的多开局套件，并在每个开局交换红黑进行成对评估。
    计分率超过阈值则接受候选模型。
    """
    
    def __init__(self, num_games: int = 40, num_simulations: int = 200,
                 inference_batch_size: int = 64,
                 inference_cache_size: int = 10000,
                 acceptance_threshold: float = 0.55, opening_seed: int = 20240814):
        """
        Args:
            num_games: 评估对弈局数
            num_simulations: MCTS 模拟次数
            inference_batch_size: MCTS GPU 叶节点推理批大小
            inference_cache_size: 每个参评模型的 LRU 局面缓存容量
            acceptance_threshold: 候选模型最低计分率
            opening_seed: 固定开局套件的随机种子
        """
        if num_games < 2 or num_games % 2 != 0:
            raise ValueError("num_games 必须是大于等于 2 的偶数，以便交换红黑成对评估")
        if not 0.0 <= acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold 必须在 0 到 1 之间")
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.inference_batch_size = inference_batch_size
        self.inference_cache_size = inference_cache_size
        self.acceptance_threshold = acceptance_threshold
        self.opening_seed = opening_seed
        self.move_index = get_move_index()
    
    def evaluate(self, candidate_model: PolicyValueNet,
                 reference_model: PolicyValueNet) -> Dict:
        """
        评估候选模型相对于当前基准模型的表现。
        
        Args:
            candidate_model: 刚完成一轮训练的候选模型
            reference_model: 最近一次通过评估的基准模型
            
        Returns:
            评估结果字典
        """
        candidate_wins = 0
        reference_wins = 0
        draws = 0
        termination_counts = {}
        candidate_cache = InferenceCache(self.inference_cache_size)
        reference_cache = InferenceCache(self.inference_cache_size)
        
        for game_idx in range(self.num_games):
            # 每两局使用同一开局并交换红黑，消除先手和单一开局偏差。
            opening_game = self._create_opening(game_idx // 2)
            if game_idx % 2 == 0:
                red_model, black_model = candidate_model, reference_model
                red_cache, black_cache = candidate_cache, reference_cache
                candidate_is_red = True
            else:
                red_model, black_model = reference_model, candidate_model
                red_cache, black_cache = reference_cache, candidate_cache
                candidate_is_red = False
            
            result, reason = self._play_evaluation_game(
                red_model, black_model, opening_game, red_cache, black_cache
            )
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
            
            if result == 'red_wins':
                if candidate_is_red:
                    candidate_wins += 1
                else:
                    reference_wins += 1
            elif result == 'black_wins':
                if candidate_is_red:
                    reference_wins += 1
                else:
                    candidate_wins += 1
            else:
                draws += 1
            
            if (game_idx + 1) % 5 == 0:
                print(f"  评估进度: {game_idx+1}/{self.num_games}")
        
        total = candidate_wins + reference_wins + draws
        win_rate = candidate_wins / total if total > 0 else 0
        score_rate = (candidate_wins + 0.5 * draws) / total if total > 0 else 0
        
        return {
            'candidate_wins': candidate_wins,
            'reference_wins': reference_wins,
            'draws': draws,
            'win_rate': win_rate,
            'score_rate': score_rate,
            # 全和时 50% 不再被视为候选模型通过。
            'accepted': (
                candidate_wins > reference_wins
                and score_rate >= self.acceptance_threshold
            ),
            'termination_counts': termination_counts,
        }

    def _create_opening(self, opening_index: int) -> Game:
        """生成固定、可复现的非终局开局；每个开局保持红方走棋。"""
        game = Game()
        rng = random.Random(self.opening_seed + opening_index)
        opening_plies = 8 + 4 * (opening_index % 5)
        for _ in range(opening_plies):
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            rng.shuffle(legal_moves)
            moved = False
            for move in legal_moves:
                game.make_move(*move, validate=False)
                is_over, _ = game.is_game_over()
                if not is_over:
                    moved = True
                    break
                game.undo_move()
            if not moved:
                break
        if not game.red_to_move:
            game.undo_move()
        return game

    def _play_evaluation_game(self, red_model: PolicyValueNet,
                              black_model: PolicyValueNet,
                              opening_game: Game = None,
                              red_cache: InferenceCache = None,
                              black_cache: InferenceCache = None) -> Tuple[str, str]:
        """执行一局评估对弈"""
        game = opening_game.copy() if opening_game is not None else Game()
        red_mcts = MCTS(
            red_model, num_simulations=self.num_simulations,
            inference_batch_size=self.inference_batch_size,
            inference_cache=red_cache,
        )
        black_mcts = MCTS(
            black_model, num_simulations=self.num_simulations,
            inference_batch_size=self.inference_batch_size,
            inference_cache=black_cache,
        )
        
        for _ in range(200):
            if game.red_to_move:
                move_idx, _ = red_mcts.search(game, temperature=0.01)
            else:
                move_idx, _ = black_mcts.search(game, temperature=0.01)
            
            if move_idx < 0:
                result, reason = game_outcome(
                    game, material_adjudication_threshold=None,
                )
                return result, reason
            
            move = self.move_index.index_to_move(move_idx)
            if move is None:
                return 'draw', 'invalid_move'
            
            fr, fc, tr, tc = move
            game.make_move((fr, fc), (tr, tc), validate=False)
            
            is_over, result = game.is_game_over()
            if is_over:
                result, reason = game_outcome(
                    game, material_adjudication_threshold=None,
                )
                return result, reason
        
        result, reason = game_outcome(
            game, hit_move_limit=True, material_adjudication_threshold=None,
        )
        return result, reason


class TrainingPipeline:
    """
    完整的训练流水线
    
    整合自对弈、训练、评估的完整流程。
    """
    
    def __init__(self,
                 num_blocks: int = 6,
                 channels: int = 128,
                 policy_channels: int = 4,
                 num_simulations: int = 400,
                 initial_simulations: int = 100,
                 simulation_ramp_iterations: int = 30,
                 inference_batch_size: int = 64,
                 inference_cache_size: int = 10000,
                 self_play_workers: int = 0,
                 inference_server_batch_size: int = 256,
                 inference_server_wait_ms: float = 5.0,
                 eval_simulations: int = 200,
                 eval_games: int = 20,
                 self_play_games: int = 32,
                 replay_ratio: float = 4.0,
                 min_training_steps: int = 32,
                 max_training_steps: int = 128,
                 batch_size: int = 256,
                 data_workers: int = 4,
                 learning_rate: float = 3e-4,
                 buffer_size: int = 50000,
                 recent_window_size: int = 10000,
                 recent_sample_fraction: float = 0.5,
                 material_adjudication_threshold: float = 0.10,
                 evaluate_every: int = 5,
                 acceptance_threshold: float = 0.55,
                 save_dir: str = 'models',
                 resume_from: str = None,
                 load_buffer: str = None,
                 num_iterations: int = 100):
        """
        Args:
            num_blocks: 残差块数量
            channels: 特征通道数
            num_simulations: MCTS 模拟次数
            inference_batch_size: MCTS GPU 叶节点推理批大小
            inference_cache_size: 当前模型权重的 LRU 推理缓存容量
            self_play_workers: 并行生成棋局的 CPU actor 进程数
            inference_server_batch_size: 主进程集中式 GPU 推理最大 batch
            inference_server_wait_ms: 集中推理合批等待时间（毫秒）
            eval_simulations: 模型评估时每步 MCTS 模拟次数
            self_play_games: 每轮自对弈局数
            replay_ratio: 每条新样本的目标重放次数
            batch_size: 训练批大小
            data_workers: 训练 DataLoader 工作进程数
            learning_rate: 初始学习率
            buffer_size: 经验回放缓冲区大小
            recent_window_size: 分层采样时视为近期数据的末尾样本数
            recent_sample_fraction: 每轮训练从近期数据抽取的目标比例
            material_adjudication_threshold: 无进展/超时对局的子力裁定阈值
            evaluate_every: 每隔多少轮评估一次
            acceptance_threshold: 候选模型接受的最低计分率
            save_dir: 模型保存目录
            resume_from: 从指定 checkpoint 继续训练
            load_buffer: 从 replay 目录或 manifest 加载数据
            num_iterations: 总训练迭代次数（用于余弦退火计算）
        """
        self.num_blocks = num_blocks
        self.channels = channels
        self.policy_channels = policy_channels
        self.num_simulations = num_simulations
        self.initial_simulations = min(initial_simulations, num_simulations)
        self.simulation_ramp_iterations = max(1, simulation_ramp_iterations)
        self.inference_batch_size = inference_batch_size
        self.inference_cache_size = max(0, inference_cache_size)
        self.self_play_workers = resolve_self_play_workers(
            self_play_workers, self_play_games
        )
        self.inference_server_batch_size = max(
            inference_batch_size, inference_server_batch_size
        )
        self.inference_server_wait_ms = max(0.0, inference_server_wait_ms)
        self.eval_simulations = eval_simulations
        self.eval_games = eval_games
        self.self_play_games = self_play_games
        self.replay_ratio = max(0.0, replay_ratio)
        self.min_training_steps = max(1, min_training_steps)
        self.max_training_steps = max(self.min_training_steps, max_training_steps)
        self.batch_size = batch_size
        self.data_workers = max(0, data_workers)
        self.learning_rate = learning_rate
        self.buffer_size = max(1, buffer_size)
        self.replay_window_size = self.buffer_size
        self.recent_window_size = max(1, recent_window_size)
        if not 0.0 <= recent_sample_fraction <= 1.0:
            raise ValueError("recent_sample_fraction 必须在 0 到 1 之间")
        self.recent_sample_fraction = recent_sample_fraction
        self.material_adjudication_threshold = max(
            0.0, material_adjudication_threshold
        )
        self.evaluate_every = evaluate_every
        self.save_dir = save_dir
        self.start_iteration = 1

        # 不允许“从零训练”静默覆盖或混入已有训练产物。
        if not resume_from and os.path.isdir(save_dir):
            existing_artifacts = []
            for name in os.listdir(save_dir):
                if name == 'latest_model.pt' or (
                    name.startswith('checkpoint_iter_') and name.endswith('.pt')
                ):
                    existing_artifacts.append(os.path.join(save_dir, name))
            existing_manifest = os.path.join(save_dir, 'replay', 'manifest.json')
            if os.path.exists(existing_manifest):
                existing_artifacts.append(existing_manifest)
            if existing_artifacts:
                raise FileExistsError(
                    f"保存目录 {save_dir} 已包含训练产物；从零训练请指定新的 "
                    "--save-dir，继续训练请显式使用 --resume"
                )

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 初始化模型；续训时检查点中的网络结构是唯一数据源。
        device = get_default_device()
        resume_checkpoint = None
        if resume_from:
            if not os.path.exists(resume_from):
                raise FileNotFoundError(f"checkpoint 不存在: {resume_from}")
            resume_checkpoint = read_checkpoint(resume_from, device)
            model_config = resume_checkpoint['model_config']
            self.num_blocks = model_config['num_blocks']
            self.channels = model_config['channels']
            self.policy_channels = model_config['policy_channels']
        self.model = create_model(
            self.num_blocks,
            self.channels,
            self.policy_channels,
            device=device,
        )
        self.device = device

        # 训练器（需要在 resume 之前创建，以便加载优化器状态）
        # 学习率每个训练迭代更新一次，而非随不断增长的 buffer 长度变化。
        total_training_iterations = num_iterations
        self.trainer = Trainer(
            self.model, learning_rate=learning_rate,
            total_training_iterations=total_training_iterations,
        )

        # 断点续训：加载 checkpoint
        if resume_checkpoint is not None:
            completed_iteration = self.trainer.restore_checkpoint(
                resume_checkpoint, resume_from
            )
            self.start_iteration = completed_iteration + 1
            print(f"[训练] 从迭代 {self.start_iteration} 继续训练")

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=self.replay_window_size)
        self.replay_dir = os.path.join(save_dir, 'replay')
        self.replay_manifest_path = os.path.join(self.replay_dir, 'manifest.json')

        # 加载 replay buffer
        buffer_loaded = False
        if load_buffer:
            if not os.path.exists(load_buffer):
                raise FileNotFoundError(f"replay 路径不存在: {load_buffer}")
            self._load_replay_source(load_buffer)
            buffer_loaded = True
        elif resume_from:
            checkpoint_dir = os.path.dirname(resume_from)
            auto_manifest = os.path.join(checkpoint_dir, 'replay', 'manifest.json')
            if os.path.exists(auto_manifest):
                self._load_replay_shards(auto_manifest)
                buffer_loaded = True

        if not buffer_loaded:
            print(f"[训练] replay buffer 为空，从零开始收集数据")

        self.self_play_worker = SelfPlayWorker(
            self.model, num_simulations=self.initial_simulations,
            inference_batch_size=inference_batch_size,
            inference_cache_size=self.inference_cache_size,
            num_workers=self.self_play_workers,
            inference_server_batch_size=self.inference_server_batch_size,
            inference_server_wait_ms=self.inference_server_wait_ms,
            material_adjudication_threshold=(
                self.material_adjudication_threshold
            ),
        )

        # 评估器
        self.evaluator = ModelEvaluator(
            num_games=self.eval_games, num_simulations=eval_simulations,
            inference_batch_size=inference_batch_size,
            inference_cache_size=self.inference_cache_size,
            acceptance_threshold=acceptance_threshold,
        )

        self.best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        if resume_from and os.path.exists(self.best_model_path):
            best_checkpoint = read_checkpoint(self.best_model_path, self.device)
            self.best_model_state = {
                key: value.clone()
                for key, value in best_checkpoint['model_state_dict'].items()
            }
        else:
            self.best_model_state = {
                key: value.clone() for key, value in self.model.state_dict().items()
            }

    def _save_replay_shard(self, new_data: List[Tuple], iteration: int):
        """只保存本轮新增样本，并通过 manifest 管理最近的分片。"""
        if not new_data:
            return
        os.makedirs(self.replay_dir, exist_ok=True)
        manifest = {'shards': []}
        if os.path.exists(self.replay_manifest_path):
            with open(self.replay_manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

        shard_name = f"shard_iter_{iteration:05d}_{time.time_ns()}.pkl"
        shard_path = os.path.join(self.replay_dir, shard_name)
        temp_shard_path = f"{shard_path}.tmp"
        with open(temp_shard_path, 'wb') as f:
            pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_shard_path, shard_path)
        manifest['shards'].append({'file': shard_name, 'count': len(new_data)})

        total = sum(item['count'] for item in manifest['shards'])
        removed_shards = []
        while total > self.replay_window_size and len(manifest['shards']) > 1:
            shard = manifest['shards'].pop(0)
            total -= shard['count']
            removed_shards.append(shard['file'])

        manifest['sample_count'] = total
        temp_manifest = self.replay_manifest_path + '.tmp'
        with open(temp_manifest, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(temp_manifest, self.replay_manifest_path)
        for shard_name_to_remove in removed_shards:
            shard_path_to_remove = os.path.join(self.replay_dir, shard_name_to_remove)
            if os.path.exists(shard_path_to_remove):
                os.remove(shard_path_to_remove)
        print(
            f"[训练] replay 增量分片已保存: {shard_name} "
            f"(+{len(new_data)}，磁盘保留约 {total} 条)"
        )

    def _load_replay_source(self, path: str):
        """加载 replay 目录或 manifest。"""
        if os.path.isdir(path):
            manifest_path = os.path.join(path, 'manifest.json')
            self._load_replay_shards(manifest_path)
        elif path.endswith('.json'):
            self._load_replay_shards(path)
        else:
            raise ValueError("replay 路径必须是目录或 manifest.json")

    def _load_replay_shards(self, manifest_path: str):
        """按 manifest 顺序恢复增量 replay 分片。"""
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        replay_dir = os.path.dirname(manifest_path)
        for shard in manifest.get('shards', []):
            with open(os.path.join(replay_dir, shard['file']), 'rb') as f:
                samples = pickle.load(f)
            self.replay_buffer.extend(samples)
        print(
            f"[训练] replay 分片已加载: {manifest_path} "
            f"({len(self.replay_buffer)} 条样本)"
        )

    def _simulations_for_iteration(self, iteration: int) -> int:
        progress = min(1.0, max(0.0, (iteration - 1) / self.simulation_ramp_iterations))
        return round(
            self.initial_simulations
            + progress * (self.num_simulations - self.initial_simulations)
        )

    def _training_steps_for_new_data(self, sample_count: int) -> int:
        requested = int(np.ceil(sample_count * self.replay_ratio / self.batch_size))
        return min(self.max_training_steps, max(self.min_training_steps, requested))

    def run(self, num_iterations: int = 100):
        """
        运行完整的训练流程。

        Args:
            num_iterations: 训练迭代次数
        """
        print("=" * 60)
        print("中国象棋 AI 训练开始")
        print("=" * 60)
        print(f"配置:")
        print(
            f"  模型: ResNet-{self.num_blocks}x{self.channels} "
            f"(策略头 {self.policy_channels} 通道, WDL 价值头)"
        )
        print(
            f"  MCTS 模拟次数: {self.initial_simulations} -> "
            f"{self.num_simulations}"
        )
        print(f"  MCTS 推理批大小: {self.inference_batch_size}")
        print(f"  自对弈 CPU actors: {self.self_play_workers}")
        print(f"  集中式推理最大 batch: {self.inference_server_batch_size}")
        print(f"  集中式推理合批等待: {self.inference_server_wait_ms:.1f} ms")
        print(f"  推理缓存容量: {self.inference_cache_size}")
        print(f"  评估 MCTS 模拟次数: {self.eval_simulations}")
        print(f"  候选模型接受门槛: {self.evaluator.acceptance_threshold:.0%}")
        print(f"  每轮自对弈: {self.self_play_games} 局")
        print(
            f"  训练/新样本比: {self.replay_ratio:.1f}x，"
            f"batch 范围 {self.min_training_steps}-{self.max_training_steps}"
        )
        print(f"  训练批大小: {self.batch_size}")
        print(f"  Replay 窗口: {self.replay_window_size} 条")
        print(
            f"  近期样本: 末尾 {self.recent_window_size} 条，"
            f"训练占比 {self.recent_sample_fraction:.0%}"
        )
        print(
            "  无进展/超时子力裁定阈值: "
            f"{self.material_adjudication_threshold:.1%}"
        )
        print(f"  数据加载进程: {self.data_workers}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  总迭代次数: {num_iterations}")
        print("=" * 60)

        if not os.path.exists(self.best_model_path):
            self.trainer.save_model(
                self.best_model_path, self.start_iteration - 1,
                self.best_model_state,
            )

        for iteration in range(self.start_iteration, num_iterations + 1):
            print(f"\n{'='*60}")
            print(f"迭代 {iteration}/{num_iterations}")
            print(f"{'='*60}")

            start_time = time.time()

            current_simulations = self._simulations_for_iteration(iteration)
            self.self_play_worker.num_simulations = current_simulations
            print(f"  本轮 MCTS 模拟次数: {current_simulations}")

            # 步骤1：自对弈生成训练数据
            print("\n[步骤1] 自对弈中...")
            self_play_start = time.time()
            game_data = self.self_play_worker.play_multiple_games(self.self_play_games)
            self_play_elapsed = time.time() - self_play_start
            
            # 添加到经验回放缓冲区
            self.replay_buffer.extend(game_data)
            print(f"  新增样本: {len(game_data)}，缓冲区大小: {len(self.replay_buffer)}")
            print(f"  自对弈耗时: {self_play_elapsed:.1f} 秒")
            
            # 步骤2：训练神经网络
            print("\n[步骤2] 训练神经网络...")
            training_start = time.time()
            if len(self.replay_buffer) >= self.batch_size:
                training_steps = self._training_steps_for_new_data(len(game_data))
                dataset = GameDataset(list(self.replay_buffer), augment=True)
                sample_weights = build_replay_sample_weights(
                    len(dataset), self.recent_window_size,
                    self.recent_sample_fraction,
                )
                sampler = WeightedRandomSampler(
                    sample_weights,
                    num_samples=training_steps * self.batch_size,
                    replacement=True,
                )
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.batch_size,
                    sampler=sampler,
                    num_workers=self.data_workers,
                    pin_memory=self.device == 'cuda',
                    persistent_workers=self.data_workers > 0,
                    drop_last=True,
                )

                metrics = self.trainer.train_iteration(dataloader, iteration)
                print(f"  {training_steps} batches: loss={metrics['loss']:.4f}, "
                      f"policy={metrics['policy_loss']:.4f}, "
                      f"wdl={metrics['value_loss']:.4f}")
                if metrics['skipped_batches']:
                    print(
                        f"  AMP 梯度溢出恢复: 跳过 {metrics['skipped_batches']} 个 batch，"
                        f"成功更新 {metrics['successful_batches']} 次，"
                        f"当前 scale={metrics['amp_scale']:.0f}"
                    )
            else:
                print(f"  样本不足（需要至少 {self.batch_size}），跳过训练")
            print(f"  网络训练耗时: {time.time() - training_start:.1f} 秒")
            
            # 步骤3：评估新模型
            if iteration % self.evaluate_every == 0:
                print("\n[步骤3] 评估新模型...")
                evaluation_start = time.time()
                # 创建当前基准模型副本（使用上次评估通过的权重）
                reference_model = create_model(
                    self.num_blocks,
                    self.channels,
                    self.policy_channels,
                    device=self.device,
                )
                reference_model.load_state_dict(self.best_model_state)

                eval_result = self.evaluator.evaluate(self.model, reference_model)
                print(f"  纯胜率: {eval_result['win_rate']:.1%}")
                print(f"  计分率（和棋计半分）: {eval_result['score_rate']:.1%}")
                print(f"  候选模型胜: {eval_result['candidate_wins']}, "
                      f"基准模型胜: {eval_result['reference_wins']}, "
                      f"和棋: {eval_result['draws']}")
                print(f"  评估终局统计: {eval_result['termination_counts']}")
                print(f"  模型评估耗时: {time.time() - evaluation_start:.1f} 秒")

                if eval_result['accepted']:
                    print("  ✅ 新模型已接受")
                    self.best_model_state = {
                        key: value.clone()
                        for key, value in self.model.state_dict().items()
                    }
                    self.trainer.save_model(
                        self.best_model_path, iteration, self.best_model_state
                    )
                else:
                    print("  ❌ 本轮未刷新 best；latest 保留并继续学习")
            
            # 保存检查点
            if iteration % 5 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_iter_{iteration}.pt')
                self.trainer.save_checkpoint(checkpoint_path, iteration)

            # 保存最新模型
            latest_path = os.path.join(self.save_dir, 'latest_model.pt')
            self.trainer.save_checkpoint(latest_path, iteration)

            # 增量保存本轮 replay 分片，避免每轮重写整个缓冲区。
            replay_save_start = time.time()
            self._save_replay_shard(game_data, iteration)
            print(f"  replay 保存耗时: {time.time() - replay_save_start:.1f} 秒")
            
            elapsed = time.time() - start_time
            print(f"\n本轮耗时: {elapsed:.1f} 秒")
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)


def train_from_scratch():
    """从零开始训练的入口函数"""
    pipeline = TrainingPipeline(
        num_blocks=6,
        channels=128,
        policy_channels=4,
        num_simulations=400,   # MCTS 模拟次数
        initial_simulations=100,
        inference_batch_size=64, # MCTS GPU 推理批大小
        inference_cache_size=10000, # 当前模型权重的 LRU 局面数
        eval_simulations=200,   # 评估使用较少模拟，降低门控成本
        self_play_games=32,
        replay_ratio=4.0,
        batch_size=256,        # 批大小（RTX 5070 12GB 足够）
        data_workers=4,        # 并行准备训练 batch
        learning_rate=3e-4,
        buffer_size=50000,
        evaluate_every=2,      # 每2轮评估一次
        save_dir='models',     # 模型保存目录
        num_iterations=100,    # 总迭代次数
    )

    pipeline.run(num_iterations=100)


if __name__ == '__main__':
    train_from_scratch()
