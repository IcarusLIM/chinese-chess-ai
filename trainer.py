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
  3. 评估新模型
     - 新模型 vs 旧模型对弈
     - 胜率 > 55% 则接受新模型
  4. 重复

关键超参数：
  - 每次自对弈生成多少局
  - 每局最多走多少步
  - 训练的 epoch 数
  - 学习率和调度策略

硬件优化：
  - RTX 5070 (12GB VRAM) 优化配置
  - 混合精度训练 (AMP)
  - 批量推理加速 MCTS
"""

import os
import time
import random
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from collections import deque

from game import Game
from network import PolicyValueNet, create_model
from mcts import MCTS, MCTSEvaluator
from move_index import get_move_index
from constants import BOARD_ROWS, BOARD_COLS


class GameDataset(Dataset):
    """
    棋局数据集
    
    存储自对弈产生的训练样本，每个样本包含：
    - board_tensor: 棋盘状态 [15, 10, 9]
    - policy: MCTS 策略分布 [num_moves]
    - value: 游戏结果（+1 红胜, -1 黑胜, 0 和棋）
    """
    
    def __init__(self, data: List[Tuple]):
        """
        Args:
            data: 样本列表，每个元素为 (board_tensor, policy, value)
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board_tensor, policy, value = self.data[idx]
        return (
            torch.tensor(board_tensor, dtype=torch.float32),
            torch.tensor(policy, dtype=torch.float32),
            torch.tensor(value, dtype=torch.float32),
        )


class SelfPlayWorker:
    """
    自对弈工作器
    
    负责执行自对弈并收集训练数据。
    使用 MCTS 搜索选择走法，记录每步的训练样本。
    """
    
    def __init__(self,
                 model: PolicyValueNet,
                 num_simulations: int = 400,
                 max_moves: int = 200,
                 temperature_threshold: int = 30,
                 material_weight: float = 0.0):
        """
        Args:
            model: 当前模型
            num_simulations: MCTS 模拟次数
            max_moves: 每局最大走子数（超过判和）
            temperature_threshold: 走子数超过此值后温度降为 0.01
            material_weight: MCTS 叶节点材料评估混合权重
        """
        self.model = model
        self.num_simulations = num_simulations
        self.max_moves = max_moves
        self.temperature_threshold = temperature_threshold
        self.material_weight = material_weight
        self.move_index = get_move_index()
    
    def play_one_game(self) -> List[Tuple]:
        """
        执行一局自对弈，收集训练数据。
        
        Returns:
            训练样本列表，每个元素为 (board_tensor, policy, value)
            其中 value 是最终游戏结果（从红方视角）
        """
        game = Game()
        mcts = MCTS(self.model, num_simulations=self.num_simulations,
                    material_weight=self.material_weight)
        
        # 存储每一步的数据
        states = []    # 棋盘状态
        policies = []  # MCTS 策略
        
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
            states.append(game.get_board_tensor())
            policies.append(policy)
            
            # 执行走子
            move = self.move_index.index_to_move(move_idx)
            if move is None:
                break
            fr, fc, tr, tc = move
            game.make_move((fr, fc), (tr, tc), validate=False)
            move_count += 1
            
            # 检查游戏是否结束
            is_over, result = game.is_game_over()
            if is_over:
                break
        
        # 确定游戏结果
        is_over, result = game.is_game_over()
        if result == 'red_wins':
            final_value = 1.0
        elif result == 'black_wins':
            final_value = -1.0
        else:
            final_value = 0.0  # 和棋或超时
        
        # 构建训练样本
        # 价值从当前走棋方的视角：红方走棋时为 final_value，黑方走棋时翻转
        training_data = []
        for i, (state, policy) in enumerate(zip(states, policies)):
            # 第 i 步时的走棋方（红方先走，交替进行）
            if i % 2 == 0:
                value = final_value  # 红方走棋
            else:
                value = -final_value  # 黑方走棋
            training_data.append((state, policy, value))
        
        return training_data
    
    def play_multiple_games(self, num_games: int) -> List[Tuple]:
        """
        执行多局自对弈，收集训练数据。
        
        Args:
            num_games: 对弈局数
            
        Returns:
            所有训练样本的列表
        """
        all_data = []
        for i in range(num_games):
            game_data = self.play_one_game()
            all_data.extend(game_data)
            if (i + 1) % 3 == 0:
                print(f"  自对弈进度: {i+1}/{num_games}，累计样本数: {len(all_data)}")
        return all_data


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
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 lr_milestones: List[int] = None,
                 lr_gamma: float = 0.1):
        """
        Args:
            model: 策略-价值网络
            learning_rate: 初始学习率
            weight_decay: L2 正则化系数
            lr_milestones: 学习率衰减的 epoch 节点
            lr_gamma: 学习率衰减因子
        """
        self.model = model
        self.learning_rate = learning_rate
        
        # 优化器：使用 SGD + 动量（比 Adam 更稳定）
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        
        # 学习率调度器：在指定 epoch 衰减学习率
        if lr_milestones is None:
            lr_milestones = [50, 100, 150]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=lr_gamma
        )
        
        # 混合精度训练（RTX 5070 优化）
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 训练统计
        self.training_history = []
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        训练一个 epoch。
        
        Args:
            dataloader: 训练数据加载器
            epoch: 当前 epoch 编号
            
        Returns:
            训练指标字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for batch_idx, (states, target_policies, target_values) in enumerate(dataloader):
            device = next(self.model.parameters()).device
            # 移动到计算设备
            states = states.to(device, non_blocking=True)
            target_policies = target_policies.to(device, non_blocking=True)
            target_values = target_values.to(device, non_blocking=True)
            
            # channels_last 内存格式
            states = states.to(memory_format=torch.channels_last)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 混合精度前向传播（仅 CUDA 启用）
            use_amp = device.type == 'cuda'
            with torch.amp.autocast('cuda', enabled=use_amp):
                policy_logits, value_pred = self.model(states)
                
                # 策略损失：交叉熵
                # 使用 log_softmax + NLLLoss 更数值稳定
                policy_loss = -torch.mean(
                    torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1)
                )
                
                # 价值损失：均方误差
                value_loss = F.mse_loss(value_pred.squeeze(-1), target_values)
                
                # 总损失
                loss = policy_loss + value_loss
            
            # 混合精度反向传播
            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / max(num_batches, 1)
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        
        metrics = {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'lr': self.scheduler.get_last_lr()[0],
        }
        
        self.training_history.append(metrics)
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, additional_info: Dict = None):
        """
        保存模型检查点。
        
        Args:
            path: 保存路径
            epoch: 当前 epoch
            additional_info: 额外保存的信息
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
        }
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
        print(f"[训练] 检查点已保存: {path}")
    
    def load_checkpoint(self, path: str) -> dict:
        """
        加载模型检查点。

        Args:
            path: 检查点文件路径

        Returns:
            checkpoint 中的 additional_info 字典（包含 iteration 等）
        """
        device = next(self.model.parameters()).device
        checkpoint = torch.load(path, map_location=device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', [])

        epoch = checkpoint['epoch']
        print(f"[训练] 检查点已加载: {path} (epoch {epoch})")

        # 返回额外信息（包含 iteration 等）
        additional = {}
        for key, value in checkpoint.items():
            if key not in ('model_state_dict', 'optimizer_state_dict',
                           'scheduler_state_dict', 'training_history', 'epoch'):
                additional[key] = value
        return additional


class ModelEvaluator:
    """
    模型评估器
    
    通过让新旧模型对弈来评估模型强度。
    胜率超过阈值则接受新模型。
    """
    
    def __init__(self, num_games: int = 20, num_simulations: int = 200):
        """
        Args:
            num_games: 评估对弈局数
            num_simulations: MCTS 模拟次数
        """
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.move_index = get_move_index()
    
    def evaluate(self, new_model: PolicyValueNet, old_model: PolicyValueNet) -> Dict:
        """
        评估新模型相对于旧模型的胜率。
        
        Args:
            new_model: 新训练的模型
            old_model: 旧模型（基线）
            
        Returns:
            评估结果字典
        """
        new_wins = 0
        old_wins = 0
        draws = 0
        
        for game_idx in range(self.num_games):
            # 交替先后手
            if game_idx % 2 == 0:
                red_model, black_model = new_model, old_model
                new_is_red = True
            else:
                red_model, black_model = old_model, new_model
                new_is_red = False
            
            result = self._play_evaluation_game(red_model, black_model)
            
            if result == 'red_wins':
                if new_is_red:
                    new_wins += 1
                else:
                    old_wins += 1
            elif result == 'black_wins':
                if new_is_red:
                    old_wins += 1
                else:
                    new_wins += 1
            else:
                draws += 1
            
            if (game_idx + 1) % 5 == 0:
                print(f"  评估进度: {game_idx+1}/{self.num_games}")
        
        total = new_wins + old_wins + draws
        win_rate = new_wins / total if total > 0 else 0
        
        return {
            'new_wins': new_wins,
            'old_wins': old_wins,
            'draws': draws,
            'win_rate': win_rate,
            'accepted': win_rate > 0.55,
        }
    
    def _play_evaluation_game(self, red_model: PolicyValueNet, black_model: PolicyValueNet) -> str:
        """执行一局评估对弈"""
        game = Game()
        red_mcts = MCTS(red_model, num_simulations=self.num_simulations)
        black_mcts = MCTS(black_model, num_simulations=self.num_simulations)
        
        for _ in range(200):
            if game.red_to_move:
                move_idx, _ = red_mcts.search(game, temperature=0.01)
            else:
                move_idx, _ = black_mcts.search(game, temperature=0.01)
            
            if move_idx < 0:
                return 'draw'
            
            move = self.move_index.index_to_move(move_idx)
            if move is None:
                return 'draw'
            
            fr, fc, tr, tc = move
            game.make_move((fr, fc), (tr, tc), validate=False)
            
            is_over, result = game.is_game_over()
            if is_over:
                return result
        
        return 'draw'


class TrainingPipeline:
    """
    完整的训练流水线
    
    整合自对弈、训练、评估的完整流程。
    """
    
    def __init__(self,
                 num_blocks: int = 10,
                 channels: int = 256,
                 num_simulations: int = 400,
                 self_play_games: int = 25,
                 training_epochs: int = 5,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 buffer_size: int = 100000,
                 evaluate_every: int = 5,
                 save_dir: str = 'models',
                 resume_from: str = None,
                 load_buffer: str = None,
                 material_warmup: int = 0):
        """
        Args:
            num_blocks: 残差块数量
            channels: 特征通道数
            num_simulations: MCTS 模拟次数
            self_play_games: 每轮自对弈局数
            training_epochs: 每轮训练的 epoch 数
            batch_size: 训练批大小
            learning_rate: 初始学习率
            buffer_size: 经验回放缓冲区大小
            evaluate_every: 每隔多少轮评估一次
            save_dir: 模型保存目录
            resume_from: 从指定 checkpoint 继续训练
            load_buffer: 从指定文件加载 replay buffer
            material_warmup: 材料评估 warmup 迭代数（0=不启用）
        """
        self.num_blocks = num_blocks
        self.channels = channels
        self.num_simulations = num_simulations
        self.self_play_games = self_play_games
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.evaluate_every = evaluate_every
        self.save_dir = save_dir
        self.material_warmup = material_warmup
        self.start_iteration = 1

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 初始化模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = create_model(num_blocks, channels, device=device)
        self.device = device

        # 训练器（需要在 resume 之前创建，以便加载优化器状态）
        self.trainer = Trainer(self.model, learning_rate=learning_rate)

        # 断点续训：加载 checkpoint
        if resume_from and os.path.exists(resume_from):
            additional = self.trainer.load_checkpoint(resume_from)
            if isinstance(additional, dict):
                self.start_iteration = additional.get('iteration', 1) + 1
            print(f"[训练] 从迭代 {self.start_iteration} 继续训练")

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=buffer_size)

        # 加载 replay buffer
        buffer_loaded = False
        if load_buffer and os.path.exists(load_buffer):
            self._load_replay_buffer(load_buffer)
            buffer_loaded = True
        elif resume_from:
            # 自动查找同目录下的 replay_buffer.pkl
            auto_buffer = os.path.join(os.path.dirname(resume_from), 'replay_buffer.pkl')
            if os.path.exists(auto_buffer):
                self._load_replay_buffer(auto_buffer)
                buffer_loaded = True

        if not buffer_loaded:
            print(f"[训练] replay buffer 为空，从零开始收集数据")

        # 自对弈工作器（material_weight 在 run() 中动态设置）
        self.self_play_worker = SelfPlayWorker(
            self.model, num_simulations=num_simulations
        )

        # 评估器
        self.evaluator = ModelEvaluator(num_games=20, num_simulations=num_simulations)

    def _save_replay_buffer(self):
        """将 replay buffer 保存到磁盘。"""
        buffer_path = os.path.join(self.save_dir, 'replay_buffer.pkl')
        try:
            with open(buffer_path, 'wb') as f:
                pickle.dump(list(self.replay_buffer), f)
            print(f"[训练] replay buffer 已保存: {buffer_path} ({len(self.replay_buffer)} 条样本)")
        except Exception as e:
            print(f"[训练] replay buffer 保存失败: {e}")

    def _load_replay_buffer(self, path: str):
        """从磁盘加载 replay buffer。"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.replay_buffer.extend(data)
            print(f"[训练] replay buffer 已加载: {path} ({len(self.replay_buffer)} 条样本)")
        except Exception as e:
            print(f"[训练] replay buffer 加载失败: {e}")

    def _get_material_weight(self, iteration: int) -> float:
        """
        根据当前迭代计算材料评估混合权重。

        Args:
            iteration: 当前迭代编号（从 1 开始）

        Returns:
            material_weight: 0.0（纯网络）到 1.0（纯材料评估）
        """
        if self.material_warmup <= 0:
            return 0.0
        if iteration >= self.material_warmup:
            return 0.0
        # 线性衰减：从 1.0 衰减到 0.0
        return 1.0 - (iteration - 1) / self.material_warmup

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
        print(f"  模型: ResNet-{self.num_blocks}x{self.channels}")
        print(f"  MCTS 模拟次数: {self.num_simulations}")
        print(f"  每轮自对弈: {self.self_play_games} 局")
        print(f"  训练批大小: {self.batch_size}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  总迭代次数: {num_iterations}")
        if self.material_warmup > 0:
            print(f"  材料评估 warmup: {self.material_warmup} 轮")
        print("=" * 60)

        # 记录上次评估通过时的状态（用于评估失败时回滚）
        accepted_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        accepted_optimizer_state = copy.deepcopy(self.trainer.optimizer.state_dict())
        accepted_scheduler_state = copy.deepcopy(self.trainer.scheduler.state_dict())

        for iteration in range(self.start_iteration, num_iterations + 1):
            print(f"\n{'='*60}")
            print(f"迭代 {iteration}/{num_iterations}")
            print(f"{'='*60}")

            start_time = time.time()

            # 动态设置材料评估权重
            material_weight = self._get_material_weight(iteration)
            self.self_play_worker.material_weight = material_weight
            if material_weight > 0:
                print(f"  材料评估权重: {material_weight:.2f}")

            # 步骤1：自对弈生成训练数据
            print("\n[步骤1] 自对弈中...")
            game_data = self.self_play_worker.play_multiple_games(self.self_play_games)
            
            # 添加到经验回放缓冲区
            self.replay_buffer.extend(game_data)
            print(f"  新增样本: {len(game_data)}，缓冲区大小: {len(self.replay_buffer)}")
            
            # 步骤2：训练神经网络
            print("\n[步骤2] 训练神经网络...")
            if len(self.replay_buffer) >= self.batch_size * 10:
                dataset = GameDataset(list(self.replay_buffer))
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                )
                
                for epoch in range(1, self.training_epochs + 1):
                    metrics = self.trainer.train_epoch(dataloader, epoch)
                    print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}, "
                          f"policy={metrics['policy_loss']:.4f}, "
                          f"value={metrics['value_loss']:.4f}")
            else:
                print(f"  样本不足（需要至少 {self.batch_size * 10}），跳过训练")
            
            # 步骤3：评估新模型
            if iteration % self.evaluate_every == 0:
                print("\n[步骤3] 评估新模型...")
                # 创建旧模型副本（使用上次评估通过的权重）
                old_model = create_model(self.num_blocks, self.channels, device=self.device)
                old_model.load_state_dict(accepted_model_state)

                eval_result = self.evaluator.evaluate(self.model, old_model)
                print(f"  胜率: {eval_result['win_rate']:.1%}")
                print(f"  新模型胜: {eval_result['new_wins']}, "
                      f"旧模型胜: {eval_result['old_wins']}, "
                      f"和棋: {eval_result['draws']}")

                if eval_result['accepted']:
                    print("  ✅ 新模型已接受")
                    # 更新 accepted 状态为当前模型
                    accepted_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    accepted_optimizer_state = copy.deepcopy(self.trainer.optimizer.state_dict())
                    accepted_scheduler_state = copy.deepcopy(self.trainer.scheduler.state_dict())
                else:
                    print("  ❌ 新模型未达标，回滚到上次通过的模型")
                    self.model.load_state_dict(accepted_model_state)
                    self.trainer.optimizer.load_state_dict(accepted_optimizer_state)
                    self.trainer.scheduler.load_state_dict(accepted_scheduler_state)
            
            # 保存检查点
            additional_info = {'iteration': iteration}
            if iteration % 5 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_iter_{iteration}.pt')
                self.trainer.save_checkpoint(checkpoint_path, iteration, additional_info)

            # 保存最新模型
            latest_path = os.path.join(self.save_dir, 'latest_model.pt')
            self.trainer.save_checkpoint(latest_path, iteration, additional_info)

            # 保存 replay buffer
            self._save_replay_buffer()
            
            elapsed = time.time() - start_time
            print(f"\n本轮耗时: {elapsed:.1f} 秒")
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)


def train_from_scratch():
    """从零开始训练的入口函数"""
    pipeline = TrainingPipeline(
        num_blocks=10,         # 残差块数量
        channels=256,          # 特征通道数
        num_simulations=400,   # MCTS 模拟次数
        self_play_games=25,    # 每轮自对弈局数
        training_epochs=5,     # 每轮训练 epoch 数
        batch_size=256,        # 批大小（RTX 5070 12GB 足够）
        learning_rate=0.001,   # 初始学习率
        buffer_size=100000,    # 经验回放缓冲区大小
        evaluate_every=5,      # 每5轮评估一次
        save_dir='models',     # 模型保存目录
    )
    
    pipeline.run(num_iterations=100)


if __name__ == '__main__':
    train_from_scratch()
