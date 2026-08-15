"""
神经网络模型模块
================
实现 AlphaZero 风格的策略-价值网络（Policy-Value Network）。

网络架构：
  输入 → 卷积层 → 残差块 × N → 策略头 + 价值头

  - 策略头（Policy Head）：输出每个走法的概率分布
  - 价值头（WDL Head）：输出当前走棋方的负/和/胜概率

设计选择：
  - 使用 ResNet 残差块而非纯 Transformer，因为：
    1. 棋盘是固定大小的空间结构，CNN 天然适合
    2. ResNet 在 AlphaZero/Leela Chess Zero 中已验证有效
    3. 训练更稳定，收敛更快
  - 可选：在残差块后加 Transformer 注意力层捕捉长距离依赖

硬件优化：
  - 针对 RTX 5070 (12GB VRAM) 优化
  - 支持混合精度训练 (AMP)
  - 使用 channels_last 内存格式加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from constants import BOARD_ROWS, BOARD_COLS
from encoding import NUM_INPUT_PLANES
from move_index import get_move_index


class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）
    
    结构：
      输入 → Conv → BN → ReLU → Conv → BN → (+输入) → ReLU
    
    残差连接让网络可以训练更深的模型，避免梯度消失。
    每个残差块包含两个 3×3 卷积层和批归一化。
    """
    
    def __init__(self, channels: int):
        """
        Args:
            channels: 特征通道数（如 256）
        """
        super().__init__()
        # 第一个卷积：3×3，保持空间尺寸不变
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 第二个卷积：3×3
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入特征图 [batch, channels, rows, cols]
            
        Returns:
            输出特征图，形状与输入相同
        """
        residual = x  # 保存输入用于残差连接
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual  # 残差连接：输出 = F(x) + x
        out = F.relu(out)
        
        return out


class PolicyValueNet(nn.Module):
    """
    策略-价值网络
    
    输入：当前走棋方视角的状态张量 [batch, 19, 10, 9]
    
    输出：
      - policy: [batch, num_moves] 走法概率分布
      - wdl: [batch, 3] 负/和/胜 logits
    
    网络结构：
      输入(19通道) → Conv2d(128) → ResBlock×6 →
        ├→ 策略头: Conv(4) → FC(num_moves)
        └→ WDL 头: Conv(3) → FC(256) → FC(3)
    """
    
    def __init__(self, num_blocks: int = 6, channels: int = 128,
                 policy_channels: int = 4):
        """
        Args:
            num_blocks: 残差块数量（更多块 = 更强但更慢）
            channels: 特征通道数（更多通道 = 更强但更占显存）
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        self.policy_channels = policy_channels
        
        # 获取走法总数（网络输出维度）
        mi = get_move_index()
        self.num_moves = mi.num_moves
        
        # === 输入层 ===
        # 将规范化棋盘、上一步和规则状态映射到特征空间。
        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_INPUT_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        
        # === 残差块堆叠 ===
        # 这是网络的"躯干"，负责提取棋盘特征
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        
        # === 策略头（Policy Head） ===
        # 输出每个走法的概率
        # 使用 1×1 卷积降低通道数，然后全连接到走法数
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_channels * BOARD_ROWS * BOARD_COLS, self.num_moves),
        )
        
        # === 价值头（Value Head） ===
        # 从当前走棋方视角输出 [loss, draw, win] logits。
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),                    # 展平：3 × 10 × 9 = 270
            nn.Linear(3 * BOARD_ROWS * BOARD_COLS, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            x: 棋盘状态张量 [batch, 19, 10, 9]
            
        Returns:
            (policy, wdl):
            - policy: [batch, num_moves] 走法概率（未经 softmax）
            - wdl: [batch, 3] 负/和/胜 logits
        """
        # 提取特征
        features = self.input_conv(x)
        features = self.residual_blocks(features)
        
        # 策略头：输出走法 logits（在 MCTS 中会加 softmax）
        policy = self.policy_head(features)
        
        # 价值头：输出局面评估
        wdl = self.value_head(features)
        
        return policy, wdl
    
    def predict(self, board_tensor: 'list') -> Tuple[list, float]:
        """
        单个局面的推理接口（不使用 DataLoader）。
        
        Args:
            board_tensor: 19×10×9 的三维数组（来自 Game.get_board_tensor()）
            
        Returns:
            (policy_probs, value):
            - policy_probs: 长度为 num_moves 的概率列表
            - value: 局面评估值 [-1, 1]
        """
        policies, values = self.predict_batch([board_tensor])
        return policies[0], values[0]

    def predict_batch(self, board_tensors: List['list']) -> Tuple[np.ndarray, np.ndarray]:
        """批量推理接口；一次 GPU 调用评估多个 MCTS 叶节点。"""
        if len(board_tensors) == 0:
            return (
                np.empty((0, self.num_moves), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        self.eval()
        device = next(self.parameters()).device
        states = np.asarray(board_tensors, dtype=np.float32)
        x = torch.from_numpy(states).to(device)
        x = x.to(memory_format=torch.channels_last)

        use_amp = device.type == 'cuda'
        with torch.inference_mode(), torch.amp.autocast(
            'cuda', dtype=torch.float16, enabled=use_amp
        ):
            policy_logits, wdl_logits = self.forward(x)
            policy_probs = F.softmax(policy_logits, dim=1)
            wdl_probs = F.softmax(wdl_logits, dim=1)

        # 每个 batch 只同步一次 GPU；有限值检查在 CPU 上完成。
        policies_cpu = policy_probs.float().cpu().numpy()
        wdl_cpu = wdl_probs.float().cpu().numpy()
        values_cpu = wdl_cpu[:, 2] - wdl_cpu[:, 0]
        if not np.isfinite(policies_cpu).all() or not np.isfinite(values_cpu).all():
            raise FloatingPointError("模型推理输出出现 NaN/Inf，请勿继续使用该检查点")
        return policies_cpu, values_cpu
    
    def count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> str:
        """获取模型信息摘要"""
        total_params = self.count_parameters()
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"PolicyValueNet:\n"
            f"  残差块数: {self.num_blocks}\n"
            f"  特征通道: {self.channels}\n"
            f"  策略头通道: {self.policy_channels}\n"
            f"  走法数量: {self.num_moves}\n"
            f"  总参数量: {total_params:,}\n"
            f"  可训练参数: {trainable_params:,}\n"
            f"  模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)"
        )


def get_default_device() -> str:
    """选择当前机器上可用的最佳 PyTorch 设备。"""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def create_model(num_blocks: int = 6, channels: int = 128,
                 policy_channels: int = 4, device: str = None) -> PolicyValueNet:
    """
    创建并初始化策略-价值网络。
    
    Args:
        num_blocks: 残差块数量
            - 5块: 快速原型，适合调试
            - 10块: 标准配置，适合正式训练
            - 20块: 更强但训练更慢
        channels: 特征通道数
            - 128: 轻量级，适合快速实验
            - 256: 标准配置
            - 384/512: 更强但需要更多显存
        device: 计算设备（None 时依次选择 CUDA、Apple MPS、CPU）
        
    Returns:
        初始化好的 PolicyValueNet 模型
    """
    device = device or get_default_device()
    model = PolicyValueNet(
        num_blocks=num_blocks,
        channels=channels,
        policy_channels=policy_channels,
    )
    
    # 移动到 GPU
    if device == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        model = model.cuda()
        # 使用 channels_last 内存格式（NVIDIA GPU 优化）
        model = model.to(memory_format=torch.channels_last)
        print(f"[模型] 已创建并移动到 CUDA 设备: {torch.cuda.get_device_name()}")
    elif device == 'mps' and torch.backends.mps.is_available():
        model = model.to('mps')
        model = model.to(memory_format=torch.channels_last)
        print("[模型] 已创建并移动到 Apple MPS 设备")
    else:
        model = model.to('cpu')
        print("[模型] 已创建，使用 CPU 推理")
    
    print(model.get_model_info())
    return model


def read_checkpoint(path: str, device: str):
    """读取训练检查点；其字段结构由当前训练流程统一定义。"""
    return torch.load(path, map_location=device)


def create_model_from_checkpoint(path: str, device: str) -> PolicyValueNet:
    """按检查点记录的网络结构创建模型并加载权重。"""
    checkpoint = read_checkpoint(path, device)
    config = checkpoint['model_config']
    model = create_model(
        num_blocks=config['num_blocks'],
        channels=config['channels'],
        policy_channels=config['policy_channels'],
        device=device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[模型] 已加载检查点: {path}")
    return model


if __name__ == '__main__':
    # 测试：创建模型并进行一次前向传播
    device = get_default_device()
    model = create_model(device=device)
    
    # 模拟输入
    dummy_input = torch.randn(1, NUM_INPUT_PLANES, BOARD_ROWS, BOARD_COLS)
    if device != 'cpu':
        dummy_input = dummy_input.to(device)
    dummy_input = dummy_input.to(memory_format=torch.channels_last)
    
    policy, wdl = model(dummy_input)
    print(f"\n测试前向传播:")
    print(f"  策略输出形状: {policy.shape}")
    print(f"  WDL 输出形状: {wdl.shape}")
