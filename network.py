"""
神经网络模型模块
================
实现 AlphaZero 风格的策略-价值网络（Policy-Value Network）。

网络架构：
  输入 → 卷积层 → 残差块 × N → 策略头 + 价值头

  - 策略头（Policy Head）：输出每个走法的概率分布
  - 价值头（Value Head）：输出当前局面的评估值 [-1, 1]
    - +1 表示红方必胜
    - -1 表示黑方必胜
    - 0 表示均势

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
from typing import Tuple
from constants import BOARD_ROWS, BOARD_COLS
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
    
    输入：棋盘状态张量 [batch, 15, 10, 9]
      - 15 个通道：红方7种棋子 + 黑方7种棋子 + 当前走棋方
      - 每个通道是该种棋子的位置热力图
    
    输出：
      - policy: [batch, num_moves] 走法概率分布
      - value: [batch, 1] 局面评估值 [-1, 1]
    
    网络结构：
      输入(15通道) → Conv2d(256) → ResBlock×10 →
        ├→ 策略头: Conv(32) → FC(num_moves) → policy
        └→ 价值头: Conv(3) → FC(256) → FC(1) → tanh → value
    """
    
    def __init__(self, num_blocks: int = 10, channels: int = 256):
        """
        Args:
            num_blocks: 残差块数量（更多块 = 更强但更慢）
            channels: 特征通道数（更多通道 = 更强但更占显存）
        """
        super().__init__()
        
        # 获取走法总数（网络输出维度）
        mi = get_move_index()
        self.num_moves = mi.num_moves
        
        # === 输入层 ===
        # 将 15 通道的棋盘状态映射到高维特征空间（14 棋子 + 1 走棋方）
        self.input_conv = nn.Sequential(
            nn.Conv2d(15, channels, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),  # 降维
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),                    # 展平：32 × 10 × 9 = 2880
            nn.Linear(32 * BOARD_ROWS * BOARD_COLS, self.num_moves),  # 全连接
        )
        
        # === 价值头（Value Head） ===
        # 输出局面评估值 [-1, 1]
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),                    # 展平：3 × 10 × 9 = 270
            nn.Linear(3 * BOARD_ROWS * BOARD_COLS, 256),
            nn.ReLU(),
            nn.Linear(256, 1),               # 输出单个标量
            nn.Tanh(),                       # 限制在 [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            x: 棋盘状态张量 [batch, 15, 10, 9]
            
        Returns:
            (policy, value):
            - policy: [batch, num_moves] 走法概率（未经 softmax）
            - value: [batch, 1] 局面评估 [-1, 1]
        """
        # 提取特征
        features = self.input_conv(x)
        features = self.residual_blocks(features)
        
        # 策略头：输出走法 logits（在 MCTS 中会加 softmax）
        policy = self.policy_head(features)
        
        # 价值头：输出局面评估
        value = self.value_head(features)
        
        return policy, value
    
    def predict(self, board_tensor: 'list') -> Tuple[list, float]:
        """
        单个局面的推理接口（不使用 DataLoader）。
        
        Args:
            board_tensor: 15×10×9 的三维列表（来自 Game.get_board_tensor()）
            
        Returns:
            (policy_probs, value):
            - policy_probs: 长度为 num_moves 的概率列表
            - value: 局面评估值 [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            # 转换为张量并添加 batch 维度
            x = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0)
            x = x.to(next(self.parameters()).device)
            
            # channels_last 内存格式（RTX 5070 优化）
            x = x.to(memory_format=torch.channels_last)
            
            policy, value = self.forward(x)
            
            # 转换为概率分布
            policy_probs = F.softmax(policy, dim=1).squeeze(0).cpu().tolist()
            value_scalar = value.item()
        
        return policy_probs, value_scalar
    
    def count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> str:
        """获取模型信息摘要"""
        total_params = self.count_parameters()
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"PolicyValueNet:\n"
            f"  残差块数: {len(self.residual_blocks)}\n"
            f"  特征通道: {self.residual_blocks[0].conv1.in_channels}\n"
            f"  走法数量: {self.num_moves}\n"
            f"  总参数量: {total_params:,}\n"
            f"  可训练参数: {trainable_params:,}\n"
            f"  模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)"
        )


def create_model(num_blocks: int = 10, channels: int = 256, device: str = 'cuda') -> PolicyValueNet:
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
        device: 计算设备 ('cuda' 或 'cpu')
        
    Returns:
        初始化好的 PolicyValueNet 模型
    """
    model = PolicyValueNet(num_blocks=num_blocks, channels=channels)
    
    # 移动到 GPU
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        # 使用 channels_last 内存格式（NVIDIA GPU 优化）
        model = model.to(memory_format=torch.channels_last)
        print(f"[模型] 已创建并移动到 CUDA 设备: {torch.cuda.get_device_name()}")
    else:
        print("[模型] 已创建，使用 CPU 推理")
    
    print(model.get_model_info())
    return model


if __name__ == '__main__':
    # 测试：创建模型并进行一次前向传播
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(num_blocks=5, channels=128, device=device)
    
    # 模拟输入
    dummy_input = torch.randn(1, 14, BOARD_ROWS, BOARD_COLS)
    if device == 'cuda':
        dummy_input = dummy_input.cuda()
    dummy_input = dummy_input.to(memory_format=torch.channels_last)
    
    policy, value = model(dummy_input)
    print(f"\n测试前向传播:")
    print(f"  策略输出形状: {policy.shape}")
    print(f"  价值输出形状: {value.shape}")
    print(f"  价值输出: {value.item():.4f}")
