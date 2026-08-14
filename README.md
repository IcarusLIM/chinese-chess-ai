# 中国象棋 AI

> 从零开始训练的中国象棋 AI，基于 AlphaZero 架构（MCTS + 神经网络）

## 项目简介

这是一个完整的中国象棋 AI 系统，实现了以下功能：

1. **完整的象棋引擎** — 规则验证、合法走法生成、将军/将死检测
2. **神经网络模型** — ResNet 策略-价值网络，预测走法概率和局面评估
3. **蒙特卡洛树搜索 (MCTS)** — AlphaZero 风格的搜索算法
4. **自对弈训练** — 无需棋谱，从零开始自我学习
5. **Web 对弈界面** — 支持人机对弈、棋谱回放、局面分析、棋谱下载

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    AlphaZero 架构                        │
│                                                         │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐       │
│   │  自对弈   │────▶│ 训练数据  │────▶│ 模型训练  │       │
│   │ (MCTS)   │     │ (缓冲区) │     │ (PyTorch)│       │
│   └────┬─────┘     └──────────┘     └────┬─────┘       │
│        │                                  │              │
│        ▼                                  ▼              │
│   ┌──────────┐                    ┌──────────┐          │
│   │  象棋引擎 │                    │ 策略-价值 │          │
│   │ (game.py) │◀──────────────────│  网络     │          │
│   └──────────┘    局面评估         └──────────┘          │
│                                                         │
│   ┌──────────────────────────────────────┐              │
│   │         Web 界面 (Flask)              │              │
│   │   对弈 │ 回放 │ 分析 │ 下载          │              │
│   └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

## 文件结构

```
chinese-chess-ai/
├── main.py              # 主程序入口（训练/Web/控制台）
├── constants.py         # 棋盘常量定义
├── game.py              # 象棋游戏引擎（规则、走法、状态）
├── move_index.py        # 走法 ↔ 索引双向映射
├── network.py           # 神经网络模型（ResNet 策略-价值网络）
├── mcts.py              # 蒙特卡洛树搜索（MCTS）
├── trainer.py           # 自对弈训练流水线
├── requirements.txt     # Python 依赖
├── README.md            # 本文档
├── web/
│   ├── app.py           # Flask Web 服务器
│   └── templates/
│       └── index.html   # 前端界面（棋盘 + 交互）
├── models/              # 模型检查点与增量 replay 分片
│   └── replay/          # manifest.json + shard_iter_*.pkl
├── logs/                # 训练日志
└── data/                # 训练数据
```

## 快速开始

### 环境要求

- Python 3.9+
- NVIDIA GPU（推荐 RTX 5070 或同等级，12GB+ VRAM）
- CUDA 12.x

### 安装

```bash
# 克隆项目
cd chinese-chess-ai

# 安装依赖
pip install -r requirements.txt
```

### 运行 Web 对弈（使用随机模型）

```bash
python main.py web --simulations 100
```

然后在浏览器打开 `http://localhost:5000`

Web 棋局默认保存在 `instance/games.sqlite3`，服务重启或使用多个 worker 时仍可共享。
生产环境可通过 `CHINESE_CHESS_DB` 指定数据库路径，并通过
`CHINESE_CHESS_SECRET_KEY` 配置固定的 Flask 会话密钥。

### 从零开始训练

```bash
# 完整训练（100轮迭代，每轮25局自对弈，前80轮材料评估 warmup）
python main.py train

# 快速实验（减少参数）
python main.py train --iterations 20 --blocks 5 --channels 128 --simulations 100 --self-play-games 10

# 训练完成后启动 Web 对弈
python main.py web --model models/latest_model.pt
```

### 断点续训

训练过程中会自动保存检查点和经验回放数据：

| 文件 | 说明 |
|------|------|
| `models/latest_model.pt` | 每轮迭代更新的最新模型 |
| `models/checkpoint_iter_N.pt` | 每 5 轮保存的检查点 |
| `models/replay/manifest.json` | replay 分片清单和样本数量 |
| `models/replay/shard_iter_*.pkl` | 每轮新增的紧凑 replay 样本 |

checkpoint 会保存网络的残差块数和通道数，控制台与 Web 加载时会据此自动重建模型，
无需再次手动指定网络结构。

从检查点继续训练：

```bash
# 加载模型、优化器状态，并从上次迭代编号 +1 继续
python main.py train --resume models/latest_model.pt

# 指定迭代次数（在已完成的迭代基础上继续）
python main.py train --resume models/latest_model.pt --iterations 150
```

`--resume` 会自动加载与检查点同目录下的 `replay/manifest.json`。若 replay 在其他路径，可单独加载目录或 manifest：

```bash
python main.py train --resume models/latest_model.pt --load-buffer models/replay
```

仅加载 replay buffer、模型仍随机初始化（一般不推荐）：

```bash
python main.py train --load-buffer models/replay/manifest.json
```

重新训练时建议使用一个全新的保存目录：

```bash
python main.py train --save-dir models_fresh
```

若未传 `--resume`，程序发现保存目录中已有 checkpoint 或 replay 也会立即中止，
避免随机初始化模型意外覆盖已有模型或混入历史样本。

### 材料评估 Warmup

训练初期神经网络尚未收敛，MCTS 叶节点评估几乎随机。启用 `--material-warmup` 后，前 N 轮自对弈会在 MCTS 中混合**材料评估**（基于棋子价值的启发式）与网络评估，并随迭代线性衰减至纯网络评估：

```
叶节点价值 = (1 - w) × 网络评估 + w × 材料评估
w = 1.0 → 0.0（第 1 轮到第 N 轮线性衰减，之后 w = 0）
```

```bash
# 前 80 轮启用材料评估 warmup，帮助早期自对弈获得更合理的搜索信号
python main.py train --material-warmup 80

# 组合使用：从检查点续训 + warmup
python main.py train --resume models/latest_model.pt --material-warmup 80
```

新模型默认使用 80 轮 warmup；模型已有一定棋力时可显式设为 `0`。

### 控制台对弈（调试用）

```bash
python main.py play --model models/latest_model.pt
```

## 核心模块详解

### 1. 游戏引擎 (`game.py`)

实现了完整的中国象棋规则：

| 棋子 | 走法规则 |
|------|----------|
| 帅/将 | 九宫格内走一步（上下左右） |
| 仕/士 | 九宫格内斜走一步 |
| 相/象 | 走"田"字，不过河，蹩象眼不能走 |
| 马    | 走"日"字，蹩马腿不能走 |
| 车    | 沿直线走任意距离 |
| 炮    | 不吃子时同车，吃子需隔一子（炮架） |
| 兵/卒 | 未过河只能前进，过河后可左右 |

特殊规则：
- **将军检测**：走子后不能让己方将/帅被将
- **飞将规则**：将帅不能面对面（同列中间无子）
- **将死判定**：被将且无合法走法
- **困毙判定**：未被将但无合法走法（中国象棋判负）
- **和棋规则**：60回合无吃子 或 三次重复局面

### 2. 神经网络 (`network.py`)

```
输入: [batch, 15, 10, 9]  ← 14种棋子通道 + 1个走棋方通道
  │
  ▼
Conv2d(15→256) + BN + ReLU    ← 输入层
  │
  ▼
ResBlock × 10                  ← 10个残差块（特征提取）
  │
  ├──▶ 策略头: Conv(256→32) → FC(2880→2238)  ← 走法概率
  │
  └──▶ 价值头: Conv(256→3) → FC(270→256→1) → tanh  ← 局面评估
```

- **策略头**：输出每个可能走法的概率（2238 种走法）
- **价值头**：输出当前走棋方视角的局面评估值 [-1, +1]

### 3. 蒙特卡洛树搜索 (`mcts.py`)

使用 PUCT（Predictor Upper Confidence Bound applied to Trees）算法：

```
选择分数 = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
```

- `Q(s,a)`: 走法的平均价值（从模拟中学习）
- `P(s,a)`: 神经网络给出的先验概率
- `N(s)`: 父节点访问次数
- `c_puct`: 探索常数（默认1.5）

搜索流程：
1. 从根节点开始，用 PUCT 选择路径
2. 到达叶节点时，用神经网络评估
3. 将评估值沿路径回传，更新统计信息
4. 重复 400 次模拟后，根据访问次数选择走法

### 4. 训练流程 (`trainer.py`)

```
for 迭代 in range(100):
    1. 自对弈：当前模型 × 当前模型 → 生成训练数据
       （可选：材料评估 warmup 混合启发式评估）
    2. 训练：从经验回放缓冲区随机采样固定数量的 batch
    3. 评估：在固定多开局中交换红黑，候选模型 vs 当前基准模型成对对弈
       - 计分率（胜=1、和=0.5、负=0）> 55% → 接受候选模型
       - 否则 → 回滚到上次通过的模型
    4. 保存：latest_model.pt + 本轮新增 replay 分片 + manifest
```

训练损失函数：
```
Loss = 策略损失(交叉熵) + 价值损失(MSE) + L2正则化
```

每轮迭代结束后更新 `latest_model.pt`，并只写入本轮新增 replay 分片，不再重复序列化整个缓冲区。

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_blocks` | 10 | 残差块数量，越多越强但越慢 |
| `channels` | 256 | 特征通道数，越多越强但越占显存 |
| `num_simulations` | 400 | MCTS 模拟次数，越多越精确 |
| `inference_batch_size` | 64 | 单次 GPU 推理合并的 MCTS 叶节点数 |
| `inference_cache_size` | 10000 | 当前模型权重的 LRU 网络评估缓存容量 |
| `eval_simulations` | 200 | 模型评估时每步 MCTS 模拟次数 |
| `self_play_games` | 25 | 每轮自对弈局数 |
| `training_steps` | 500 | 每轮从 replay 随机训练的 batch 数 |
| `batch_size` | 256 | 训练批大小（RTX 5070 12GB 足够） |
| `data_workers` | 4 | 训练数据加载进程数 |
| `learning_rate` | 0.001 | 初始学习率，随训练衰减 |
| `c_puct` | 1.5 | MCTS 探索常数 |
| `temperature` | 1.0→0.01 | 温度参数，训练时先高后低 |
| `--resume` | — | 从指定 checkpoint 继续训练（恢复模型、优化器、迭代编号） |
| `--load-buffer` | — | 从 replay 目录或 manifest 加载数据 |
| `--material-warmup` | 80 | 材料评估 warmup 轮数（0 = 不启用） |

## RTX 5070 训练建议

MCTS 默认以 64 个叶节点为一批执行 FP16 GPU 推理，并在实际落子后复用搜索子树。
400 次模拟通常只需要约 7 次批量叶节点推理，而不是逐叶执行约 400 次推理。
同一组模型权重还会使用有界 LRU 缓存复用重复局面的原始网络输出；模型更新后自动清空。

不同 CUDA/PyTorch 版本和棋局平均长度差异很大，因此不再给出未经实测的每轮耗时。
训练日志会分别输出自对弈、网络训练和模型评估耗时。显存充足但 GPU 利用率偏低时，
可尝试 `--inference-batch-size 128`；显存不足时降为 32。

| 配置 | blocks | channels | simulations | 推理批大小 | 训练 batch |
|------|--------|----------|-------------|------------|------------|
| 快速实验 | 5 | 128 | 100 | 32 | 128 |
| 标准训练 | 10 | 256 | 400 | 64 | 256 |
| 高强度 | 15 | 384 | 800 | 64 | 256 |

### Replay 与训练吞吐优化

- 棋盘以 `uint8` 保存，价值以 `int8` 保存。
- MCTS 策略只保存非零动作索引及其 `float16` 概率。
- 2238 位合法走法掩码使用 `packbits` 压缩。
- 每轮只写入新增 replay 分片，并通过 manifest 原子更新清单。
- 磁盘只保留最近窗口的分片，旧分片按清单安全清理。
- 每轮固定随机训练 500 个 batch，耗时不再随 replay buffer 线性增长。
- 与模型权重绑定的 LRU 缓存使用 Zobrist hash 复用重复局面的网络输出。

在初始局面的实测样本中，紧凑格式从约 16.6 KB 降至约 1.6 KB；
dense Python 浮点列表的对象开销更高，因此实际内存降幅通常更大。

## Web 界面功能

- **人机对弈**：点击棋子 → 显示合法走法 → 点击目标位置
- **悔棋**：撤销玩家和 AI 各一步
- **局面分析**：AI 推荐 top-5 最佳走法
- **棋谱下载**：导出 JSON 格式的完整棋谱
- **棋盘翻转**：切换红方/黑方视角

## 从零训练的预期时间线

| 迭代 | 预期水平 |
|------|----------|
| 1-10 | 学会基本走法规则，避免送子 |
| 10-30 | 学会基本战术（吃子、保护） |
| 30-60 | 学会开局和简单组合 |
| 60-100 | 具有一定棋力，能击败初学者 |

> **注意**：AlphaZero 风格训练的耗时主要取决于实际对局长度和 MCTS 叶节点数量。
> 请以日志中的分阶段计时为准，先运行一轮快速配置验证吞吐量。

## 常见问题

**Q: 训练时 GPU 内存不足怎么办？**
减小 `batch_size`（如 128 或 64）或减小 `channels`（如 128）。

**Q: AI 走子太慢怎么办？**
减小 `num_simulations`（如 100）或使用更小的模型（blocks=5, channels=128）。

**Q: 如何继续训练已有模型？**
使用 `--resume` 加载检查点，会自动恢复模型权重、优化器状态和迭代编号，并尝试加载同目录下的 `replay/manifest.json`：

```bash
python main.py train --resume models/latest_model.pt
```

**Q: 续训时 replay buffer 为空怎么办？**
若 checkpoint 同目录没有 replay，训练会以空 buffer 继续，并在样本达到 `batch_size × 10`
后开始更新。也可显式指定数据：`--load-buffer models/replay`；路径不存在或数据缺失时会直接报错。

**Q: 材料评估 warmup 是什么？**
训练早期网络输出接近随机，MCTS 搜索质量差。`--material-warmup N` 在前 N 轮将棋子价值启发式混入 MCTS 叶节点评估，随轮次线性衰减至 0，帮助生成更高质量的早期自对弈数据。默认值为 80；已有一定棋力的续训可显式传入 `--material-warmup 0`。

**Q: 为什么 AI 走子看起来是随机的？**
未训练的模型输出确实是随机的。需要先运行 `python main.py train` 进行训练。

## 许可证

MIT License
