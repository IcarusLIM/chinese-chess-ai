# 中国象棋 AI

一个面向实际自对弈训练的 AlphaZero 风格中国象棋项目，包含规则引擎、批量 MCTS、PyTorch 策略/WDL 网络、多 actor 自对弈、replay buffer、模型评估和 Web 对弈界面。

## 快速开始

需要 Python 3.9+ 和 PyTorch。NVIDIA GPU 使用 CUDA，Apple Silicon 使用 MPS，其他环境回退到 CPU。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 从零训练
.venv/bin/python main.py train --save-dir models

# 继续训练
.venv/bin/python main.py train --resume models/latest_model.pt

# 人机对弈
.venv/bin/python main.py web --model models/best_model.pt
```

训练目录已有 checkpoint 或 replay 时，从零训练会拒绝覆盖；请使用 `--resume` 或新的 `--save-dir`。

## 系统设计

### 状态与走法

网络始终从当前走棋方视角观察局面。黑方走棋时棋盘旋转 180°，同时交换己方/对方语义。输入为 19 个 `10×9` 平面：

- 0–6：当前方七种棋子；
- 7–13：对方七种棋子；
- 14–15：上一步走法的起点和终点；
- 16：当前局面重复次数；
- 17：无吃子半回合计数；
- 18：当前方是否被将军。

策略使用同一规范坐标。训练时会同步左右镜像棋盘、策略和合法走法掩码。

### 网络

默认模型是 `6 × 128` 残差 CNN，约 300 万参数：

```text
[19, 10, 9]
    → Conv(128) → 6 个残差块
        ├─ 策略头: Conv(4) → 2238 个走法 logits
        └─ WDL 头: Conv(3) → FC(256) → [负, 和, 胜]
```

MCTS 使用 `P(win) - P(loss)` 作为标量价值。WDL 头直接学习和棋，不人为降低和棋样本的采样比例。

### MCTS 与自对弈

- PUCT 使用网络策略作为先验，使用 WDL 价值回溯；
- 自对弈根节点加 Dirichlet 噪声，评估和对弈不加；
- 多个 CPU actor 构建棋局和搜索树，主进程将叶节点集中成 GPU batch；
- 推理缓存键覆盖棋盘、上一步、重复次数和半回合计数；
- 搜索次数默认在前 30 轮从 100 线性增长到 400。

MCTS 不混入材料分数，也不禁止返回历史局面。重复局面和长将由规则引擎判定。

### 训练和模型生命周期

损失为策略交叉熵加 WDL 交叉熵，优化器为 AdamW，并使用梯度裁剪、CUDA AMP 和余弦学习率。

每轮更新数根据新样本数动态计算：

```text
steps = clamp(ceil(新样本数 × replay_ratio / batch_size), 32, 128)
```

默认 `replay_ratio=4`，replay 保留最近 50,000 条样本；50% 采样概率分配给最近 10,000 条，其余覆盖整个窗口。

- `latest_model.pt`：每轮更新，包含模型、优化器和调度器，用于继续训练；
- `best_model.pt`：只在候选模型通过评估时更新，用于对弈和部署。

评估使用固定多开局、红黑互换和严格规则，不做子力裁定。全部和棋不促成。未通过评估时 latest 不回滚，best 保持不变。

自对弈的最大步数/无进展截断可以用子力阈值生成训练标签，默认为 `0.10`；用 `--material-adjudication-threshold 0` 可关闭。

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--blocks` | 6 | 残差块数 |
| `--channels` | 128 | 主干通道数 |
| `--policy-channels` | 4 | 策略头通道数 |
| `--simulations` | 400 | 最终 MCTS 模拟次数 |
| `--initial-simulations` | 100 | 初始 MCTS 模拟次数 |
| `--self-play-games` | 32 | 每轮自对弈局数 |
| `--self-play-workers` | 0 | 自动，最多 8 个 actor |
| `--batch-size` | 256 | 训练 batch |
| `--replay-ratio` | 4.0 | 新样本目标重放次数 |
| `--buffer-size` | 50000 | replay 窗口 |
| `--lr` | 0.0003 | AdamW 初始学习率 |
| `--acceptance-threshold` | 0.55 | best 模型促成计分率 |

```bash
.venv/bin/python main.py train --help
```

## 测试和基准

```bash
PYTHONPYCACHEPREFIX=/tmp/chinese_chess_pycache \
  .venv/bin/python -m unittest discover -s tests -v

.venv/bin/python benchmark_self_play.py \
  --games 8 --moves 40 --simulations 100 --workers 4
```

## 主要文件

| 文件 | 职责 |
|---|---|
| `game.py` | 规则、合法走法、将军、长将/重复和终局判定 |
| `encoding.py` | 当前方视角编码、坐标转换和镜像映射 |
| `move_index.py` | 2238 个策略走法的稳定索引 |
| `network.py` | 轻量残差策略/WDL 网络 |
| `mcts.py` | 批量 PUCT、子树复用和推理缓存 |
| `trainer.py` | 多 actor 自对弈、replay、训练、评估和保存 |
| `main.py` | 训练、Web 和控制台入口 |
