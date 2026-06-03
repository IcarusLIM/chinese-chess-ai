"""
蒙特卡洛树搜索（MCTS）模块
============================
实现 AlphaZero 风格的 MCTS 搜索算法。

核心思想：
  通过模拟大量可能的走子序列，结合神经网络的评估，
  选择最优的走法。与传统 MCTS 不同，这里使用神经网络
  来指导搜索（而非随机模拟）。

搜索流程（PUCT 算法）：
  1. 选择（Select）：从根节点开始，使用 PUCT 公式选择最优子节点
  2. 扩展（Expand）：到达叶节点时，用神经网络评估并扩展
  3. 评估（Evaluate）：神经网络输出 (策略概率, 局面价值)
  4. 回溯（Backup）：将评估值沿路径回传，更新统计信息

PUCT 公式：
  Q(s,a) + c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))
  
  其中：
  - Q(s,a): 动作价值（平均回报）
  - P(s,a): 先验概率（来自神经网络）
  - N(s): 父节点访问次数
  - N(s,a): 该动作的访问次数
  - c_puct: 探索常数（控制探索与利用的平衡）
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from game import Game
from network import PolicyValueNet
from move_index import get_move_index


class MCTSNode:
    """
    MCTS 树节点
    
    每个节点代表一个棋盘局面，存储：
    - 该局面下每个合法走法的统计信息
    - 节点的访问次数
    - 节点的评估值
    
    属性：
    - N(s,a): 每个走法的访问次数
    - W(s,a): 每个走法的累计价值
    - Q(s,a): 每个走法的平均价值 = W / N
    - P(s,a): 每个走法的先验概率（来自神经网络）
    """
    
    def __init__(self, prior: float = 0.0):
        """
        Args:
            prior: 该节点的先验概率（从父节点继承）
        """
        # 走法统计信息（key: 走法索引, value: 统计值）
        self.N: Dict[int, int] = {}      # 访问次数
        self.W: Dict[int, float] = {}    # 累计价值
        self.Q: Dict[int, float] = {}    # 平均价值
        self.P: Dict[int, float] = {}    # 先验概率
        
        # 子节点（key: 走法索引, value: MCTSNode）
        self.children: Dict[int, MCTSNode] = {}
        
        # 该节点的先验概率
        self.prior = prior
        
        # 是否已扩展（已用神经网络评估过）
        self.is_expanded = False
    
    def select_child(self, c_puct: float, parent_visits: int) -> Tuple[int, 'MCTSNode']:
        """
        使用 PUCT 公式选择最优子节点。
        
        PUCT = Q(s,a) + c_puct × P(s,a) × √(N(parent)) / (1 + N(s,a))
        
        Args:
            c_puct: 探索常数（越大越倾向探索）
            parent_visits: 父节点的总访问次数
            
        Returns:
            (best_move_index, best_child_node)
        """
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move_idx, child in self.children.items():
            # Q 值：平均价值
            q = self.Q.get(move_idx, 0.0)
            
            # 探索奖励：先验概率 × sqrt(父节点访问次数) / (1 + 该动作访问次数)
            n = self.N.get(move_idx, 0)
            u = c_puct * self.P.get(move_idx, 0.0) * math.sqrt(parent_visits) / (1 + n)
            
            score = q + u
            
            if score > best_score:
                best_score = score
                best_move = move_idx
                best_child = child
        
        return best_move, best_child
    
    def expand(self, legal_move_indices: List[int], priors: List[float]):
        """
        扩展节点：为所有合法走法创建子节点。
        
        Args:
            legal_move_indices: 合法走法的索引列表
            priors: 对应的先验概率列表
        """
        for idx, move_idx in enumerate(legal_move_indices):
            if move_idx not in self.children:
                self.N[move_idx] = 0
                self.W[move_idx] = 0.0
                self.Q[move_idx] = 0.0
                self.P[move_idx] = priors[idx] if idx < len(priors) else 1.0 / len(legal_move_indices)
                self.children[move_idx] = MCTSNode(prior=priors[idx] if idx < len(priors) else 0.0)
        
        self.is_expanded = True
    
    def backup(self, move_idx: int, value: float):
        """
        回溯更新：将评估值沿路径回传。
        
        Args:
            move_idx: 走法索引
            value: 评估值（从当前走棋方的视角）
        """
        self.N[move_idx] = self.N.get(move_idx, 0) + 1
        self.W[move_idx] = self.W.get(move_idx, 0.0) + value
        self.Q[move_idx] = self.W[move_idx] / self.N[move_idx]


class MCTS:
    """
    蒙特卡洛树搜索引擎
    
    结合神经网络进行搜索，核心流程：
    1. 从根节点开始，使用 PUCT 选择最优路径
    2. 到达叶节点时，用神经网络评估
    3. 将评估值沿路径回传
    
    搜索结束后，根据访问次数选择走法：
    - 训练时：按概率采样（带温度参数）
    - 对弈时：选择访问次数最多的走法
    """
    
    def __init__(self,
                 model: PolicyValueNet,
                 num_simulations: int = 400,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 material_weight: float = 0.0):
        """
        Args:
            model: 策略-价值网络
            num_simulations: 每步搜索的模拟次数
                - 100: 快速但不精确
                - 400: 标准配置
                - 800+: 更精确但更慢
            c_puct: PUCT 探索常数
                - 1.0: 较保守
                - 1.5: 标准配置
                - 3.0: 较激进的探索
            dirichlet_alpha: Dirichlet 噪声参数（控制探索）
            dirichlet_epsilon: 噪声混合比例
            material_weight: 材料评估混合权重（0.0=纯网络，1.0=纯材料评估）
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.material_weight = material_weight
        self.move_index = get_move_index()
    
    def search(self, game: Game, temperature: float = 1.0) -> Tuple[int, List[float]]:
        """
        执行 MCTS 搜索，返回最佳走法和策略分布。
        
        Args:
            game: 当前游戏状态
            temperature: 温度参数
                - 1.0: 按概率采样（探索性强）
                - 0.1: 接近贪心（几乎选最好的）
                - 0.01: 纯贪心（选访问次数最多的）
                
        Returns:
            (best_move_index, policy):
            - best_move_index: 选择的走法索引
            - policy: 长度为 num_moves 的概率分布
        """
        root = MCTSNode()
        
        # 获取合法走法
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            # 没有合法走法（游戏结束），返回空策略
            return -1, [0.0] * self.move_index.num_moves
        
        legal_indices = []
        for (fr, fc), (tr, tc) in legal_moves:
            idx = self.move_index.move_to_index((fr, fc, tr, tc))
            if idx is not None:
                legal_indices.append(idx)
        
        if not legal_indices:
            return -1, [0.0] * self.move_index.num_moves
        
        # 用神经网络评估根节点
        board_tensor = game.get_board_tensor()
        policy_probs, value = self.model.predict(board_tensor)
        
        # 提取合法走法的先验概率并归一化
        legal_priors = [policy_probs[idx] for idx in legal_indices]
        prior_sum = sum(legal_priors)
        if prior_sum > 0:
            legal_priors = [p / prior_sum for p in legal_priors]
        else:
            legal_priors = [1.0 / len(legal_indices)] * len(legal_indices)
        
        # 在根节点添加 Dirichlet 噪声（促进探索）
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
        noisy_priors = [
            (1 - self.dirichlet_epsilon) * p + self.dirichlet_epsilon * n
            for p, n in zip(legal_priors, noise)
        ]
        
        # 扩展根节点
        root.expand(legal_indices, noisy_priors)
        
        # 执行模拟
        for _ in range(self.num_simulations):
            self._simulate(game, root)
        
        # 根据访问次数计算策略分布
        policy = self._compute_policy(root, temperature)
        
        # 选择走法
        if temperature < 0.01:
            # 贪心：选择访问次数最多的
            best_move = max(root.N, key=lambda k: root.N[k])
        else:
            # 按概率采样
            visits = np.array([root.N.get(i, 0) for i in legal_indices])
            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum() if visits.sum() > 0 else np.ones(len(visits)) / len(visits)
            best_idx = np.random.choice(len(legal_indices), p=probs)
            best_move = legal_indices[best_idx]
        
        return best_move, policy
    
    def _simulate(self, game: Game, node: MCTSNode):
        """
        执行一次模拟：从根节点走到叶节点，然后评估。
        
        Args:
            game: 当前游戏状态（会在此基础上模拟走子）
            node: 当前节点
        """
        # 选择阶段：沿树向下选择直到叶节点
        search_path = [(node, -1)]
        current_game = game.copy()

        while node.is_expanded and node.children:
            move_idx, node = node.select_child(self.c_puct, sum(node.N.values()) if node.N else 0)
            search_path.append((node, move_idx))

            # 在模拟棋盘上执行走子
            move = self.move_index.index_to_move(move_idx)
            if move is None:
                return
            fr, fc, tr, tc = move
            current_game.make_move((fr, fc), (tr, tc))
        
        # 检查游戏是否结束
        is_over, result = current_game.is_game_over()
        
        if is_over:
            # 游戏结束，从当前走棋方视角返回
            if result == 'red_wins':
                value = 1.0 if current_game.red_to_move else -1.0
            elif result == 'black_wins':
                value = -1.0 if current_game.red_to_move else 1.0
            else:
                value = 0.0
        else:
            # 未结束，用神经网络评估（可选混合材料评估）
            board_tensor = current_game.get_board_tensor()
            policy_probs, network_value = self.model.predict(board_tensor)

            if self.material_weight > 0:
                material_value = current_game.evaluate_material_normalized()
                value = (1 - self.material_weight) * network_value + self.material_weight * material_value
            else:
                value = network_value
            
            # 扩展叶节点
            legal_moves = current_game.get_legal_moves()
            legal_indices = []
            for (fr, fc), (tr, tc) in legal_moves:
                idx = self.move_index.move_to_index((fr, fc, tr, tc))
                if idx is not None:
                    legal_indices.append(idx)
            
            if legal_indices:
                legal_priors = [policy_probs[idx] for idx in legal_indices]
                prior_sum = sum(legal_priors)
                if prior_sum > 0:
                    legal_priors = [p / prior_sum for p in legal_priors]
                else:
                    legal_priors = [1.0 / len(legal_indices)] * len(legal_indices)
                node.expand(legal_indices, legal_priors)
        
        # 回溯阶段：沿路径回传价值
        # value 为叶节点走棋方视角，走子交替，到叶节点的
        # 距离为偶数 → 同色 → 用 value，奇数 → 异色 → 用 -value
        depth = len(search_path) - 1  # 叶节点深度
        for i in range(depth, 0, -1):
            _, move_idx = search_path[i]
            parent_node, _ = search_path[i - 1]
            if (depth - i + 1) % 2 == 0:
                parent_node.backup(move_idx, value)
            else:
                parent_node.backup(move_idx, -value)
    
    def _compute_policy(self, root: MCTSNode, temperature: float) -> List[float]:
        """
        根据根节点的访问次数计算策略分布。
        
        Args:
            root: 根节点
            temperature: 温度参数
            
        Returns:
            长度为 num_moves 的概率分布
        """
        policy = [0.0] * self.move_index.num_moves
        
        if not root.N:
            return policy
        
        total_visits = sum(root.N.values())
        if total_visits == 0:
            return policy
        
        if temperature < 0.01:
            # 贪心策略：只给最佳走法概率 1
            best_move = max(root.N, key=lambda k: root.N[k])
            policy[best_move] = 1.0
        else:
            # 带温度的策略分布
            for move_idx, visits in root.N.items():
                policy[move_idx] = (visits ** (1.0 / temperature))
            
            # 归一化
            policy_sum = sum(policy)
            if policy_sum > 0:
                policy = [p / policy_sum for p in policy]
        
        return policy


class MCTSEvaluator:
    """
    MCTS 评估器（用于对弈和分析）
    
    封装了 MCTS 搜索和走法选择逻辑，提供简洁的接口。
    """
    
    def __init__(self, model: PolicyValueNet, num_simulations: int = 400):
        """
        Args:
            model: 策略-价值网络
            num_simulations: 搜索模拟次数
        """
        self.mcts = MCTS(model, num_simulations=num_simulations)
        self.move_index = get_move_index()
    
    def select_move(self, game: Game, temperature: float = 0.01) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        为当前局面选择最佳走法。
        
        Args:
            game: 当前游戏状态
            temperature: 温度（对弈时接近0，训练时为1）
            
        Returns:
            ((from_row, from_col), (to_row, to_col))
        """
        move_idx, policy = self.mcts.search(game, temperature)
        
        if move_idx < 0:
            return None
        
        move = self.move_index.index_to_move(move_idx)
        if move is None:
            return None
        
        fr, fc, tr, tc = move
        return ((fr, fc), (tr, tc))
    
    def get_policy(self, game: Game, temperature: float = 1.0) -> List[float]:
        """
        获取当前局面的策略分布（用于训练数据收集）。
        
        Args:
            game: 当前游戏状态
            temperature: 温度参数
            
        Returns:
            长度为 num_moves 的概率分布
        """
        _, policy = self.mcts.search(game, temperature)
        return policy
    
    def analyze_position(self, game: Game, top_k: int = 5) -> List[dict]:
        """
        分析当前局面，返回 top-k 最佳走法及评估。
        
        Args:
            game: 当前游戏状态
            top_k: 返回前 k 个最佳走法
            
        Returns:
            走法分析列表，每个元素包含：
            - move: 走法描述
            - visits: 访问次数
            - value: 评估值
            - prior: 先验概率
        """
        # 执行搜索
        move_idx, policy = self.mcts.search(game, temperature=0.01)
        
        # 收集根节点的统计信息
        root = self.mcts.mcts_root  # 需要在 search 中保存根节点引用
        
        # 简化实现：基于策略概率排序
        legal_moves = game.get_legal_moves()
        move_scores = []
        
        for (fr, fc), (tr, tc) in legal_moves:
            idx = self.move_index.move_to_index((fr, fc, tr, tc))
            if idx is not None:
                move_scores.append({
                    'move': ((fr, fc), (tr, tc)),
                    'move_str': f"({fr},{fc})->({tr},{tc})",
                    'prior': policy[idx],
                    'score': policy[idx],
                })
        
        # 按分数排序
        move_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return move_scores[:top_k]
