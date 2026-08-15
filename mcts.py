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
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from game import Game
from network import PolicyValueNet
from move_index import get_move_index


class InferenceCache:
    """与一组固定模型权重绑定的 LRU 推理缓存。"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max(0, max_size)
        self._data = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, position_hash: int):
        if self.max_size <= 0 or position_hash not in self._data:
            self.misses += 1
            return None
        self.hits += 1
        value = self._data.pop(position_hash)
        self._data[position_hash] = value
        return value

    def put(self, position_hash: int, policy, value: float):
        if self.max_size <= 0:
            return
        if position_hash in self._data:
            self._data.pop(position_hash)
        self._data[position_hash] = (
            np.asarray(policy, dtype=np.float16), np.float16(value)
        )
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._data)


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
    
    def __init__(self):
        # 走法统计信息（key: 走法索引, value: 统计值）
        self.N: Dict[int, int] = {}      # 访问次数
        self.W: Dict[int, float] = {}    # 累计价值
        self.Q: Dict[int, float] = {}    # 平均价值
        self.P: Dict[int, float] = {}    # 先验概率
        
        # 子节点（key: 走法索引, value: MCTSNode）
        self.children: Dict[int, MCTSNode] = {}
        
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
        for move_idx, prior in zip(legal_move_indices, priors):
            if move_idx not in self.children:
                self.N[move_idx] = 0
                self.W[move_idx] = 0.0
                self.Q[move_idx] = 0.0
                self.P[move_idx] = prior
                self.children[move_idx] = MCTSNode()
        
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

    def add_virtual_loss(self, move_idx: int, virtual_loss: float):
        """批量选择期间临时降低该分支吸引力，避免所有叶节点重复。"""
        self.N[move_idx] = self.N.get(move_idx, 0) + 1
        self.W[move_idx] = self.W.get(move_idx, 0.0) - virtual_loss
        self.Q[move_idx] = self.W[move_idx] / self.N[move_idx]

    def revert_virtual_loss(self, move_idx: int, virtual_loss: float):
        """叶节点完成评估后移除临时 virtual loss。"""
        self.N[move_idx] -= 1
        self.W[move_idx] += virtual_loss
        self.Q[move_idx] = (
            self.W[move_idx] / self.N[move_idx] if self.N[move_idx] > 0 else 0.0
        )


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
                 add_root_noise: bool = False,
                 inference_batch_size: int = 64,
                 virtual_loss: float = 1.0,
                 inference_cache: Optional[InferenceCache] = None,
                 material_weight: float = 0.0,
                 avoid_repetition: bool = True,
                 repetition_material_threshold: float = -1.0):
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
            add_root_noise: 是否在根节点加入 Dirichlet 噪声（仅自对弈训练使用）
            inference_batch_size: 每次 GPU 推理合并的叶节点数
            virtual_loss: 批内选择时的临时损失，用于分散搜索路径
            inference_cache: 与当前模型权重绑定的共享推理缓存
            material_weight: 材料评估混合权重（0.0=纯网络，1.0=纯材料评估）
            avoid_repetition: 有其他选择时避免主动走回历史局面
            repetition_material_threshold: 启用重复规避的最低材料评估
        """
        if num_simulations < 1:
            raise ValueError("num_simulations 必须大于 0")
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_root_noise = add_root_noise
        self.inference_batch_size = max(1, inference_batch_size)
        self.virtual_loss = virtual_loss
        self.inference_cache = inference_cache
        self.material_weight = material_weight
        self.avoid_repetition = avoid_repetition
        self.repetition_material_threshold = repetition_material_threshold
        self.move_index = get_move_index()
        self._cached_root: Optional[MCTSNode] = None
        self._cached_root_hash: Optional[int] = None
        self.last_legal_indices: List[int] = []
        self.inference_calls = 0
        self.inference_positions = 0
        self.repetition_moves_avoided = 0
    
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
        if game.get_adjudicated_result() is not None:
            self.last_legal_indices = []
            return -1, [0.0] * self.move_index.num_moves

        # 当前局面若正好是上一步选中后的子节点，则复用整棵子树。
        reused_root = (
            self._cached_root is not None and self._cached_root_hash == game.current_hash
        )
        if reused_root:
            root = self._cached_root
        else:
            root = MCTSNode()

        # 获取合法走法（同时暴露给自对弈数据收集，避免重复生成）
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            self.last_legal_indices = []
            return -1, [0.0] * self.move_index.num_moves

        legal_indices = self._legal_indices(legal_moves)
        self.last_legal_indices = legal_indices
        if not legal_indices:
            return -1, [0.0] * self.move_index.num_moves

        if not root.is_expanded:
            policies, _ = self._predict_with_cache(
                [game.get_board_tensor()], [game.current_hash]
            )
            root.expand(legal_indices, self._normalized_priors(policies[0], legal_indices))

        # 根节点噪声只用于自对弈；推理和模型评估保持确定性。
        if self.add_root_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
            for move_idx, n in zip(legal_indices, noise):
                root.P[move_idx] = (
                    (1 - self.dirichlet_epsilon) * root.P.get(move_idx, 0.0)
                    + self.dirichlet_epsilon * float(n)
                )

        # 批量执行模拟：一次 GPU 调用评估多个叶节点。
        existing_visits = sum(root.N.values()) if reused_root else 0
        simulations_remaining = max(0, self.num_simulations - existing_visits)
        # 自对弈的新根噪声需要至少影响一个新 batch，避免完全沿用旧统计。
        if reused_root and self.add_root_noise:
            simulations_remaining = max(
                min(self.inference_batch_size, self.num_simulations), simulations_remaining
            )
        while simulations_remaining > 0:
            current_batch = min(self.inference_batch_size, simulations_remaining)
            self._simulate_batch(game, root, current_batch)
            simulations_remaining -= current_batch
        
        best_move, policy = self._select_root_move(
            game, root, legal_indices, temperature
        )

        # 记录选中子树及其预期哈希；调用方真正落子后下一次 search 会命中。
        move = self.move_index.index_to_move(best_move)
        if move is not None and best_move in root.children:
            fr, fc, tr, tc = move
            game.make_move((fr, fc), (tr, tc), validate=False)
            self._cached_root = root.children[best_move]
            self._cached_root_hash = game.current_hash
            game.undo_move()

        return best_move, policy

    def prepare_search(self, game: Game, temperature: float = 1.0) -> dict:
        """准备一次搜索，供多个棋局共享同一个神经网络推理批次。"""
        if game.get_adjudicated_result() is not None:
            self.last_legal_indices = []
            return {'finished': True, 'result': (-1, [0.0] * self.move_index.num_moves)}

        reused_root = (
            self._cached_root is not None and self._cached_root_hash == game.current_hash
        )
        root = self._cached_root if reused_root else MCTSNode()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            self.last_legal_indices = []
            return {'finished': True, 'result': (-1, [0.0] * self.move_index.num_moves)}

        legal_indices = self._legal_indices(legal_moves)
        self.last_legal_indices = legal_indices
        if not legal_indices:
            return {'finished': True, 'result': (-1, [0.0] * self.move_index.num_moves)}

        return {
            'finished': False,
            'game': game,
            'temperature': temperature,
            'root': root,
            'legal_indices': legal_indices,
            'reused_root': reused_root,
            'needs_root_evaluation': not root.is_expanded,
        }

    def finish_prepared_search(self, context: dict) -> Tuple[int, List[float]]:
        """完成已执行模拟的搜索并缓存选中走法对应的子树。"""
        root = context['root']
        game = context['game']
        temperature = context['temperature']
        legal_indices = context['legal_indices']
        best_move, policy = self._select_root_move(
            game, root, legal_indices, temperature
        )

        move = self.move_index.index_to_move(best_move)
        if move is not None and best_move in root.children:
            fr, fc, tr, tc = move
            game.make_move((fr, fc), (tr, tc), validate=False)
            self._cached_root = root.children[best_move]
            self._cached_root_hash = game.current_hash
            game.undo_move()
        return best_move, policy

    def select_leaf_for_batch(self, game: Game, root: MCTSNode,
                              leaf_metadata: dict) -> dict:
        """选择一个叶节点；调用方可汇总多个棋局后统一执行推理。"""
        node = root
        search_path = [(root, -1)]
        virtual_edges = []

        while node.is_expanded and node.children:
            parent = node
            move_idx, node = parent.select_child(
                self.c_puct, sum(parent.N.values()) if parent.N else 0
            )
            parent.add_virtual_loss(move_idx, self.virtual_loss)
            virtual_edges.append((parent, move_idx))
            search_path.append((node, move_idx))

            move = self.move_index.index_to_move(move_idx)
            if move is None:
                raise RuntimeError(f"MCTS 树中出现无效走法索引: {move_idx}")
            fr, fc, tr, tc = move
            game.make_move((fr, fc), (tr, tc), validate=False)

        leaf_key = id(node)
        metadata = leaf_metadata.get(leaf_key)
        if metadata is None:
            legal_moves = game.get_legal_moves()
            legal_indices = self._legal_indices(legal_moves)
            if not legal_indices:
                metadata = {'legal_indices': legal_indices, 'value': -1.0}
            else:
                terminal_result = game.get_adjudicated_result()
                if terminal_result is not None:
                    metadata = {
                        'legal_indices': legal_indices,
                        'value': self._result_value(game, terminal_result),
                    }
                else:
                    metadata = {
                        'legal_indices': legal_indices,
                        'value': None,
                        'state': game.get_board_tensor(),
                        'position_hash': game.current_hash,
                        'material_value': (
                            game.evaluate_material_normalized()
                            if self.material_weight > 0 else 0.0
                        ),
                    }
            leaf_metadata[leaf_key] = metadata

        for _ in range(len(search_path) - 1):
            game.undo_move()

        return {
            'mcts': self,
            'node': node,
            'path': search_path,
            'virtual_edges': virtual_edges,
            'metadata': metadata,
        }

    def complete_leaf_record(self, record: dict):
        """用集中推理的结果扩展一个叶节点并完成回传。"""
        for parent, move_idx in record['virtual_edges']:
            parent.revert_virtual_loss(move_idx, self.virtual_loss)

        metadata = record['metadata']
        if metadata['value'] is None:
            network_value = metadata['network_value']
            value = (
                (1 - self.material_weight) * network_value
                + self.material_weight * metadata['material_value']
            )
            record['node'].expand(
                metadata['legal_indices'],
                self._normalized_priors(
                    metadata['policy_probs'], metadata['legal_indices']
                ),
            )
        else:
            value = metadata['value']
        self._backup_path(record['path'], value)

    def _simulate_batch(self, game: Game, root: MCTSNode, batch_size: int):
        """选择一批叶节点，统一推理后分别扩展和回传。"""
        records = []
        pending_states = []
        pending_hashes = []
        leaf_metadata = {}

        for _ in range(batch_size):
            node = root
            search_path = [(root, -1)]
            virtual_edges = []

            while node.is_expanded and node.children:
                parent = node
                move_idx, node = parent.select_child(
                    self.c_puct, sum(parent.N.values()) if parent.N else 0
                )
                parent.add_virtual_loss(move_idx, self.virtual_loss)
                virtual_edges.append((parent, move_idx))
                search_path.append((node, move_idx))

                move = self.move_index.index_to_move(move_idx)
                if move is None:
                    raise RuntimeError(f"MCTS 树中出现无效走法索引: {move_idx}")
                fr, fc, tr, tc = move
                game.make_move((fr, fc), (tr, tc), validate=False)

            # 同一批可能多次选中同一未扩展节点；规则生成与网络推理只做一次。
            leaf_key = id(node)
            metadata = leaf_metadata.get(leaf_key)
            if metadata is None:
                legal_moves = game.get_legal_moves()
                legal_indices = self._legal_indices(legal_moves)
                if not legal_indices:
                    value = -1.0
                    prediction_index = None
                    material_value = 0.0
                else:
                    terminal_result = game.get_adjudicated_result()
                    if terminal_result is not None:
                        value = self._result_value(game, terminal_result)
                        prediction_index = None
                        material_value = 0.0
                    else:
                        value = None
                        prediction_index = len(pending_states)
                        pending_states.append(game.get_board_tensor())
                        pending_hashes.append(game.current_hash)
                        material_value = (
                            game.evaluate_material_normalized()
                            if self.material_weight > 0 else 0.0
                        )
                metadata = (legal_indices, value, prediction_index, material_value)
                leaf_metadata[leaf_key] = metadata
            else:
                legal_indices, value, prediction_index, material_value = metadata

            records.append({
                'node': node,
                'path': search_path,
                'virtual_edges': virtual_edges,
                'legal_indices': legal_indices,
                'value': value,
                'prediction_index': prediction_index,
                'material_value': material_value,
            })

            for _ in range(len(search_path) - 1):
                game.undo_move()

        if pending_states:
            policy_batch, value_batch = self._predict_with_cache(
                pending_states, pending_hashes
            )
        else:
            policy_batch, value_batch = None, None

        for record in records:
            for parent, move_idx in record['virtual_edges']:
                parent.revert_virtual_loss(move_idx, self.virtual_loss)

            prediction_index = record['prediction_index']
            if prediction_index is not None:
                policy_probs = policy_batch[prediction_index]
                network_value = float(value_batch[prediction_index])
                value = (
                    (1 - self.material_weight) * network_value
                    + self.material_weight * record['material_value']
                )
                legal_indices = record['legal_indices']
                record['node'].expand(
                    legal_indices, self._normalized_priors(policy_probs, legal_indices)
                )
            else:
                value = record['value']

            self._backup_path(record['path'], value)

    def _predict_with_cache(self, states: list, position_hashes: List[int]):
        """批量查询 LRU，仅把未命中的局面提交给 GPU。"""
        policies = [None] * len(states)
        values = [None] * len(states)
        miss_positions = {}
        miss_states = []
        miss_hashes = []

        for position, (state, position_hash) in enumerate(zip(states, position_hashes)):
            cached = (
                self.inference_cache.get(position_hash)
                if self.inference_cache is not None else None
            )
            if cached is None:
                # 多盘棋可能到达相同局面，同一批内只推理一次。
                if position_hash not in miss_positions:
                    miss_positions[position_hash] = []
                    miss_states.append(state)
                    miss_hashes.append(position_hash)
                miss_positions[position_hash].append(position)
            else:
                cached_policy, cached_value = cached
                policies[position] = np.asarray(cached_policy, dtype=np.float32)
                values[position] = float(cached_value)

        if miss_states:
            miss_policies, miss_values = self.model.predict_batch(miss_states)
            self.inference_calls += 1
            self.inference_positions += len(miss_states)
            for offset, position_hash in enumerate(miss_hashes):
                policy = miss_policies[offset]
                value = float(miss_values[offset])
                for position in miss_positions[position_hash]:
                    policies[position] = policy
                    values[position] = value
                if self.inference_cache is not None:
                    self.inference_cache.put(position_hash, policy, value)

        return np.asarray(policies, dtype=np.float32), np.asarray(values, dtype=np.float32)

    def _backup_path(self, search_path: list, value: float):
        """从叶节点当前走棋方视角交替翻转价值并回传。"""
        depth = len(search_path) - 1
        for i in range(depth, 0, -1):
            _, move_idx = search_path[i]
            parent_node, _ = search_path[i - 1]
            parent_node.backup(
                move_idx, value if (depth - i + 1) % 2 == 0 else -value
            )

    @staticmethod
    def _result_value(game: Game, result: str) -> float:
        """将红黑胜负转换为当前走棋方视角的价值。"""
        if result == 'draw':
            return 0.0
        red_won = result == 'red_wins'
        return 1.0 if red_won == game.red_to_move else -1.0

    def _legal_indices(self, legal_moves: list) -> List[int]:
        indices = []
        for (fr, fc), (tr, tc) in legal_moves:
            idx = self.move_index.move_to_index((fr, fc, tr, tc))
            if idx is not None:
                indices.append(idx)
        return indices

    @staticmethod
    def _normalized_priors(policy_probs, legal_indices: List[int]) -> List[float]:
        priors = [float(policy_probs[idx]) for idx in legal_indices]
        prior_sum = sum(priors)
        if prior_sum > 0:
            return [p / prior_sum for p in priors]
        return [1.0 / len(legal_indices)] * len(legal_indices)
    
    def _select_root_move(self, game: Game, root: MCTSNode,
                          legal_indices: List[int], temperature: float):
        """在实际落子前根据对局历史排除主动重复。"""
        candidates = self._non_repeating_root_moves(game, legal_indices)
        policy = self._compute_policy(root, temperature, candidates)
        if temperature <= 0.01:
            best_move = max(candidates, key=lambda move_idx: root.N.get(move_idx, 0))
        else:
            visits = np.array(
                [root.N.get(move_idx, 0) for move_idx in candidates],
                dtype=np.float64,
            )
            probabilities = self._temperature_probs(visits, temperature)
            best_move = candidates[np.random.choice(len(candidates), p=probabilities)]
        return best_move, policy

    def _non_repeating_root_moves(self, game: Game,
                                  legal_indices: List[int]) -> List[int]:
        """返回不会立即回到历史局面的根走法。"""
        if (
            not self.avoid_repetition
            or len(game.hash_history) < 2
            or game.evaluate_material_normalized()
            < self.repetition_material_threshold
        ):
            return legal_indices

        historical_hashes = set(game.hash_history)
        non_repeating = []
        for move_idx in legal_indices:
            move = self.move_index.index_to_move(move_idx)
            if move is None:
                continue
            fr, fc, tr, tc = move
            game.make_move((fr, fc), (tr, tc), validate=False)
            repeats = game.current_hash in historical_hashes
            game.undo_move()
            if not repeats:
                non_repeating.append(move_idx)

        if non_repeating and len(non_repeating) < len(legal_indices):
            self.repetition_moves_avoided += len(legal_indices) - len(non_repeating)
            return non_repeating
        return legal_indices

    def _compute_policy(self, root: MCTSNode, temperature: float,
                        move_indices: Optional[List[int]] = None) -> List[float]:
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

        candidates = (
            [move_idx for move_idx in move_indices if move_idx in root.N]
            if move_indices is not None else list(root.N)
        )
        if not candidates:
            return policy
        
        total_visits = sum(root.N.values())
        if total_visits == 0:
            return policy
        
        if temperature <= 0.01:
            # 贪心策略：只给最佳走法概率 1
            best_move = max(candidates, key=lambda move_idx: root.N[move_idx])
            policy[best_move] = 1.0
        else:
            # 带温度的策略分布
            visits = np.array([root.N[i] for i in candidates], dtype=np.float64)
            probs = self._temperature_probs(visits, temperature)
            for move_idx, probability in zip(candidates, probs):
                policy[move_idx] = float(probability)
        
        return policy

    @staticmethod
    def _temperature_probs(visits: np.ndarray, temperature: float) -> np.ndarray:
        """稳定地计算 N**(1/T)，避免低温下溢出或溢出。"""
        if visits.sum() <= 0:
            return np.ones(len(visits), dtype=np.float64) / len(visits)
        positive = visits > 0
        log_weights = np.full(len(visits), -np.inf, dtype=np.float64)
        log_weights[positive] = np.log(visits[positive]) / max(temperature, 1e-8)
        log_weights[positive] -= np.max(log_weights[positive])
        weights = np.zeros(len(visits), dtype=np.float64)
        weights[positive] = np.exp(log_weights[positive])
        return weights / weights.sum()


def search_many(requests: List[Tuple[MCTS, Game, float]],
                inference_batch_size: int = 64) -> List[Tuple[int, List[float]]]:
    """并行推进多棵 MCTS 树，把不同棋局的叶节点合并为 GPU batch。

    这里的“并行”指搜索调度交错进行；规则生成仍在当前进程完成，但神经网络
    不再为每盘棋分别等待一个小 batch。所有请求必须使用同一组模型权重。
    """
    if not requests:
        return []
    model = requests[0][0].model
    if any(mcts.model is not model for mcts, _, _ in requests):
        raise ValueError("search_many 的所有请求必须共享同一个模型实例")

    contexts = []
    results = [None] * len(requests)
    for index, (mcts, game, temperature) in enumerate(requests):
        context = mcts.prepare_search(game, temperature)
        context['index'] = index
        context['mcts'] = mcts
        contexts.append(context)
        if context['finished']:
            results[index] = context['result']

    active = [context for context in contexts if not context['finished']]
    if not active:
        return results

    # 未扩展根节点也跨棋局集中推理。相同初始局面会在批内去重。
    root_contexts = [c for c in active if c['needs_root_evaluation']]
    if root_contexts:
        predictor = root_contexts[0]['mcts']
        policies, _ = predictor._predict_with_cache(
            [c['game'].get_board_tensor() for c in root_contexts],
            [c['game'].current_hash for c in root_contexts],
        )
        for context, policy in zip(root_contexts, policies):
            context['root'].expand(
                context['legal_indices'],
                context['mcts']._normalized_priors(policy, context['legal_indices']),
            )

    for context in active:
        mcts = context['mcts']
        root = context['root']
        legal_indices = context['legal_indices']
        if mcts.add_root_noise:
            noise = np.random.dirichlet([mcts.dirichlet_alpha] * len(legal_indices))
            for move_idx, value in zip(legal_indices, noise):
                root.P[move_idx] = (
                    (1 - mcts.dirichlet_epsilon) * root.P.get(move_idx, 0.0)
                    + mcts.dirichlet_epsilon * float(value)
                )

        existing_visits = sum(root.N.values()) if context['reused_root'] else 0
        remaining = max(0, mcts.num_simulations - existing_visits)
        if context['reused_root'] and mcts.add_root_noise:
            remaining = max(
                min(mcts.inference_batch_size, mcts.num_simulations), remaining
            )
        context['remaining'] = remaining
        context['leaf_metadata'] = {}

    batch_limit = max(1, inference_batch_size)
    while any(context['remaining'] > 0 for context in active):
        records = []
        # Round-robin 选择，防止长棋局或高分支棋局独占一个 batch。
        while len(records) < batch_limit:
            added = False
            for context in active:
                if context['remaining'] <= 0:
                    continue
                records.append(context['mcts'].select_leaf_for_batch(
                    context['game'], context['root'], context['leaf_metadata']
                ))
                context['remaining'] -= 1
                added = True
                if len(records) >= batch_limit:
                    break
            if not added:
                break

        unique_metadata = []
        seen_metadata = set()
        for record in records:
            metadata = record['metadata']
            if metadata['value'] is None and id(metadata) not in seen_metadata:
                seen_metadata.add(id(metadata))
                unique_metadata.append(metadata)

        if unique_metadata:
            predictor = records[0]['mcts']
            policies, values = predictor._predict_with_cache(
                [metadata['state'] for metadata in unique_metadata],
                [metadata['position_hash'] for metadata in unique_metadata],
            )
            for metadata, policy, value in zip(unique_metadata, policies, values):
                metadata['policy_probs'] = policy
                metadata['network_value'] = float(value)

        for record in records:
            record['mcts'].complete_leaf_record(record)

    for context in active:
        results[context['index']] = context['mcts'].finish_prepared_search(context)
    return results


class MCTSEvaluator:
    """
    MCTS 评估器（用于对弈和分析）
    
    封装了 MCTS 搜索和走法选择逻辑，提供简洁的接口。
    """
    
    def __init__(self, model: PolicyValueNet, num_simulations: int = 400,
                 inference_batch_size: int = 64,
                 inference_cache_size: int = 10000):
        """
        Args:
            model: 策略-价值网络
            num_simulations: 搜索模拟次数
            inference_batch_size: GPU 叶节点推理批大小
            inference_cache_size: 当前模型的 LRU 局面缓存容量
        """
        self.inference_cache = InferenceCache(inference_cache_size)
        self.mcts = MCTS(
            model, num_simulations=num_simulations,
            inference_batch_size=inference_batch_size,
            inference_cache=self.inference_cache,
        )
        self.move_index = get_move_index()
    
    def select_move(
        self, game: Game, temperature: float = 0.01
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        为当前局面选择最佳走法。
        
        Args:
            game: 当前游戏状态
            temperature: 温度（对弈时接近0，训练时为1）
            
        Returns:
            ((from_row, from_col), (to_row, to_col))
        """
        move_idx, _ = self.mcts.search(game, temperature)
        
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
        # 分析需要完整的访问次数分布，而不是对弈模式的一热贪心策略。
        _, policy = self.mcts.search(game, temperature=1.0)

        # 基于 MCTS 策略概率排序
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
