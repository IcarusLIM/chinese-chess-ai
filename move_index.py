"""
走法索引映射模块
================
构建所有可能走法与整数索引之间的双向映射。

索引设计：
  - 将所有可能的走法（约2000+种）分配唯一的整数索引 0, 1, 2, ...
  - 神经网络的策略头输出一个长度等于总走法数的概率分布
  - 在 MCTS 中，通过索引快速查找对应的走法

走法编码：
  每个走法用 (from_row, from_col, to_row, to_col) 四元组表示
  所有可能的走法通过遍历所有棋子类型和所有起始位置生成
"""

from typing import List, Tuple, Dict, Optional
from constants import *


# 走法类型定义：(行偏移, 列偏移)
# 用于将、仕、象等单步棋子的走法生成
KING_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ADVISOR_DELTAS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
BISHOP_DELTAS = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

# 马的走法：(行偏移, 列偏移, 马腿行偏移, 马腿列偏移)
KNIGHT_DELTAS = [
    (-2, -1, -1, 0), (-2, 1, -1, 0),
    (2, -1, 1, 0), (2, 1, 1, 0),
    (-1, -2, 0, -1), (-1, 2, 0, 1),
    (1, -2, 0, -1), (1, 2, 0, 1),
]

# 直线方向（车、炮用）
STRAIGHT_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_board(r: int, c: int) -> bool:
    """判断坐标是否在棋盘内"""
    return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS


def _in_red_palace(r: int, c: int) -> bool:
    """判断是否在红方九宫格"""
    return 0 <= r <= 2 and 3 <= c <= 5


def _in_black_palace(r: int, c: int) -> bool:
    """判断是否在黑方九宫格"""
    return 7 <= r <= 9 and 3 <= c <= 5


def generate_all_moves() -> List[Tuple[int, int, int, int]]:
    """
    生成中国象棋中所有可能的走法（不考虑棋盘上是否有棋子）。
    
    这是一个静态枚举：对于每个起始位置和每种可能的走法模式，
    生成所有合法的目标位置。结果是一个固定的走法列表，
    用于建立走法↔索引的双向映射。
    
    Returns:
        所有可能走法的列表，每个元素为 (from_row, from_col, to_row, to_col)
    """
    moves = []
    move_set = set()  # 用于去重
    
    def add(fr, fc, tr, tc):
        """添加一条走法（自动去重）"""
        if _in_board(fr, fc) and _in_board(tr, tc):
            key = (fr, fc, tr, tc)
            if key not in move_set:
                move_set.add(key)
                moves.append(key)
    
    # 遍历所有可能的起始位置
    for fr in range(BOARD_ROWS):
        for fc in range(BOARD_COLS):
            
            # === 将/帅 ===
            # 在九宫格内走一步（上下左右）
            for dr, dc in KING_DELTAS:
                tr, tc = fr + dr, fc + dc
                if _in_board(tr, tc):
                    # 红帅只能在红方九宫格
                    if _in_red_palace(fr, fc) and _in_red_palace(tr, tc):
                        add(fr, fc, tr, tc)
                    # 黑将只能在黑方九宫格
                    if _in_black_palace(fr, fc) and _in_black_palace(tr, tc):
                        add(fr, fc, tr, tc)
            
            # === 仕/士 ===
            # 在九宫格内斜走一步
            for dr, dc in ADVISOR_DELTAS:
                tr, tc = fr + dr, fc + dc
                if _in_board(tr, tc):
                    if _in_red_palace(fr, fc) and _in_red_palace(tr, tc):
                        add(fr, fc, tr, tc)
                    if _in_black_palace(fr, fc) and _in_black_palace(tr, tc):
                        add(fr, fc, tr, tc)
            
            # === 相/象 ===
            # 走"田"字，不过河
            for dr, dc in BISHOP_DELTAS:
                tr, tc = fr + dr, fc + dc
                if _in_board(tr, tc):
                    # 红相不过河（行 0-4）
                    if fr <= 4 and tr <= 4:
                        add(fr, fc, tr, tc)
                    # 黑象不过河（行 5-9）
                    if fr >= 5 and tr >= 5:
                        add(fr, fc, tr, tc)
            
            # === 马 ===
            # 走"日"字（8个方向）
            for dr, dc, _, _ in KNIGHT_DELTAS:
                tr, tc = fr + dr, fc + dc
                if _in_board(tr, tc):
                    add(fr, fc, tr, tc)
            
            # === 车 ===
            # 沿直线走任意距离
            for dr, dc in STRAIGHT_DIRS:
                for dist in range(1, 10):
                    tr, tc = fr + dr * dist, fc + dc * dist
                    if not _in_board(tr, tc):
                        break
                    add(fr, fc, tr, tc)
            
            # === 炮 ===
            # 不吃子时与车相同；吃子时需隔一子
            # 这里只枚举所有可能的起止位置对（不考虑中间状态）
            for dr, dc in STRAIGHT_DIRS:
                for dist in range(1, 10):
                    tr, tc = fr + dr * dist, fc + dc * dist
                    if not _in_board(tr, tc):
                        break
                    add(fr, fc, tr, tc)
            
            # === 兵/卒 ===
            # 未过河只能前进，过河后可以左右
            if fr >= 3 and fr <= 8:  # 红兵
                add(fr, fc, fr + 1, fc)  # 前进（行号增大）
            if fr >= 5:  # 红兵（已过河）
                add(fr, fc, fr, fc - 1)  # 左
                add(fr, fc, fr, fc + 1)  # 右

            if fr <= 6 and fr >= 1:  # 黑卒（未过河）
                add(fr, fc, fr - 1, fc)  # 前进（行号减小）
            if fr <= 4:  # 黑卒（未过河）
                add(fr, fc, fr, fc - 1)  # 左
                add(fr, fc, fr, fc + 1)  # 右
    
    return moves


class MoveIndex:
    """
    走法 ↔ 索引 双向映射管理器
    
    功能：
    - 维护一个全局的走法列表
    - 提供 走法→索引 和 索引→走法 的快速查询
    - 在游戏开始时初始化，之后只读
    
    使用示例：
        mi = MoveIndex()
        idx = mi.move_to_index((0, 0, 0, 3))  # 车从(0,0)走到(0,3)
        move = mi.index_to_move(idx)            # 反向查询
        print(f"总走法数: {mi.num_moves}")
    """
    
    def __init__(self):
        """初始化走法索引映射"""
        # 生成所有可能的走法
        self.all_moves = generate_all_moves()
        self.num_moves = len(self.all_moves)
        
        # 构建 走法→索引 的哈希表
        self._move_to_idx: Dict[Tuple[int, int, int, int], int] = {}
        for idx, move in enumerate(self.all_moves):
            self._move_to_idx[move] = idx
        
        print(f"[MoveIndex] 初始化完成，共 {self.num_moves} 种可能走法")
    
    def move_to_index(self, move: Tuple[int, int, int, int]) -> Optional[int]:
        """
        将走法转换为整数索引。
        
        Args:
            move: (from_row, from_col, to_row, to_col)
            
        Returns:
            走法对应的索引，如果走法不存在返回 None
        """
        return self._move_to_idx.get(move)
    
    def index_to_move(self, idx: int) -> Optional[Tuple[int, int, int, int]]:
        """
        将整数索引转换为走法。
        
        Args:
            idx: 走法索引
            
        Returns:
            (from_row, from_col, to_row, to_col) 或 None（索引越界时）
        """
        if 0 <= idx < self.num_moves:
            return self.all_moves[idx]
        return None
    
    def get_legal_move_mask(self, game) -> List[int]:
        """
        获取当前局面下合法走法的索引列表。
        
        这个方法用于 MCTS 中过滤非法走法：
        神经网络输出所有走法的概率，但只有合法走法的概率有效。
        
        Args:
            game: Game 对象
            
        Returns:
            合法走法的索引列表
        """
        legal_moves = game.get_legal_moves()
        legal_indices = []
        for (fr, fc), (tr, tc) in legal_moves:
            idx = self.move_to_index((fr, fc, tr, tc))
            if idx is not None:
                legal_indices.append(idx)
        return legal_indices
    
    def get_legal_move_mask_array(self, game) -> List[float]:
        """
        获取合法走法的掩码数组（用于神经网络输出过滤）。
        
        Returns:
            长度为 num_moves 的列表，合法走法位置为 1.0，非法为 0.0
        """
        mask = [0.0] * self.num_moves
        legal_moves = game.get_legal_moves()
        for (fr, fc), (tr, tc) in legal_moves:
            idx = self.move_to_index((fr, fc, tr, tc))
            if idx is not None:
                mask[idx] = 1.0
        return mask


# 全局单例（在模块加载时创建一次）
_move_index = None

def get_move_index() -> MoveIndex:
    """获取全局 MoveIndex 单例"""
    global _move_index
    if _move_index is None:
        _move_index = MoveIndex()
    return _move_index


if __name__ == '__main__':
    # 测试：打印走法索引信息
    mi = get_move_index()
    print(f"总走法数: {mi.num_moves}")
    print(f"前10种走法:")
    for i in range(min(10, mi.num_moves)):
        print(f"  索引 {i}: {mi.index_to_move(i)}")
