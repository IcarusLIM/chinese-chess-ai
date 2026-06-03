"""
中国象棋游戏引擎
================
实现完整的中国象棋规则，包括：
  - 棋盘状态管理
  - 合法走法生成（所有7种棋子的走法规则）
  - 将军/被将检测
  - 将死/困毙判定
  - 和棋判定（自然和棋、60回合无吃子）
  - Zobrist哈希（用于局面重复检测）

核心类：
  - Game: 单局游戏状态，支持走子、悔棋、局面序列化
"""

import copy
from typing import List, Tuple, Optional, Set
from constants import *


class Game:
    """
    中国象棋游戏状态管理器
    
    维护一局棋的完整状态，包括：
    - 当前棋盘（10x9 二维数组）
    - 当前走棋方
    - 走子历史（用于悔棋和回放）
    - 局面哈希历史（用于三次重复和棋判定）
    
    用法示例：
        game = Game()
        game.print_board()                    # 打印棋盘
        moves = game.get_legal_moves()        # 获取所有合法走法
        game.make_move((0, 0), (0, 3))        # 走车
        game.undo_move()                      # 悔棋
    """
    
    def __init__(self):
        """
        初始化一局新棋。
        棋盘使用深拷贝，避免修改全局常量 INITIAL_BOARD。
        """
        # 棋盘状态：board[row][col]，row=0 红方底线
        self.board = [row[:] for row in INITIAL_BOARD]
        
        # 当前走棋方：True=红方，False=黑方
        self.red_to_move = True
        
        # 走子历史：[(from_pos, to_pos, captured_piece, halfmove_before, fullmove_before), ...]
        self.move_history: List[Tuple[Tuple[int, int], Tuple[int, int], int, int, int]] = []
        
        # 局面哈希历史（用于三次重复和棋判定）
        self.hash_history: List[int] = []
        
        # 当前局面的Zobrist哈希值
        self.current_hash = self._compute_hash()
        self.hash_history.append(self.current_hash)
        
        # 半步计数器（用于60回合无吃子和棋规则）
        self.halfmove_clock = 0
        
        # 总走子步数
        self.fullmove_number = 1
    
    def _compute_hash(self) -> int:
        """
        计算当前棋盘的Zobrist哈希值。
        
        Zobrist哈希原理：
        - 为每个(棋子, 位置)组合分配一个随机64位整数
        - 棋盘哈希 = 所有棋子对应随机数的异或(XOR)
        - 走子时只需异或上新旧位置的值，无需重新计算整个棋盘
        
        Returns:
            64位整数哈希值
        """
        h = 0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece = self.board[row][col]
                if piece != EMPTY:
                    h ^= ZOBRIST_TABLE[piece][row][col]
        if self.red_to_move:
            h ^= ZOBRIST_RED_TURN
        return h
    
    def piece_at(self, row: int, col: int) -> int:
        """获取指定位置的棋子编码"""
        return self.board[row][col]
    
    def is_red(self, piece: int) -> bool:
        """判断棋子是否为红方"""
        return piece in RED_PIECES
    
    def is_black(self, piece: int) -> bool:
        """判断棋子是否为黑方"""
        return piece in BLACK_PIECES
    
    def is_own_piece(self, piece: int) -> bool:
        """判断棋子是否为当前走棋方的棋子"""
        if self.red_to_move:
            return piece in RED_PIECES
        else:
            return piece in BLACK_PIECES
    
    def is_enemy_piece(self, piece: int) -> bool:
        """判断棋子是否为对方的棋子"""
        if self.red_to_move:
            return piece in BLACK_PIECES
        else:
            return piece in RED_PIECES
    
    def in_board(self, row: int, col: int) -> bool:
        """判断坐标是否在棋盘范围内"""
        return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS
    
    def in_red_palace(self, row: int, col: int) -> bool:
        """判断坐标是否在红方九宫格内"""
        return row in RED_PALACE_ROWS and col in RED_PALACE_COLS
    
    def in_black_palace(self, row: int, col: int) -> bool:
        """判断坐标是否在黑方九宫格内"""
        return row in BLACK_PALACE_ROWS and col in BLACK_PALACE_COLS
    
    def in_own_half(self, row: int) -> bool:
        """判断坐标是否在己方半场"""
        if self.red_to_move:
            return row in RED_HALF_ROWS
        else:
            return row in BLACK_HALF_ROWS
    
    # ============================================================
    # 将军检测
    # ============================================================
    
    def find_king(self, is_red: bool) -> Optional[Tuple[int, int]]:
        """
        查找指定方的将/帅位置。
        
        Args:
            is_red: True=查找红帅，False=查找黑将
            
        Returns:
            (row, col) 位置元组，如果未找到返回 None（理论上不应发生）
        """
        king = R_KING if is_red else B_KING
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == king:
                    return (row, col)
        return None
    
    def is_in_check(self, is_red: bool) -> bool:
        """
        判断指定方是否被将军（处于被将状态）。
        
        检测逻辑：
        1. 找到己方将/帅的位置
        2. 检查对方所有棋子是否能攻击到该位置
        
        Args:
            is_red: True=检查红方是否被将，False=检查黑方是否被将
            
        Returns:
            True 表示被将
        """
        king_pos = self.find_king(is_red)
        if king_pos is None:
            return True  # 将/帅已被吃，视为被将
        
        kr, kc = king_pos
        
        # 检查对方车是否能攻击到将/帅
        opponent_rook = B_ROOK if is_red else R_ROOK
        if self._is_attacked_by_rook(kr, kc, opponent_rook):
            return True
        
        # 检查对方炮是否能攻击到将/帅
        opponent_cannon = B_CANNON if is_red else R_CANNON
        if self._is_attacked_by_cannon(kr, kc, opponent_cannon):
            return True
        
        # 检查对方马是否能攻击到将/帅
        opponent_knight = B_KNIGHT if is_red else R_KNIGHT
        if self._is_attacked_by_knight(kr, kc, opponent_knight):
            return True
        
        # 检查对方兵/卒是否能攻击到将/帅
        opponent_pawn = B_PAWN if is_red else R_PAWN
        if self._is_attacked_by_pawn(kr, kc, opponent_pawn):
            return True
        
        # 检查飞将（将帅面对面，中间无棋子）
        if self._is_facing_kings():
            return True
        
        return False
    
    def _is_attacked_by_rook(self, tr: int, tc: int, rook_piece: int) -> bool:
        """检查目标位置是否被指定类型的车攻击"""
        # 四个方向：上下左右
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = tr + dr, tc + dc
            while self.in_board(r, c):
                p = self.board[r][c]
                if p != EMPTY:
                    if p == rook_piece:
                        return True
                    break  # 被其他棋子挡住
                r += dr
                c += dc
        return False
    
    def _is_attacked_by_cannon(self, tr: int, tc: int, cannon_piece: int) -> bool:
        """检查目标位置是否被指定类型的炮攻击"""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = tr + dr, tc + dc
            found_screen = False  # 是否已找到炮架
            while self.in_board(r, c):
                p = self.board[r][c]
                if not found_screen:
                    if p != EMPTY:
                        found_screen = True  # 找到炮架
                else:
                    if p != EMPTY:
                        if p == cannon_piece:
                            return True
                        break  # 炮架后面不是炮
                r += dr
                c += dc
        return False
    
    def _is_attacked_by_knight(self, tr: int, tc: int, knight_piece: int) -> bool:
        """检查目标位置是否被指定类型的马攻击"""
        # 马的8个攻击来源位置（考虑蹩马腿）
        # (dr, dc): 马腿位置相对于目标的偏移
        knight_attacks = [
            (-2, -1, -1, 0), (-2, 1, -1, 0),   # 上方两个
            (2, -1, 1, 0), (2, 1, 1, 0),        # 下方两个
            (-1, -2, 0, -1), (-1, 2, 0, 1),     # 左右上
            (1, -2, 0, -1), (1, 2, 0, 1),       # 左右下
        ]
        for dr, dc, leg_dr, leg_dc in knight_attacks:
            r, c = tr - dr, tc - dc
            if self.in_board(r, c) and self.board[r][c] == knight_piece:
                # 检查蹩马腿
                leg_r, leg_c = r + leg_dr, c + leg_dc
                if self.board[leg_r][leg_c] == EMPTY:
                    return True
        return False
    
    def _is_attacked_by_pawn(self, tr: int, tc: int, pawn_piece: int) -> bool:
        """检查目标位置是否被指定类型的兵攻击"""
        # 兵的攻击方向取决于其所属阵营
        if pawn_piece == R_PAWN:
            # 红兵：未过河只能前进，过河后可以左右
            directions = [(-1, 0)]  # 红兵向上走
            # 检查目标位置前方和两侧是否有红兵
            for dr, dc in [(-1, 0), (0, -1), (0, 1)]:
                r, c = tr + dr, tc + dc
                if self.in_board(r, c) and self.board[r][c] == R_PAWN:
                    return True
        else:
            # 黑卒：向下走
            for dr, dc in [(1, 0), (0, -1), (0, 1)]:
                r, c = tr + dr, tc + dc
                if self.in_board(r, c) and self.board[r][c] == B_PAWN:
                    return True
        return False
    
    def _is_facing_kings(self) -> bool:
        """
        检查飞将：将帅是否在同一列且中间无棋子。
        这是中国象棋的特殊规则：将帅不能直接面对面。
        """
        red_king = self.find_king(True)
        black_king = self.find_king(False)
        if red_king is None or black_king is None:
            return False
        
        rr, rc = red_king
        br, bc = black_king
        
        if rc != bc:  # 不在同一列
            return False
        
        # 检查中间是否有棋子
        min_r = min(rr, br)
        max_r = max(rr, br)
        for r in range(min_r + 1, max_r):
            if self.board[r][rc] != EMPTY:
                return False
        
        return True
    
    # ============================================================
    # 合法走法生成
    # ============================================================
    
    def get_legal_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        获取当前走棋方的所有合法走法。
        
        合法走法 = 伪合法走法 + 走子后不被将军
        这确保了走子后己方将/帅安全。
        
        Returns:
            走法列表，每个元素为 ((from_row, from_col), (to_row, to_col))
        """
        pseudo_moves = self._get_pseudo_legal_moves()
        legal_moves = []
        
        for move in pseudo_moves:
            fr, fc = move[0]
            tr, tc = move[1]
            
            # 临时走子
            captured = self.board[tr][tc]
            moving_piece = self.board[fr][fc]
            self.board[tr][tc] = moving_piece
            self.board[fr][fc] = EMPTY
            
            # 检查走子后是否被将
            in_check = self.is_in_check(self.red_to_move)
            
            # 撤销走子
            self.board[fr][fc] = moving_piece
            self.board[tr][tc] = captured
            
            if not in_check:
                legal_moves.append(move)
        
        return legal_moves
    
    def _get_pseudo_legal_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        获取伪合法走法（不考虑走子后是否被将）。
        比 get_legal_moves 快，用于快速生成候选走法。
        """
        moves = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece = self.board[row][col]
                if piece == EMPTY:
                    continue
                if not self.is_own_piece(piece):
                    continue
                
                # 根据棋子类型生成走法
                piece_type = piece if piece <= 7 else piece - 7
                
                if piece_type == R_KING:
                    self._gen_king_moves(row, col, moves)
                elif piece_type == R_ADVISOR:
                    self._gen_advisor_moves(row, col, moves)
                elif piece_type == R_BISHOP:
                    self._gen_bishop_moves(row, col, moves)
                elif piece_type == R_KNIGHT:
                    self._gen_knight_moves(row, col, moves)
                elif piece_type == R_ROOK:
                    self._gen_rook_moves(row, col, moves)
                elif piece_type == R_CANNON:
                    self._gen_cannon_moves(row, col, moves)
                elif piece_type == R_PAWN:
                    self._gen_pawn_moves(row, col, moves)
        
        return moves
    
    def _gen_king_moves(self, row: int, col: int, moves: list):
        """
        生成将/帅的走法。
        规则：只能在九宫格内走一步（上下左右）。
        """
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if not self.in_board(r, c):
                continue
            # 检查是否在己方九宫格内
            if self.red_to_move:
                if not self.in_red_palace(r, c):
                    continue
            else:
                if not self.in_black_palace(r, c):
                    continue
            
            target = self.board[r][c]
            if target == EMPTY or self.is_enemy_piece(target):
                moves.append(((row, col), (r, c)))
    
    def _gen_advisor_moves(self, row: int, col: int, moves: list):
        """
        生成仕/士的走法。
        规则：在九宫格内沿斜线走一步。
        """
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = row + dr, col + dc
            if not self.in_board(r, c):
                continue
            if self.red_to_move:
                if not self.in_red_palace(r, c):
                    continue
            else:
                if not self.in_black_palace(r, c):
                    continue
            
            target = self.board[r][c]
            if target == EMPTY or self.is_enemy_piece(target):
                moves.append(((row, col), (r, c)))
    
    def _gen_bishop_moves(self, row: int, col: int, moves: list):
        """
        生成相/象的走法。
        规则：走"田"字（斜走两格），不能过河，不能被蹩象眼。
        """
        for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            r, c = row + dr, col + dc
            if not self.in_board(r, c):
                continue
            # 不能过河
            if self.red_to_move and r > 4:
                continue
            if not self.red_to_move and r < 5:
                continue
            
            # 象眼位置（"田"字中心）
            eye_r, eye_c = row + dr // 2, col + dc // 2
            if self.board[eye_r][eye_c] != EMPTY:
                continue  # 被蹩象眼
            
            target = self.board[r][c]
            if target == EMPTY or self.is_enemy_piece(target):
                moves.append(((row, col), (r, c)))
    
    def _gen_knight_moves(self, row: int, col: int, moves: list):
        """
        生成马的走法。
        规则：走"日"字（先直后斜），需要检查蹩马腿。
        蹩马腿：如果马前进方向上紧邻的一格有棋子，则不能走。
        """
        # (dr, dc, leg_dr, leg_dc)
        # dr, dc: 最终目标偏移
        # leg_dr, leg_dc: 马腿位置偏移
        knight_moves = [
            (-2, -1, -1, 0), (-2, 1, -1, 0),  # 上方
            (2, -1, 1, 0), (2, 1, 1, 0),       # 下方
            (-1, -2, 0, -1), (-1, 2, 0, 1),    # 左右上
            (1, -2, 0, -1), (1, 2, 0, 1),      # 左右下
        ]
        
        for dr, dc, leg_dr, leg_dc in knight_moves:
            r, c = row + dr, col + dc
            if not self.in_board(r, c):
                continue
            
            # 检查蹩马腿
            leg_r, leg_c = row + leg_dr, col + leg_dc
            if self.board[leg_r][leg_c] != EMPTY:
                continue  # 马腿被绊
            
            target = self.board[r][c]
            if target == EMPTY or self.is_enemy_piece(target):
                moves.append(((row, col), (r, c)))
    
    def _gen_rook_moves(self, row: int, col: int, moves: list):
        """
        生成车的走法。
        规则：沿直线（横/竖）走任意距离，遇到棋子停下（敌方可吃）。
        """
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            while self.in_board(r, c):
                target = self.board[r][c]
                if target == EMPTY:
                    moves.append(((row, col), (r, c)))
                elif self.is_enemy_piece(target):
                    moves.append(((row, col), (r, c)))
                    break  # 吃子后停下
                else:
                    break  # 己方棋子，停下
                r += dr
                c += dc
    
    def _gen_cannon_moves(self, row: int, col: int, moves: list):
        """
        生成炮的走法。
        规则：
        - 不吃子时：沿直线走任意距离（与车相同）
        - 吃子时：必须隔一个棋子（炮架）才能吃
        """
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            found_screen = False
            while self.in_board(r, c):
                target = self.board[r][c]
                if not found_screen:
                    if target == EMPTY:
                        moves.append(((row, col), (r, c)))
                    else:
                        found_screen = True  # 找到炮架
                else:
                    if target != EMPTY:
                        if self.is_enemy_piece(target):
                            moves.append(((row, col), (r, c)))
                        break  # 无论是否吃子，都停下
                r += dr
                c += dc
    
    def _gen_pawn_moves(self, row: int, col: int, moves: list):
        """
        生成兵/卒的走法。
        规则：
        - 未过河：只能前进
        - 过河后：可以前进或左右移动，不能后退
        """
        if self.red_to_move:
            # 红兵向上走（行号增大）
            directions = [(1, 0)]  # 前进
            if row >= 5:  # 已过河（在黑方半场）
                directions.extend([(0, -1), (0, 1)])  # 可以左右走
        else:
            # 黑卒向下走（行号减小）
            directions = [(-1, 0)]  # 前进
            if row <= 4:  # 已过河（在红方半场）
                directions.extend([(0, -1), (0, 1)])
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if self.in_board(r, c):
                target = self.board[r][c]
                if target == EMPTY or self.is_enemy_piece(target):
                    moves.append(((row, col), (r, c)))
    
    # ============================================================
    # 走子操作
    # ============================================================
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                  *, validate: bool = True) -> int:
        """
        执行一步走子。
        
        Args:
            from_pos: 起始位置 (row, col)
            to_pos: 目标位置 (row, col)
            validate: 是否校验走法合法性。MCTS 等已过滤合法走法的
                热路径应传 False，避免重复调用 get_legal_moves()。
            
        Returns:
            被吃掉的棋子编码（如果没有吃子返回 EMPTY）
            
        Raises:
            ValueError: 如果走法不合法
        """
        fr, fc = from_pos
        tr, tc = to_pos
        
        # 验证起始位置有己方棋子
        moving_piece = self.board[fr][fc]
        if moving_piece == EMPTY or not self.is_own_piece(moving_piece):
            raise ValueError(f"起始位置 ({fr},{fc}) 没有己方棋子")
        
        # 验证目标位置不是己方棋子
        target_piece = self.board[tr][tc]
        if target_piece != EMPTY and self.is_own_piece(target_piece):
            raise ValueError(f"目标位置 ({tr},{tc}) 有己方棋子")

        if validate and (from_pos, to_pos) not in self.get_legal_moves():
            raise ValueError(f"不合法的走法: ({fr},{fc}) → ({tr},{tc})")
        
        # 执行走子
        self.board[tr][tc] = moving_piece
        self.board[fr][fc] = EMPTY
        
        # 保存历史（含走子前的计数器，供悔棋恢复）
        self.move_history.append((
            from_pos, to_pos, target_piece,
            self.halfmove_clock, self.fullmove_number,
        ))
        
        # 更新Zobrist哈希
        self.current_hash ^= ZOBRIST_TABLE[moving_piece][fr][fc]  # 移除旧位置
        self.current_hash ^= ZOBRIST_TABLE[moving_piece][tr][tc]  # 添加新位置
        if target_piece != EMPTY:
            self.current_hash ^= ZOBRIST_TABLE[target_piece][tr][tc]  # 移除被吃的棋子
        self.current_hash ^= ZOBRIST_RED_TURN  # 切换走棋方
        
        self.hash_history.append(self.current_hash)
        
        # 更新半步计数器
        if target_piece != EMPTY or moving_piece in (R_PAWN, B_PAWN):
            self.halfmove_clock = 0  # 吃子或兵走子，重置计数器
        else:
            self.halfmove_clock += 1
        
        # 切换走棋方
        self.red_to_move = not self.red_to_move
        
        # 更新总步数
        if not self.red_to_move:
            self.fullmove_number += 1
        
        return target_piece
    
    def undo_move(self) -> bool:
        """
        悔棋（撤销最后一步）。
        
        Returns:
            True 表示悔棋成功，False 表示没有历史可撤销
        """
        if not self.move_history:
            return False
        
        from_pos, to_pos, captured, halfmove_before, fullmove_before = self.move_history.pop()
        fr, fc = from_pos
        tr, tc = to_pos
        
        moving_piece = self.board[tr][tc]
        
        # 撤销走子
        self.board[fr][fc] = moving_piece
        self.board[tr][tc] = captured
        
        # 切换走棋方
        self.red_to_move = not self.red_to_move

        # 恢复计数器
        self.halfmove_clock = halfmove_before
        self.fullmove_number = fullmove_before
        
        # 撤销哈希
        self.hash_history.pop()
        self.current_hash = self.hash_history[-1] if self.hash_history else self._compute_hash()
        
        return True
    
    # ============================================================
    # 游戏结束判定
    # ============================================================
    
    def is_checkmate(self) -> bool:
        """
        判断当前走棋方是否被将死。
        将死 = 被将军 + 没有合法走法。
        """
        if not self.is_in_check(self.red_to_move):
            return False
        return len(self.get_legal_moves()) == 0
    
    def is_stalemate(self) -> bool:
        """
        判断当前走棋方是否被困毙（无子可动且未被将军）。
        中国象棋中，困毙算输（不同于国际象棋的和棋）。
        """
        if self.is_in_check(self.red_to_move):
            return False
        return len(self.get_legal_moves()) == 0
    
    def is_draw(self) -> bool:
        """
        判断是否和棋。
        和棋条件：
        1. 60回合（120半步）无吃子
        2. 三次重复局面
        """
        # 60回合无吃子
        if self.halfmove_clock >= 120:
            return True
        
        # 三次重复局面
        if len(self.hash_history) >= 5:
            current = self.hash_history[-1]
            count = sum(1 for h in self.hash_history if h == current)
            if count >= 3:
                return True
        
        return False
    
    def is_game_over(self) -> Tuple[bool, Optional[str]]:
        """
        判断游戏是否结束。
        
        Returns:
            (is_over, result):
            - (True, 'red_wins'): 红方胜
            - (True, 'black_wins'): 黑方胜
            - (True, 'draw'): 和棋
            - (False, None): 游戏未结束
        """
        # 检查将死
        if self.is_checkmate():
            winner = 'black_wins' if self.red_to_move else 'red_wins'
            return (True, winner)
        
        # 检查困毙
        if self.is_stalemate():
            winner = 'black_wins' if self.red_to_move else 'red_wins'
            return (True, winner)
        
        # 检查和棋
        if self.is_draw():
            return (True, 'draw')
        
        return (False, None)
    
    # ============================================================
    # 棋盘显示
    # ============================================================
    
    def print_board(self):
        """
        在控制台打印当前棋盘。
        使用中文棋子名称显示，红方在下方。
        """
        print("\n  0 1 2 3 4 5 6 7 8")
        print("  ───────────────────")
        for row in range(BOARD_ROWS - 1, -1, -1):
            line = f"{row}│"
            for col in range(BOARD_COLS):
                piece = self.board[row][col]
                name = PIECE_NAMES.get(piece, '?')
                line += f"{name} " if name else "· "
            print(line)
        print("  ───────────────────")
        print(f"走棋方: {'红方' if self.red_to_move else '黑方'}")
        
        if self.is_in_check(self.red_to_move):
            print("⚠️  将军！")
    
    def to_fen(self) -> str:
        """
        将当前局面转换为 FEN (Forsyth-Edwards Notation) 字符串。
        中国象棋 FEN 格式示例：
        rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w
        
        Returns:
            FEN 字符串
        """
        fen = ""
        for row in range(BOARD_ROWS - 1, -1, -1):
            empty_count = 0
            for col in range(BOARD_COLS):
                piece = self.board[row][col]
                if piece == EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen += PIECE_SYMBOLS.get(piece, '?')
            if empty_count > 0:
                fen += str(empty_count)
            if row > 0:
                fen += "/"
        
        fen += " w" if self.red_to_move else " b"
        return fen
    
    def get_board_tensor(self) -> 'list':
        """
        将棋盘状态编码为神经网络输入格式。

        使用 15 个通道（plane），每个通道是一个 10x9 的矩阵：
        - 通道 0-6：红方的 帅、仕、相、马、车、炮、兵（二值）
        - 通道 7-13：黑方的 将、士、象、马、车、炮、卒（二值）
        - 通道 14：当前走棋方（红方走棋全 1，黑方走棋全 0）

        Returns:
            15x10x9 的三维列表
        """
        tensor = [[[0.0] * BOARD_COLS for _ in range(BOARD_ROWS)] for _ in range(15)]

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece = self.board[row][col]
                if piece != EMPTY:
                    channel = piece - 1  # 编码 1-14 → 通道 0-13
                    tensor[channel][row][col] = 1.0

        # 第 15 通道：当前走棋方
        turn_value = 1.0 if self.red_to_move else 0.0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                tensor[14][row][col] = turn_value

        return tensor
    
    def copy(self) -> 'Game':
        """创建游戏状态的深拷贝"""
        new_game = Game.__new__(Game)
        new_game.board = [row[:] for row in self.board]
        new_game.red_to_move = self.red_to_move
        new_game.move_history = self.move_history[:]
        new_game.hash_history = self.hash_history[:]
        new_game.current_hash = self.current_hash
        new_game.halfmove_clock = self.halfmove_clock
        new_game.fullmove_number = self.fullmove_number
        return new_game

    def evaluate_material_normalized(self) -> float:
        """
        归一化的材料评估，用于混合评估信号。

        排除将/帅（始终存在），将材料差归一化到 [-1, 1]。
        返回值从当前走棋方视角：正数表示当前方优势。

        Returns:
            float: [-1, 1] 的局面评估值
        """
        # 初始非将子力总值（双方各 480，共 960）
        INITIAL_MATERIAL_NO_KING = 960

        balance = 0.0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece = self.board[row][col]
                if piece != EMPTY and piece not in (R_KING, B_KING):
                    value = PIECE_VALUES.get(piece, 0)
                    # 兵过河后价值提升
                    if piece == R_PAWN and row >= 5:
                        value += PAWN_VALUE_CROSSED_RIVER
                    elif piece == B_PAWN and row <= 4:
                        value += PAWN_VALUE_CROSSED_RIVER

                    if piece in RED_PIECES:
                        balance += value
                    else:
                        balance -= value

        # 归一化到 [-1, 1]
        normalized = max(-1.0, min(1.0, balance / INITIAL_MATERIAL_NO_KING))

        # 从当前走棋方视角返回
        return normalized if self.red_to_move else -normalized
