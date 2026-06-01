"""
棋盘常量定义模块
================
定义中国象棋棋盘的尺寸、棋子编码、初始布局等核心常量。
所有其他模块都应从此处导入常量，确保一致性。

棋盘坐标系：
    列(col): 0-8，从左到右（从红方视角）
    行(row): 0-9，从下到上（红方在下方 0-4，黑方在上方 5-9）
    
棋子编码：
    0  = 空位
    1-7  = 红方棋子（帅、仕、相、马、车、炮、兵）
    8-14 = 黑方棋子（将、士、象、马、车、炮、卒）
"""

# ============================================================
# 棋盘尺寸
# ============================================================
BOARD_ROWS = 10   # 行数（纵向）
BOARD_COLS = 9    # 列数（横向）
NUM_SQUARES = BOARD_ROWS * BOARD_COLS  # 总格数 90

# ============================================================
# 空位
# ============================================================
EMPTY = 0

# ============================================================
# 红方棋子编码（1-7）
# ============================================================
R_KING   = 1   # 帅
R_ADVISOR = 2  # 仕
R_BISHOP = 3   # 相
R_KNIGHT = 4   # 马
R_ROOK   = 5   # 车
R_CANNON = 6   # 炮
R_PAWN   = 7   # 兵

# ============================================================
# 黑方棋子编码（8-14）
# ============================================================
B_KING   = 8   # 将
B_ADVISOR = 9  # 士
B_BISHOP = 10  # 象
B_KNIGHT = 11  # 马
B_ROOK   = 12  # 车
B_CANNON = 13  # 炮
B_PAWN   = 14  # 卒

# ============================================================
# 棋子显示名称（用于 Web 界面渲染）
# ============================================================
PIECE_NAMES = {
    EMPTY: '',
    R_KING: '帅', R_ADVISOR: '仕', R_BISHOP: '相',
    R_KNIGHT: '马', R_ROOK: '车', R_CANNON: '炮', R_PAWN: '兵',
    B_KING: '将', B_ADVISOR: '士', B_BISHOP: '象',
    B_KNIGHT: '馬', B_ROOK: '車', B_CANNON: '砲', B_PAWN: '卒',
}

# ============================================================
# 棋子符号（用于控制台显示和棋谱记录）
# ============================================================
PIECE_SYMBOLS = {
    EMPTY: '·',
    R_KING: 'K', R_ADVISOR: 'A', R_BISHOP: 'B',
    R_KNIGHT: 'N', R_ROOK: 'R', R_CANNON: 'C', R_PAWN: 'P',
    B_KING: 'k', B_ADVISOR: 'a', B_BISHOP: 'b',
    B_KNIGHT: 'n', B_ROOK: 'r', B_CANNON: 'c', B_PAWN: 'p',
}

# ============================================================
# 红方棋子集合 / 黑方棋子集合
# ============================================================
RED_PIECES   = {R_KING, R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK, R_CANNON, R_PAWN}
BLACK_PIECES = {B_KING, B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK, B_CANNON, B_PAWN}

# 红方兵种名称 → 编码
PIECE_TYPE_MAP = {
    'king': R_KING, 'advisor': R_ADVISOR, 'bishop': R_BISHOP,
    'knight': R_KNIGHT, 'rook': R_ROOK, 'cannon': R_CANNON, 'pawn': R_PAWN,
}

# ============================================================
# 初始棋盘布局（标准中国象棋开局）
# ============================================================
# board[row][col]，row=0 是红方底线（棋盘下方）
INITIAL_BOARD = [
    [R_ROOK, R_KNIGHT, R_BISHOP, R_ADVISOR, R_KING, R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK],
    [EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY],
    [EMPTY,  R_CANNON,  EMPTY,    EMPTY,     EMPTY,  EMPTY,     EMPTY,    R_CANNON,  EMPTY],
    [R_PAWN, EMPTY,     R_PAWN,   EMPTY,     R_PAWN, EMPTY,     R_PAWN,   EMPTY,     R_PAWN],
    [EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY],
    [EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY],
    [B_PAWN, EMPTY,     B_PAWN,   EMPTY,     B_PAWN, EMPTY,     B_PAWN,   EMPTY,     B_PAWN],
    [EMPTY,  B_CANNON,  EMPTY,    EMPTY,     EMPTY,  EMPTY,     EMPTY,    B_CANNON,  EMPTY],
    [EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY],
    [B_ROOK, B_KNIGHT, B_BISHOP, B_ADVISOR, B_KING, B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK],
]

# ============================================================
# 红方九宫格范围（帅/仕的活动范围）
# ============================================================
RED_PALACE_ROWS = (0, 1, 2)
RED_PALACE_COLS = (3, 4, 5)

# 黑方九宫格范围（将/士的活动范围）
BLACK_PALACE_ROWS = (7, 8, 9)
BLACK_PALACE_COLS = (3, 4, 5)

# ============================================================
# 红方半场行范围 / 黑方半场行范围
# ============================================================
RED_HALF_ROWS   = (0, 1, 2, 3, 4)
BLACK_HALF_ROWS = (5, 6, 7, 8, 9)

# ============================================================
# 棋子价值评估（用于简单局面评估和MCTS引导）
# ============================================================
PIECE_VALUES = {
    R_KING: 10000, R_ADVISOR: 20, R_BISHOP: 20,
    R_KNIGHT: 40, R_ROOK: 90, R_CANNON: 45, R_PAWN: 10,
    B_KING: 10000, B_ADVISOR: 20, B_BISHOP: 20,
    B_KNIGHT: 40, B_ROOK: 90, B_CANNON: 45, B_PAWN: 10,
}

# 兵过河前后的价值差异
PAWN_VALUE_CROSSED_RIVER = 20  # 过河兵价值提升

# ============================================================
# Zobrist 哈希相关
# ============================================================
import random
random.seed(42)  # 固定随机种子，确保可复现

# zobrist_table[piece][row][col] = 随机64位整数
ZOBRIST_TABLE = [
    [[random.getrandbits(64) for _ in range(BOARD_COLS)]
     for _ in range(BOARD_ROWS)]
    for _ in range(15)  # 0-14 共15种棋子（含空位）
]

# 轮到红方走棋的哈希增量
ZOBRIST_RED_TURN = random.getrandbits(64)
