"""神经网络状态与走法的规范化编码。

网络始终从当前走棋方视角观察棋盘：当前方位于棋盘下方，棋子通道 0-6；
对手位于上方，棋子通道 7-13。黑方走棋时旋转棋盘并交换双方语义。
"""

from functools import lru_cache
from typing import Tuple

import numpy as np

from constants import BOARD_COLS, BOARD_ROWS, EMPTY
from move_index import get_move_index


NUM_INPUT_PLANES = 19


def canonical_position(row: int, col: int, red_to_move: bool) -> Tuple[int, int]:
    if red_to_move:
        return row, col
    return BOARD_ROWS - 1 - row, BOARD_COLS - 1 - col


def canonical_move(move: Tuple[int, int, int, int], red_to_move: bool):
    fr, fc, tr, tc = move
    fr, fc = canonical_position(fr, fc, red_to_move)
    tr, tc = canonical_position(tr, tc, red_to_move)
    return fr, fc, tr, tc


def encode_game(game) -> np.ndarray:
    """编码当前局面为 ``[19, 10, 9]`` float32 张量。"""
    tensor = np.zeros((NUM_INPUT_PLANES, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = game.board[row][col]
            if piece == EMPTY:
                continue
            cr, cc = canonical_position(row, col, game.red_to_move)
            piece_is_red = piece <= 7
            is_current = piece_is_red == game.red_to_move
            piece_type = (piece - 1) % 7
            tensor[piece_type + (0 if is_current else 7), cr, cc] = 1.0

    if game.move_history:
        from_pos, to_pos = game.move_history[-1][0], game.move_history[-1][1]
        fr, fc = canonical_position(*from_pos, game.red_to_move)
        tr, tc = canonical_position(*to_pos, game.red_to_move)
        tensor[14, fr, fc] = 1.0
        tensor[15, tr, tc] = 1.0

    repetition_count = sum(
        position_hash == game.current_hash for position_hash in game.hash_history
    )
    tensor[16, :, :] = max(0, min(repetition_count - 1, 2)) / 2.0
    tensor[17, :, :] = min(game.halfmove_clock, 120) / 120.0
    tensor[18, :, :] = 1.0 if game.is_in_check(game.red_to_move) else 0.0
    return tensor


def inference_cache_key(game) -> tuple:
    """返回覆盖全部网络输入语义的不可变缓存键。"""
    last_move = None
    if game.move_history:
        from_pos, to_pos = game.move_history[-1][0], game.move_history[-1][1]
        last_move = canonical_move((*from_pos, *to_pos), game.red_to_move)
    repetition_count = sum(
        position_hash == game.current_hash for position_hash in game.hash_history
    )
    return (
        game.current_hash,
        last_move,
        min(repetition_count, 3),
        min(game.halfmove_clock, 120),
    )


def canonical_move_index(move, red_to_move: bool) -> int:
    return get_move_index().move_to_index(canonical_move(move, red_to_move))


@lru_cache(maxsize=1)
def mirror_index_map() -> np.ndarray:
    """返回规范化走法索引在左右镜像后的映射。"""
    move_index = get_move_index()
    mapping = np.empty(move_index.num_moves, dtype=np.int64)
    for index, (fr, fc, tr, tc) in enumerate(move_index.all_moves):
        mirrored = (fr, BOARD_COLS - 1 - fc, tr, BOARD_COLS - 1 - tc)
        mapped = move_index.move_to_index(mirrored)
        if mapped is None:
            raise RuntimeError(f"走法缺少镜像映射: {(fr, fc, tr, tc)}")
        mapping[index] = mapped
    return mapping


def policy_to_canonical(policy, red_to_move: bool) -> np.ndarray:
    """把绝对坐标策略转换到当前方规范坐标。"""
    source = np.asarray(policy, dtype=np.float32)
    if red_to_move:
        return source.copy()
    move_index = get_move_index()
    target = np.zeros_like(source)
    for index in np.flatnonzero(source):
        move = move_index.index_to_move(int(index))
        target[canonical_move_index(move, False)] = source[index]
    return target


def legal_mask_to_canonical(legal_indices, red_to_move: bool) -> np.ndarray:
    move_index = get_move_index()
    mask = np.zeros(move_index.num_moves, dtype=np.bool_)
    for index in legal_indices:
        move = move_index.index_to_move(int(index))
        mask[canonical_move_index(move, red_to_move)] = True
    return mask
