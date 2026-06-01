"""
Web 服务器模块
==============
提供基于 Flask 的 Web 界面，支持：
  - 人机对弈（PvE）
  - 棋谱回放
  - 棋局分析
  - 棋谱下载

API 端点：
  - GET  /              渲染主页面
  - POST /api/new_game  创建新游戏
  - POST /api/move       玩家走子
  - POST /api/ai_move    AI 走子
  - POST /api/undo       悔棋
  - GET  /api/state      获取当前状态
  - POST /api/analyze    分析当前局面
  - GET  /api/download   下载棋谱
"""

import os
import json
import time
import uuid
from flask import Flask, render_template, request, jsonify, session
from game import Game
from network import create_model, PolicyValueNet
from mcts import MCTS, MCTSEvaluator
from move_index import get_move_index
from constants import PIECE_NAMES, PIECE_SYMBOLS, BOARD_ROWS, BOARD_COLS

# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(__name__)
app.secret_key = os.urandom(24)

# AI 引擎（共享，无状态）
ai_evaluator = None
move_index = None

# 每会话游戏状态
games = {}
MAX_GAMES = 100


def get_game():
    """获取当前会话的游戏实例，不存在则自动创建"""
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    sid = session['sid']
    if sid not in games:
        if len(games) >= MAX_GAMES:
            oldest = next(iter(games))
            del games[oldest]
        games[sid] = Game()
    return games[sid]


def init_ai(model_path: str = None, num_simulations: int = 200):
    """
    初始化 AI 引擎。
    
    Args:
        model_path: 模型文件路径（None 则使用随机初始化的模型）
        num_simulations: MCTS 模拟次数（越多越强但越慢）
    """
    global ai_evaluator, move_index
    
    move_index = get_move_index()
    
    # 创建模型
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    model = create_model(num_blocks=10, channels=256, device=device)
    
    # 加载训练好的模型（如果存在）
    if model_path and os.path.exists(model_path):
        checkpoint = __import__('torch').load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Web] 已加载模型: {model_path}")
        else:
            print(f"[Web] 模型格式不正确，使用随机初始化")
    else:
        print(f"[Web] 未找到模型文件，使用随机初始化（AI 走子将随机）")
    
    ai_evaluator = MCTSEvaluator(model, num_simulations=num_simulations)
    print(f"[Web] AI 引擎初始化完成，MCTS 模拟次数: {num_simulations}")


def game_state_to_dict(game: Game) -> dict:
    """
    将游戏状态转换为 JSON 可序列化的字典。
    
    Args:
        game: Game 对象
        
    Returns:
        包含完整游戏状态的字典
    """
    # 棋盘数据
    board_data = []
    for row in range(BOARD_ROWS - 1, -1, -1):  # 从上到下（黑方视角）
        row_data = []
        for col in range(BOARD_COLS):
            piece = game.board[row][col]
            row_data.append({
                'type': piece,
                'name': PIECE_NAMES.get(piece, ''),
                'is_red': piece in {1, 2, 3, 4, 5, 6, 7},
                'is_black': piece in {8, 9, 10, 11, 12, 13, 14},
            })
        board_data.append(row_data)
    
    # 游戏状态
    is_over, result = game.is_game_over()
    
    return {
        'board': board_data,
        'red_to_move': game.red_to_move,
        'move_count': len(game.move_history),
        'is_check': game.is_in_check(game.red_to_move),
        'is_game_over': is_over,
        'result': result,
        'fen': game.to_fen(),
        'history': [
            {
                'from': {'row': fr, 'col': fc},
                'to': {'row': tr, 'col': tc},
                'piece': PIECE_NAMES.get(game.board[tr][tc] if len(game.move_history) > i else 0, '?'),
            }
            for i, ((fr, fc), (tr, tc), _) in enumerate(game.move_history)
        ] if game.move_history else [],
    }


# ============================================================
# API 端点
# ============================================================

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """
    创建新游戏。
    
    Returns:
        新游戏的初始状态
    """
    sid = session.get('sid', str(uuid.uuid4()))
    session['sid'] = sid
    games[sid] = Game()
    return jsonify({
        'success': True,
        'state': game_state_to_dict(games[sid]),
    })


@app.route('/api/move', methods=['POST'])
def player_move():
    """
    玩家走子。
    
    请求体:
        {
            "from": {"row": 0, "col": 0},
            "to": {"row": 0, "col": 3}
        }
    
    Returns:
        走子后的游戏状态
    """
    current_game = get_game()

    data = request.json
    from_pos = (data['from']['row'], data['from']['col'])
    to_pos = (data['to']['row'], data['to']['col'])
    
    # 验证走法合法性
    legal_moves = current_game.get_legal_moves()
    if (from_pos, to_pos) not in legal_moves:
        return jsonify({
            'success': False,
            'error': '不合法的走法',
        })
    
    # 执行走子
    captured = current_game.make_move(from_pos, to_pos)
    
    return jsonify({
        'success': True,
        'captured': PIECE_NAMES.get(captured, ''),
        'state': game_state_to_dict(current_game),
    })


@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """
    AI 走子。
    
    Returns:
        AI 的走法和走子后的游戏状态
    """
    current_game = get_game()

    if ai_evaluator is None:
        return jsonify({
            'success': False,
            'error': 'AI 未初始化',
        })
    
    # 检查游戏是否已结束
    is_over, _ = current_game.is_game_over()
    if is_over:
        return jsonify({
            'success': False,
            'error': '游戏已结束',
        })
    
    # AI 思考
    start_time = time.time()
    move = ai_evaluator.select_move(current_game, temperature=0.01)
    think_time = time.time() - start_time
    
    if move is None:
        return jsonify({
            'success': False,
            'error': 'AI 无法找到合法走法',
        })
    
    from_pos, to_pos = move
    
    # 执行走子
    captured = current_game.make_move(from_pos, to_pos)
    
    return jsonify({
        'success': True,
        'move': {
            'from': {'row': from_pos[0], 'col': from_pos[1]},
            'to': {'row': to_pos[0], 'col': to_pos[1]},
        },
        'captured': PIECE_NAMES.get(captured, ''),
        'think_time': round(think_time, 2),
        'state': game_state_to_dict(current_game),
    })


@app.route('/api/undo', methods=['POST'])
def undo_move():
    """
    悔棋（撤销两步：玩家的走子 + AI 的走子）。
    
    Returns:
        悔棋后的游戏状态
    """
    current_game = get_game()

    # 撤销两步（玩家 + AI）
    undone = 0
    for _ in range(2):
        if current_game.undo_move():
            undone += 1
    
    return jsonify({
        'success': True,
        'undone': undone,
        'state': game_state_to_dict(current_game),
    })


@app.route('/api/state')
def get_state():
    """获取当前游戏状态"""
    current_game = get_game()

    return jsonify({
        'success': True,
        'state': game_state_to_dict(current_game),
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_position():
    """
    分析当前局面，返回 AI 推荐的走法。
    
    Returns:
        推荐走法列表（按推荐度排序）
    """
    current_game = get_game()

    if ai_evaluator is None:
        return jsonify({'success': False, 'error': 'AI 未初始化'})
    
    # 获取分析结果
    top_moves = ai_evaluator.analyze_position(current_game, top_k=5)
    
    # 格式化结果
    analysis = []
    for i, move_info in enumerate(top_moves):
        fr, fc = move_info['move'][0]
        tr, tc = move_info['move'][1]
        analysis.append({
            'rank': i + 1,
            'from': {'row': fr, 'col': fc},
            'to': {'row': tr, 'col': tc},
            'score': round(move_info['score'], 4),
            'prior': round(move_info['prior'], 4),
            'description': f"{PIECE_NAMES.get(current_game.board[fr][fc], '?')} "
                          f"({fr},{fc})→({tr},{tc})",
        })
    
    return jsonify({
        'success': True,
        'analysis': analysis,
    })


@app.route('/api/download')
def download_game():
    """
    下载棋谱。
    
    Returns:
        JSON 格式的棋谱数据
    """
    current_game = get_game()

    # 构建棋谱
    game_record = {
        'format': 'ChineseChess_v1',
        'result': '',
        'moves': [],
    }
    
    is_over, result = current_game.is_game_over()
    if result == 'red_wins':
        game_record['result'] = '1-0'
    elif result == 'black_wins':
        game_record['result'] = '0-1'
    else:
        game_record['result'] = '1/2-1/2'
    
    # 重建每一步的棋盘状态
    temp_game = Game()
    for i, ((fr, fc), (tr, tc), captured) in enumerate(current_game.move_history):
        piece = temp_game.board[fr][fc]
        game_record['moves'].append({
            'number': i + 1,
            'from': {'row': fr, 'col': fc},
            'to': {'row': tr, 'col': tc},
            'piece': PIECE_NAMES.get(piece, '?'),
            'captured': PIECE_NAMES.get(captured, ''),
            'fen': temp_game.to_fen(),
        })
        temp_game.make_move((fr, fc), (tr, tc))
    
    return jsonify({
        'success': True,
        'game_record': game_record,
    })


@app.route('/api/replay', methods=['POST'])
def replay_move():
    """
    棋谱回放：跳转到指定步数。
    
    请求体:
        {"move_index": 5}
    
    Returns:
        指定步数时的棋盘状态
    """
    current_game = get_game()
    data = request.json
    move_index = data.get('move_index', 0)

    # 重建棋盘到指定步数
    temp_game = Game()
    for i, ((fr, fc), (tr, tc), _) in enumerate(current_game.move_history):
        if i >= move_index:
            break
        temp_game.make_move((fr, fc), (tr, tc))
    
    return jsonify({
        'success': True,
        'state': game_state_to_dict(temp_game),
        'current_move': move_index,
    })


# ============================================================
# 主程序入口
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='中国象棋 AI Web 服务器')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS 模拟次数')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 初始化 AI
    init_ai(model_path=args.model, num_simulations=args.simulations)
    
    # 启动服务器
    print(f"\n[Web] 服务器启动于 http://{args.host}:{args.port}")
    print(f"[Web] 在浏览器中打开 http://localhost:{args.port} 开始对弈")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
