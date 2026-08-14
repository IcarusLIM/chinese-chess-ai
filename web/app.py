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
import secrets
import sqlite3
from flask import Flask, render_template, request, jsonify, session
from game import Game
from network import create_model, create_model_from_checkpoint, get_default_device
from mcts import MCTSEvaluator
from move_index import get_move_index
from constants import PIECE_NAMES, PIECE_SYMBOLS, BOARD_ROWS, BOARD_COLS

# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(__name__)
os.makedirs(app.instance_path, exist_ok=True)


def _load_secret_key() -> str:
    """优先使用环境变量，否则使用跨重启持久化的本地密钥。"""
    configured = os.environ.get('CHINESE_CHESS_SECRET_KEY')
    if configured:
        return configured
    key_path = os.path.join(app.instance_path, 'secret_key')
    try:
        with open(key_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        key = secrets.token_hex(32)
        # 独占创建，避免多个 worker 首次启动时相互覆盖。
        try:
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(key)
            return key
        except FileExistsError:
            with open(key_path, 'r', encoding='utf-8') as f:
                return f.read().strip()


app.secret_key = _load_secret_key()

# AI 引擎（共享，无状态）
ai_evaluator = None
move_index = None

# 每会话游戏状态持久化到 SQLite，可跨进程和服务重启共享。
MAX_GAMES = 100


def _serialize_game(game: Game) -> str:
    return json.dumps({
        'board': game.board,
        'red_to_move': game.red_to_move,
        'move_history': game.move_history,
        'hash_history': game.hash_history,
        'current_hash': game.current_hash,
        'halfmove_clock': game.halfmove_clock,
        'fullmove_number': game.fullmove_number,
    }, separators=(',', ':'))


def _deserialize_game(payload: str) -> Game:
    data = json.loads(payload)
    game = Game.__new__(Game)
    game.board = data['board']
    game.red_to_move = data['red_to_move']
    game.move_history = [
        (tuple(item[0]), tuple(item[1]), item[2], item[3], item[4])
        for item in data['move_history']
    ]
    game.hash_history = data['hash_history']
    game.current_hash = data['current_hash']
    game.halfmove_clock = data['halfmove_clock']
    game.fullmove_number = data['fullmove_number']
    return game


class GameStore:
    """SQLite 棋局仓库；每次操作使用独立连接以支持多 worker。"""

    def __init__(self, path: str, max_games: int):
        self.path = path
        self.max_games = max_games
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS games ('
                'sid TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL)'
            )

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def load(self, sid: str):
        with self._connect() as conn:
            row = conn.execute('SELECT state FROM games WHERE sid = ?', (sid,)).fetchone()
        return _deserialize_game(row[0]) if row else None

    def save(self, sid: str, game: Game):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO games(sid, state, updated_at) VALUES (?, ?, ?) '
                'ON CONFLICT(sid) DO UPDATE SET '
                'state=excluded.state, updated_at=excluded.updated_at',
                (sid, _serialize_game(game), time.time()),
            )
            count = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
            if count > self.max_games:
                conn.execute(
                    'DELETE FROM games WHERE sid IN ('
                    'SELECT sid FROM games ORDER BY updated_at ASC LIMIT ?) ',
                    (count - self.max_games,),
                )


database_path = os.environ.get(
    'CHINESE_CHESS_DB', os.path.join(app.instance_path, 'games.sqlite3')
)
game_store = GameStore(database_path, MAX_GAMES)


def get_game():
    """获取当前会话的游戏实例，不存在则自动创建"""
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    sid = session['sid']
    game = game_store.load(sid)
    if game is None:
        game = Game()
        game_store.save(sid, game)
    return game


def save_game(game: Game):
    """保存当前会话棋局。"""
    game_store.save(session['sid'], game)


def init_ai(model_path: str = None, num_simulations: int = 200,
            inference_batch_size: int = 64,
            inference_cache_size: int = 10000):
    """
    初始化 AI 引擎。
    
    Args:
        model_path: 模型文件路径（None 则使用随机初始化的模型）
        num_simulations: MCTS 模拟次数（越多越强但越慢）
        inference_batch_size: GPU 叶节点推理批大小
        inference_cache_size: 当前模型的 LRU 局面缓存容量
    """
    global ai_evaluator, move_index
    
    move_index = get_move_index()
    
    # 创建模型
    device = get_default_device()

    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        model = create_model_from_checkpoint(model_path, device)
    else:
        model = create_model(num_blocks=10, channels=256, device=device)
        print(f"[Web] 未指定模型文件，使用随机初始化（AI 走子将随机）")
    
    ai_evaluator = MCTSEvaluator(
        model, num_simulations=num_simulations,
        inference_batch_size=inference_batch_size,
        inference_cache_size=inference_cache_size,
    )
    print(
        f"[Web] AI 引擎初始化完成，MCTS 模拟次数: {num_simulations}，"
        f"推理批大小: {inference_batch_size}"
    )


def _build_move_history(game: Game) -> list:
    """通过回放棋局构建走子历史（含正确的棋子名称）。"""
    if not game.move_history:
        return []

    history = []
    temp = Game()
    for from_pos, to_pos, captured, _, _ in game.move_history:
        fr, fc = from_pos
        tr, tc = to_pos
        piece = temp.board[fr][fc]
        history.append({
            'from': {'row': fr, 'col': fc},
            'to': {'row': tr, 'col': tc},
            'piece': PIECE_NAMES.get(piece, '?'),
            'captured': PIECE_NAMES.get(captured, ''),
        })
        temp.make_move(from_pos, to_pos, validate=False)
    return history


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
        'history': _build_move_history(game),
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
    game = Game()
    game_store.save(sid, game)
    return jsonify({
        'success': True,
        'state': game_state_to_dict(game),
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
    captured = current_game.make_move(from_pos, to_pos, validate=False)
    save_game(current_game)
    
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
    captured = current_game.make_move(from_pos, to_pos, validate=False)
    save_game(current_game)
    
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
    save_game(current_game)
    
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
    for i, (from_pos, to_pos, captured, _, _) in enumerate(current_game.move_history):
        fr, fc = from_pos
        tr, tc = to_pos
        piece = temp_game.board[fr][fc]
        game_record['moves'].append({
            'number': i + 1,
            'from': {'row': fr, 'col': fc},
            'to': {'row': tr, 'col': tc},
            'piece': PIECE_NAMES.get(piece, '?'),
            'captured': PIECE_NAMES.get(captured, ''),
            'fen': temp_game.to_fen(),
        })
        temp_game.make_move((fr, fc), (tr, tc), validate=False)
    
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
    for i, (from_pos, to_pos, _, _, _) in enumerate(current_game.move_history):
        if i >= move_index:
            break
        temp_game.make_move(from_pos, to_pos, validate=False)
    
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
    parser.add_argument('--inference-batch-size', type=int, default=64, help='MCTS GPU 叶节点推理批大小')
    parser.add_argument('--inference-cache-size', type=int, default=10000, help='LRU 局面缓存容量')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 初始化 AI
    init_ai(
        model_path=args.model, num_simulations=args.simulations,
        inference_batch_size=args.inference_batch_size,
        inference_cache_size=args.inference_cache_size,
    )
    
    # 启动服务器
    print(f"\n[Web] 服务器启动于 http://{args.host}:{args.port}")
    print(f"[Web] 在浏览器中打开 http://localhost:{args.port} 开始对弈")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
