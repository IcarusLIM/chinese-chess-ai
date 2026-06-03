"""
主程序入口
==========
统一的命令行入口，支持三种运行模式：
  - train: 自对弈训练
  - web:   启动 Web 对弈界面
  - play:  控制台对弈（调试用）

用法：
  python main.py train                                        # 从零开始训练
  python main.py train --material-warmup 80                   # 启用材料评估 warmup
  python main.py train --resume models/latest_model.pt        # 从 checkpoint 续训
  python main.py train --iterations 50                        # 训练50轮
  python main.py web                                          # 启动 Web 界面
  python main.py web --model models/latest_model.pt           # 加载模型后启动
  python main.py play                                         # 控制台对弈
"""

import argparse
import sys
import os

# 确保当前目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_train(args):
    """训练模式"""
    from trainer import TrainingPipeline
    
    print("=" * 60)
    print("中国象棋 AI - 训练模式")
    print("=" * 60)
    
    pipeline = TrainingPipeline(
        num_blocks=args.blocks,
        channels=args.channels,
        num_simulations=args.simulations,
        self_play_games=args.self_play_games,
        training_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        resume_from=args.resume,
        load_buffer=args.load_buffer,
        material_warmup=args.material_warmup,
    )

    pipeline.run(num_iterations=args.iterations)


def cmd_web(args):
    """Web 对弈模式"""
    from web.app import app, init_ai
    
    print("=" * 60)
    print("中国象棋 AI - Web 对弈模式")
    print("=" * 60)
    
    init_ai(model_path=args.model, num_simulations=args.simulations)
    
    print(f"\n在浏览器中打开: http://localhost:{args.port}")
    print("按 Ctrl+C 停止服务器\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_play(args):
    """控制台对弈模式（调试用）"""
    from game import Game
    from network import create_model
    from mcts import MCTSEvaluator
    from move_index import get_move_index
    import torch
    
    print("=" * 60)
    print("中国象棋 AI - 控制台对弈")
    print("=" * 60)
    print("输入走法格式: from_row from_col to_row to_col")
    print("例如: 0 0 0 3（车从(0,0)走到(0,3)）")
    print("输入 'quit' 退出，'undo' 悔棋")
    print("=" * 60)
    
    # 初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(num_blocks=10, channels=256, device=device)
    
    if args.model and os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载模型: {args.model}")
    
    evaluator = MCTSEvaluator(model, num_simulations=args.simulations)
    game = Game()
    
    while True:
        game.print_board()
        
        is_over, result = game.is_game_over()
        if is_over:
            print(f"\n游戏结束！结果: {result}")
            break
        
        if game.red_to_move:
            # 玩家（红方）走子
            print("\n轮到你走棋（红方）:")
            cmd = input("> ").strip()
            
            if cmd == 'quit':
                break
            if cmd == 'undo':
                game.undo_move()
                game.undo_move()
                continue
            
            try:
                parts = cmd.split()
                if len(parts) != 4:
                    print("格式错误，请输入: from_row from_col to_row to_col")
                    continue
                fr, fc, tr, tc = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                game.make_move((fr, fc), (tr, tc))
            except Exception as e:
                print(f"走子失败: {e}")
        else:
            # AI（黑方）走子
            print("\nAI 思考中...")
            move = evaluator.select_move(game, temperature=0.01)
            if move:
                game.make_move(*move, validate=False)
                print(f"AI 走: {move[0]} → {move[1]}")
            else:
                print("AI 无法走子")
                break


def main():
    parser = argparse.ArgumentParser(
        description='中国象棋 AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py train                                        # 从零开始训练
  python main.py train --material-warmup 80                   # 启用 80 轮材料评估 warmup
  python main.py train --resume models/latest_model.pt        # 从 checkpoint 续训
  python main.py train --iterations 50 --blocks 5             # 快速训练
  python main.py web                                          # 启动 Web 界面
  python main.py web --model models/latest_model.pt           # 带模型启动
  python main.py play                                         # 控制台对弈
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='运行模式')
    
    # === train 子命令 ===
    train_parser = subparsers.add_parser('train', help='自对弈训练')
    train_parser.add_argument('--iterations', type=int, default=100, help='训练迭代次数')
    train_parser.add_argument('--blocks', type=int, default=10, help='残差块数量')
    train_parser.add_argument('--channels', type=int, default=256, help='特征通道数')
    train_parser.add_argument('--simulations', type=int, default=400, help='MCTS 模拟次数')
    train_parser.add_argument('--self-play-games', type=int, default=25, help='每轮自对弈局数')
    train_parser.add_argument('--epochs', type=int, default=5, help='每轮训练 epoch 数')
    train_parser.add_argument('--batch-size', type=int, default=256, help='训练批大小')
    train_parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    train_parser.add_argument('--save-dir', type=str, default='models', help='模型保存目录')
    train_parser.add_argument('--resume', type=str, default=None, help='从指定 checkpoint 继续训练')
    train_parser.add_argument('--load-buffer', type=str, default=None, help='从指定文件加载 replay buffer')
    train_parser.add_argument('--material-warmup', type=int, default=0, help='材料评估 warmup 迭代数（0=不启用）')
    
    # === web 子命令 ===
    web_parser = subparsers.add_parser('web', help='Web 对弈界面')
    web_parser.add_argument('--model', type=str, default=None, help='模型文件路径')
    web_parser.add_argument('--simulations', type=int, default=200, help='MCTS 模拟次数')
    web_parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    web_parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    web_parser.add_argument('--debug', action='store_true', help='调试模式')
    
    # === play 子命令 ===
    play_parser = subparsers.add_parser('play', help='控制台对弈')
    play_parser.add_argument('--model', type=str, default=None, help='模型文件路径')
    play_parser.add_argument('--simulations', type=int, default=200, help='MCTS 模拟次数')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'web':
        cmd_web(args)
    elif args.command == 'play':
        cmd_play(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
