"""快速比较串行与多 actor 自对弈吞吐。"""

import argparse
import time

import numpy as np

from network import create_model, get_default_device
from trainer import SelfPlayWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--moves', type=int, default=20)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=1)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--server-wait-ms', type=float, default=5.0)
    args = parser.parse_args()

    model = create_model(
        args.blocks, args.channels, policy_channels=4,
        device=get_default_device(),
    )
    common = dict(
        num_simulations=args.simulations,
        inference_batch_size=args.batch_size,
        inference_cache_size=10000,
        max_moves=args.moves,
    )

    np.random.seed(20240815)
    serial = SelfPlayWorker(model, num_workers=1, **common)
    start = time.perf_counter()
    serial_samples = []
    for _ in range(args.games):
        serial_samples.extend(serial.play_one_game())
    serial_seconds = time.perf_counter() - start

    np.random.seed(20240815)
    parallel = SelfPlayWorker(
        model,
        num_workers=args.workers,
        inference_server_batch_size=args.batch_size * args.workers,
        inference_server_wait_ms=args.server_wait_ms,
        **common,
    )
    start = time.perf_counter()
    parallel_samples = parallel.play_multiple_games(args.games)
    parallel_seconds = time.perf_counter() - start

    print(f"串行: {serial_seconds:.2f}s / {len(serial_samples)} 条样本")
    print(f"并行: {parallel_seconds:.2f}s / {len(parallel_samples)} 条样本")
    print(f"加速比: {serial_seconds / parallel_seconds:.2f}x")


if __name__ == '__main__':
    main()
