import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn


DATA_FILE = Path('/workspace/kame_putt_ws/shot_trials_model.csv')
MODEL_FILE = Path('/workspace/kame_putt_ws/shot_speed_nn.pt')

SPEED_MIN = 0.10
SPEED_MAX = 1.20
SPEED_BIN_COUNT = 12
SAMPLES_PER_BIN = 2

_MODEL_CACHE = None
_TRAINING_PLAN = None


def _valid_rows():
    """移動距離と停止誤差の整合性を確認して、有効な記録だけ返す。"""
    if not DATA_FILE.exists():
        return []

    valid = []
    with DATA_FILE.open(newline='') as f:
        for row in csv.DictReader(f):
            target = float(row['target_distance'])
            speed = round(float(row['initial_speed']), 2)
            traveled = float(row['travel_distance'])
            error = float(row['error'])

            # 動かなかった試行は記録には残すが、距離から初速を学ぶNNからは除外
            if traveled <= 0.001:
                continue

            # 目標距離がほぼゼロの試行や、整合しない測定値も除外
            consistency_error = abs(abs(target - traveled) - error)
            if target <= 0.05 or consistency_error > 0.05:
                continue

            valid.append({
                'target_distance': target,
                'initial_speed': speed,
                'travel_distance': traveled,
                'error': error,
            })

    return valid


def choose_exploration_speed():
    """速度範囲全体から、ランダムな初速を選んで試行する。"""
    global _TRAINING_PLAN

    if _TRAINING_PLAN is None:
        plan = []
        bin_width = (SPEED_MAX - SPEED_MIN) / SPEED_BIN_COUNT

        for bin_index in range(SPEED_BIN_COUNT):
            low = SPEED_MIN + bin_index * bin_width
            if bin_index == SPEED_BIN_COUNT - 1:
                high = SPEED_MAX
            else:
                high = low + bin_width - 0.01

            candidates = [
                round(low + step * 0.01, 2)
                for step in range(int(round((high - low) * 100)) + 1)
            ]
            plan.extend(random.sample(candidates, SAMPLES_PER_BIN))

        # 動き出ししきい値0.20付近を確認する初速も追加
        plan.extend([0.10, 0.15, 0.19, 0.20, 0.21, 0.25])
        random.shuffle(plan)
        _TRAINING_PLAN = plan

    if not _TRAINING_PLAN:
        raise RuntimeError('ランダム探索の24試行は完了しているよ')

    return _TRAINING_PLAN.pop(), 'random exploration'


def _build_network():
    return nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
    )


def train_and_save_model():
    """CSVの試行データでNNを学習し、重みと正規化情報を保存する。"""
    rows = _valid_rows()
    if len(rows) < 3:
        raise RuntimeError('学習に使える有効な記録が3件未満だよ')

    # 同じ初速の繰り返しは、平均した移動距離にまとめる
    distances_by_speed = defaultdict(list)
    for row in rows:
        distances_by_speed[row['initial_speed']].append(
            row['travel_distance']
        )

    speeds = sorted(distances_by_speed)
    traveled = [
        sum(distances_by_speed[speed]) / len(distances_by_speed[speed])
        for speed in speeds
    ]

    # NNの入力は実測移動距離、出力は初速
    x = torch.tensor(traveled, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(speeds, dtype=torch.float32).reshape(-1, 1)

    x_mean = x.mean()
    x_scale = torch.clamp(x.std(unbiased=False), min=1e-6)
    y_mean = y.mean()
    y_scale = torch.clamp(y.std(unbiased=False), min=1e-6)

    x_norm = (x - x_mean) / x_scale
    y_norm = (y - y_mean) / y_scale

    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = _build_network()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.0001
    )

    for _ in range(4000):
        prediction = model(x_norm)
        loss = torch.mean((prediction - y_norm) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    checkpoint = {
        'state_dict': model.state_dict(),
        'x_mean': float(x_mean.item()),
        'x_scale': float(x_scale.item()),
        'y_mean': float(y_mean.item()),
        'y_scale': float(y_scale.item()),
        'min_speed': float(min(speeds)),
        'max_speed': float(max(speeds)),
        'training_loss': float(loss.item()),
        'training_points': len(speeds),
    }

    torch.save(checkpoint, MODEL_FILE)
    global _MODEL_CACHE
    _MODEL_CACHE = None

    excluded = 0
    if DATA_FILE.exists():
        with DATA_FILE.open(newline='') as f:
            total_rows = sum(1 for _ in csv.DictReader(f))
        excluded = total_rows - len(rows)

    print(
        f'NN学習完了: 有効試行={len(rows)}, '
        f'異なる初速={len(speeds)}, 除外={excluded}, '
        f'loss={loss.item():.4f}'
    )
    print(f'モデル保存先: {MODEL_FILE}')


def choose_initial_speed(target_distance):
    """保存済みNNに目標距離を入力し、初速を推論する。学習はしない。"""
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        if not MODEL_FILE.exists():
            raise RuntimeError(
                '保存済みNNがないよ。先にtrain_and_save_model()を実行してね'
            )

        checkpoint = torch.load(
            MODEL_FILE, map_location='cpu', weights_only=True
        )
        model = _build_network()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        _MODEL_CACHE = (model, checkpoint)

    model, checkpoint = _MODEL_CACHE
    x = torch.tensor([[float(target_distance)]], dtype=torch.float32)
    x_norm = (x - checkpoint['x_mean']) / checkpoint['x_scale']

    with torch.no_grad():
        speed_norm = model(x_norm)
        speed = (
            speed_norm * checkpoint['y_scale'] + checkpoint['y_mean']
        ).item()

    speed = max(
        checkpoint['min_speed'],
        min(checkpoint['max_speed'], speed),
    )
    return round(float(speed), 2), 'saved NN'


def record_trial(target_distance, initial_speed, travel_distance, error):
    """実測した試行結果をCSVへ保存する。"""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not DATA_FILE.exists()

    with DATA_FILE.open('a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'target_distance',
                'initial_speed',
                'travel_distance',
                'error',
            ])
        writer.writerow([
            f'{target_distance:.2f}',
            f'{initial_speed:.2f}',
            f'{travel_distance:.3f}',
            f'{error:.3f}',
        ])
