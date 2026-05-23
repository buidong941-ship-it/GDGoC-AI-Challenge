"""Shared training utilities used across scripts.

Keep this module lightweight and dependency-free so it can be imported from
multiple training entrypoints without circular imports.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Any
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model_fn(model, optimizer, global_step, epsilon, lr, input_spec, num_actions, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epsilon": epsilon,
        "lr": lr,
        "input_dim": input_spec,  # backward-compatible key name
        "input_shape": input_spec,
        "input_spec": input_spec,
        "num_actions": num_actions,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def plot_loss(loss_history, save_path=None):
    if plt is None:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def plot_rewards(reward_history, save_path=None):
    if plt is None:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Rewards')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def plot_win_rates(win_history, save_path=None):
    if plt is None:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(win_history, label='Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('DQN Training Win Rates')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.asarray(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_moving_average(data, window_size=10, save_path=None):
    if plt is None:
        return
    ma_data = moving_average(data, window_size)
    plt.figure(figsize=(10, 5))
    plt.plot(ma_data, label=f'Moving Average (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('DQN Training Moving Average')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()
