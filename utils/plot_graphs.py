import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.args import parse_args
from utils.trainer_utils import plot_stats

if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    assert len(checkpoint['train_stats']) == len(checkpoint['val_stats']), 'train_stat len must be equal to val_stats len'
    plot_stats(checkpoint['train_stats'], checkpoint['val_stats'], len(checkpoint['train_stats']), 'FODFs prediction')
