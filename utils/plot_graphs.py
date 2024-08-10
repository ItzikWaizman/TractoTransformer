from trainer_utils import *
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_stats.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    checkpoint = torch.load(checkpoint_path)
    assert len(checkpoint['train_stats']) == len(checkpoint['val_stats']), 'train_stat len must be equal to val_stats len'
    plot_stats(checkpoint['train_stats'], checkpoint['val_stats'], len(checkpoint['train_stats']), 'FODFs prediction')

