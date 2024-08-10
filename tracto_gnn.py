import os
import argparse
import logging
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import signal
from config import Parameters
from trainer import TractoGNNTrainer
from tracker import Tracker
from utils.trainer_utils import plot_stats

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size)

    abs_path = os.path.abspath(__file__)
    dname = os.path.dirname(abs_path)
    log_path = os.path.join(dname, '.log')
    
    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    params = Parameters().params
    trainer = TractoGNNTrainer(logger=logger, params=params, rank=rank, world_size=world_size)

    if args.train:
        train_stats, val_stats = trainer.train()
        if rank == 0 and train_stats is not None:
            checkpoint = torch.load(params['checkpoint_path'])
            plot_stats(checkpoint['train_stats'], checkpoint['val_stats'], len(checkpoint['train_stats']), 'FODFs prediction')

    if args.track:
        tracker = Tracker(logger=logger, params=params, device=rank)
        tracker.track()

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", required=False, help='Whether to start training phase')
    parser.add_argument('--track', action="store_true", required=False, help='Whether to start inference phase')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    world_size =  min(1, torch.cuda.device_count())
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
