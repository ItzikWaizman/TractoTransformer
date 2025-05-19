import os
import logging
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from config.args import parse_args
from trainers.trainer import TractoTransformerTrainer
from trackers.tracker import Tracker

def setup(rank, world_size, socket_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = socket_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size, args.socket_port)

    abs_path = os.path.abspath(__file__)
    dname = os.path.dirname(abs_path)
    log_path = os.path.join(dname, '.log')
    
    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    if args.train:
        trainer = TractoTransformerTrainer(logger=logger, params=args, rank=rank, world_size=world_size)
        trainer.train()

    if args.track:
        tracker = Tracker(logger=logger, params=args)
        tracker.track()

    cleanup()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    world_size =  min(args.world_size, torch.cuda.device_count())
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
