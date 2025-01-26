import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    """ Pre-Process Parameters """
    parser.add_argument('--raw_subjects_directory', type=str, default='/home/itzikwei/tract_inferno_dataset/chosen_subjects', help='Path to a folder containing a subset of TractoInferno subjects.')
    parser.add_argument('--processed_data_directory', type=str, default='/home/itzikwei/processed_tract_inferno_dataset', help='Destination to save processed data ready for training.')
    parser.add_argument('--streamline_stepsize', type=float, default=3.5, help='Distance (mm) between consecutive points in the streamline for the resampling script.')

    """ General Parameters """
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Whether to use GPU or CPU as the device.', choices=['cuda', 'cpu'])

    """ Data Parameters """
    parser.add_argument('--num_gradients', type=int, default=65, help='Number of gradient directions in the data.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Portion of the subjects that will used for train data.')

    args = parser.parse_args()
    return args