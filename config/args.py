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
    parser.add_argument('--batch_size', type=int, default=1000, help='Data loader batch size.')
    parser.add_argument('--soft_labels_gaussian_var', type=float, default=0.1, help='Variance of the Gaussian distribution used to generate soft labels.')

    """ Model Parameters """
    parser.add_argument('--num_decoder_layers', type=int, default=8, help='Number of transformer decoder layers.')
    parser.add_argument('--nhead', type=int, default=13, help='Number of heads in the multi head self attention of the TrasfoemerEncoderLayer.')
    parser.add_argument('--ff_dim', type=int, default=512, help='Dimension of the feed-forward network in Transformer Encoder layer.')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout probability.')
    parser.add_argument('--max_positions', type=int, default=250, help='Number of positions to encode by Positional Encoder layer')
    parser.add_argument('--output_size', type=int, default=725, help='The output size of the network.')

    """ Training Parameters """
    parser.add_argument('--trained_model_path', type=str, default='trained_model/model.pt', help='Path for saving the model after training.')
    parser.add_argument('--save_checkpoints', type=bool, default=True, help='Whether to save model checkpoints during training or not.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/checkpoint_cnn_tracto_inferno.pth', help='Path to save the training checkpoints.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Training learning rate.')
    parser.add_argument('--decay_lr', type=bool, default=True, help='Whether to use learning rate decay during training.')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Lower bound for learning rate decay. Only valid when decay_lr==True.')
    parser.add_argument('--decay_lr_patience', type=int, default=10, help='Number of epochs to wait before learning rate decay is applied. Only valid when decay_lr==True.')
    parser.add_argument('--decay_lr_factor', type=float, default=0.7, help='The factor by which the learning rate is decayed. Only valid when decay_lr==True.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--k1', type=int, default=4, help='K in top k accuracy computation.')
    parser.add_argument('--k2', type=int, default=7, help='K in top k accuracy computation.')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Whether to use early stopping when validation does not improve.')
    parser.add_argument('--improvement_threshold', type=float, default=0.1, help='Minimal value of improvement to avoid decay learning rate or early stopping. Only valid when decay_lr==True or early_stopping==True.')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Number of epochs to wait before training is terminated when validation performance does not improve. Only valid when early_stopping==True.')
    parser.add_argument('--train_val_ratio', type=float, default=0.8, help='Training/Validation split ratio for training.')
    parser.add_argument('--load_checkpoint', type=bool, default=True, help='Whether to continue training from previous checkpoint or start a new one.')

    args = parser.parse_args()
    return args
