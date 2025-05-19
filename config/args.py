import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    """ Pre-Process Parameters """
    parser.add_argument('--raw_subjects_directory', type=str, default='.', help='Path to a folder containing a subset of TractoInferno subjects.')
    parser.add_argument('--processed_data_directory', type=str, default='.', help='Destination to save processed data ready for training.')
    parser.add_argument('--streamline_stepsize', type=float, default=3.5, help='Distance (mm) between consecutive points in the streamline for the resampling script.')

    """ General Parameters """
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Whether to use GPU or CPU as the device.', choices=['cuda', 'cpu'])
    parser.add_argument('--train', action="store_true", help='Run TractoTransformer script with train mode.')
    parser.add_argument('--track',  action="store_true", help='Run TractoTransformer script with track mode.')
    parser.add_argument('--cuda_devices', type=str, default="1,2", help='Avaliable GPU indices to use.')
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--socket_port', type=str, default='12357', help='Number of GPUs to use')

    """ Data Parameters """
    parser.add_argument('--num_gradients', type=int, default=100, help='Number of gradient directions in the data.')
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
    parser.add_argument('--trained_model_path', type=str, default='.', help='Path for saving the model after training.')
    parser.add_argument('--save_checkpoints', type=bool, default=True, help='Whether to save model checkpoints during training or not.')
    parser.add_argument('--checkpoint_path', type=str, default='.', help='Path to save the training checkpoints.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Training learning rate.')
    parser.add_argument('--decay_lr', type=bool, default=True, help='Whether to use learning rate decay during training.')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Lower bound for learning rate decay. Only valid when decay_lr==True.')
    parser.add_argument('--decay_lr_patience', type=int, default=5, help='Number of epochs to wait before learning rate decay is applied. Only valid when decay_lr==True.')
    parser.add_argument('--decay_lr_factor', type=float, default=0.6, help='The factor by which the learning rate is decayed. Only valid when decay_lr==True.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--k1', type=int, default=4, help='K in top k accuracy computation.')
    parser.add_argument('--k2', type=int, default=7, help='K in top k accuracy computation.')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Whether to use early stopping when validation does not improve.')
    parser.add_argument('--improvement_threshold', type=float, default=0.3, help='Minimal value of improvement to avoid decay learning rate or early stopping. Only valid when decay_lr==True or early_stopping==True.')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Number of epochs to wait before training is terminated when validation performance does not improve. Only valid when early_stopping==True.')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='Whether to continue training from previous checkpoint or start a new one.')
    parser.add_argument('--save_model', type=bool, default=True, help='Decide whether do save the model.')

    """ Tracking parameters """
    parser.add_argument('--num_seeds', type=int, default=1000000, help='Number of initial points to start tracking from.')
    parser.add_argument('--track_batch_size', type=int, default=1000, help='Batch size for tracking.')
    parser.add_argument('--angular_threshold', type=float, default=120.0, help='If the angle between 2 consecutive tracking steps is greater than this threshold (degrees), tracking is terminated.')
    parser.add_argument('--fa_threshold', type=float, default=0.1, help='Fractional anisotropy threshold to terminate the tracking.')
    parser.add_argument('--max_track_len', type=int, default=210, help='Maximum allowed length of a streamline.')
    parser.add_argument('--min_streamline_len', type=int, default=3, help='Minimum allowed length of a steamline.')
    parser.add_argument('--tracking_step_size', type=float, default=3.5, help='Tracking step size.')
    parser.add_argument('--save_tracking', type=bool, default=True, help='Whether to save tracking output.')
    parser.add_argument('--mask_dilation', type=bool, default=True, help='Whether to perform white matter mask dilation.')
    parser.add_argument('--trk_file_saving_path', type=str, default=".", help='Path to save output tractography.')
    parser.add_argument('--track_mode', type=str, choices=['comparison', 'inference'], default='comparison', help='Whether to perform tracking in order to compare to a reference algorithm or not. The difference is in seed points generation.')
    parser.add_argument('--test_subject', type=str, default='.', help='Number of initial points to start tracking from.')
    parser.add_argument('--normalize_brain', action="store_true", help='Whether to normalize the brain.')

    args = parser.parse_args()
    return args
