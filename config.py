import torch


class Parameters(object):

    def __init__(self):
        self.params = dict()

        """ Model Parameters """

        # num_transformer_encoder_layers - Number of transformer encoder layers.
        self.params['num_transformer_encoder_layers'] = 6

        # num_transformer_decoder_layers - Number of transformer decoder layers.
        self.params['num_transformer_decoder_layers'] = 8

        # nhead - Number of heads in the Multi Head Self Attention mechanism of the TransformerEncoderLayer.
        self.params['nhead'] = 10

        # transformer_feed_forward_dim - Dimension of the feedforward network in TransformerEncoder layer.
        self.params['transformer_feed_forward_dim'] = 512

        # dropout_rate - Probability to execute a dropout
        self.params['dropout_rate'] = 0.3

        # max_streamline_len - Upper bound of an expected streamline length. Used for positional encoding.
        self.params['max_streamline_len'] = 250

        # output_size - Decoder output features size.
        self.params['output_size'] = 725

        # model_weights_save_dir - (string) Path for saving the model's files after training is done.
        self.params['trained_model_path'] = ""

        """ Training Parameters """
        # save_checkpoints - (bool) Whether to save model checkpoints during training or not
        self.params['save_checkpoints'] = True

        # checkpoint_path - (string) a path to save the checkpoint
        self.params['checkpoint_path'] = ""
 
        # learning_rate -(float) Initial learning rate in training phase.
        self.params['learning_rate'] = 0.001

        # min_lr - (float) min value of learning rate
        self.params['min_lr'] = 7e-5

        # batch_size - (int) Data batch size for training.
        self.params['batch_size'] = 100

        # epochs - (int) Number of training epochs.
        self.params['epochs'] = 100

        # top k accuracy computation
        self.params['k1'] = 7

        self.params['k2'] = 4

        # decay_LR - (bool) Whether to use learning rate decay.
        self.params['decay_LR'] = True

        # decay_LR_patience - (int) Number of training epochs to wait in case validation performance does not improve
        # before learning rate decay is applied.
        self.params['decay_LR_patience'] = 10

        # decay_factor - (float [0, 1]) In an LR decay step, the existing LR will be multiplied by this factor.
        self.params['decay_factor'] = 0.8

        # early_stopping - (bool) Whether to use early stopping.
        self.params['early_stopping'] = True

        # threshold - (float) min value of improvement we require between epochs
        self.params['threshold'] = 0.1

        # early_stopping - (int) Number of epochs to wait before training is terminated when validation performance
        # does not improve.
        self.params['early_stopping_patience'] = 20

        # train_val_ratio - (float [0, 1]) Training/Validation split ratio for training.
        self.params['train_val_ratio'] = 0.8

        # device - Device for training, GPU if available and otherwise CPU.
        self.params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load_checkpoint - (bool) Whether to start a new training or continue from privious checkpoint
        self.params['load_checkpoint'] = True

        """ Data Parameters """

        # subject_folder - (string) Path to subject folder containing related data
        self.params['train_subject_folder'] = ''

        self.params['val_subject_folder'] = ''
        
        self.params['test_subject_folder'] =''

        self.params['num_of_gradients'] = 50
        
        """ Tracking parameters """
        # num_seeds - (int) number of initial points to start stracking from.
        self.params['num_seeds'] = 100000

        # track_batch_size - (int)  batch size to perform tracking.
        self.params['track_batch_size'] = 500

        # angular_threshold - (float) if the angle between 2 consecutive steps in tracking is greater than angular threshold (in degrees), tracking is treminated.
        self.params['angular_threshold'] = 45.0
        
        # fa_threshold - (float) fractional anisotropy threshold to terminate the tracking.
        self.params['fa_threshold'] = 0.2

        # max_sequence_length - (int) max allowed length of streamline.
        self.params['max_sequence_length'] = 150

        # min_streamline_length - (int) min allowed length of streamline.
        self.params['min_streamline_length'] = 3

        # tracking_step_size - (float) step size in which we will make a step in the corresponding direction during tracking.
        self.params['tracking_step_size'] = 0.5

        # save_tracking - (bool) a boolean to determine whether to save the tracking or not.
        self.params['save_tracking'] = True

        # trk_file_saving_path - (string) path to save the tractography
        self.params['trk_file_saving_path'] = ''
    
