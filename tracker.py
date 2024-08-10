import torch
import torch.nn as nn
import nibabel as nib
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import *
from data_handling import *
from models.network import TractoTransformer
from utils.tracker_utils import *
import time

class Tracker(nn.Module):
    def __init__(self, logger, params):
        super(Tracker, self).__init__()
        self.data_handler = SubjectDataHandler(logger=logger, params=params, mode=TRACK, device=params['device'])
        self.data_handler.dwi = self.data_handler.dwi.to(params['device'])
        self.causality_mask = self.data_handler.causality_mask.to(params['device'])
        self.affine, self.inverse_affine = self.data_handler.affine, self.data_handler.inverse_affine
        self.wm_mask =self.data_handler.wm_mask
        self.fa_map = self.data_handler.fa_map
        self.num_seeds = params['num_seeds']
        self.track_batch_size = params['track_batch_size']
        self.angular_threshold = params['angular_threshold']
        self.fa_threshold = params['fa_threshold']
        self.max_sequence_length = params['max_sequence_length']
        self.min_streamline_length = params['min_streamline_length']
        self.step_size = params['tracking_step_size']
        self.save_tracking = params['save_tracking']
        self.trk_file_path = params['trk_file_saving_path']
        self.device = params['device']
        self.model = TractoTransformer(logger=logger, params=params)
        self.tracking_mode = params['tracking_mode']
        self.load_model(params['trained_model_path'])

    def load_model(self, checkpoint_path=''):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

    def streamlines_tracking(self, seed_points, sphere):
        """
        Parameters: 
        - seed_points: Tensor of shape [batch_size, seq_length, 3] initialized to zeros except for seed_points[:, 0, :].
        - sphere: sphere points that models the fodf classes.

        Returns: 
        - Tensor of shape [batch_size, max_sequence_length, 3]  
        """
        streamlines = seed_points.clone()
        batch_size, max_sequence_length = streamlines.size(0), streamlines.size(1)

        padding_mask = torch.ones(batch_size, max_sequence_length, dtype=torch.bool, device=self.device) # True where the values are padded.
        padding_mask[:, 0] = False # The first step are the seed points and these points are not zero padded.
        terminated_streamlines = torch.zeros(batch_size, dtype=torch.bool) # A boolean mask to indicate which streamlines have been terminated.

        step = 0
        while step < max_sequence_length:
            # Get the fodfs from the model
            voxel_streamlines = ras_to_voxel(streamlines, self.inverse_affine).to(self.device)
            log_fodfs = self.model(self.data_handler.dwi, voxel_streamlines, padding_mask, self.causality_mask)
            fodfs = torch.exp(log_fodfs)

            # Calculate the next positions and the terminated streamlines of the current iteration from fodf.
            next_positions, terminated_in_curr_iter = get_next_step_from_fodf(fodfs, 
                                                                              streamlines,
                                                                              step, 
                                                                              sphere, 
                                                                              self.step_size,
                                                                              self)
            # Update terminated_streamlines, padding_mask and streamlines
            terminated_streamlines |= terminated_in_curr_iter
            padding_mask[:, step+1] &= terminated_streamlines.to(self.device) # Clear the masking from streamlines that were not calssified as EoF.
            streamlines[~terminated_streamlines, step+1, :] = next_positions[~terminated_streamlines, :]
            
            # Increase step size
            step = step +1 

            if torch.all(terminated_streamlines):
                break

        lengths = (~padding_mask).sum(dim=1)
        return streamlines, lengths
    
    def track(self):
        seed_points = init_seeds(self.wm_mask, self.num_seeds, self.affine, self.max_sequence_length)
        self.num_seeds = seed_points.size(0)
        all_streamlines = []
        sphere = get_sphere('repulsion724')

        batch_num = 0
        for start_idx in range(0, self.num_seeds, self.track_batch_size):
            end_idx = min(start_idx + self.track_batch_size, self.num_seeds)
            seed_batch = seed_points[start_idx:end_idx]
            batch_streamlines, batch_lengths = self.streamlines_tracking(seed_batch, sphere)
            streamlines_list = create_streamlines_from_tensor(ras_to_voxel(batch_streamlines, self.inverse_affine), batch_lengths)
            all_streamlines.extend(streamlines_list)
            batch_num += 1

        filtered_streamlines = filter_short_streamlines(all_streamlines, self.min_streamline_length)
        tractogram = Tractogram(streamlines=filtered_streamlines, affine_to_rasmm=self.affine)

        header = self.data_handler.tractogram_header

        trk_file = nib.streamlines.TrkFile(tractogram, header=header)
        if self.save_tracking:
            nib.streamlines.save(trk_file, self.trk_file_path)
        
    