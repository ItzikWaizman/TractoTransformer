import torch
import os
import nibabel as nib
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import *
from utils.tracker_utils import *
from utils.data.data_utils import *
from models.network import TractoTransformer
import time
from tqdm import tqdm
from models.network import *

class Tracker(object):
    def __init__(self, logger, params):
        super(Tracker, self).__init__()
        self.params = params
        self.model = self.load_trained_model(logger).to(self.params.device)
        self.model.eval()
        self.test_subject_data_paths = extract_subject_paths(self.params.test_subject)
        self.dwi, self.wm_mask, self.fa_map, self.affine, self.inverse_affine = self.load_test_subject_data()
        if params.normalize_brain:
            self.dwi = normalize_brain(self.dwi)
        self.reference_tractogram, self.streamlines_lengths, self.tractogram_header = (None, None, None) if params.track_mode == 'inference' else self.load_ref_tract()
        self.directions = self.reference_tractogram[:, 1:, :] - self.reference_tractogram[:, :-1, :]

        # DEBUG CODE FOR ISMRM
        self.tractogram_affine = torch.tensor(self.tractogram_header['voxel_to_rasmm'], dtype=torch.float32)
        self.tractogram_inverse_affine = torch.inverse(self.tractogram_affine)

    def load_trained_model(self, logger):
        model_data = torch.load(self.params.trained_model_path)
        network = TractoTransformer(logger, self.params, model_data['max_streamline_length'])
        network.load_state_dict(model_data['model_state_dict'])
    
        return network

    def load_test_subject_data(self):
        dwi = nib.load(self.test_subject_data_paths['dwi_data'])
        dwi_data = torch.tensor(dwi.get_fdata(), dtype=torch.float32, device=self.params.device)
        dwi_data = dwi_data.unsqueeze(0)

        fa_map = nib.load(self.test_subject_data_paths['fa'])
        fa_map_data = torch.tensor(fa_map.get_fdata(), dtype=torch.float32)

        affine = torch.tensor(dwi.affine, dtype=torch.float32)
        inverse_affine = torch.inverse(affine)

        mask = nib.load(self.test_subject_data_paths['wm_mask'])
        mask_data = torch.tensor(mask.get_fdata(), dtype=torch.bool)

        if self.params.mask_dilation:
            mask_data = mask_dilation(mask.get_fdata())

        return dwi_data, mask_data, fa_map_data, affine, inverse_affine

    def load_ref_tract(self):
        streamlines_padded_len = self.model.causality_mask.shape[0]
        tractography_folder = self.test_subject_data_paths['tractography_folder']
        return get_streamline_tensor(tractography_folder, streamlines_padded_len)
        #return get_streamline_tensor(tractography_folder, self.params.max_track_len) #streamlines_padded_len)

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

        padding_mask = torch.ones(batch_size, max_sequence_length, dtype=torch.bool, device=self.params.device) # True where the values are padded.
        padding_mask[:, 0] = False # The first step are the seed points and these points are not zero padded.
        terminated_streamlines = torch.zeros(batch_size, dtype=torch.bool) # A boolean mask to indicate which streamlines have been terminated.

        step = 0
        with torch.no_grad():
            while step < max_sequence_length:
                # Get the fodfs from the model
                voxel_streamlines = ras_to_voxel(streamlines, self.inverse_affine).to(self.params.device)
                indices = torch.zeros(self.params.track_batch_size, dtype=torch.int32, device=self.params.device)
                log_fodfs = self.model(self.dwi, voxel_streamlines, padding_mask, indices)
                fodfs = torch.exp(log_fodfs)

                # Calculate the next positions and the terminated streamlines of the current iteration from fodf.
                next_positions, terminated_in_curr_iter = get_next_step_from_fodf(fodfs,
                                                                                streamlines,
                                                                                step, 
                                                                                sphere, 
                                                                                self)
                # Update terminated_streamlines, padding_mask and streamlines
                terminated_streamlines |= terminated_in_curr_iter
                padding_mask[:, step+1] &= terminated_streamlines.to(self.params.device) # Clear the masking from streamlines that were not calssified as EoF.
                streamlines[~terminated_streamlines, step+1, :] = next_positions[~terminated_streamlines, :]
                
                # Increase step size
                step = step +1 

                if torch.all(terminated_streamlines):
                    break

        lengths = (~padding_mask).sum(dim=1)
        return streamlines, lengths

    def track(self):
        seed_points = init_seeds(self.params, self.wm_mask, self.affine, self.reference_tractogram, self.streamlines_lengths)
        num_streamlines = seed_points.size(0)
        all_streamlines = []
        sphere = get_sphere('repulsion724')

        batch_num = 0
        for start_idx in tqdm(range(0, num_streamlines, self.params.track_batch_size), desc="Tracking Streamlines"):
            end_idx = min(start_idx + self.params.track_batch_size, num_streamlines)
            seed_batch = seed_points[start_idx:end_idx]
            start = time.time()
            batch_streamlines, batch_lengths = self.streamlines_tracking(seed_batch, sphere)
            end = time.time()
            print(f"TractTransformer: batch_time = {end-start}, batch={batch_num}")

            streamlines_list = create_streamlines_from_tensor(voxel_to_ras(batch_streamlines,self.tractogram_inverse_affine), batch_lengths) #create_streamlines_from_tensor(voxel_to_ras(batch_streamlines,self.inverse_affine), batch_lengths)
            all_streamlines.extend(streamlines_list)
            batch_num += 1

        filtered_streamlines = filter_short_streamlines(all_streamlines, self.params.min_streamline_len)
        tractogram = Tractogram(streamlines=filtered_streamlines, affine_to_rasmm=self.tractogram_affine)
        header = self.tractogram_header

        trk_file = nib.streamlines.TrkFile(tractogram, header=header)
        if self.params.save_tracking:
            nib.streamlines.save(trk_file, self.params.trk_file_saving_path)
