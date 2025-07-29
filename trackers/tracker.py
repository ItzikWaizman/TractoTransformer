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
        if self.reference_tractogram is not None:
            self.directions = self.reference_tractogram[:, 1:, :] - self.reference_tractogram[:, :-1, :]

        # DEBUG CODE FOR ISMRM
        if self.tractogram_header is not None:
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
        streamlines_padded_len = self.model.max_sequence_length  # Use the stored max_sequence_length
        tractography_folder = self.test_subject_data_paths['tractography_folder']
        return get_streamline_tensor(tractography_folder, streamlines_padded_len)

    def streamlines_tracking(self, seed_points, sphere):
        """
        Parameters: 
        - seed_points: Tensor of shape [batch_size, 3] - just the initial seed points.
        - sphere: sphere points that models the fodf classes.

        Returns: 
        - streamlines: Tensor of shape [batch_size, actual_length, 3]
        - lengths: Tensor of shape [batch_size] with actual streamline lengths
        """
        batch_size = seed_points.size(0)
        max_sequence_length = self.model.max_sequence_length
        
        # Initialize streamlines tensor to store all positions
        streamlines = torch.zeros(batch_size, max_sequence_length, 3, device=self.params.device)
        streamlines[:, 0, :] = seed_points  # Set seed points as first position
        
        # Track termination status
        terminated_streamlines = torch.zeros(batch_size, dtype=torch.bool, device=self.params.device)
        
        # Initialize variables for KV-cache
        past_kvs = None
        current_length = 1  # We start with seed points (length 1)
        
        with torch.no_grad():
            for step in range(max_sequence_length - 1):  # -1 because we start with seed points
                # Prepare current input - only the positions we've generated so far
                current_streamlines = streamlines[:, :current_length, :].clone()
                
                # Convert to voxel coordinates
                voxel_streamlines = ras_to_voxel(current_streamlines, self.inverse_affine).to(self.params.device)
                
                # Create brain indices (assuming single brain for all streamlines)
                indices = torch.zeros(batch_size, dtype=torch.int32, device=self.params.device)
                
                # Create padding mask for current input (no padding for valid positions)
                current_padding_mask = torch.zeros(batch_size, current_length, dtype=torch.bool, device=self.params.device)
                
                # Get model predictions
                if past_kvs is None:
                    # First forward pass - process all positions so far
                    log_fodfs, past_kvs = self.model(
                        self.dwi, 
                        voxel_streamlines, 
                        current_padding_mask, 
                        indices, 
                        past_kvs=None, 
                        use_cache=True
                    )
                    # Only take the prediction for the last position
                    log_fodfs = log_fodfs[:, -1:, :]
                else:
                    # Subsequent passes - only process the new position
                    new_position = voxel_streamlines[:, -1:, :]  # Only the latest position
                    new_padding_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.params.device)
                    
                    log_fodfs, past_kvs = self.model(
                        self.dwi,
                        new_position,
                        new_padding_mask,
                        indices,
                        past_kvs=past_kvs,
                        use_cache=True
                    )
                
                # Convert to probabilities
                fodfs = torch.exp(log_fodfs.squeeze(1))  # Remove sequence dimension since we only have 1 step
                
                # Calculate next positions for non-terminated streamlines
                next_positions, terminated_in_curr_iter = get_next_step_from_fodf(
                    fodfs.unsqueeze(1),  # Add sequence dimension back for compatibility
                    current_streamlines,
                    step, 
                    sphere, 
                    self
                )
                
                # Update termination status
                terminated_streamlines |= terminated_in_curr_iter
                
                # Update streamlines with next positions (only for non-terminated ones)
                active_mask = ~terminated_streamlines
                if active_mask.any():
                    streamlines[active_mask, current_length, :] = next_positions[active_mask, :]
                
                # Increment current length for active streamlines
                current_length += 1
                
                # For terminated streamlines, we need to update the KV cache to reflect that
                # they won't contribute new tokens, but we keep the cache for consistency
                
                # Break if all streamlines are terminated
                if torch.all(terminated_streamlines):
                    break
        
        # Calculate actual lengths for each streamline
        lengths = torch.zeros(batch_size, dtype=torch.long, device=self.params.device)
        for i in range(batch_size):
            # Find the first zero position (excluding the first position which is the seed)
            non_zero_mask = torch.any(streamlines[i, 1:, :] != 0, dim=1)
            if non_zero_mask.any():
                lengths[i] = non_zero_mask.sum() + 1  # +1 for the seed point
            else:
                lengths[i] = 1  # Only the seed point
        
        return streamlines, lengths

    def track(self):
        seed_points = init_seeds(self.params, self.wm_mask, self.affine, self.reference_tractogram, self.streamlines_lengths)
        
        # Extract just the seed points (first position) if seed_points has sequence dimension
        if len(seed_points.shape) == 3:
            seed_points = seed_points[:, 0, :]  # Take only the first position
        
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

            # Convert to RAS coordinates and create streamline list
            ras_streamlines = voxel_to_ras(batch_streamlines, self.tractogram_inverse_affine if hasattr(self, 'tractogram_inverse_affine') else self.inverse_affine)
            streamlines_list = create_streamlines_from_tensor(ras_streamlines, batch_lengths)
            all_streamlines.extend(streamlines_list)
            batch_num += 1

        # Filter short streamlines and create tractogram
        filtered_streamlines = filter_short_streamlines(all_streamlines, self.params.min_streamline_len)
        
        affine_to_use = self.tractogram_affine if hasattr(self, 'tractogram_affine') else self.affine
        tractogram = Tractogram(streamlines=filtered_streamlines, affine_to_rasmm=affine_to_use)
        
        header = self.tractogram_header if hasattr(self, 'tractogram_header') and self.tractogram_header is not None else {}
        
        trk_file = nib.streamlines.TrkFile(tractogram, header=header)
        if self.params.save_tracking:
            nib.streamlines.save(trk_file, self.params.trk_file_saving_path)
        
        return filtered_streamlines