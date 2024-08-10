import torch
import scipy.ndimage as ndimage
import numpy as np
from nibabel.streamlines import Tractogram, TrkFile
from utils.common_utils import *

def get_next_step_from_fodf(fodf, streamlines, step, sphere, step_size, tracker):
    """
    Computes the next step in the direction of the most probable directions from a FODF tensor.
    
    Parameters:
    - fodf: Tensor of shape [batch_size, 724] representing the FODF.
    - current_positions: Tensor of shape [batch_size, 3] indicating the points where the fodfs were calculated.
    - sphere: Sphere object containing vertices (directions).
    - step_size: Scalar representing the step size to scale the direction vectors.

    Returns:
    - steps: PyTorch tensor of shape [batch_size, 3] representing the next steps.
    """
    # Get current positions
    current_positions = streamlines[:, step, :]

    # Get the indices of the most probable directions from the FODF
    if tracker.tracking_mode == 'deterministic':
        most_probable_directions = torch.argmax(fodf[:, step, :], dim=1).cpu()
    else:
        most_probable_directions = sample_from_fodf(fodf[:, 0, :]).cpu()

    # Model's decision for streamlines termination
    model_termination_decisions = (most_probable_directions == 724)

    # Clamp indices to avoid out-of-bounds indexing in sphere points
    clamped_directions = torch.clamp(most_probable_directions, max=723).cpu().numpy()

    # Retrieve the corresponding direction vectors from the sphere
    direction_vectors = sphere.vertices[clamped_directions]

    # Scale the direction vectors by the step size to compute the steps
    steps = direction_vectors * step_size
    steps = torch.tensor(steps, dtype=current_positions.dtype, device=current_positions.device)

    # Add the steps to the current positions to get the next positions
    next_positions = current_positions + steps

    # Check if any of the manual stopping criteria are met
    terminated_streamlines = calc_stopping_mask(tracker, streamlines, next_positions, step+1) | model_termination_decisions

    return next_positions, terminated_streamlines

def init_seeds(wm_mask, num_of_seeds, affine, max_seq_length):
    seeds = torch.zeros(num_of_seeds, max_seq_length, 3)
    white_matter_indices = torch.nonzero(wm_mask == 1)
    sampled_indices = white_matter_indices[torch.randint(len(white_matter_indices), (num_of_seeds,))]
    ras_coords = voxel_to_ras(sampled_indices, affine)
    seeds[:, 0, :] = ras_coords
    return seeds

def mask_dilation(wm_mask):
    SE = ndimage.generate_binary_structure(3, 1)
    out_mask = torch.from_numpy(ndimage.binary_dilation(wm_mask, structure=SE))
    return out_mask

def create_streamlines_from_tensor(streamlines_tensor, lengths):
    """
    Creates a list of streamlines from a PyTorch tensor.

    Parameters:
    - streamlines_tensor: PyTorch tensor of shape [batch_size, max_seq_length, 3].
    - lengths: List or tensor of actual lengths of each streamline in the batch.

    Returns:
    - streamlines_list: List of NumPy arrays, each representing a streamline.
    """
    # Ensure lengths is a NumPy array
    lengths = lengths.cpu().numpy() if torch.is_tensor(lengths) else np.array(lengths)
    
    # Convert the PyTorch tensor to a list of NumPy arrays
    streamlines_list = []
    for i, length in enumerate(lengths):
        streamline = streamlines_tensor[i, :length, :].cpu().numpy()
        streamlines_list.append(streamline)
    
    return streamlines_list


def calc_stopping_mask(tracker, streamlines, current_points, step):
    """
    Check if any stopping criteria are met.

    Parameters:
    tracker: Tracker object.
    streamlines: [batch_size, max_sequence_length, 3] batch of streamlines tracked up to value specified in step parameter.
    step: Current step in the tracking. for j > step, streamlines[:, j, :] is not valid yet.

    Returns:
    [batch_size]: Boolean mask to indicate which steramlines should be terminated.
    """
    # Get 3 last steps in the batch of streamlines.
    current_points = current_points
    previous_points = streamlines[:, step-1, :] if step > 0 else None
    preprevious_points = streamlines[:, step-2, :] if step > 1 else None
    
    # Convert rasmm coords to voxel coords
    current_voxels = ras_to_voxel(current_points, tracker.inverse_affine)
    
    # Initialize a tensor for the stopping criteria results
    batch_size = current_points.size(0)
    stop_criteria_met = torch.zeros(batch_size, dtype=torch.bool, device=current_points.device)
    
    # Check each stopping criterion
    if step >= tracker.max_sequence_length-1:
        stop_criteria_met |= torch.ones(batch_size, dtype=torch.bool, device=current_points.device)
    
    stop_criteria_met |= outside_image_bounds(current_voxels, tracker.wm_mask.shape)
    stop_criteria_met |= outside_white_matter_mask(current_voxels, tracker.wm_mask)
    stop_criteria_met |= angular_threshold_exceeded(current_points, previous_points, preprevious_points, tracker.angular_threshold)
    stop_criteria_met |= fa_threshold_exceeded(current_voxels, tracker.fa_map, tracker.fa_threshold)
    stop_criteria_met |= entropy_threshold_exceeded(current_points)
    
    return stop_criteria_met

def angular_threshold_exceeded(current_points, previous_points, preprevious_points, angular_threshold):
    """
    Check if the angular deviation between the current and previous directions exceeds the threshold.

    Parameters:
    current_points: Batch of current points coordinates [batch_size, 3] in RASmm coords.
    previous_points: Batch of previous points coordinates [batch_size, 3] in RASmm coords.
    preprevious_points: Batch of preprevious points coordinates [batch_size, 3] in RASmm coords. 
    angular_threshold (float): Angular threshold value in degrees.

    Returns:
    [batch_size]: Boolean mask. True if angular threshold is exceeded, False otherwise.
    """
    if previous_points is None or preprevious_points is None: 
         return torch.zeros(current_points.size(0), dtype=torch.bool, device=current_points.device)
    
    # Get directions between 3 last steps
    current_directions = current_points - previous_points
    previous_directions = previous_points - preprevious_points

    # Calculate the norms of the corresponding directions
    current_directions_norms = torch.norm(current_directions, dim=1)
    previous_directions_norms = torch.norm(previous_directions, dim=1)

    # Calculate cosine angles
    cosine_angles = torch.sum(current_directions*previous_directions, dim=1) / (current_directions_norms*previous_directions_norms)
    cosine_angles = torch.clamp(cosine_angles, -1.0, 1.0)

    # Calculate corresponding angles
    angles = torch.acos(cosine_angles) * 180.0 / torch.pi

    return (angles > angular_threshold)

def fa_threshold_exceeded(current_voxels, fa_map, fa_threshold):
    """
    Check if the FA values at the current voxels are below the threshold.

    Parameters:
    current_voxels Tensor [batch_size, 3]: Current batch voxels coordinates.
    fa_map Tensor: FA map.
    fa_threshold (float): FA threshold value.

    Returns:
    [batch_size]: Bool mask. True if FA threshold is exceeded, False otherwise.
    """
    if fa_map is None:
        return torch.zeros(current_voxels.size(0), dtype=torch.bool, device=current_voxels.device)
    
    x_max, y_max, z_max = fa_map.shape
    
    # Clamp the coordinates to be within valid range
    x_coords = current_voxels[:, 0].clamp(0, x_max - 1).int()
    y_coords = current_voxels[:, 1].clamp(0, y_max - 1).int()
    z_coords = current_voxels[:, 2].clamp(0, z_max - 1).int()
    
    return (fa_map[x_coords, y_coords, z_coords] < fa_threshold)

def outside_white_matter_mask(current_voxels, wm_mask):
    """
    Check if the current voxel is outside the white matter mask.

    Parameters:
    current_voxels Tensor [batch_size, 3]: Current batch voxels coordinates.
    wm_mask Tensor: White matter mask.

    Returns:
    [batch_size]: Boolean mask. True if outside the white matter mask, False otherwise.
    """
    x_max, y_max, z_max = wm_mask.shape
    
    # Clamp the coordinates to be within valid range
    x_coords = current_voxels[:, 0].clamp(0, x_max - 1).int()
    y_coords = current_voxels[:, 1].clamp(0, y_max - 1).int()
    z_coords = current_voxels[:, 2].clamp(0, z_max - 1).int()

    return (wm_mask[x_coords, y_coords, z_coords] < 1)

def outside_image_bounds(current_voxels, image_shape):
    """
    Check if the current voxel is outside the image bounds.

    Parameters:
    current_voxels [batch_size, 3]: Current voxels coordinates.
    image_shape: Shape of the DWI image (x, y, z).

    Returns:
    [batch_size]: Boolean mask. True if outside the image bounds, False otherwise.
    """
    x_max, y_max, z_max = image_shape
    x_coords, y_coords, z_coords = current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]
    mask_x = (x_coords < 0) | (x_coords > x_max)
    mask_y = (y_coords < 0) | (y_coords > y_max)
    mask_z = (z_coords < 0) | (z_coords > z_max)
    
    return (mask_x | mask_y | mask_z)

def entropy_threshold_exceeded(current_points):
    return torch.zeros(current_points.size(0), dtype=torch.bool, device=current_points.device)

def filter_short_streamlines(streamlines, min_length):
    """
    Filters out streamlines that are shorter than the specified minimum length.

    Parameters:
    - streamlines (list of np.ndarray): List of streamlines, each represented as a numpy array of shape (N, 3), where N is the number of points in the streamline.
    - min_length (int): Minimum number of points a streamline must have to be retained.

    Returns:
    - list of np.ndarray: Filtered list of streamlines.
    """
    filtered_streamlines = [s for s in streamlines if len(s) >= min_length]
    return filtered_streamlines

def sample_from_fodf(fodf):
    """
    INPUT: pdf of a single time step - tensor of size N x #out_directions (N is the batch size)
           n - how many samples to draw from each pdf

    OUTPUT: selected_dirs - the indices of directions sampled from the pdfs, a vector of size N
    """

    # Sample 1 direction from each FODF in the batch
    selected_dirs = torch.multinomial(fodf, 1, replacement=True)

    # Return as a 1D tensor of indices
    return selected_dirs.squeeze(1)