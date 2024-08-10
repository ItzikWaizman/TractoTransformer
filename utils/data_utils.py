import os
import glob
import torch
import numpy as np
from dipy.reconst.shm import sph_harm_lookup
from nibabel import streamlines
import torch.nn.functional as F


def extract_subject_paths(subject_folder):
    # Check if the subject folder exists
    if not os.path.exists(subject_folder):
        print(f"Error: Subject folder not found - {subject_folder}")
        return None

    # Extract bvals, bvecs, and dwi_data paths
    dwi_folder = os.path.join(subject_folder, "dwi")
    dwi_data_path = glob.glob(os.path.join(dwi_folder, "*.nii*"))[0]
    bval_path = glob.glob(os.path.join(dwi_folder, "*.bval*"))[0]
    bvec_path = glob.glob(os.path.join(dwi_folder, "*.bvec*"))[0]

    # Extract white matter mask path
    mask_folder = os.path.join(subject_folder, "mask")
    wm_mask_path = glob.glob(os.path.join(mask_folder, "*mask_wm*"))[0]

    # Extract fractional anisotropy
    fa_folder = os.path.join(subject_folder, "dti")
    fa_path = glob.glob(os.path.join(fa_folder, "*fa*"))[0]

    # Extract tractography folder path
    tractography_folder = os.path.join(subject_folder, "tractography")

    # Return the extracted paths
    return {
        "dwi_data": dwi_data_path,
        "bvals": bval_path,
        "bvecs": bvec_path,
        "wm_mask": wm_mask_path, 
        "tractography_folder": tractography_folder,
        "fa" : fa_path
    }


def sample_signal_from_sh(data_sh, sh_order, sphere):
    sph_harm_basis = sph_harm_lookup.get("tournier07")
    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    Ba = torch.tensor(Ba, dtype=torch.float32)

    X, Y, Z, C = data_sh.shape
    data_sh_reshaped = data_sh.reshape(-1, C)

    data_sampled = torch.matmul(data_sh_reshaped, Ba.T)
    data_sampled = data_sampled.reshape(X,Y,Z, -1)
    return data_sampled

def load_tractogram(tractography_folder, augment_streamlines=False):
    folder_path = tractography_folder

    # Get a list of all .trk files in the specified folder
    trk_files = [file for file in os.listdir(folder_path) if file.endswith(".trk")]
    tractogram_header = None

    merged_streamlines = []
    # Iterate over the .trk files and merge them
    for trk_file in trk_files:
        current_tractogram = streamlines.load(os.path.join(folder_path, trk_file))
        if tractogram_header is None:
            tractogram_header = current_tractogram.header
        merged_streamlines.extend(current_tractogram.streamlines)

    if augment_streamlines:
        for streamline in merged_streamlines.copy():
            reversed_streamline = streamline[::-1].copy()  # Reverse the streamline
            merged_streamlines.append(reversed_streamline)  # Append the reversed streamline

    return merged_streamlines, tractogram_header


def prepare_streamlines_for_training(subject, save_dir_name="torch_streamlines", save_filename="streamlines.pt"):
    """
    Prepares and saves streamlines for training, or loads them if already saved.

    Parameters:
    - subject - SubjectDataHandler object.
    - save_dir: Directory where the tensor and lengths will be saved.
    - save_filename: Filename for saving the tensor and lengths.

    Returns:
    - padded_streamlines: torch tensor of padded streamlines
    """

    save_dir = os.path.join(subject.paths_dictionary['tractography_folder'], save_dir_name)
    save_path = os.path.join(save_dir, save_filename)

    if os.path.exists(save_path):
        data = torch.load(save_path)
        return data['padded_streamlines_tensor'], data['lengths'], data['tractogram_header']

    # If the data does not exist, proceed to load and process the streamlines
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare streamlines and lengths
    np_streamlines, tractogram_header = load_tractogram(subject.paths_dictionary['tractography_folder'])
    torch_streamlines = []
    for streamline in np_streamlines:
        torch_streamline = torch.tensor(streamline, dtype=torch.float32)
        torch_streamlines.append(torch_streamline)

    max_seq_len = max(len(sl) for sl in torch_streamlines)
    padded_streamlines = torch.zeros(len(torch_streamlines), max_seq_len + 1, 3)

    for i, streamline in enumerate(torch_streamlines):
        length = len(streamline)
        padded_streamlines[i, :length, :] = streamline

    lengths = torch.tensor([len(sl) for sl in torch_streamlines], dtype=torch.int)

    # Save padded_streamlines and lengths
    torch.save({'padded_streamlines_tensor': padded_streamlines, 'lengths': lengths, 'tractogram_header': tractogram_header}, save_path)
    
    return padded_streamlines, lengths, tractogram_header


def generate_labels(streamline, actual_length, sphere_points, EoF, sigma=0.1):
    # Truncate the streamline to the actual length + 1 (The streamlines are padded enough to ensure it is safe)
    padded_length = streamline.size(0)
    streamline = streamline[:actual_length+1]
    
    # Calculate direction unit vectors between consecutive points
    directions = streamline[1:] - streamline[:-1]
    directions = directions / directions.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity for all direction vectors with all sphere points
    # directions: (actual_length, 3), sphere_points: (725, 3)
    # cosine_similarity: (actual_length, 725)
    cosine_similarity = torch.matmul(directions, sphere_points.t())
    
    # Convert cosine similarity to distance on the unit sphere
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    distances = torch.acos(cosine_similarity)

    # Apply gaussian kernel
    gaussian_weights = torch.exp(-distances**2 / (2 * sigma**2))
    gaussian_weights[:actual_length-1, 724] = 0 # Set the probability of EoF to zero for any point on the streamline except the last one.

    # Generate FODFs
    fodfs = gaussian_weights / gaussian_weights.sum(dim=1, keepdim=True)
    fodfs[actual_length-1, :] = EoF # Set the fodf of the last point to be the fodf of EoF

    fodfs_padded = torch.zeros((padded_length, 725), device=streamline.device)
    fodfs_padded[:actual_length, :] = fodfs

    return fodfs_padded

