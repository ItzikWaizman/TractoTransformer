import os
import glob
import torch
from nibabel import streamlines
from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from dipy.core.sphere import Sphere
import numpy as np

EoF = 724

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
    fa_path = None
    if os.path.exists(fa_folder):
        fa_path = glob.glob(os.path.join(fa_folder, "*fa*"))[0]

    # Extract tractography folder path
    tractography_folder = os.path.join(subject_folder, "tractography")
    tractography_resampled = tractography_folder + "_resampled"

    if os.path.exists(tractography_resampled):
        tractography_folder = tractography_resampled

    # Return the extracted paths
    return {
        "dwi_data": dwi_data_path,
        "bvals": bval_path,
        "bvecs": bvec_path,
        "wm_mask": wm_mask_path,
        "tractography_folder": tractography_folder,
        "fa": fa_path
    }

def load_tractogram(tractography_folder, reverse_streamlines=True):
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

    if reverse_streamlines:
        for streamline in merged_streamlines.copy():
            reversed_streamline = streamline[::-1].copy()  # Reverse the streamline
            merged_streamlines.append(reversed_streamline)  # Append the reversed streamline

    return merged_streamlines, tractogram_header

def get_streamline_tensor(tractography_folder, padded_length):
    """
    Prepares streamlines for training - converts them to a torch tensor of padded streamlines.

    Parameters:
    - paths_dictionary - subject paths dictionary.
    - padded_length - length of the padded streamlines. Should be max_streamline_length + 1 over the entire dataset

    Returns:
    - padded_streamlines: torch tensor of padded streamlines
    - streamline_lengths: original lengths of the streamlines
    - tractography_header: tractography header
    """

    # Prepare streamlines
    np_streamlines, tractogram_header = load_tractogram(tractography_folder)

    padded_streamlines = torch.zeros(len(np_streamlines), padded_length, 3, dtype=torch.float32)
    streamline_lengths = []

    for i, np_streamline in enumerate(np_streamlines):
        length = len(np_streamline)
        streamline_lengths.append(length)
        padded_streamlines[i, :length, :] = torch.tensor(np_streamline, dtype=torch.float32)

    streamline_lengths = torch.tensor(streamline_lengths, dtype=torch.int)

    return padded_streamlines, streamline_lengths, tractogram_header

def load_tractograms(tractography_folder, reverse_streamlines=True):
    folder_path = tractography_folder

    # Get a list of all .trk files in the specified folder
    trk_files = [file for file in os.listdir(folder_path) if file.endswith(".trk")]
    tractogram_header = None

    bundles = []
    bundle_names = []

    for trk_file in trk_files:
        file_path = os.path.join(folder_path, trk_file)
        current_tractogram = streamlines.load(file_path)

        if tractogram_header is None:
            tractogram_header = current_tractogram.header

        bundle_name = trk_file.split("__")[1].replace(".trk", "")
        bundle_streamlines = []
        bundle_streamlines.extend(current_tractogram.streamlines)

        if reverse_streamlines:
            for streamline in bundle_streamlines.copy():
                reversed_streamline = streamline[::-1].copy()  # Reverse the streamline
                bundle_streamlines.append(reversed_streamline)  # Append the reversed streamline

        bundles.append(bundle_streamlines)
        bundle_names.append(bundle_name)

    return bundles, bundle_names, tractogram_header

def get_streamline_tensors(tractography_folder, padded_length):
    # Prepare streamlines
    np_bundles, bundle_names, tractogram_header = load_tractograms(tractography_folder)

    padded_streamline_bundles = []
    bundles_sreamline_lengths = []

    for np_streamlines in np_bundles:
        padded_streamlines = torch.zeros(len(np_streamlines), padded_length, 3, dtype=torch.float32)
        streamline_lengths = []

        for i, np_streamline in enumerate(np_streamlines):
            length = len(np_streamline)
            streamline_lengths.append(length)
            padded_streamlines[i, :length, :] = torch.tensor(np_streamline, dtype=torch.float32)

        streamline_lengths = torch.tensor(streamline_lengths, dtype=torch.int)
        padded_streamline_bundles.append(padded_streamlines)
        bundles_sreamline_lengths.append(streamline_lengths)

    return padded_streamline_bundles, bundles_sreamline_lengths, bundle_names, tractogram_header

def get_streamline_labels(streamline, actual_size, sphere):
    """
    Assigns labels to each point in the streamline based on the closest sphere vector
    that represents the direction of the streamline at that point.

    Parameters:
    - streamline: torch tensor of shape [nun_streamlines, max_streamline_len, 3]
    - actual_size: actual sizes of the streamlines (excluding padding) [num_streamlines,]
    - sphere: sphere object

    Returns:
    - labels: torch tensor of shape [num_streamlines, max_streamline_len]
    """

    # Initialize the labels tensor with EoF value for all points (default for padded points)
    labels = torch.full((streamline.shape[0],), EoF, dtype=torch.int64)

    # Get the direction vectors between consecutive points (vectorized)
    directions = streamline[1:actual_size] - streamline[:actual_size-1]

    # Normalize the direction vectors (vectorized)
    norms = directions.norm(dim=1, keepdim=True)
    directions_unit = directions / norms

    # Find the closest unit vectors for all the direction vectors (vectorized)
    sphere_vectors = torch.tensor(sphere.vertices, dtype=torch.float32)  # Shape: [num_sphere_vectors, 3]

    # Compute cosine similarity between all directions and sphere vectors
    cos_sim = torch.matmul(directions_unit, sphere_vectors.T)  # Shape: [actual_size-1, num_sphere_vectors]

    # Find the index of the maximum similarity for each direction vector
    closest_indices = torch.argmax(cos_sim, dim=1)  # Shape: [actual_size-1,]

    # Assign the closest indices to the labels for valid (non-padded) points
    labels[:actual_size-1] = closest_indices

    return labels

def build_soft_labels_tensor(sigma=0.1):
    """
    Constructs a tensor that maps labels (shepre vector index or EoF index) to corresponding 
    Gaussian weighted soft labels (log probability vectors over the sphere vectors or EoF).

    Parameters:
    - sigma - standard deviation of the gaussian weights
    
    Returns:
    - soft_labels_tensor: tensor of soft labels [num_sphere_vectors+1, num_sphere_vectors+1]
    """

    sphere = get_sphere('repulsion724')

    # Get the sphere vectors
    sphere_vectors = torch.tensor(sphere.vertices, dtype=torch.float32)
    num_sphere_vectors = sphere_vectors.shape[0]

    soft_labels_tensor = torch.zeros(num_sphere_vectors+1, num_sphere_vectors+1)

    # Compute cosine similarity between all pairs of sphere vectors
    cosine_similarity = torch.matmul(sphere_vectors, sphere_vectors.T)
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)

    # Convert cosine similarity to distance on the unit sphere
    distances = torch.acos(cosine_similarity)

    # Compute gaussian weighted soft labels
    gaussian_weights = torch.exp(-distances**2 / (2 * sigma**2))
    soft_labels_tensor[:num_sphere_vectors, :num_sphere_vectors] = gaussian_weights / gaussian_weights.sum(dim=1, keepdim=True)

    # End of Fiber (EoF) soft label is a one-hot vector
    soft_labels_tensor[num_sphere_vectors, num_sphere_vectors] = 1.0

    return soft_labels_tensor

def normalize_brain(brain):
    epsilon = 1e-6
    b0 = brain[..., 0:1]

    # Normalize all gradient directions, including b0 itself
    brain /= (b0 + epsilon)

    return torch.clamp(brain, min=0, max=1)

def get_spherical_harmonics_coefficients(dwi_weights, bvecs, sh_order, smooth=0.006):
    # Extract diffusion weights and normalize by the b0.
    bvecs = bvecs[1:, :]
    weights = dwi_weights[..., 1:]

    # Assuming all directions are on the hemisphere.
    raw_sphere = Sphere(xyz=bvecs)

    # Fit SH to signal
    sph_harm_basis = sph_harm_lookup.get("tournier07")
    Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
    data_sh = np.dot(weights, invB.T)
    return data_sh

def resample_and_normalize_dwi(dwi, bvecs, sh_order=12, smooth=0, directions=None):
    if directions is None:
        directions = bvecs

    data_sh = get_spherical_harmonics_coefficients(dwi, bvecs, sh_order=sh_order, smooth=smooth)

    sphere = get_sphere('repulsion100')
    if directions is not None:
        sphere = Sphere(xyz=directions[1:])

    sph_harm_basis = sph_harm_lookup.get("tournier07")
    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = torch.tensor(np.dot(data_sh, Ba.T), dtype=torch.float32)

    tensor_shape = (dwi.shape[0], dwi.shape[1], dwi.shape[2], len(directions))
    resampled_dwi = torch.zeros(tensor_shape)
    resampled_dwi[..., 1:] = data_resampled
    resampled_dwi[..., 0] = dwi[..., 0]

    return normalize_brain(resampled_dwi)
