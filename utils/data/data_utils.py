import os
import glob
import torch
from nibabel import streamlines
from dipy.data import get_sphere

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

    # Extract fodf
    fodf_folder = os.path.join(subject_folder, "fodf")
    fodf_path = glob.glob(os.path.join(fodf_folder, "*fodf*"))[0]

    # Extract fractional anisotropy
    fa_folder = os.path.join(subject_folder, "dti")
    fa_path = glob.glob(os.path.join(fa_folder, "*fa*"))[0]

    # Extract tractography folder path
    tractography_folder = os.path.join(subject_folder, "tractography")
    tractography_resampled = tractography_folder + "_resampled"

    if os.path.exists(tractography_resampled):
        tractography_folder = tractography_resampled

    # Extract spherical harmonics path
    sh_folder = os.path.join(subject_folder, "sh")
    sh_path = glob.glob(os.path.join(sh_folder, "*sh.nii*"))[0]

    # Return the extracted paths
    return {
        "dwi_data": dwi_data_path,
        "bvals": bval_path,
        "bvecs": bvec_path,
        "wm_mask": wm_mask_path,
        "tractography_folder": tractography_folder,
        "sh": sh_path,
        "fodf": fodf_path,
        "fa": fa_path
    }


def load_tractogram(tractography_folder):
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
    np_streamlines, _ = load_tractogram(tractography_folder)

    padded_streamlines = torch.zeros(len(np_streamlines), padded_length, 3, dtype=torch.float32)
    streamline_lengths = []

    for i, np_streamline in enumerate(np_streamlines):
        length = len(np_streamline)
        streamline_lengths.append(length)
        padded_streamlines[i, :length, :] = torch.tensor(np_streamline, dtype=torch.float32)

    streamline_lengths = torch.tensor(streamline_lengths, dtype=torch.int)

    return padded_streamlines, streamline_lengths


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
