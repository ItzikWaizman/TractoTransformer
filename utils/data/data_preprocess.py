import os
import io
import subprocess
import nibabel as nib
import lmdb
import torch
from config.args import parse_args
from dipy.data import get_sphere
from utils.data.data_utils import *
from utils.common_utils import ras_to_voxel
from tqdm import tqdm

def resample_streamlines(subjects_directory, step_size_mm):
    # Iterate through all subject directories in the given directory
    for subject_name in tqdm(os.listdir(subjects_directory)):
        subject_path = os.path.join(subjects_directory, subject_name)
        tractography_folder = os.path.join(subject_path, "tractography")
        resampled_tractography_folder = tractography_folder + "_resampled"

        # Skip if resampled folder already exists
        if os.path.exists(resampled_tractography_folder):
            continue

        os.makedirs(resampled_tractography_folder)
        trk_files = [file for file in os.listdir(tractography_folder) if file.endswith(".trk")]

        for trk_file in trk_files:
            tractogram_file_path = os.path.join(tractography_folder, trk_file)
            new_tractogram_file_path = os.path.join(resampled_tractography_folder, trk_file)

            # Run the streamline resampling command with the specified step size
            subprocess.run([
                "utils/streamline-resample-venv/bin/python",
                "utils/data/scil_resample_streamlines.py",
                tractogram_file_path,
                new_tractogram_file_path,
                "--step_size",
                str(step_size_mm),
            ])


def determine_max_streamline_length(subjects_directory):
    max_streamline_length = 0

    for subject_name in os.listdir(subjects_directory):
        subject_path = os.path.join(subjects_directory, subject_name)
        tractography_folder = os.path.join(subject_path, "tractography") + "_resampled"
        streamlines, _ = load_tractogram(tractography_folder)
        max_len = max(len(sl) for sl in streamlines)
        max_streamline_length = max(max_streamline_length, max_len)

    return max_streamline_length


def prepare_subject_data(streamline_tensor, streamline_lengths, inverse_affine, subject_name, save_path, subject_index):
    # Get sphere from dipy.Sphere
    sphere = get_sphere('repulsion724')

    # Open the LMDB environment without specifying map_size
    lmdb_file = os.path.join(save_path, f"idx_{subject_index}_subject_{subject_name}.lmdb")
    map_size = 10 * 1024 * 1024 * 1024 # 10 GB
    env = lmdb.open(lmdb_file, map_size=map_size)
    with env.begin(write=True) as txn:
        for streamline_idx in range(streamline_tensor.shape[0]):
            streamline_voxels = ras_to_voxel(streamline_tensor[streamline_idx], inverse_affine=inverse_affine)
            streamline_len = streamline_lengths[streamline_idx]
            labels = get_streamline_labels(streamline_tensor[streamline_idx], streamline_len, sphere)
            padding_mask = torch.arange(streamline_tensor.size(1)) >= streamline_len

            # Convert to a dictionary or a format you wish to store
            data_unit = {
                'streamline_voxels': streamline_voxels,
                'labels': labels,
                'streamline_len': streamline_len.item(),
                'padding_mask': padding_mask,
                'brain_idx': subject_index
            }

            # Serialize the data to a byte string using BytesIO
            buffer = io.BytesIO()
            torch.save(data_unit, buffer)
            serialized_data = buffer.getvalue()

            # Put serialized data into the LMDB transaction
            txn.put(f"{subject_index}_{subject_name}_{streamline_idx}".encode(), serialized_data)

    env.close()

def process_data():
    print("Preprocess script beginning")

    # Get preprocess and relevant data args
    args = parse_args()
    subjects_dir_path = args.raw_subjects_directory

    # Resample streamlines to constant distance between consecutive points
    #resample_streamlines(subjects_dir_path, step_size_mm=args.streamline_stepsize)

    streamlines_padded_length = 60 #determine_max_streamline_length(subjects_dir_path) + 1
    print(f"The length of the padded streamline sequence should be {streamlines_padded_length}")

    for i, subject_name in tqdm(enumerate(os.listdir(subjects_dir_path))):
        subject_path = os.path.join(subjects_dir_path, subject_name)
        subject_name = subject_path.split('/')[-1]
        subject_paths_dictionary = extract_subject_paths(subject_path)

        dwi_data = nib.load(subject_paths_dictionary['dwi_data'])
        affine = torch.tensor(dwi_data.affine, dtype=torch.float32)
        inverse_affine = torch.inverse(affine)

        # Streamlines and labels
        padded_streamlines, streamline_lengths = get_streamline_tensor(subject_paths_dictionary['tractography_folder'], streamlines_padded_length)
        prepare_subject_data(padded_streamlines, streamline_lengths, inverse_affine, subject_name, subject_path, i)

    print("Data preparation completed successfully")


if __name__ == "__main__":
    process_data()
