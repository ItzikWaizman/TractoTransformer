import os
import random
import shutil
import torch
import glob
import nibabel as nib
from config.args import parse_args

def prepare_data(args):
    raw_subjects_directory = args.raw_subjects_directory
    processed_data_directory = args.processed_data_directory
    train_ratio = args.train_ratio

    subject_dirs = [d for d in os.listdir(raw_subjects_directory) if os.path.isdir(os.path.join(raw_subjects_directory, d))]

    # Extract the subject idx from the .lmdb file names (inside each subject's directory)
    subject_ids = {}
    for subject_dir in subject_dirs:
        subject_path = os.path.join(raw_subjects_directory, subject_dir)

    # Find the LMDB file in the subject directory
        lmdb_file = glob.glob(os.path.join(subject_path, "*.lmdb*"))[0]
        subject_idx = int(lmdb_file.split("/")[-1].split("_")[1])  # Extract the idx from the file name (idx_0_subject_sub-1029.lmdb -> 0)
        subject_ids[subject_idx] = subject_dir

    # Total subjects
    num_subjects = len(subject_ids)

    # Shuffle subject IDs without losing their mapping
    subject_ids_list = list(subject_ids.items())  # [(subject_name, idx), ...]
    random.shuffle(subject_ids_list)  # Shuffle the list of tuples

    # Split the shuffled list into train, val, and test
    test_size = 1
    train_size = int((num_subjects - test_size) * train_ratio)
    val_size = num_subjects - test_size - train_size

    # Assert that there is at least one subject in the validation set
    assert val_size >= 1, "Validation set must contain at least one subject."

    # Extract train, val, and test subjects while keeping their mapping
    train_subjects = dict(subject_ids_list[:train_size])  # {subject_name: idx, ...}
    val_subjects = dict(subject_ids_list[train_size:train_size + val_size])
    test_subjects = dict(subject_ids_list[train_size + val_size:])


    # Define the unified tensor sizes for dwi_data
    max_x, max_y, max_z = 0, 0, 0
    for subject_idx in list(train_subjects.keys()) + list(val_subjects.keys()):
        subject_dir = os.path.join(raw_subjects_directory, f"{subject_ids[subject_idx]}")
        dwi_data_path = glob.glob(os.path.join(subject_dir,"dwi", "*.nii*"))[0]
        dwi_data = nib.load(dwi_data_path)
        dwi_shape = dwi_data.shape
        max_x = max(max_x, dwi_shape[0])
        max_y = max(max_y, dwi_shape[1])
        max_z = max(max_z, dwi_shape[2])

    # Create unified tensors for train and validation
    unified_train_tensor = torch.zeros(len(train_subjects), max_x, max_y, max_z, args.num_gradients)
    unified_val_tensor = torch.zeros(len(val_subjects), max_x, max_y, max_z, args.num_gradients)

    # Create maps for indexing
    train_idx_map = {subject_idx: i for i, subject_idx in enumerate(train_subjects)}
    val_idx_map = {subject_idx: i for i, subject_idx in enumerate(val_subjects)}

    # Load DWI data for train and validation
    for subject_idx in list(train_subjects.keys()) + list(val_subjects.keys()):
        subject_dir = os.path.join(raw_subjects_directory, f"{subject_ids[subject_idx]}")
        dwi_data_path = glob.glob(os.path.join(subject_dir,"dwi", "*.nii*"))[0]
        dwi_data = nib.load(dwi_data_path)
        dwi = torch.tensor(dwi_data.get_fdata(), dtype=torch.float32)

        # Select appropriate tensor based on subject type (train or val)
        if subject_idx in train_subjects:
            tensor = unified_train_tensor[train_idx_map[subject_idx]]
        else:
            tensor = unified_val_tensor[val_idx_map[subject_idx]]

        # Copy the DWI data to the corresponding place in the unified tensor
        tensor[:dwi.shape[0], :dwi.shape[1], :dwi.shape[2], :] = dwi

    train_dwi_dir = os.path.join(processed_data_directory, "train", "dwi")
    val_dwi_dir = os.path.join(processed_data_directory, "val", "dwi")

    # Create directories if they don't exist
    os.makedirs(train_dwi_dir, exist_ok=True)
    os.makedirs(val_dwi_dir, exist_ok=True)

    # Save the unified tensors
    torch.save(unified_train_tensor, os.path.join(train_dwi_dir, "train_dwi_data.pt"))
    torch.save(unified_val_tensor, os.path.join(val_dwi_dir, "val_dwi_data.pt"))

    # Save the index mapping dictionaries
    torch.save(train_idx_map, os.path.join(train_dwi_dir, "train_idx_map.pt"))
    torch.save(val_idx_map, os.path.join(val_dwi_dir, "val_idx_map.pt"))

    # Move the LMDB files to their corresponding directories
    train_dir_shards = os.path.join(processed_data_directory, "train", "shards")
    val_dir_shards = os.path.join(processed_data_directory, "val", "shards")

    # Create the directories if they don't exist
    os.makedirs(train_dir_shards, exist_ok=True)
    os.makedirs(val_dir_shards, exist_ok=True)

    # Move the LMDB files
    for subject_idx in train_subjects:
        lmdb_file = os.path.join(raw_subjects_directory, subject_ids[subject_idx], f"idx_{subject_idx}_subject_{subject_ids[subject_idx]}.lmdb")
        os.rename(lmdb_file, os.path.join(train_dir_shards, os.path.basename(lmdb_file)))

    for subject_idx in val_subjects:
        lmdb_file = os.path.join(raw_subjects_directory, subject_ids[subject_idx],  f"idx_{subject_idx}_subject_{subject_ids[subject_idx]}.lmdb")
        os.rename(lmdb_file, os.path.join(val_dir_shards, os.path.basename(lmdb_file)))

    # Move the test data to the test directory
    test_dir = os.path.join(processed_data_directory, "test")
    os.makedirs(test_dir, exist_ok=True)

    for subject_idx in test_subjects:
        subject_dir = os.path.join(raw_subjects_directory, f"{subject_ids[subject_idx]}")
        for folder in ["dwi", "fodf", "mask", "sh", "tractography_resampled"]:
            src_dir = os.path.join(subject_dir, folder)
            dest_dir = os.path.join(test_dir, folder)
            os.rename(src_dir, dest_dir)

if __name__ == "__main__":
    args = parse_args()
    prepare_data(args)
