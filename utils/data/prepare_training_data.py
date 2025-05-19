import os
import random
import shutil
import torch
import glob
import nibabel as nib
from config.args import parse_args
from dipy.io import read_bvals_bvecs
from utils.data.data_utils import *

def prepare_data(args):
    raw_subjects_directory = args.raw_subjects_directory
    processed_data_directory = args.processed_data_directory

    partitions = ["trainset", "validset", "testset"]
    subject_ids = {partition: {} for partition in partitions}

    for partition in partitions:
        partition_path = os.path.join(raw_subjects_directory, partition)
        if not os.path.isdir(partition_path):
            continue

        subject_dirs = [d for d in os.listdir(partition_path) if os.path.isdir(os.path.join(partition_path, d))]

        for subject_dir in subject_dirs:
            subject_path = os.path.join(partition_path, subject_dir)
            lmdb_file = glob.glob(os.path.join(subject_path, "*.lmdb*"))[0]
            subject_idx = int(lmdb_file.split("/")[-1].split("_")[1])
            subject_ids[partition][subject_idx] = subject_dir

    # Define the unified tensor sizes for dwi_data (train & val only)
    max_x, max_y, max_z = 0, 0, 0
    print("Calculating unified dimensions")
    for partition in ["trainset", "validset"]:
        for subject_idx in subject_ids[partition]:
            subject_dir = os.path.join(raw_subjects_directory, partition, subject_ids[partition][subject_idx])
            dwi_data_path = glob.glob(os.path.join(subject_dir, "dwi", "*.nii*"))[0]
            dwi_data = nib.load(dwi_data_path)
            dwi_shape = dwi_data.shape
            max_x = max(max_x, dwi_shape[0])
            max_y = max(max_y, dwi_shape[1])
            max_z = max(max_z, dwi_shape[2])

    unified_tensors = {
        "trainset": torch.zeros(len(subject_ids["trainset"]), max_x, max_y, max_z, args.num_gradients, device=args.device),
        "validset": torch.zeros(len(subject_ids["validset"]), max_x, max_y, max_z, args.num_gradients, device=args.device),
    }

    index_maps = {
        "trainset": {subject_idx: i for i, subject_idx in enumerate(subject_ids["trainset"])},
        "validset": {subject_idx: i for i, subject_idx in enumerate(subject_ids["validset"])},
    }
    print("Iterating through the brains")

    # Load DWI data into unified tensors
    bvecs = None

    for partition in ["trainset", "validset"]:
        for subject_idx in subject_ids[partition]:
            subject_dir = os.path.join(raw_subjects_directory, partition, subject_ids[partition][subject_idx])
            print("subject dir = ", subject_dir)
            dwi_data_path = glob.glob(os.path.join(subject_dir, "dwi", "*.nii*"))[0]
            bvals_path, bvecs_path = glob.glob(os.path.join(subject_dir, "dwi", "*.bval*"))[0], glob.glob(os.path.join(subject_dir, "dwi", "*.bvec*"))[0]
            subject_bvals, subject_bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
            print("Updating bvecs")
            if bvecs is None:
                bvecs = subject_bvecs
            print("loading dwi_data")
            dwi_data = nib.load(dwi_data_path)
            print("extracting fdata")
            dwi = torch.tensor(dwi_data.get_fdata(), dtype=torch.float32)
            print("Resampling brain of ", subject_dir)
            dwi = resample_and_normalize_dwi(dwi, subject_bvecs, sh_order=12, smooth=0, directions=None)

            tensor = unified_tensors[partition][index_maps[partition][subject_idx]]
            tensor[:dwi.shape[0], :dwi.shape[1], :dwi.shape[2], :] = dwi

    for subject_idx in subject_ids["testset"]:
        subject_dir = os.path.join(raw_subjects_directory, "testset", subject_ids["testset"][subject_idx])
        dwi_data_path = glob.glob(os.path.join(subject_dir, "dwi", "*.nii*"))[0]
        bvals_path, bvecs_path = glob.glob(os.path.join(subject_dir, "dwi", "*.bval*"))[0], glob.glob(os.path.join(subject_dir, "dwi", "*.bvec*"))[0]
        subject_bvals, subject_bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
        dwi_data = nib.load(dwi_data_path)
        dwi = torch.tensor(dwi_data.get_fdata(), dtype=torch.float32)
        print("Resampling brain of ", subject_dir)
        dwi = resample_and_normalize_dwi(dwi, subject_bvecs, sh_order=12, smooth=0, directions=bvecs)
        new_dwi_data = nib.Nifti1Image(dwi, affine=dwi_data.affine, header=dwi_data.header)
        nib.save(new_dwi_data, os.path.join(subject_dir, "dwi", "Diffusion.nii"))

    # Save tensors & index maps
    for partition in ["trainset", "validset"]:
        partition_dwi_dir = os.path.join(processed_data_directory, partition, "dwi")
        os.makedirs(partition_dwi_dir, exist_ok=True)
        torch.save(unified_tensors[partition], os.path.join(partition_dwi_dir, f"{partition}_dwi_data.pt"))
        torch.save(index_maps[partition], os.path.join(partition_dwi_dir, f"{partition}_idx_map.pt"))

    # Move LMDB files
    for partition in ["trainset", "validset"]:
        shards_dir = os.path.join(processed_data_directory, partition, "shards")
        os.makedirs(shards_dir, exist_ok=True)

        for subject_idx in subject_ids[partition]:
            subject_dir = os.path.join(raw_subjects_directory, partition, subject_ids[partition][subject_idx])
            for lmdb_file in glob.glob(os.path.join(subject_dir, "*.lmdb*")):
                print("Moving file: ", lmdb_file)
                os.rename(lmdb_file, os.path.join(shards_dir, os.path.basename(lmdb_file)))

    # Move test set data
    test_dir = os.path.join(processed_data_directory, "testset")
    os.makedirs(test_dir, exist_ok=True)

    for subject_idx in subject_ids["testset"]:
        subject_dir = os.path.join(raw_subjects_directory, "testset", subject_ids["testset"][subject_idx])

        for folder in ["anat", "dti", "dwi", "fodf", "mask", "sh", "tractography_resampled"]:
            src_dir = os.path.join(subject_dir, folder)
            dest_dir = os.path.join(test_dir, folder)
            os.rename(src_dir, dest_dir)

    print("Data preparation completed successfully.")

if __name__ == "__main__":
    args = parse_args()
    prepare_data(args)
