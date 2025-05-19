import lmdb
import os
import torch
from io import BytesIO
from torch.utils.data import Dataset

class DataHandler:
    def __init__(self, args):
        """
        Initializes the DataHandler by loading train and validation data
        and their respective index maps. Also creates Dataset instances.

        Args:
            processed_data_directory (str): Path to the directory containing
                                            processed train/val/test data.
        """
        processed_data_directory = args.processed_data_directory

        self.device = args.device
        self.train_dir = os.path.join(processed_data_directory, "trainset")
        self.val_dir = os.path.join(processed_data_directory, "validset")

        # Load train and validation brains
        self.train_brains = torch.load(os.path.join(self.train_dir, "dwi", "trainset_dwi_data.pt"), map_location=args.device)
        self.val_brains = torch.load(os.path.join(self.val_dir, "dwi", "validset_dwi_data.pt"), map_location=args.device)

        # Load train and validation subject maps
        self.train_subjects_map = torch.load(os.path.join(self.train_dir, "dwi", "trainset_idx_map.pt"))
        self.val_subjects_map = torch.load(os.path.join(self.val_dir, "dwi", "validset_idx_map.pt"))

        # Create Dataset instances
        self.train_dataset = ShardsDataset(os.path.join(self.train_dir, "shards"), self.train_subjects_map)
        self.val_dataset = ShardsDataset(os.path.join(self.val_dir, "shards"), self.val_subjects_map)

        # Get max_sequence_length
        self.max_sequence_length = self.train_dataset.get_max_seq_length()


class ShardsDataset(Dataset):
    def __init__(self, shards_directory, subject_id_to_dwi_entry):
        self.subject_id_to_dwi_entry = subject_id_to_dwi_entry
        self.shards_directory = shards_directory
        self.shard_files = [os.path.join(shards_directory, f) for f in os.listdir(shards_directory) if f.endswith(".lmdb")]
        self.data = []  # List of all data units (keys) in LMDB
        self._load_all_keys()

    def _load_all_keys(self):
        """Load all keys (subject_idx_streamline_idx) from the LMDB files."""
        for lmdb_file in self.shard_files:
            env = lmdb.open(lmdb_file, readonly=True)
            with env.begin() as txn:
                cursor = txn.cursor()
                keys = [key.decode() for key, _ in cursor]
                self.data.extend([(lmdb_file, key) for key in keys])
            env.close()

    def __len__(self):
        """Return the total number of data units across all shards."""
        return len(self.data)

    def __getitem__(self, idx):
        """Randomly fetch a data unit from a random LMDB file."""
        lmdb_dir, key = self.data[idx]
        env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            # Retrieve the data unit for the given key
            raw_data = txn.get(key.encode())
            data_unit = torch.load(BytesIO(raw_data))
        env.close()

        return (
            data_unit['streamline_voxels'],
            data_unit['labels'],
            data_unit['streamline_len'],
            data_unit['padding_mask'],
            self.subject_id_to_dwi_entry[data_unit['brain_idx']]
        )

    def get_max_seq_length(self):
        lmdb_dir, key = self.data[0]
        env = lmdb.open(lmdb_dir)
        with env.begin() as txn:
            # Retrieve the data unit for the given key
            raw_data = txn.get(key.encode())
            data_unit = torch.load(BytesIO(raw_data))
        env.close()

        return data_unit['streamline_voxels'].shape[0]
