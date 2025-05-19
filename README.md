# TractoTransformer

**TractoTransformer** is a deep learning framework for tractography based on transformer and cnn models. This repository includes everything required to preprocess diffusion MRI (dMRI) data, train the TractoTransformer model, and perform tractography inference.

---

## 📁 Repository Overview

This repository contains:

- Preprocessing utilities to prepare dMRI data for training and inference
- Training pipeline for the TractoTransformer model
- Inference module for performing tractography

---

## 🔧 Preprocessing Pipeline

The preprocessing is performed in two sequential steps, both located in the `utils/` directory.

### 1. `data_preprocess.py`

This script processes raw subject data by resampling their reference streamlines to a uniform step size and organizing the output into `.lmdb` files.

**Usage**
```bash
python utils/data/data_preprocess.py --raw_subjects_directory /path/to/raw_subjects
```

**Arguments**
- `--raw_subjects_directory`: Path to a folder with the following structure:
  ```
  /path/to/raw_subjects/
      ├── trainset/
      ├── validset/
      └── testset/
  ```
  Each set should contain subject folders with raw diffusion and tractography data.

**Output**
- `.lmdb` files per subject with resampled reference streamlines saved in-place.

---

### 2. `prepare_training_data.py`

This script further processes the data by:
- Resampling DWI volumes to a constant number of gradient directions
- Concatenating subject brains into a new dimension
- Creating tensors for training and validation sets
- Aggregating `.lmdb` files for training and validation to be loaded efficiently at runtime

**Usage**
```bash
python utils/data/prepare_training_data.py --raw_subjects_directory /path/to/raw_subjects --processed_data_directory /path/to/processed_data
```

**Arguments**
- `--raw_subjects_directory`: Path to the previously preprocessed data
- `--processed_data_directory`: Destination path for the final, training-ready data

---

## 🚀 Model Training and Inference

Once data is prepared, you can either train the TractoTransformer model or perform tractography.

### 🔹 Train Mode
```bash
python main.py --train
```

### 🔹 Inference / Tractography Mode
```bash
python main.py --track
```

> ⚙️ **Note**: Be sure to configure all necessary parameters in the `args.py` file before running training or inference.

---

## 📌 Notes

- Input data must be structured in `trainset`, `validset`, and `testset` subdirectories before running `data_preprocess.py`.
- The preprocessing pipeline outputs optimized `.lmdb` and tensor formats for efficient use during training and inference.

---

## 📁 Example Directory Structure

```
/raw_subjects/
    ├── trainset/
    │   ├── subject1/
    │   ├── subject2/
    │   └── ...
    ├── validset/
    │   ├── subjectA/
    │   └── ...
    └── testset/
        ├── subjectX/
        └── ...

/processed_data/
    ├── trainset/
    │   ├── dwi/
    │   └── shards/
    ├── validset/
    │   ├── dwi/
    │   └── shards
    |── testset/
        
```

---

## 🧠 About TractoTransformer

TractoTransformer leverages transformer architectures to model diffusion pathways in the brain, aiming to improve the accuracy and generalizability of tractography predictions. This repository offers an end-to-end pipeline from raw dMRI data to fiber tracking results.
