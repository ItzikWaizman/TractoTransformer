import nibabel as nib
from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from dipy.reconst.shm import sph_harm_lookup, sph_harm_ind_list, sh_to_sf, sf_to_sh, smooth_pinv
from dipy.io import read_bvals_bvecs
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import *
from utils.common_utils import *


# Data Handler Modes
TRAIN, VALIDATION, TRACK = 0, 1, 2

class SubjectDataHandler(object):
    def __init__(self, logger, params, mode, device):
        logger.info(f"Create SubjectDataHandler object with mode {mode}")
        self.mode = mode
        self.logger = logger
        self.device = device
        self.paths_dictionary = extract_subject_paths(self.get_subject_folder(mode, params))
        self.wm_mask = self.load_mask()
        self.dwi, self.affine, self.inverse_affine, self.fa_map = self.load_subject_data(mode)
        self.bvals, self.bvecs = read_bvals_bvecs(self.paths_dictionary['bvals'], self.paths_dictionary['bvecs'])
        self.dwi = self.resample_dwi()         # Resample to gather 100 directions
        self.logger.info("SubjectDataHandler: Preparing streamlines")
        self.tractogram, self.lengths, self.tractogram_header = prepare_streamlines_for_training(self)
        self.dataset = self.create_dataloaders(batch_size=params['batch_size'])
        self.causality_mask  = torch.triu(torch.ones(self.tractogram.size(1), self.tractogram.size(1)), diagonal=1).bool()
        if self.mode is TRACK:
            self.causality_mask  = torch.triu(torch.ones(params['max_sequence_length'], params['max_sequence_length']), diagonal=1).bool()
        
    def get_subject_folder(self, mode, params):
        if mode == TRAIN:
            return params['train_subject_folder']
        elif mode == VALIDATION:
            return params['val_subject_folder']
        else:
            return params['test_subject_folder']

    def load_subject_data(self, mode):
        dwi_data = nib.load(self.paths_dictionary['dwi_data'])
        dwi = torch.tensor(dwi_data.get_fdata(), dtype=torch.float32)

        fa_map = nib.load(self.paths_dictionary['fa'])
        fa_map = torch.tensor(fa_map.get_fdata(), dtype=torch.float32) if mode is TRACK else None

        affine = torch.tensor(dwi_data.affine, dtype=torch.float32)
        return dwi, affine, torch.inverse(affine), fa_map

    @staticmethod
    def normalize_dwi(weights, b0):
        """ Normalize dwi by the first b0.
        Parameters:
        -----------
        weights : ndarray of shape (X, Y, Z, #gradients)
            Diffusion weighted images.
        b0 : ndarray of shape (X, Y, Z)
            B0 image.
        Returns
        -------
        ndarray
            Diffusion weights normalized by the B0.
        """
        b0 = b0[..., None]  # Easier to work if it is a 4D array.

        # Make sure in every voxels weights are lower than ones from the b0.
        nb_erroneous_voxels = np.sum(weights > b0)
        if nb_erroneous_voxels != 0:
            weights = np.minimum(weights, b0)

        # Normalize dwi using the b0.
        weights_normed = weights / b0
        weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

        return weights_normed

    def get_spherical_harmonics_coefficients(self, dwi_weights, bvals, bvecs, sh_order=8, smooth=0.006):
        """ Compute coefficients of the spherical harmonics basis.
        Parameters
        -----------
        dwi_weights : `nibabel.NiftiImage` object
            Diffusion signal as weighted images (4D).
        bvals : ndarray shape (N,)
            B-values used with each direction.
        bvecs : ndarray shape (N, 3)
            Directions of the diffusion signal. Directions are
            assumed to be only on the hemisphere.
        sh_order : int, optional
            SH order. Default: 8
        smooth : float, optional
            Lambda-regularization in the SH fit. Default: 0.006.
        Returns
        -------
        sh_coeffs : ndarray of shape (X, Y, Z, #coeffs)
            Spherical harmonics coefficients at every voxel. The actual number of
            coeffs depends on `sh_order`.
        """

        # Exract the averaged b0.
        b0_idx = bvals == 0
        b0 = dwi_weights[..., b0_idx].mean(axis=3) + 1e-10

        # Extract diffusion weights and normalize by the b0.
        bvecs = bvecs[np.logical_not(b0_idx)]
        weights = dwi_weights[..., np.logical_not(b0_idx)]
        weights = self.normalize_dwi(weights, b0)

        # Assuming all directions are on the hemisphere.
        raw_sphere = HemiSphere(xyz=bvecs)

        # Fit SH to signal
        sph_harm_basis = sph_harm_lookup.get("tournier07")
        Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
        L = -n * (n + 1)
        invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
        data_sh = np.dot(weights, invB.T)
        return data_sh
    
    def resample_dwi(self, sh_order=4, smooth=0):
        data_sh = self.get_spherical_harmonics_coefficients(self.dwi.numpy(), self.bvals, self.bvecs,
                                                            sh_order=sh_order, smooth=smooth)
        sphere = get_sphere('repulsion100')

        sph_harm_basis = sph_harm_lookup.get("tournier07")
        Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
        data_resampled = torch.tensor(np.dot(data_sh, Ba.T), dtype=torch.float32)

        return data_resampled

    def load_mask(self):
        mask_path = self.paths_dictionary['wm_mask']
        mask = nib.load(mask_path)
        mask = torch.tensor(mask.get_fdata(), dtype=torch.float32)
        return mask
    
    def create_dataloaders(self):
        dataset = StreamlineDataset(self.tractogram, self.lengths, self.inverse_affine, self.mode, self.device)
        return dataset

class StreamlineDataset(Dataset):
    def __init__(self, streamlines, lengths, inverse_affine, mode, device):
        permutation = torch.arange(0, streamlines.size(0)-1)
        permutation = permutation[torch.randperm(permutation.size(0))]
        self.streamlines = streamlines
        self.lengths = lengths
        sphere = get_sphere('repulsion724')
        self.sphere_points = torch.zeros((725, 3), dtype=torch.float32)
        self.sphere_points[:724, :] = torch.tensor(sphere.vertices)

        EoF = torch.zeros(725, dtype=torch.float32)
        EoF[724] = 1
        self.EoF = EoF
        self.inverse_affine = inverse_affine

        self.sphere_points = self.sphere_points.to(device)
        self.streamlines = self.streamlines.to(device)
        self.EoF = self.EoF.to(device)
        self.inverse_affine = self.inverse_affine.to(device)

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, idx):
        """
        Parameters:
        - idx (int): Index of the streamline to fetch.
            
        Returns:
        - tuple: (streamline, label, seq_length, padding_mask)
        """
        streamline = self.streamlines[idx]
        streamline_voxels = ras_to_voxel(streamline, inverse_affine=self.inverse_affine)
        seq_length = self.lengths[idx]
        label = generate_labels(streamline, seq_length, self.sphere_points, self.EoF)
        padding_mask = torch.arange(streamline.size(0), device=streamline.device) >= seq_length
        return streamline_voxels, label, seq_length, padding_mask
