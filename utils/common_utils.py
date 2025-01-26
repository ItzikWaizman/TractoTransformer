import torch

def ras_to_voxel(ras_coords, inverse_affine):
    """
    INPUT: ras_coords - a tensor of size (B, N, 3) or (N, 3)
           inverse_affine - the inverse affine transformation matrix, tensor of size (4, 4)

    OUTPUT: voxel_coords - the transformed coordinates, tensor of size (B, N, 3) or (N, 3)
    """
    if len(ras_coords.shape) == 2:
        # Append a column of ones for homogeneous coordinates
        ones_column = torch.ones((ras_coords.shape[0], 1), dtype=torch.float32, device=ras_coords.device)
        ras_homogeneous = torch.cat((ras_coords, ones_column), dim=1)

        # Apply inverse affine transformation to convert RAS to voxel coordinates
        voxel_coords_homogeneous = torch.matmul(ras_homogeneous, inverse_affine.T)

        # Remove homogeneous coordinate and round to get voxel indices
        voxel_coords = torch.round(voxel_coords_homogeneous[:, :3]).to(torch.int)
    elif len(ras_coords.shape) == 3:
        B, N, _ = ras_coords.shape
        voxel_coords = torch.zeros((B, N, 3), dtype=torch.int, device=ras_coords.device)

        for i in range(B):
            ones_column = torch.ones((ras_coords[i].shape[0], 1), dtype=torch.float32, device=ras_coords.device)
            ras_homogeneous = torch.cat((ras_coords[i], ones_column), dim=1)

            # Apply inverse affine transformation to convert RAS to voxel coordinates
            voxel_coords_homogeneous = torch.matmul(ras_homogeneous, inverse_affine.T)

            # Remove homogeneous coordinate and round to get voxel indices
            voxel_coords[i] = torch.round(voxel_coords_homogeneous[:, :3]).to(torch.int)
    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    return voxel_coords


def voxel_to_ras(voxel_indices, affine):
    """
    INPUT: voxel_indices - a tensor of size (B, N, 3) or (N, 3)
           affine - the affine transformation matrix, tensor of size (4, 4)

    OUTPUT: ras_coords - the transformed coordinates, tensor of size (B, N, 3) or (N, 3)
    """

    if len(voxel_indices.shape) == 2:
        # Add a 1 for homogeneous coordinates
        ones_column = torch.ones(voxel_indices.shape[0], 1, dtype=torch.float32)
        voxel_homogeneous = torch.cat((voxel_indices, ones_column), dim=1)  # Shape: (N, 4)

        # Perform affine transformation
        ras_coords_homogeneous = torch.matmul(affine, voxel_homogeneous.T).T  # Shape: (N, 4)

        ras_coords = ras_coords_homogeneous[:, :3]

    elif len(voxel_indices.shape) == 3:
        B, N, _ = voxel_indices.shape
        ras_coords = torch.zeros((B, N, 3), dtype=torch.float32)

        for i in range(B):
            ones_column = torch.ones(voxel_indices[i].shape[0], 1, dtype=torch.float32)
            voxel_homogeneous = torch.cat((voxel_indices[i], ones_column), dim=1)  # Shape: (N, 4)

            # Perform affine transformation
            ras_coords_homogeneous = torch.matmul(affine, voxel_homogeneous.T).T  # Shape: (N, 4)

            ras_coords[i] = ras_coords_homogeneous[:, :3]

    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    return ras_coords


def calculate_streamlines_lengths(streamlines):
    """
    Calculate the lengths of a batch of streamlines.

    INPUT:
        streamlines - a tensor of shape [num_streamlines, max_seq_len, 3], representing a batch of sequences of points in 3D space.

    OUTPUT:
        lengths - a tensor of shape [num_streamlines], containing the total length of each streamline.
    """
    # Calculate the differences between consecutive points for each streamline
    differences = streamlines[:, 1:, :] - streamlines[:, :-1, :]

    # Calculate the Euclidean distance for each step in each streamline
    step_lengths = torch.norm(differences, dim=2)

    # Sum the step lengths to get the total length for each streamline
    lengths = torch.sum(step_lengths, dim=1)

    return lengths
