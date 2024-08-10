import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import *

class TractoTransformer(nn.Module):
    def __init__(self, logger, params):
        super(TractoTransformer, self).__init__()
        logger.info("Create TractoGNN model object")

        # Build decoder-only transformer model
        self.positional_encoding = PositionalEncoding(d_model=params['num_of_gradients'], max_len=params['max_streamline_len'])
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(embed_dim=params['num_of_gradients'],
                                                                     num_heads=params['nhead'],
                                                                     ff_dim=params['transformer_feed_forward_dim'],
                                                                     dropout=params['dropout_rate']) for _ in range(params['num_transformer_decoder_layers'])])
        
        # Use Linear network as a projection to output_size.
        self.projection = nn.Linear(params['num_of_gradients'], params['output_size'])
        self.dropout = nn.Dropout(params['dropout_rate'])

        self.cnn3d_layer = CNN3DLayer(num_of_gradients=params['num_of_gradients'])

    def forward(self, dwi_data, streamline_voxels_batch, padding_mask, causality_mask):

        center_features = self.cnn3d_layer(dwi_data, streamline_voxels_batch)
        #center_features = dwi_data[streamline_voxels_batch[:, :, 0], streamline_voxels_batch[:, :, 1], streamline_voxels_batch[:, :, 2]]
        # Apply positional encoding
        x = self.dropout(self.positional_encoding(center_features))
        
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, causality_mask, padding_mask)

        outputs = self.projection(x)
        log_probabilities = F.log_softmax(outputs, dim=-1)

        return log_probabilities


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ff = PositionWiseFeedForward(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causality_mask, padding_mask):
        # Self-attention with masking
        attn_output, _ = self.self_attn(x, x, x, attn_mask=causality_mask, key_padding_mask=padding_mask, is_causal=True)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # Feed-forward network
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).float()
        div_term = (10000 ** (_2i / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)[..., : encoding[:, 1::2].size(-1)]
        
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        """
        Adds positional encoding to input tensor x.
        
        Parameters:
            x (Tensor): Input tensor of shape [batch_size, seq_length, d_model].
        
        Returns:
            Tensor: The input tensor augmented with positional encodings.
        """
        return x + self.encoding[:x.size(1), :].to(x.device).unsqueeze(0)
    

class CNN3DLayer(nn.Module):
    def __init__(self, num_of_gradients):
        super(CNN3DLayer, self).__init__()
        self.cnn3d = nn.Conv3d(in_channels=num_of_gradients, out_channels=num_of_gradients, kernel_size=3, padding=1)
        
        # Save the offsets during initialization
        self.offsets = torch.tensor([
            [-1, -1, -1], [-1, -1,  0], [-1, -1,  1],
            [-1,  0, -1], [-1,  0,  0], [-1,  0,  1],
            [-1,  1, -1], [-1,  1,  0], [-1,  1,  1],
             [0, -1, -1], [0, -1,  0], [0, -1,  1],
             [0,  0, -1], [0,  0,  0], [0,  0,  1],
             [0,  1, -1], [0,  1,  0], [0,  1,  1],
             [1, -1, -1], [1, -1,  0], [1, -1,  1],
             [1,  0, -1], [1,  0,  0], [1,  0,  1],
             [1,  1, -1], [1,  1,  0], [1,  1,  1]
        ]).view(1, 1, 27, 3)
        
    def forward(self, dwi_data, streamline_voxels_batch, white_matter_mask=None):
        batch_size, max_sequence_length, _ = streamline_voxels_batch.shape
        num_of_gradients = dwi_data.shape[3]

        # Move offsets to the correct device
        offsets = self.offsets.to(dwi_data.device)

        # Initialize a tensor to hold the 3x3x3 patches
        patches = torch.zeros((batch_size, max_sequence_length, 27, num_of_gradients), device=dwi_data.device)

        # Expand streamline_voxels_batch and add offsets
        expanded_voxels = streamline_voxels_batch.unsqueeze(2) + offsets  # Shape: (batch_size, max_sequence_length, 27, 3)

        # Split the expanded_voxels into x, y, z coordinates
        x, y, z = expanded_voxels[..., 0], expanded_voxels[..., 1], expanded_voxels[..., 2]

        # Ensure the voxel indices are within valid range
        valid_mask = (x >= 0) & (x < dwi_data.shape[0]) & (y >= 0) & (y < dwi_data.shape[1]) & (z >= 0) & (z < dwi_data.shape[2])

        # Include the white matter mask in the validity check
        if white_matter_mask is not None:
            valid_mask &= white_matter_mask[x, y, z]

        # Flatten the indices and mask for efficient indexing
        flat_x = x.flatten()
        flat_y = y.flatten()
        flat_z = z.flatten()
        flat_valid_mask = valid_mask.flatten()

        # Extract DWI values for the valid patches
        valid_indices = torch.nonzero(flat_valid_mask).squeeze()
        valid_patches = dwi_data[flat_x[valid_indices], flat_y[valid_indices], flat_z[valid_indices]]

        # Fill the valid patches into the corresponding locations in the patches tensor
        patches.view(-1, num_of_gradients)[valid_indices] = valid_patches

        # Reshape to (batch_size * max_sequence_length, num_of_gradients, 3, 3, 3) for CNN input
        patches = patches.view(batch_size * max_sequence_length, 3, 3, 3, num_of_gradients).permute(0, 4, 1, 2, 3)

        # Apply the 3D CNN to the patches
        cnn_features = self.cnn3d(patches)  # Shape: (batch_size * max_sequence_length, num_of_gradients, 3, 3, 3)

        # Extract the center voxel features from the CNN output
        center_features = cnn_features[:, :, 1, 1, 1]  # Shape: (batch_size * max_sequence_length, num_of_gradients)

        # Reshape center_features back to (batch_size, max_sequence_length, num_of_gradients)
        center_features = center_features.view(batch_size, max_sequence_length, num_of_gradients)

        return center_features