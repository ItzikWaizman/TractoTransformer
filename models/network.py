import torch
import torch.nn as nn
import torch.nn.functional as F

class TractoTransformer(nn.Module):
    def __init__(self, logger, params, max_sequence_length):
        super(TractoTransformer, self).__init__()
        logger.info("Create TractoTransformer model object")

        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(params.num_gradients, max_sequence_length+1)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(params.num_gradients,
                                                                     params.nhead,
                                                                     params.ff_dim,
                                                                     params.dropout_rate) for _ in range(params.num_decoder_layers)])

        # Fully connected layer as projection to the possible directions space
        self.projection = nn.Linear(params.num_gradients, params.output_size)

        # Dropout layer
        self.dropout = nn.Dropout(params.dropout_rate)

        #3D CNN layer
        self.cnn3d_layer = CNN3DLayer(params.num_gradients, params.device)

        # Store max sequence length for mask generation
        self.max_sequence_length = max_sequence_length
        self.device = params.device

    def _get_causality_mask(self, seq_len, cache_len=0):
        """Generate causality mask that accounts for cached sequences"""
        total_len = seq_len + cache_len
        # Create mask for the full sequence (cached + new)
        mask = torch.triu(torch.ones(seq_len, total_len, device=self.device), diagonal=cache_len+1).bool()
        return mask

    def forward(self, dwi_data, streamline_voxels_batch, padding_mask, brain_indices, past_kvs=None, use_cache=False):
        # Use 3D CNN to account for dwi values in the environment of each voxel in a streamline
        center_features = self.cnn3d_layer(dwi_data, streamline_voxels_batch, brain_indices)

        # Apply positional encoding
        batch_size, seq_len, _ = center_features.shape
        
        # Adjust positional encoding for cached sequences
        if past_kvs is not None and len(past_kvs) > 0:
            # Get the cache length from the first layer's past_kv
            cache_len = past_kvs[0][0].shape[1] if past_kvs[0] is not None else 0
            # Apply positional encoding starting from cache_len position
            pos_start = cache_len
            x = center_features + self.positional_encoding.encoding[pos_start:pos_start+seq_len, :].unsqueeze(0).to(center_features.device)
        else:
            cache_len = 0
            pos_start = 0
            x = center_features + self.positional_encoding.encoding[pos_start:pos_start+seq_len, :].unsqueeze(0).to(center_features.device)
        
        x = self.dropout(x)

        # Generate causality mask accounting for cache
        causality_mask = self._get_causality_mask(seq_len, cache_len)
        
        # Adjust padding mask for cached sequences if needed
        if past_kvs is not None and len(past_kvs) > 0 and cache_len > 0:
            # Extend padding mask to account for cached tokens (assume cached tokens are not padded)
            cache_padding = torch.zeros(batch_size, cache_len, dtype=torch.bool, device=padding_mask.device)
            extended_padding_mask = torch.cat([cache_padding, padding_mask], dim=1)
        else:
            extended_padding_mask = padding_mask

        # Apply decoder layers
        new_kvs = [] if use_cache else None
        for i, decoder_layer in enumerate(self.decoder_layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = decoder_layer(x, causality_mask, extended_padding_mask, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        # Project to the number of possible directions
        outputs = self.projection(x)

        # Return probability distribution
        log_probabilities = F.log_softmax(outputs, dim=-1)

        if use_cache:
            return log_probabilities, new_kvs
        else:
            return log_probabilities


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ff = PositionWiseFeedForward(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causality_mask, padding_mask, past_kv=None, use_cache=False):
        # Prepare query, key, value for self-attention
        query = x
        
        if past_kv is not None:
            # Concatenate past keys and values with current input
            past_key, past_value = past_kv
            key = torch.cat([past_key, x], dim=1)
            value = torch.cat([past_value, x], dim=1)
        else:
            key = x
            value = x

        # Self-attention
        attn_output, attn_weights = self.self_attn(
            query,                          # Current tokens as query
            key,                           # Past + current tokens as key
            value,                         # Past + current tokens as value
            attn_mask=causality_mask,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        # Residual connection and layer norm
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # Feed-forward network
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        # Prepare new KV cache if needed
        if use_cache:
            if past_kv is not None:
                # Update the cache with new keys and values
                new_key = torch.cat([past_kv[0], x], dim=1)
                new_value = torch.cat([past_kv[1], x], dim=1)
            else:
                # Initialize cache with current keys and values
                new_key = x
                new_value = x
            
            new_kv = (new_key, new_value)
            return x, new_kv
        else:
            return x, None


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_positions):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_positions, d_model)

        position = torch.arange(max_positions).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).float()
        div_term = (10000 ** (_2i / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)[..., : encoding[:, 1::2].size(-1)]

        self.register_buffer('encoding', encoding)

    def forward(self, x):
        return x + self.encoding[:x.size(1), :].to(x.device).unsqueeze(0)


class CNN3DLayer(nn.Module):
    def __init__(self, num_channels, device):
        super(CNN3DLayer, self).__init__()
        self.num_channels = num_channels
        self.cnn3d = nn.Conv3d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, padding=1)

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
        ], device=device).view(1, 1, 27, 3)

    def forward(self, dwi_data, streamline_voxels_batch, brain_indices, white_matter_mask=None):
        batch_size, max_sequence_length, _ = streamline_voxels_batch.shape
        num_gradients = self.num_channels

        # Initialize a tensor to hold the 3x3x3 patches
        patches = torch.zeros((batch_size, max_sequence_length, 27, num_gradients), device=dwi_data.device)

        # Expand streamline_voxels_batch and add offsets
        expanded_voxels = streamline_voxels_batch.unsqueeze(2) + self.offsets  # Shape: (batch_size, max_sequence_length, 27, 3)

        # Split the expanded_voxels into x, y, z coordinates
        x, y, z = expanded_voxels[..., 0], expanded_voxels[..., 1], expanded_voxels[..., 2]

        # Ensure the voxel indices are within valid range
        valid_mask = (x >= 0) & (x < dwi_data.shape[1]) & (y >= 0) & (y < dwi_data.shape[2]) & (z >= 0) & (z < dwi_data.shape[3])

        # Include the white matter mask in the validity check
        if white_matter_mask is not None:
            valid_mask &= white_matter_mask[x, y, z]

        # Flatten the indices and mask for efficient indexing
        flat_x = x.flatten()
        flat_y = y.flatten()
        flat_z = z.flatten()
        flat_valid_mask = valid_mask.flatten()

        # Repeat brain_indices for each voxel in the streamline sequence
        flat_brain_indices = torch.repeat_interleave(brain_indices.to(dwi_data.device), max_sequence_length * 27).flatten()

        # Extract DWI values for the valid patches
        valid_indices = torch.nonzero(flat_valid_mask).squeeze()
        flat_brain_indices = flat_brain_indices[valid_indices]
        valid_patches = dwi_data[flat_brain_indices, flat_x[valid_indices], flat_y[valid_indices], flat_z[valid_indices]]

        # Fill the valid patches into the corresponding locations in the patches tensor
        patches.view(-1, num_gradients)[valid_indices] = valid_patches

        # Reshape to (batch_size * max_sequence_length, num_gradients, 3, 3, 3) for CNN input
        patches = patches.view(batch_size * max_sequence_length, 3, 3, 3, num_gradients).permute(0, 4, 1, 2, 3)

        # Apply the 3D CNN to the patches
        cnn_features = self.cnn3d(patches)  # Shape: (batch_size * max_sequence_length, num_gradients, 3, 3, 3)

        # Extract the center voxel features from the CNN output
        center_features = cnn_features[:, :, 1, 1, 1]  # Shape: (batch_size * max_sequence_length, num_gradients)

        # Reshape center_features back to (batch_size, max_sequence_length, num_gradients)
        center_features = center_features.view(batch_size, max_sequence_length, num_gradients)

        return center_features
        