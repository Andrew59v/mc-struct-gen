# mc_struct_embed_head.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNet3DConditional

class MCStructEmbedHead(nn.Module):
    """
    Prototype/embedding-based voxel head.

    - block embeddings: (num_blocks, embed_dim)
    - project UNet features -> embed_dim and compute dot-product logits per voxel
    - metadata: simple per-voxel classifier (1x1 conv) with logits -> sigmoid -> bits
    """
    def __init__(
        self,
        model: Optional[UNet3DConditional],
        num_blocks: int = 512,
        embed_dim: int = 32,
        meta_dim: int = 32,
        normalize_embeddings: bool = True,
        temperature: float = 1.0,
        freeze_embeddings: bool = False,
    ):
        """
        Args:
            model: optional UNet instance for convenience `forward(...)`.
            num_blocks: number of block classes (K).
            embed_dim: channels in UNet features (embeddings will match this dimension).
            meta_dim: number of binary metadata channels.
            normalize_embeddings: if True, l2-normalize embeddings and features (cosine-sim).
            temperature: scalar dividing logits (higher -> softer).
            freeze_embeddings: if True, embeddings won't be updated.
        """
        super().__init__()
        self.model = model
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.meta_dim = meta_dim
        self.normalize = normalize_embeddings
        self.temperature = float(temperature)

        self.feat_dim = embed_dim

        # learnable block embeddings (prototypes) - now same dim as features
        self.block_embeddings = nn.Parameter(torch.randn(num_blocks, embed_dim))
        if freeze_embeddings:
            self.block_embeddings.requires_grad = False

        # optional per-class bias (helps calibration)
        self.class_bias = nn.Parameter(torch.zeros(num_blocks))

        # metadata head (logits)
        self.meta_head = nn.Conv3d(self.feat_dim, meta_dim, kernel_size=1)

        # Feature modulation: metadata -> modulation parameters for blocks
        # Generate scale/shift from metadata to modulate block features (FiLM-style)
        self.meta_to_modulation = nn.Sequential(
            nn.Conv3d(meta_dim, embed_dim * 2, kernel_size=1),  # scale + shift
            nn.GroupNorm(num_groups=min(8, embed_dim), num_channels=embed_dim * 2, eps=1e-6),
            nn.SiLU(),
        )

        # init
        self._init_weights()

    def _init_weights(self):
        # convs are initialized by default, but initialize embeddings small
        nn.init.normal_(self.block_embeddings, mean=0.0, std=0.02)
        nn.init.zeros_(self.class_bias)
        # conv init for meta and projector layers handled by default conv init; optionally reinit:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_from_features(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify voxels from precomputed features.

        Args:
            features: (B, C_feat, D, H, W)

        Returns:
            block_logits: (B, K, D, H, W) torch.float - classification logits for each block type
            meta_logits: (B, M, D, H, W) torch.float - metadata prediction logits
        """
        B, C, D, H, W = features.shape

        # No projection needed - direct similarity with features (embed_dim = feat_channels)
        # optionally normalize
        if self.normalize:
            feat_flat = features.view(B, self.feat_dim, -1)
            feat_norm = F.normalize(feat_flat, dim=1)  # (B, feat_channels, N)
            emb = F.normalize(self.block_embeddings, dim=1)  # (K, feat_channels)
        else:
            feat_norm = features.view(B, self.feat_dim, -1)
            emb = self.block_embeddings  # (K, feat_channels)

        # compute dot product logits: (B, E, N) @ (E, K) -> (B, N, K) -> (B, K, N)
        feat_perm = feat_norm.permute(0, 2, 1)  # (B, N, E)
        emb_t = emb.t()  # (E, K)
        logits = torch.matmul(feat_perm, emb_t)  # (B, N, K)
        logits = logits.permute(0, 2, 1).contiguous()  # (B, K, N)

        # scale by temperature and add per-class bias
        logits = logits / (self.temperature + 1e-8)
        logits = logits + self.class_bias.view(1, -1, 1)  # (1, K, 1) broadcasts
        logits = logits.view(B, self.num_blocks, D, H, W)  # (B, K, D, H, W)

        # metadata: simple 1x1 conv over original features
        meta_logits = self.meta_head(features)  # (B, M, D, H, W)

        return logits, meta_logits
    
    def logits_to_outputs(
        self,
        logits: torch.Tensor,
        meta_logits: torch.Tensor,
        meta_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits to discrete outputs.

        Args:
            logits: (B, K, D, H, W)
            meta_logits: (B, M, D, H, W)
            meta_threshold: threshold for metadata binarization

        Returns:
            block_ids: (B, 1, D, H, W) torch.long
            meta_flags: (B, M, D, H, W) torch.float (0/1)
        """
        block_ids = torch.argmax(logits, dim=1, keepdim=True).long()  # (B, 1, D, H, W)

        meta_probs = torch.sigmoid(meta_logits)
        meta_flags = (meta_probs > meta_threshold).float()  # (B, M, D, H, W)

        return block_ids, meta_flags

    def backward_to_features(
        self,
        block_logits: torch.Tensor,
        meta_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert logits back to feature space using FiLM-style modulation.

        Reconstructs features by modulating predicted block embeddings with metadata,
        maintaining separation between blocks and metadata information.

        Args:
            block_logits: (B, K, D, H, W) - block classification logits
            meta_logits: (B, M, D, H, W) - metadata logits

        Returns:
            features: (B, F, D, H, W) - reconstructed features for UNet
        """
        # Get predicted block classes
        block_ids = torch.argmax(block_logits, dim=1)  # (B, D, H, W)
        block_ids = torch.clamp(block_ids, 0, self.num_blocks - 1)

        # Get corresponding block embeddings (base features)
        block_embeddings = self.block_embeddings[block_ids]  # (B, D, H, W, embed_dim)
        block_embeddings = block_embeddings.permute(0, 4, 1, 2, 3)  # (B, embed_dim, D, H, W)

        # Generate modulation parameters from metadata
        modulation_params = self.meta_to_modulation(meta_logits)  # (B, embed_dim * 2, D, H, W)
        scale, shift = modulation_params.chunk(2, dim=1)  # Each: (B, embed_dim, D, H, W)

        # Stabilize modulation parameters to prevent extreme values
        scale = torch.clamp(scale, -2.0, 2.0)  # Prevent extreme scaling
        shift = torch.clamp(shift, -1.0, 1.0)  # Prevent extreme shifting

        # Apply FiLM modulation: modulated = scale * base + shift
        modulated_blocks = scale * block_embeddings + shift

        # embed_dim = feat_channels, so no dimension conversion needed
        features = modulated_blocks

        return features 

    def logits_from_outputs(
        self,
        block_ids: torch.Tensor,
        metadata_flags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert discrete outputs back to logits (inverse of logits_to_outputs).
        
        Reconstructs logits from ground truth block IDs and metadata flags
        for encoding into the diffusion model.

        Args:
            block_ids: (B, 1, D, H, W) torch.long - discrete block IDs (0-511)
            metadata_flags: (B, M, D, H, W) torch.float - binary metadata (0/1)

        Returns:
            block_logits: (B, K, D, H, W) - reconstructed block classification logits
            meta_logits: (B, M, D, H, W) - reconstructed metadata logits
        """
        B, _, D, H, W = block_ids.shape
        K = self.num_blocks
        device = block_ids.device
        
        block_ids_flat = block_ids.squeeze(1).long()  # (B, D, H, W)
        block_ids_flat = torch.clamp(block_ids_flat, 0, K - 1)
        
        # Create one-hot style logits: high value for correct class, low for others
        block_logits = torch.full((B, K, D, H, W), -10.0, device=device, dtype=torch.float32)
        
        # Vectorized assignment: set correct class logits to 10.0
        b_idx = torch.arange(B, device=device)[:, None, None, None].expand(B, D, H, W)
        d_idx = torch.arange(D, device=device)[None, :, None, None].expand(B, D, H, W)
        h_idx = torch.arange(H, device=device)[None, None, :, None].expand(B, D, H, W)
        w_idx = torch.arange(W, device=device)[None, None, None, :].expand(B, D, H, W)
        
        block_logits[b_idx, block_ids_flat, d_idx, h_idx, w_idx] = 10.0
        
        # Convert metadata flags (0/1) to logits using log-odds style
        # Ensure we're working with float tensors
        metadata_float = metadata_flags.float()
        meta_logits = torch.where(
            metadata_float > 0.5,
            torch.full_like(metadata_float, 5.0),
            torch.full_like(metadata_float, -5.0)
        )
        
        return block_logits, meta_logits

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        meta_threshold: float = 0.5,
        return_logits: bool = False
    ):
        if self.model is None:
            raise RuntimeError("No UNet model attached. Use forward_from_features(...) instead or instantiate with model.")

        # UNet returns denoised latent (B, 32, D, H, W)
        latents = self.model(x, timesteps, encoder_hidden_states)
        
        logits, meta_logits = self.forward_from_features(latents)
        if return_logits:
            return logits, meta_logits
        else:
            return self.logits_to_outputs(logits, meta_logits, meta_threshold)

# ---------------------- Smoke test ----------------------
if __name__ == "__main__":
    # minimal smoke test
    B, C, D, H, W = 1, 32, 8, 8, 8
    x = torch.randn(B, C, D, H, W)
    timesteps = torch.randint(0, 1000, (B,))
    text_emb = torch.randn(B, 512)

    # create unet and attach head (unet must support return_features=True)
    unet = UNet3DConditional()
    head = MCStructEmbedHead(unet)

    block_id, meta_bits = head(x, timesteps, text_emb)

    print("block_id:", block_id.shape, block_id.min().item(), block_id.max().item())
    print("meta_bits:", meta_bits.shape)
    print(block_id)
    
