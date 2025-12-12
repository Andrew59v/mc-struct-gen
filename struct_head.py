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
        meta_dim: int = 32,
        feat_channels: Optional[int] = None,
        embed_dim: int = 64,
        projector_hidden: Optional[int] = 128,
        normalize_embeddings: bool = True,
        temperature: float = 1.0,
        freeze_embeddings: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            model: optional UNet instance for convenience `forward(...)`.
            num_blocks: number of block classes (K).
            meta_dim: number of binary metadata channels.
            feat_channels: channels in UNet features (if None, attempt to infer).
            embed_dim: dimension of the learned block embeddings.
            projector_hidden: if provided, uses a small MLP (conv1x1 -> SiLU -> conv1x1) to map features -> embed_dim.
                              if None, uses single 1x1 conv projection.
            normalize_embeddings: if True, l2-normalize embeddings and projected features (cosine-sim).
            temperature: scalar dividing logits (higher -> softer).
            freeze_embeddings: if True, embeddings won't be updated.
            dropout: dropout in projector if projector_hidden provided.
        """
        super().__init__()
        self.model = model
        self.num_blocks = num_blocks
        self.meta_dim = meta_dim
        self.embed_dim = embed_dim
        self.projector_hidden = projector_hidden
        self.normalize = normalize_embeddings
        self.temperature = float(temperature)

        # infer feat_channels if not provided
        if feat_channels is None:
            if model is not None and hasattr(model, "final_norm") and hasattr(model.final_norm, "num_channels"):
                feat_channels = model.final_norm.num_channels
            else:
                raise ValueError("feat_channels not provided and could not be inferred. "
                                 "Pass feat_channels or attach model with final_norm.")
        self.feat_channels = feat_channels

        # projector: maps feat_channels -> embed_dim per voxel
        if projector_hidden is None:
            # single 1x1 conv
            self.projector = nn.Sequential(
                nn.Conv3d(self.feat_channels, embed_dim, kernel_size=1),
                nn.Dropout(dropout) if (dropout and dropout > 0.0) else nn.Identity()
            )
        else:
            # two-layer bottleneck MLP: 1x1 conv -> act -> 1x1 conv
            self.projector = nn.Sequential(
                nn.Conv3d(self.feat_channels, projector_hidden, kernel_size=1),
                nn.GroupNorm(num_groups=min(8, max(1, projector_hidden)), num_channels=projector_hidden, eps=1e-6),
                nn.SiLU(),
                nn.Dropout(dropout) if (dropout and dropout > 0.0) else nn.Identity(),
                nn.Conv3d(projector_hidden, embed_dim, kernel_size=1),
            )

        # learnable block embeddings (prototypes)
        self.block_embeddings = nn.Parameter(torch.randn(num_blocks, embed_dim))
        if freeze_embeddings:
            self.block_embeddings.requires_grad = False

        # optional per-class bias (helps calibration)
        self.class_bias = nn.Parameter(torch.zeros(num_blocks))

        # metadata head (logits)
        self.meta_head = nn.Conv3d(self.feat_channels, meta_dim, kernel_size=1)

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

        # project features to embedding space
        proj = self.projector(features)  # (B, embed_dim, D, H, W)
        
        # optionally normalize
        if self.normalize:
            proj_flat = proj.view(B, self.embed_dim, -1)
            proj_norm = F.normalize(proj_flat, dim=1)  # (B, embed_dim, N)
            emb = F.normalize(self.block_embeddings, dim=1)  # (K, embed_dim)
        else:
            proj_norm = proj.view(B, self.embed_dim, -1)
            emb = self.block_embeddings  # (K, embed_dim)

        # compute dot product logits: (B, E, N) @ (E, K) -> (B, N, K) -> (B, K, N)
        proj_perm = proj_norm.permute(0, 2, 1)  # (B, N, E)
        emb_t = emb.t()  # (E, K)
        logits = torch.matmul(proj_perm, emb_t)  # (B, N, K)
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

        out = self.model(x, timesteps, encoder_hidden_states, return_features=True)
        if isinstance(out, (tuple, list)):
            _, features = out
        else:
            raise RuntimeError("UNet.forward did not return (out, features). Pass return_features=True.")
        
        logits, meta_logits = self.forward_from_features(features)
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
    
