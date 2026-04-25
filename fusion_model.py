import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITY_ORDER = ["emotion", "environment", "health", "gesture", "speech"]
MODALITY_DIMS = [768, 512, 512, 512, 512]
COMMON_DIM = 256
NUM_CLASSES = 8

class ModalityProjection(nn.Module):
    """Projects each modality embedding down to a common dimension."""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class FusionModelV2(nn.Module):
    """
    V2 Architecture for dynamic multimodal fusion.
    Matches the input API requirement of train_fusion_v3.py
    """
    def __init__(self, 
                 modality_dims=MODALITY_DIMS, 
                 common_dim=COMMON_DIM, 
                 num_classes=NUM_CLASSES,
                 proj_dropout=0.2, 
                 cls_dropout=0.3,
                 use_mask_features=True):
        super().__init__()
        self.num_mods = len(modality_dims)
        self.common_dim = common_dim
        self.use_mask_features = use_mask_features

        # Projection Layers
        self.projections = nn.ModuleList([
            ModalityProjection(d, common_dim, dropout=proj_dropout)
            for d in modality_dims
        ])

        # Mask aware Attention Gating
        self.attention = nn.Sequential(
            nn.Linear(common_dim * self.num_mods, common_dim),
            nn.GELU(),
            nn.Linear(common_dim, self.num_mods)
        )

        # Fusion Classification Head
        fusion_in = common_dim * self.num_mods
        if self.use_mask_features:
            fusion_in += self.num_mods  # add raw mask array as feature

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, *args):
        # args structure: emotion, env, health, gesture, speech, mask
        *embs, mask = args
        B = embs[0].shape[0]

        # 1. Project individual modalities
        projs = []
        for i in range(self.num_mods):
            # Pass through ModalityProjection [B, 256]
            projs.append(self.projections[i](embs[i]))

        # 2. Stack and Zero-out missing embeddings using Mask
        stacked = torch.stack(projs, dim=1) # [B, 5, 256]
        stacked = stacked * mask.unsqueeze(2)
        flat = stacked.view(B, -1)          # [B, 1280]

        # 3. Dynamic Attention Weights
        attn_logits = self.attention(flat)  # [B, 5]
        # Ignore dead modalities
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        # Careful with batch instances where everything is masked to avoid NaNs
        attn_weights = F.softmax(attn_logits, dim=1) 
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)

        # 4. Multiplied by Attention
        stacked_weighted = stacked * attn_weights.unsqueeze(2)
        flat_weighted = stacked_weighted.view(B, -1)   # [B, 1280]

        # 5. Append implicit mask feature mapping if enabled
        if self.use_mask_features:
            flat_weighted = torch.cat([flat_weighted, mask], dim=1) # [B, 1285]

        # 6. Classification logits
        logits = self.classifier(flat_weighted)
        
        return logits, attn_weights

