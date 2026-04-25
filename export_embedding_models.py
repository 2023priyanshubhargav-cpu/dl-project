import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import timm
from torchvision import models

# ==============================================================
# EXACT ARCHITECTURES
# The full models were pickled WITH these class definitions, 
# so they MUST be in __main__ exactly as they were trained.
# ==============================================================

# --- EMOTION ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0)
        self.cbam = CBAM(768)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def get_embedding(self, x):
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
    def forward(self, x):
        return self.classifier(self.get_embedding(x))


# --- ENVIRONMENT ---
class EnvironmentModel(nn.Module):
    def __init__(self, num_classes=147, embedding_dim=512):
        super().__init__()
        backbone = models.resnet50(weights=False)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(1024, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.GELU(), nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
    def get_embedding(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.embedding_head(x)
    def forward(self, x):
        return self.classifier(self.get_embedding(x))


# --- HEALTH ---
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dilation=1, dropout=0.2, pool=2):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout), nn.MaxPool1d(pool)
        )
    def forward(self, x):
        return self.block(x)

class HealthModel(nn.Module):
    def __init__(self, num_classes=4, embedding_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock1D(6, 64, kernel=7, pool=2),
            ConvBlock1D(64, 128, kernel=7, pool=2),
            ConvBlock1D(128, 256, kernel=7, pool=2),
            ConvBlock1D(256, 512, kernel=7, pool=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_head = nn.Sequential(
            nn.Linear(512, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.GELU(), nn.Dropout(0.3),
        )
        self.residual_proj = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
    def get_embedding(self, x):
        feat = self.encoder(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        return self.embedding_head(feat) + self.residual_proj(feat)
    def forward(self, x):
        return self.classifier(self.get_embedding(x))


# ==============================================================
# WRAPPER FOR EXPLICIT OUTPUT: (Embedding, Logits)
# ==============================================================

class TupleOutputWrapper(nn.Module):
    """
    Wraps the extracted model so that calling model(x)
    returns both the intermediate embedding AND the final logits.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # We explicitly call our named methods
        emb = self.model.get_embedding(x)
        logits = self.model.classifier(emb)
        return emb, logits


def export_as_tuple_model(pth_path, pt_path, dummy_input, name):
    if not os.path.exists(pth_path):
        print(f"  [X] {pth_path} not found. Ensure it exists in the directory.")
        return
        
    print(f"  [>] Processing {name} ({pth_path}) -> {pt_path}")
    try:
        # Load the fully pickled model (which uses the __main__ classes above)
        model = torch.load(pth_path, map_location='cpu', weights_only=False)
        model.eval()
        
        # Wrap it
        wrapper = TupleOutputWrapper(model)
        wrapper.eval()
        
        # Trace with dummy input
        scripted_model = torch.jit.trace(wrapper, dummy_input)
        
        # We delete the old one automatically
        if os.path.exists(pt_path):
            os.remove(pt_path)
            
        scripted_model.save(pt_path)
        print(f"  [+] SUCCESS: Created {pt_path} (Outputs: [Embedding, Logits])")
    except Exception as e:
        print(f"  [X] FAILED {name}: {e}")

if __name__ == "__main__":
    print("========== Re-exporting models for Inference ==========")
    print("This will overwrite the old `.pt` models with new ones that output (Embedding, Logits).")
    
    export_as_tuple_model(
        "emotion_model_full.pth", "emotion_model_scripted.pt", 
        torch.randn(1, 3, 224, 224), "Emotion"
    )
    
    export_as_tuple_model(
        "environment_model_full.pth", "environment_model_scripted.pt", 
        torch.randn(1, 3, 224, 224), "Environment"
    )

    export_as_tuple_model(
        "health_model_full.pth", "health_model_scripted.pt", 
        torch.randn(1, 6, 1250), "Health"
    )
    
    print("\n========== Cleanup Old Versions ==========")
    for f in ["emotion_model_scripted_v2.pt", "environment_model_scripted_v2.pt", 
              "health_model_scripted_v2.pt"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  [-] Removed redundant file: {f}")
            
    print("\n========== ALL DONE! You can now run realtime_fusion_8cls.py ==========")
