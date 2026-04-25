import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from fusion_update.fusion_model import MODALITY_ORDER, FusionModelV2
except ModuleNotFoundError:
    from fusion_model import MODALITY_ORDER, FusionModelV2


FUSION_CLASSES_V3 = [
    "normal",
    "monitor",
    "needs_assistance",
    "call_nurse",
    "medical_warning",
    "security_warning",
    "emergency_critical",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_embedding_file(path: Path, expected_dim: int) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    embs = obj["embeddings"].float()
    if embs.ndim != 2 or embs.shape[1] != expected_dim:
        raise ValueError(f"Embedding shape mismatch for {path}: {tuple(embs.shape)} expected [N,{expected_dim}]")
    embs = F.normalize(embs, p=2, dim=1)
    return {
        "embeddings": embs,
        "labels": obj.get("labels"),
        "classes": list(obj["classes"]) if "classes" in obj else None,
    }


def macro_f1(labels: List[int], preds: List[int], num_classes: int) -> float:
    if not labels:
        return 0.0
    y = np.array(labels, dtype=np.int64)
    p = np.array(preds, dtype=np.int64)
    eps = 1e-12
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        f1s.append(float(f1))
    return float(np.mean(f1s))


def speech_command_to_system(label_maps: dict) -> Dict[str, str]:
    return {str(k).lower(): str(v).lower() for k, v in label_maps["speech"].get("speech_to_system", {}).items()}


def find_environment_labels(all_labels: Sequence[str], keywords: Sequence[str]) -> List[str]:
    out = []
    for lbl in all_labels:
        l = lbl.lower()
        if any(k in l for k in keywords):
            out.append(lbl)
    return out


def idx_to_label_map_for_modality(modality: str, data: dict, label_map: dict) -> Dict[int, str]:
    if data.get("classes") is not None:
        return {i: str(c).lower() for i, c in enumerate(data["classes"])}

    if "idx_to_label" in label_map:
        return {int(k): str(v).lower() for k, v in label_map["idx_to_label"].items()}

    if modality == "speech" and "commands" in label_map:
        return {i: str(v).lower() for i, v in enumerate(label_map["commands"])}

    if "classes" in label_map:
        return {i: str(v).lower() for i, v in enumerate(label_map["classes"])}

    raise ValueError(f"Cannot build idx->label map for modality={modality}")


def build_label_index(modality_data: Dict[str, dict], label_maps: Dict[str, dict]) -> Dict[str, Dict[str, List[int]]]:
    out: Dict[str, Dict[str, List[int]]] = {}
    for m in MODALITY_ORDER:
        data = modality_data[m]
        labels_tensor = data.get("labels")
        if labels_tensor is None:
            raise ValueError(f"Missing labels in embedding file for modality={m}")
        idx_to_lbl = idx_to_label_map_for_modality(m, data, label_maps[m])
        bins: Dict[str, List[int]] = {}
        for i, lbl_idx in enumerate(labels_tensor.tolist()):
            lbl = idx_to_lbl.get(int(lbl_idx))
            if lbl is None:
                continue
            bins.setdefault(lbl, []).append(i)
        out[m] = bins
    return out


def scenario_label_space(label_maps: Dict[str, dict], label_index: Dict[str, Dict[str, List[int]]]) -> Dict[str, Dict[str, List[str]]]:
    speech_sys = speech_command_to_system(label_maps)
    speech_by_intent: Dict[str, List[str]] = {}
    for cmd, intent in speech_sys.items():
        speech_by_intent.setdefault(intent, []).append(cmd)

    env_all = [str(x).lower() for x in label_maps["environment"]["classes"]]
    env_medical = find_environment_labels(env_all, ["hospital", "operating_room", "nursing_home", "clinic"])
    env_security = find_environment_labels(env_all, ["street", "alley", "crosswalk", "parking", "downtown", "subway"])
    env_assist = find_environment_labels(env_all, ["bathroom", "stairs", "corridor", "bedroom", "living_room", "kitchen"])

    classes: Dict[str, Dict[str, List[str]]] = {
        "normal": {
            "emotion": ["happy", "neutral", "surprise"],
            "gesture": ["stop", "yes", "no", "calm", "cancel"],
            "health": ["baseline", "meditation", "amusement"],
            "speech": speech_by_intent.get("yes", []) + speech_by_intent.get("no", []) + speech_by_intent.get("calm", []),
            "environment": [e for e in env_all if e not in env_medical and e not in env_security],
        },
        "monitor": {
            "emotion": ["surprise", "sad", "neutral"],
            "gesture": ["attention", "unknown"],
            "health": ["baseline", "amusement"],
            "speech": speech_by_intent.get("attention", []) + speech_by_intent.get("action", []),
            "environment": env_assist + env_security,
        },
        "needs_assistance": {
            "emotion": ["sad", "disgust"],
            "gesture": ["help", "attention"],
            "health": ["stress", "baseline"],
            "speech": speech_by_intent.get("help", []) + speech_by_intent.get("action", []),
            "environment": env_assist + env_medical,
        },
        "call_nurse": {
            "emotion": ["fear", "sad", "angry"],
            "gesture": ["help", "suspicious"],
            "health": ["stress"],
            "speech": speech_by_intent.get("help", []),
            "environment": env_medical + env_assist,
        },
        "medical_warning": {
            "emotion": ["fear", "sad"],
            "gesture": ["help", "emergency", "suspicious"],
            "health": ["stress"],
            "speech": speech_by_intent.get("help", []) + speech_by_intent.get("action", []),
            "environment": env_medical,
        },
        "security_warning": {
            "emotion": ["angry", "fear"],
            "gesture": ["suspicious", "emergency"],
            "health": ["stress", "amusement"],
            "speech": speech_by_intent.get("suspicious", []) + speech_by_intent.get("attention", []),
            "environment": env_security,
        },
        "emergency_critical": {
            "emotion": ["fear", "angry"],
            "gesture": ["emergency"],
            "health": ["stress"],
            "speech": speech_by_intent.get("help", []),
            "environment": env_medical + env_security,
        },
    }

    # Ensure every class/modality has at least one valid label present in current embedding pool.
    for cname, per_mod in classes.items():
        for m in MODALITY_ORDER:
            allowed = [x.lower() for x in per_mod.get(m, [])]
            available = set(label_index[m].keys())
            filtered = [x for x in allowed if x in available]
            if not filtered:
                filtered = list(available)
            per_mod[m] = filtered
    return classes


def pattern_bank() -> Dict[str, List[List[str]]]:
    all5 = [list(MODALITY_ORDER)]
    triples = [
        ["emotion", "environment", "gesture"],
        ["emotion", "health", "gesture"],
        ["health", "gesture", "speech"],
        ["emotion", "speech", "gesture"],
        ["emotion", "environment", "health"],
    ]
    pairs = [
        ["emotion", "gesture"],
        ["gesture", "speech"],
        ["health", "speech"],
        ["gesture", "health"],
        ["emotion", "environment"],
    ]
    singles = [[m] for m in MODALITY_ORDER]
    return {"full": all5, "triple": triples, "pair": pairs, "single": singles}


def build_blueprint(
    target_per_class: int,
    full_ratio: float,
    triple_ratio: float,
    pair_ratio: float,
    single_ratio: float,
) -> List[Tuple[int, List[str]]]:
    ratios = np.array([full_ratio, triple_ratio, pair_ratio, single_ratio], dtype=np.float64)
    ratios = np.maximum(ratios, 0.0)
    if ratios.sum() == 0:
        ratios = np.array([0.6, 0.2, 0.15, 0.05], dtype=np.float64)
    ratios = ratios / ratios.sum()

    counts = (ratios * target_per_class).astype(int)
    while counts.sum() < target_per_class:
        counts[np.argmax(ratios)] += 1

    banks = pattern_bank()
    out: List[Tuple[int, List[str]]] = []
    for ci in range(len(FUSION_CLASSES_V3)):
        for group_name, n in zip(["full", "triple", "pair", "single"], counts.tolist()):
            bank = banks[group_name]
            for _ in range(n):
                out.append((ci, random.choice(bank)))
    random.shuffle(out)
    return out


class ScenarioFusionDataset(Dataset):
    def __init__(
        self,
        modality_embs: Dict[str, torch.Tensor],
        label_index: Dict[str, Dict[str, List[int]]],
        class_label_space: Dict[str, Dict[str, List[str]]],
        blueprint: List[Tuple[int, List[str]]],
        modality_drop_p: float,
        noise_std: float,
    ):
        self.modality_embs = modality_embs
        self.label_index = label_index
        self.class_label_space = class_label_space
        self.blueprint = blueprint
        self.modality_drop_p = modality_drop_p
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.blueprint)

    def _sample_emb(self, modality: str, allowed_labels: Sequence[str]) -> Optional[torch.Tensor]:
        labels = list(allowed_labels)
        random.shuffle(labels)
        for lbl in labels:
            idxs = self.label_index[modality].get(lbl, [])
            if idxs:
                idx = random.choice(idxs)
                emb = self.modality_embs[modality][idx].clone()
                if self.noise_std > 0:
                    emb = F.normalize(emb + torch.randn_like(emb) * self.noise_std, p=2, dim=0)
                return emb
        return None

    def __getitem__(self, idx: int):
        class_idx, present = self.blueprint[idx]
        class_name = FUSION_CLASSES_V3[class_idx]
        label_space = self.class_label_space[class_name]
        present_set = set(present)

        embs: List[torch.Tensor] = []
        mask: List[float] = []

        for m in MODALITY_ORDER:
            if m in present_set:
                emb = self._sample_emb(m, label_space[m])
                if emb is None:
                    emb = torch.zeros(self.modality_embs[m].shape[1], dtype=torch.float32)
                    mask.append(0.0)
                else:
                    embs.append(emb)
                    mask.append(1.0)
                    continue
            else:
                emb = torch.zeros(self.modality_embs[m].shape[1], dtype=torch.float32)
                mask.append(0.0)
            embs.append(emb)

        mask_t = torch.tensor(mask, dtype=torch.float32)

        for i in range(len(MODALITY_ORDER)):
            if mask_t[i] == 1.0 and random.random() < self.modality_drop_p:
                embs[i] = torch.zeros_like(embs[i])
                mask_t[i] = 0.0

        if mask_t.sum() == 0:
            # Keep at least one modality alive.
            for i, m in enumerate(MODALITY_ORDER):
                emb = self._sample_emb(m, label_space[m])
                if emb is not None:
                    embs[i] = emb
                    mask_t[i] = 1.0
                    break

        return (*embs, mask_t, torch.tensor(class_idx, dtype=torch.long))


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.enable_grad() if train else torch.no_grad():
        for batch in loader:
            *embs, mask, labels = [b.to(device) for b in batch]
            if train:
                optimizer.zero_grad()
            logits, _ = model(*embs, mask)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    n = max(len(all_labels), 1)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
    f1 = macro_f1(all_labels, all_preds, num_classes=len(FUSION_CLASSES_V3)) if all_labels else 0.0
    return total_loss / n, acc, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="uploads/metadata/model_manifest.json")
    parser.add_argument("--out", default="uploads/models/fusion/best_fusion_model_v3.pth")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-per-class", type=int, default=14000)
    parser.add_argument("--modality-drop-p", type=float, default=0.12)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--full-ratio", type=float, default=0.60)
    parser.add_argument("--triple-ratio", type=float, default=0.22)
    parser.add_argument("--pair-ratio", type=float, default=0.13)
    parser.add_argument("--single-ratio", type=float, default=0.05)
    parser.add_argument("--common-dim", type=int, default=256)
    parser.add_argument("--proj-dropout", type=float, default=0.2)
    parser.add_argument("--cls-dropout", type=float, default=0.3)
    parser.add_argument("--early-stop-patience", type=int, default=18)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = load_json(Path(args.manifest))

    label_maps = {
        "emotion": load_json(Path(manifest["emotion"]["labels_file"]),),
        "environment": load_json(Path(manifest["environment"]["labels_file"]),),
        "health": load_json(Path(manifest["health"]["labels_file"]),),
        "gesture": load_json(Path(manifest["gesture"]["labels_file"]),),
        "speech": load_json(Path(manifest["speech"]["labels_file"]),),
    }

    dims = {m: int(manifest[m]["embedding_dim"]) for m in MODALITY_ORDER}
    modality_data = {m: load_embedding_file(Path(manifest[m]["embedding_file"]), dims[m]) for m in MODALITY_ORDER}
    modality_embs = {m: modality_data[m]["embeddings"] for m in MODALITY_ORDER}

    print("Embedding sizes:")
    for m in MODALITY_ORDER:
        print(f"  {m:12s}: {tuple(modality_embs[m].shape)}")

    label_index = build_label_index(modality_data, label_maps)
    class_label_space = scenario_label_space(label_maps, label_index)

    print("\nAvailable labels per class/modality (post-filter):")
    for cname in FUSION_CLASSES_V3:
        print(f"  {cname}:")
        for m in MODALITY_ORDER:
            print(f"    {m:12s} -> {len(class_label_space[cname][m])}")

    blueprint = build_blueprint(
        target_per_class=args.target_per_class,
        full_ratio=args.full_ratio,
        triple_ratio=args.triple_ratio,
        pair_ratio=args.pair_ratio,
        single_ratio=args.single_ratio,
    )
    print(f"\nBlueprint samples: {len(blueprint)}")
    print(f"Blueprint distribution: {Counter(lbl for lbl, _ in blueprint)}")

    ds = ScenarioFusionDataset(
        modality_embs=modality_embs,
        label_index=label_index,
        class_label_space=class_label_space,
        blueprint=blueprint,
        modality_drop_p=args.modality_drop_p,
        noise_std=args.noise_std,
    )

    n_val = int(len(ds) * args.val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FusionModelV2(
        modality_dims=[dims[m] for m in MODALITY_ORDER],
        common_dim=args.common_dim,
        num_classes=len(FUSION_CLASSES_V3),
        proj_dropout=args.proj_dropout,
        cls_dropout=args.cls_dropout,
        use_mask_features=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best = {"val_f1": -1.0}
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(va_f1)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% f1={tr_f1:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc*100:.2f}% f1={va_f1:.4f} | lr={lr_now:.2e}"
        )

        improved = va_f1 > (best["val_f1"] + args.early_stop_min_delta)
        if improved:
            best = {
                "epoch": epoch,
                "val_f1": va_f1,
                "val_acc": va_acc,
                "model_state": model.state_dict(),
                "modality_dims": [dims[m] for m in MODALITY_ORDER],
                "modality_order": list(MODALITY_ORDER),
                "common_dim": args.common_dim,
                "fusion_classes": list(FUSION_CLASSES_V3),
                "proj_dropout": float(args.proj_dropout),
                "cls_dropout": float(args.cls_dropout),
                "use_mask_features": True,
                "notes": "v3 scenario-driven fusion with full-modality-prioritized blueprint",
                "class_label_space": class_label_space,
                "blueprint_ratios": {
                    "full": args.full_ratio,
                    "triple": args.triple_ratio,
                    "pair": args.pair_ratio,
                    "single": args.single_ratio,
                },
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: no val_f1 improvement > {args.early_stop_min_delta} "
                f"for {args.early_stop_patience} epochs."
            )
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, out)
    print(f"\nSaved best checkpoint: {out}")
    print(f"Best epoch={best['epoch']} val_acc={best['val_acc']*100:.2f}% val_f1={best['val_f1']:.4f}")


if __name__ == "__main__":
    main()