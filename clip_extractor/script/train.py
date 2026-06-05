import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =========================
# 설정값
# =========================
DATA_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "view_dataset"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "view_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = OUTPUT_DIR / "best_resnet18_atbat_view_cpu_light.pt"
CLASS_MAP_PATH = OUTPUT_DIR / "class_to_idx.json"
HISTORY_PATH = OUTPUT_DIR / "train_history_cpu_light.json"

IMG_SIZE = 224

# i5 CPU 가벼운 세팅
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

# CPU에서는 backbone freeze 추천
FREEZE_BACKBONE = True

SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=str(VAL_DIR),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model(num_classes):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size

        pbar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{correct / total:.4f}",
        })

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(loader, desc="val", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        pbar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{correct / total:.4f}",
        })

    return running_loss / total, correct / total, all_labels, all_preds, all_probs


def make_confusion_matrix(labels, preds, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for true_label, pred_label in zip(labels, preds):
        matrix[true_label][pred_label] += 1

    return matrix


def print_classification_report(labels, preds, idx_to_class, num_classes):
    matrix = make_confusion_matrix(labels, preds, num_classes)

    print("\nConfusion Matrix")
    for row in matrix:
        print(row)

    print("\nClassification Report")

    total_correct = 0
    total_count = len(labels)

    for class_idx in range(num_classes):
        class_name = idx_to_class[class_idx]

        tp = matrix[class_idx][class_idx]
        fp = sum(matrix[row][class_idx] for row in range(num_classes)) - tp
        fn = sum(matrix[class_idx][col] for col in range(num_classes)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        support = sum(matrix[class_idx])
        total_correct += tp

        print(
            f"{class_name:20s} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f} "
            f"support={support}"
        )

    accuracy = total_correct / total_count if total_count > 0 else 0.0
    print(f"\naccuracy={accuracy:.4f}")


def main():
    set_seed(SEED)

    device = get_device()
    print(f"device: {device}")
    print(f"FREEZE_BACKBONE: {FREEZE_BACKBONE}")
    print("training mode: initial training")
    print("-" * 50)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"train 폴더가 없음: {TRAIN_DIR}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"val 폴더가 없음: {VAL_DIR}")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()

    print("class_to_idx:")
    print(train_dataset.class_to_idx)
    print("-" * 50)

    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            train_dataset.class_to_idx,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"train images: {len(train_dataset)}")
    print(f"val images: {len(val_dataset)}")
    print("-" * 50)

    num_classes = len(train_dataset.classes)

    model = build_model(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, val_labels, val_preds, val_probs = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_model_state,
                "class_to_idx": train_dataset.class_to_idx,
                "img_size": IMG_SIZE,
                "model_name": "resnet18",
                "freeze_backbone": FREEZE_BACKBONE,
                "val_acc": best_val_acc,
            }, MODEL_SAVE_PATH)

            print(f"best model saved: {MODEL_SAVE_PATH}")

    print("\n" + "=" * 50)
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Best model path: {MODEL_SAVE_PATH}")
    print("=" * 50)

    model.load_state_dict(best_model_state)

    val_loss, val_acc, val_labels, val_preds, val_probs = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    print_classification_report(
        labels=val_labels,
        preds=val_preds,
        idx_to_class=idx_to_class,
        num_classes=num_classes,
    )

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"history saved: {HISTORY_PATH}")


if __name__ == "__main__":
    main()