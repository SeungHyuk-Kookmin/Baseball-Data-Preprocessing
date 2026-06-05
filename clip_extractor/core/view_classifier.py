import os
import json
import shutil
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from tqdm import tqdm

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =========================
# 설정값
# =========================
SCENE_CLIPS_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "scene_clips"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "classified_atbat_clips"

MODEL_PATH = PROJECT_ROOT / "runs" / "view_classifier" / "best_resnet18_atbat_view_cpu_light.pt"

RESULT_JSON_PATH = PROJECT_ROOT / "data" / "clip_extractor" / "classified_atbat_clips" / "classified_atbat_results.json"

# 클립에서 검사할 프레임 위치
FRAME_RATIOS = [0.20, 0.30, 0.40]

# atbat 확률 평균이 이 값 이상이면 타석으로 채택
ATBAT_THRESHOLD = 0.70

# 너무 짧거나 긴 클립 제외
MIN_CLIP_SECONDS = 1.0
MAX_CLIP_SECONDS = 20.0

IMG_SIZE = 224

# True면 기존 결과 폴더 비우고 시작
RESET_OUTPUT_DIR = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_to_idx = checkpoint["class_to_idx"]
    img_size = checkpoint.get("img_size", IMG_SIZE)

    num_classes = len(class_to_idx)

    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print("class_to_idx:")
    print(class_to_idx)
    print("-" * 50)

    # atbat class index 찾기
    atbat_idx = None
    for class_name, idx in class_to_idx.items():
        if "atbat" in class_name:
            atbat_idx = idx
            break

    if atbat_idx is None:
        raise ValueError("class_to_idx에서 atbat 클래스를 찾지 못함.")

    print(f"atbat_idx: {atbat_idx}")
    print(f"atbat_class: {idx_to_class[atbat_idx]}")
    print("-" * 50)

    return model, class_to_idx, idx_to_class, atbat_idx, img_size


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0, 0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0 or frame_count <= 0:
        return 0, 0, 0.0

    duration = frame_count / fps

    return fps, frame_count, duration


def extract_frame_at_ratio(video_path, ratio):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        return None

    frame_idx = int(frame_count * ratio)
    frame_idx = min(max(frame_idx, 0), frame_count - 1)

    # 원래 프레임 + 주변 프레임 재시도
    retry_offsets = [0, -1, 1, -2, 2, -5, 5, -10, 10]

    for offset in retry_offsets:
        target_idx = frame_idx + offset

        if target_idx < 0 or target_idx >= frame_count:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()

        if ret and frame is not None:
            cap.release()
            return frame

    cap.release()
    return None


def build_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


@torch.no_grad()
def predict_frame(frame, model, transform, device, atbat_idx):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    atbat_prob = float(probs[atbat_idx].cpu().item())
    pred_idx = int(torch.argmax(probs).cpu().item())

    return atbat_prob, pred_idx


@torch.no_grad()
def predict_clip(video_path, model, transform, device, atbat_idx):
    fps, frame_count, duration = get_video_info(video_path)

    if fps <= 0 or frame_count <= 0:
        return {
            "path": str(video_path),
            "status": "invalid_video",
            "is_atbat": False,
            "duration": duration,
            "frame_probs": [],
            "avg_atbat_prob": 0.0,
            "max_atbat_prob": 0.0,
            "min_atbat_prob": 0.0,
        }

    if duration < MIN_CLIP_SECONDS:
        return {
            "path": str(video_path),
            "status": "too_short",
            "is_atbat": False,
            "duration": duration,
            "frame_probs": [],
            "avg_atbat_prob": 0.0,
            "max_atbat_prob": 0.0,
            "min_atbat_prob": 0.0,
        }

    if duration > MAX_CLIP_SECONDS:
        return {
            "path": str(video_path),
            "status": "too_long",
            "is_atbat": False,
            "duration": duration,
            "frame_probs": [],
            "avg_atbat_prob": 0.0,
            "max_atbat_prob": 0.0,
            "min_atbat_prob": 0.0,
        }

    frame_probs = []

    for ratio in FRAME_RATIOS:
        frame = extract_frame_at_ratio(video_path, ratio)

        if frame is None:
            continue

        atbat_prob, pred_idx = predict_frame(
            frame=frame,
            model=model,
            transform=transform,
            device=device,
            atbat_idx=atbat_idx,
        )

        frame_probs.append({
            "ratio": ratio,
            "atbat_prob": atbat_prob,
            "pred_idx": pred_idx,
        })

    if not frame_probs:
        return {
            "path": str(video_path),
            "status": "frame_extract_failed",
            "is_atbat": False,
            "duration": duration,
            "frame_probs": [],
            "avg_atbat_prob": 0.0,
            "max_atbat_prob": 0.0,
            "min_atbat_prob": 0.0,
        }

    probs = [item["atbat_prob"] for item in frame_probs]

    avg_prob = sum(probs) / len(probs)
    max_prob = max(probs)
    min_prob = min(probs)

    is_atbat = avg_prob >= ATBAT_THRESHOLD

    return {
        "path": str(video_path),
        "status": "accepted" if is_atbat else "rejected",
        "is_atbat": is_atbat,
        "duration": duration,
        "frame_probs": frame_probs,
        "avg_atbat_prob": avg_prob,
        "max_atbat_prob": max_prob,
        "min_atbat_prob": min_prob,
    }


def prepare_output_dir():
    if OUTPUT_DIR.exists() and RESET_OUTPUT_DIR:
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_scene_clips():
    if not SCENE_CLIPS_DIR.exists():
        raise FileNotFoundError(f"scene clips 폴더가 없음: {SCENE_CLIPS_DIR}")

    video_files = sorted(list(SCENE_CLIPS_DIR.glob("*.mp4")))

    return video_files


def copy_accepted_clips(results):
    accepted = [item for item in results if item["is_atbat"]]

    print("-" * 50)
    print(f"채택 clip 복사 시작: {len(accepted)}개")
    print("-" * 50)

    for idx, item in enumerate(accepted, start=1):
        src_path = Path(item["path"])

        avg_prob = item["avg_atbat_prob"]
        max_prob = item["max_atbat_prob"]
        duration = item["duration"]

        dst_name = (
            f"atbat_{idx:04d}_"
            f"avg_{avg_prob:.3f}_"
            f"max_{max_prob:.3f}_"
            f"dur_{duration:.2f}_"
            f"{src_path.name}"
        )

        dst_path = OUTPUT_DIR / dst_name

        shutil.copy2(src_path, dst_path)

    print("복사 완료")


def print_summary(results):
    total = len(results)
    accepted = sum(1 for item in results if item["is_atbat"])
    rejected = total - accepted

    status_counts = {}

    for item in results:
        status = item["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\n분석 요약")
    print("-" * 50)
    print(f"전체 scene clip 수: {total}")
    print(f"타석 채택 수: {accepted}")
    print(f"제외 수: {rejected}")
    print(f"threshold: {ATBAT_THRESHOLD}")
    print()

    print("status 분포")
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {status}: {count}")

    accepted_items = [item for item in results if item["is_atbat"]]

    if accepted_items:
        avg_probs = [item["avg_atbat_prob"] for item in accepted_items]
        print()
        print("채택 clip 확률")
        print(f"- 평균 avg_atbat_prob: {sum(avg_probs) / len(avg_probs):.4f}")
        print(f"- 최소 avg_atbat_prob: {min(avg_probs):.4f}")
        print(f"- 최대 avg_atbat_prob: {max(avg_probs):.4f}")

    print("-" * 50)


def save_results(results):
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"결과 저장 완료: {RESULT_JSON_PATH}")


def main():
    prepare_output_dir()

    device = get_device()
    print(f"device: {device}")
    print(f"model: {MODEL_PATH}")
    print("-" * 50)

    model, class_to_idx, idx_to_class, atbat_idx, img_size = load_model(device)
    transform = build_transform(img_size)

    scene_files = find_scene_clips()

    print(f"scene clip 수: {len(scene_files)}")
    print("-" * 50)

    results = []

    for video_path in tqdm(scene_files, desc="scene clip classifier 필터링 중"):
        result = predict_clip(
            video_path=video_path,
            model=model,
            transform=transform,
            device=device,
            atbat_idx=atbat_idx,
        )

        results.append(result)

    print_summary(results)
    save_results(results)
    copy_accepted_clips(results)

    print("\n완료")
    print(f"출력 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()