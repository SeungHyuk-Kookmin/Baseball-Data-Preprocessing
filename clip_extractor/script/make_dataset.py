import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =========================
# 설정값
# =========================
INPUT_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "labeled_scene_clips"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "view_dataset"

CLASS_NAMES = [
    "0_atbat_view",
    "1_non_atbat_view",
]

# ambiguous는 학습에서 제외
IGNORE_CLASS_NAMES = [
    "2_ambiguous",
]

# 클립에서 뽑을 위치
FRAME_RATIOS = [0.20, 0.30, 0.40]

VAL_RATIO = 0.2
SEED = 42

# 기존 view_dataset 삭제 후 새로 만들지 여부
RESET_OUTPUT_DIR = True


def reset_output_dir():
    if OUTPUT_DIR.exists() and RESET_OUTPUT_DIR:
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val"]:
        for class_name in CLASS_NAMES:
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count


def extract_frame_at_index(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    return frame


def split_clips_by_class():
    random.seed(SEED)

    split_map = {
        "train": [],
        "val": [],
    }

    for class_name in CLASS_NAMES:
        class_dir = INPUT_DIR / class_name

        if not class_dir.exists():
            print(f"폴더 없음, 스킵: {class_dir}")
            continue

        video_files = sorted(list(class_dir.glob("*.mp4")))

        random.shuffle(video_files)

        val_count = int(len(video_files) * VAL_RATIO)

        val_files = video_files[:val_count]
        train_files = video_files[val_count:]

        for video_path in train_files:
            split_map["train"].append((class_name, video_path))

        for video_path in val_files:
            split_map["val"].append((class_name, video_path))

        print(
            f"{class_name}: "
            f"전체 {len(video_files)}개, "
            f"train {len(train_files)}개, "
            f"val {len(val_files)}개"
        )

    return split_map


def extract_frames_for_split(split_name, items):
    print(f"\n[{split_name}] 프레임 추출 시작: {len(items)} clips")

    for class_name, video_path in tqdm(items, desc=f"{split_name}"):
        frame_count = get_video_frame_count(video_path)

        if frame_count <= 0:
            print(f"프레임 수 읽기 실패, 스킵: {video_path}")
            continue

        clip_stem = video_path.stem

        for ratio in FRAME_RATIOS:
            frame_idx = int(frame_count * ratio)

            # 마지막 프레임 범위 초과 방지
            frame_idx = min(max(frame_idx, 0), frame_count - 1)

            frame = extract_frame_at_index(video_path, frame_idx)

            if frame is None:
                print(f"프레임 추출 실패: {video_path}, frame_idx={frame_idx}")
                continue

            ratio_percent = int(ratio * 100)

            output_name = f"{clip_stem}_r{ratio_percent:02d}.jpg"
            output_path = OUTPUT_DIR / split_name / class_name / output_name

            cv2.imwrite(str(output_path), frame)


def main():
    reset_output_dir()

    split_map = split_clips_by_class()

    extract_frames_for_split("train", split_map["train"])
    extract_frames_for_split("val", split_map["val"])

    print("\n완료")
    print(f"저장 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()