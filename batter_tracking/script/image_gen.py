from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =========================
# 설정
# =========================
CLIP_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "classified_atbat_clips"
DATASET_DIR = PROJECT_ROOT / "data" / "batter_tracking" / "yolo_auto"

IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "yolov8n.pt"   # 처음은 n으로 빠르게. 부족하면 yolov8s.pt
FRAME_INTERVAL = 15         # 30fps 기준 0.5초마다 1장
CONF_THRES = 0.35
IMG_SIZE = 640

PERSON_CLASS_ID_COCO = 0    # COCO에서 person은 0
TEMP_CLASS_ID = 0           # 우리가 저장할 임시 클래스 person_temp


# =========================
# 모델 로드
# =========================
model = YOLO(MODEL_PATH)


# =========================
# mp4 목록
# =========================
video_paths = sorted(list(CLIP_DIR.glob("*.mp4")))

if not video_paths:
    raise FileNotFoundError(f"mp4 파일이 없음: {CLIP_DIR}")


# =========================
# 프레임 추출 + person 자동 라벨링
# =========================
saved_images = 0
saved_labels = 0

for video_path in tqdm(video_paths, desc="Processing clips"):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARN] 영상 열기 실패: {video_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=video_path.name, leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            h, w = frame.shape[:2]

            stem = f"{video_path.stem}_f{frame_idx:06d}"
            image_path = IMAGE_DIR / f"{stem}.jpg"
            label_path = LABEL_DIR / f"{stem}.txt"

            # 이미지 저장
            cv2.imwrite(str(image_path), frame)
            saved_images += 1

            # YOLO person 검출
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                classes=[PERSON_CLASS_ID_COCO],
                verbose=False
            )

            lines = []

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for box in boxes:
                    # xyxy pixel 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # bbox sanity check
                    x1 = max(0, min(float(x1), w - 1))
                    y1 = max(0, min(float(y1), h - 1))
                    x2 = max(0, min(float(x2), w - 1))
                    y2 = max(0, min(float(y2), h - 1))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    if box_w <= 2 or box_h <= 2:
                        continue

                    # YOLO normalized xywh
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = box_w / w
                    bh = box_h / h

                    line = f"{TEMP_CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                    lines.append(line)

            # 라벨 파일 저장
            # person이 없으면 빈 txt 저장. 라벨링 툴 호환 위해 유지.
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            saved_labels += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()


print("완료")
print(f"저장 이미지 수: {saved_images}")
print(f"저장 라벨 txt 수: {saved_labels}")
print(f"이미지 폴더: {IMAGE_DIR}")
print(f"라벨 폴더: {LABEL_DIR}")