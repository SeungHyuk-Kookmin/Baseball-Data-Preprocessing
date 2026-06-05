from pathlib import Path
from ultralytics import YOLO

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =========================
# 설정
# =========================
DATASET_DIR = PROJECT_ROOT / "data" / "batter_tracking" / "yolo_labeling"
DATA_YAML = DATASET_DIR / "data.yaml"

MODEL_NAME = "yolov8n.pt"        # 빠른 실험용
# MODEL_NAME = "yolov8s.pt"      # 정확도 조금 더 원하면 이걸로

EPOCHS = 80
IMG_SIZE = 640
BATCH_SIZE = 8

PROJECT_NAME = str(PROJECT_ROOT / "runs" / "yolo")
RUN_NAME = "batter_pitcher_yolov8n"


# =========================
# 경로 확인
# =========================
if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml을 찾을 수 없음: {DATA_YAML.resolve()}")

print(f"data.yaml: {DATA_YAML.resolve()}")


# =========================
# 모델 로드
# =========================
model = YOLO(MODEL_NAME)


# =========================
# 학습
# =========================
results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device="mps",          # 맥 M칩이면 mps, 안 되면 "cpu"로 바꿔
    workers=2,
    project=PROJECT_NAME,
    name=RUN_NAME,
    patience=20,
    save=True,
    plots=True,
    pretrained=True,
    verbose=True
)

print("학습 완료")
print(f"best.pt 위치: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
print(f"last.pt 위치: {PROJECT_NAME}/{RUN_NAME}/weights/last.pt")