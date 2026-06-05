from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =========================
# 설정
# =========================
YOLO_RUN_DIR = PROJECT_ROOT / "runs" / "yolo" / "batter_pitcher_yolov8n"
MODEL_PATH_CANDIDATES = [
    YOLO_RUN_DIR / "weights" / "best.pt",
    YOLO_RUN_DIR / "weight" / "best.pt",
]

DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "classified_atbat_clips"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "batter_tracking" / "clips_result"

CONF_THRES = 0.25
IMG_SIZE = 640
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def find_best_yolo_model():
    for candidate in MODEL_PATH_CANDIDATES:
        if candidate.exists():
            return candidate

    if YOLO_RUN_DIR.exists():
        for candidate in YOLO_RUN_DIR.rglob("best.pt"):
            return candidate

    raise FileNotFoundError(
        f"YOLO best.pt를 찾을 수 없음. 검색 경로 후보: {[str(p) for p in MODEL_PATH_CANDIDATES]}"
    )


def collect_video_paths(video_dir):
    video_dir = Path(video_dir)
    video_paths = []
    for ext in VIDEO_EXTS:
        video_paths.extend(video_dir.glob(f"*{ext}"))
    return sorted(video_paths)


def run_inference(video_dir=None, output_dir=None):
    video_dir = Path(video_dir) if video_dir is not None else DEFAULT_VIDEO_DIR
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    if not video_dir.exists():
        raise FileNotFoundError(f"영상 폴더 없음: {video_dir}")

    video_paths = collect_video_paths(video_dir)
    if not video_paths:
        raise FileNotFoundError(f"폴더 안에 영상 없음: {video_dir}")

    print(f"인퍼런스할 영상 수: {len(video_paths)}")
    print(f"모델: {find_best_yolo_model()}")
    print(f"결과 저장: {output_dir}")

    model = YOLO(str(find_best_yolo_model()))
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in tqdm(video_paths, desc="YOLO Inference", unit="video"):
        run_name = video_path.stem

        model.predict(
            source=str(video_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
            verbose=False,
        )

    print("추론 완료")
    print(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    run_inference()
