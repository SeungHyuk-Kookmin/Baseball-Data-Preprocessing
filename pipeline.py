"""
2단계 야구 데이터 전처리 파이프라인

1단계: 클립 분할 + 타석 분류
  - data/raw/input의 영상을 장면 전환으로 분할
  - 각 클립을 분류해서 타석 시퀀스만 추출
  - 결과를 data/clip_extractor/classified_atbat_clips에 저장

2단계: YOLO 추론
  - classified_atbat_clips의 타석 영상들을 data/raw/videos에 복사
  - data/raw/videos의 모든 영상에 YOLO 추론 실행
  - 결과를 outputs/batter_tracking에 저장
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def main():
    print("=" * 70)
    print("STAGE 1: 클립 분할 + 타석 분류")
    print("=" * 70)

    # cut_editor: 영상을 장면 전환으로 분할
    print("\n[1-1] 영상 장면 분할 (cut_editor)")
    print("-" * 70)
    from clip_extractor.core import cut_editor
    cut_editor.main()

    # view_classifier: scene_clips를 분류해서 타석만 추출
    print("\n[1-2] 타석 시퀀스 분류 (view_classifier)")
    print("-" * 70)
    from clip_extractor.core import view_classifier
    view_classifier.main()

    print("\n" + "=" * 70)
    print("STAGE 2: YOLO 추론")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).parent
    SOURCE_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "classified_atbat_clips"
    TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "batter_tracking" / "clips_result"

    print("\n[2-1] 타석 영상 복사")
    print("-" * 70)
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 파일 삭제 후 복사
    if TARGET_DIR.exists() and list(TARGET_DIR.glob("*")):
        print(f"기존 파일 삭제: {TARGET_DIR}")
        for file in TARGET_DIR.glob("*"):
            if file.is_file():
                file.unlink()

    video_files = list(SOURCE_DIR.glob("*.mp4"))
    print(f"복사할 영상 수: {len(video_files)}")

    for src_file in tqdm(video_files, desc="Copying videos", unit="file"):
        dst_file = TARGET_DIR / src_file.name
        shutil.copy2(src_file, dst_file)

    print(f"✓ {len(video_files)}개 영상 복사 완료")

    print("\n[2-2] YOLO 추론 실행 (predict_video)")
    print("-" * 70)

    from batter_tracking.core.predict_video import run_inference

    run_inference(video_dir=TARGET_DIR, output_dir=OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("✓ 파이프라인 완료!")
    print("=" * 70)
    print(f"최종 결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
