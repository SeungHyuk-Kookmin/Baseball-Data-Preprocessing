import os
import shutil
import subprocess
from pathlib import Path

import cv2
from tqdm import tqdm
import imageio_ffmpeg

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =========================
# 설정값
# =========================
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "input" / "highlight.mp4"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "scene_clips"

# True면 기존 scene_clips 폴더 내용을 지우고 새로 저장함
RESET_OUTPUT_DIR = False

# 몇 프레임마다 장면 전환을 검사할지
# 1이면 모든 프레임, 5면 5프레임마다 한 번
FRAME_INTERVAL = 5

# 장면 전환 threshold
# 낮을수록 많이 자르고, 높을수록 확실한 전환만 자름
SCENE_CHANGE_THRESHOLD = 0.2

# 너무 짧은 scene 제거
MIN_SCENE_SECONDS = 1.0

# 너무 긴 scene 강제 분할 옵션
ENABLE_MAX_SCENE_SPLIT = False
MAX_SCENE_SECONDS = 20.0

# =========================
# 컷 보정 설정
# =========================

# 장면 전환 시점은 이미 다음 장면 첫 프레임일 수 있음.
# 그래서 이전 clip의 끝을 살짝 앞으로 당김.
# None이면 FRAME_INTERVAL / fps 만큼 자동 보정함.
# 30fps, FRAME_INTERVAL=5면 약 0.166초 보정.
CUT_BACK_SECONDS = 0.1

# True면 ffmpeg 재인코딩으로 더 정확하게 자름.
# False면 -c copy로 빠르게 자르지만, keyframe 문제로 컷이 부정확할 수 있음.
REENCODE_FOR_ACCURATE_CUT = True


def prepare_output_dir():
    output_dir = Path(OUTPUT_DIR)

    if output_dir.exists() and RESET_OUTPUT_DIR:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def run_ffmpeg_cut(video_path, start_time, end_time, output_path):
    """
    imageio-ffmpeg가 제공하는 ffmpeg 실행파일로 영상 일부 구간을 잘라 저장함.
    REENCODE_FOR_ACCURATE_CUT=True면 재인코딩으로 정확하게 자름.
    """
    duration = max(0.1, end_time - start_time)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if REENCODE_FOR_ACCURATE_CUT:
        # 정확한 컷팅용: 재인코딩
        command = [
            ffmpeg_exe,
            "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "veryfast",
            "-crf", "23",
            output_path,
        ]
    else:
        # 빠른 컷팅용: 복사
        # 단, keyframe 기준으로 잘려서 앞뒤 프레임이 섞일 수 있음.
        command = [
            ffmpeg_exe,
            "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            output_path,
        ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if result.returncode != 0:
        print(f"ffmpeg 컷팅 실패: {output_path}")


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없음: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0:
        raise ValueError("FPS 정보를 읽을 수 없음.")

    duration = total_frames / fps

    return fps, total_frames, duration


def calc_frame_hist(frame):
    """
    프레임의 HSV 히스토그램을 계산함.
    장면 전환 비교용으로 프레임을 작게 줄여서 계산함.
    """
    small = cv2.resize(frame, (320, 180))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(hist, hist)

    return hist


def is_scene_change(prev_hist, curr_hist, threshold):
    """
    이전 프레임 히스토그램과 현재 프레임 히스토그램 차이로 장면 전환 판단.
    """
    similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    diff = 1.0 - similarity

    return diff > threshold, diff


def split_long_segments(segments):
    """
    너무 긴 scene을 일정 길이로 강제 분할함.
    기본은 사용 안 함.
    """
    if not ENABLE_MAX_SCENE_SPLIT:
        return segments

    new_segments = []

    for start, end in segments:
        if end - start <= MAX_SCENE_SECONDS:
            new_segments.append((start, end))
            continue

        current = start

        while current < end:
            next_end = min(current + MAX_SCENE_SECONDS, end)
            new_segments.append((current, next_end))
            current = next_end

    return new_segments


def make_segments_from_boundaries(boundaries, fps):
    """
    장면 전환 boundary를 실제 저장할 segment로 변환함.

    핵심:
    boundaries[i + 1]은 다음 장면의 시작 시점일 수 있으므로,
    마지막 segment가 아니라면 end를 살짝 앞으로 당긴다.
    """
    segments = []

    if CUT_BACK_SECONDS is None:
        cut_back = FRAME_INTERVAL / fps
    else:
        cut_back = CUT_BACK_SECONDS

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # 마지막 영상 끝이 아니라면 다음 장면 첫 프레임이 들어가지 않도록 end 보정
        if i < len(boundaries) - 2:
            end = max(start, end - cut_back)

        if end - start >= MIN_SCENE_SECONDS:
            segments.append((start, end))

    return segments


def analyze_scene_cuts(video_path):
    """
    영상에서 장면 전환 지점을 찾고 scene segment 리스트를 반환함.
    JSON 저장 없이 매번 새로 분석함.
    """
    fps, total_frames, video_duration = get_video_info(video_path)

    print(f"영상 경로: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"총 프레임: {total_frames}")
    print(f"영상 길이: {video_duration:.2f}초")
    print("-" * 50)

    cap = cv2.VideoCapture(video_path)

    scene_cut_times = []
    prev_hist = None
    frame_idx = 0

    pbar = tqdm(
        total=total_frames,
        desc="장면 전환 분석 중",
        unit="frame",
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_time = frame_idx / fps

        pbar.update(1)
        pbar.set_postfix(
            {
                "time": f"{current_time:.1f}s",
                "cuts": len(scene_cut_times),
            }
        )

        if frame_idx % FRAME_INTERVAL == 0:
            curr_hist = calc_frame_hist(frame)

            if prev_hist is not None:
                changed, diff = is_scene_change(
                    prev_hist,
                    curr_hist,
                    SCENE_CHANGE_THRESHOLD,
                )

                if changed:
                    scene_cut_times.append(current_time)

            prev_hist = curr_hist

        frame_idx += 1

    pbar.close()
    cap.release()

    print("-" * 50)
    print(f"감지된 장면 전환 수: {len(scene_cut_times)}")

    boundaries = [0.0] + scene_cut_times + [video_duration]

    segments = make_segments_from_boundaries(boundaries, fps)
    segments = split_long_segments(segments)

    print(f"최종 scene segment 수: {len(segments)}")
    print("-" * 50)

    return segments


def cut_scene_segments(video_path, segments):
    if not segments:
        print("저장할 scene segment가 없음.")
        return

    print(f"scene clip 저장 시작: {len(segments)}개")
    print("-" * 50)

    for idx, (start, end) in enumerate(segments, start=1):
        output_path = os.path.join(
            OUTPUT_DIR,
            f"scene_{idx:04d}_{start:.2f}_{end:.2f}.mp4",
        )

        print(
            f"[{idx:04d}] "
            f"{start:.2f}s ~ {end:.2f}s "
            f"저장: {output_path}"
        )

        run_ffmpeg_cut(
            video_path=video_path,
            start_time=start,
            end_time=end,
            output_path=output_path,
        )

    print("-" * 50)
    print("scene clip 저장 완료")


def main():
    video_path = str(Path(VIDEO_PATH))

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일이 없음: {video_path}")

    prepare_output_dir()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"사용 ffmpeg: {ffmpeg_exe}")
    print("-" * 50)

    print(f"FRAME_INTERVAL: {FRAME_INTERVAL}")
    print(f"SCENE_CHANGE_THRESHOLD: {SCENE_CHANGE_THRESHOLD}")
    print(f"REENCODE_FOR_ACCURATE_CUT: {REENCODE_FOR_ACCURATE_CUT}")
    print(f"CUT_BACK_SECONDS: {CUT_BACK_SECONDS}")
    print("-" * 50)

    segments = analyze_scene_cuts(video_path)
    cut_scene_segments(video_path, segments)


if __name__ == "__main__":
    main()