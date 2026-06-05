from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =========================
# 설정값
# =========================
INPUT_DIR = PROJECT_ROOT / "data" / "clip_extractor" / "classified_atbat_clips"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "batter_tracking"

YOLO_MODEL = "yolov8n.pt"                    # 빠른 모델. 정확도 필요하면 yolov8s.pt
PERSON_CLASS_ID = 0                          # COCO person class

CONF_THRES = 0.35                            # person 검출 confidence
IOU_KEEP_THRES = 0.05                        # 이전 bbox와 최소 IoU
SMOOTH_WINDOW = 5                            # bbox 이동평균 프레임 수

CROP_MARGIN = 0.20                           # 타자 crop 여백 비율
MIN_BOX_AREA_RATIO = 0.01                    # 너무 작은 사람 제거
MAX_MISSING_FRAMES = 10                      # 검출 실패 허용 프레임 수

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

# 맥/OpenCV에서 mp4 저장이 안 되는 경우가 많아서 1차 결과는 avi + MJPG로 저장
OUTPUT_VIDEO_EXT = ".avi"
OUTPUT_FOURCC = "MJPG"

# crop 영상은 프레임마다 bbox 크기가 달라지므로 고정 크기로 저장
CROP_SIZE = 512


# =========================
# 타자 ROI 설정
# =========================
# "left", "right", "center", "full" 중 선택
# 타자가 화면 오른쪽에 주로 있으면 "right"
# 타자가 화면 왼쪽에 주로 있으면 "left"
BATTER_ROI_MODE = "center"

# center 기준: 화면 x좌표 25% ~ 75% 영역만 후보로 사용
ROI_X_MIN_RATIO = 0.30
ROI_X_MAX_RATIO = 0.70

# y축 기준: 너무 위쪽 관중/전광판 사람 제거용
ROI_Y_MIN_RATIO = 0.15
ROI_Y_MAX_RATIO = 1.00

# ROI 후보가 하나도 없을 때 전체 후보를 쓸지 여부
# False면 ROI 밖 사람은 절대 선택하지 않음
# True면 ROI 안에 후보가 없을 때만 전체 후보로 fallback
ALLOW_FALLBACK_FULL_FRAME = False


# =========================
# 추적 점수 설정
# =========================
# 이전 bbox와 현재 후보 bbox의 유사도를 계산할 때 사용
TRACK_IOU_WEIGHT = 0.60
TRACK_DIST_WEIGHT = 0.30
TRACK_AREA_WEIGHT = 0.10

# 중심점 거리 점수 감쇠 정도
# 값이 작을수록 멀리 떨어진 후보를 강하게 배제
DIST_SCALE = 200.0

# 최종 tracking score가 이 값보다 낮으면 선택하지 않음
TRACK_SCORE_THRES = 0.15


# =========================
# 유틸 함수
# =========================
def compute_iou(box_a, box_b):
    """
    box: [x1, y1, x2, y2]
    """
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(1, (x2 - x1) * (y2 - y1))


def clip_box(box, width, height):
    x1, y1, x2, y2 = box

    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))

    # 혹시 좌표가 뒤집히는 경우 방지
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return [x1, y1, x2, y2]


def expand_box(box, width, height, margin=0.2):
    """
    bbox 주변에 여백 추가
    """
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    mx = bw * margin
    my = bh * margin

    new_box = [
        x1 - mx,
        y1 - my,
        x2 + mx,
        y2 + my,
    ]

    return clip_box(new_box, width, height)


def smooth_box(box_queue):
    """
    최근 bbox들의 이동평균
    """
    arr = np.array(box_queue, dtype=np.float32)
    mean_box = arr.mean(axis=0)
    return mean_box.tolist()


def get_person_boxes(result, frame_width, frame_height):
    """
    YOLO 결과에서 person bbox만 추출
    """
    boxes = []

    if result.boxes is None:
        return boxes

    for b in result.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())

        if cls_id != PERSON_CLASS_ID:
            continue

        if conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        x1, y1, x2, y2 = clip_box([x1, y1, x2, y2], frame_width, frame_height)

        area = (x2 - x1) * (y2 - y1)
        frame_area = frame_width * frame_height
        area_ratio = area / frame_area

        if area_ratio < MIN_BOX_AREA_RATIO:
            continue

        boxes.append({
            "box": [x1, y1, x2, y2],
            "conf": conf,
            "area": area,
            "area_ratio": area_ratio,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })

    return boxes


def filter_boxes_by_roi(person_boxes, frame_width, frame_height):
    """
    타자가 있을 법한 화면 영역 안의 person bbox만 남김.
    bbox 중심점 기준으로 필터링.
    """
    if BATTER_ROI_MODE == "full":
        return person_boxes

    filtered = []

    for item in person_boxes:
        cx = item["cx"]
        cy = item["cy"]

        x_ratio = cx / frame_width
        y_ratio = cy / frame_height

        if x_ratio < ROI_X_MIN_RATIO:
            continue

        if x_ratio > ROI_X_MAX_RATIO:
            continue

        if y_ratio < ROI_Y_MIN_RATIO:
            continue

        if y_ratio > ROI_Y_MAX_RATIO:
            continue

        filtered.append(item)

    return filtered


def choose_initial_batter(person_boxes, frame_width, frame_height):
    """
    초기 타자 선택 로직.
    ROI 안에 들어온 사람 중 bbox 면적이 큰 사람을 우선 선택.
    """
    if len(person_boxes) == 0:
        return None

    # bbox 면적이 큰 순서로 선택
    person_boxes = sorted(person_boxes, key=lambda x: x["area"], reverse=True)
    selected = person_boxes[0]
    selected["iou_with_prev"] = None
    selected["track_score"] = None
    return selected


def choose_by_tracking_score(person_boxes, prev_box):
    """
    이전 bbox와 현재 후보 bbox의 유사도를 기반으로 타자 선택.
    IoU만 쓰면 다른 사람으로 튈 수 있어서 중심점 거리와 bbox 크기 유사도도 같이 사용.
    """
    if len(person_boxes) == 0 or prev_box is None:
        return None

    prev_center = box_center(prev_box)
    prev_area = box_area(prev_box)

    best = None
    best_score = -1.0

    for item in person_boxes:
        box = item["box"]

        iou = compute_iou(box, prev_box)

        center = box_center(box)
        dist = np.linalg.norm(center - prev_center)

        # 가까울수록 1에 가까움, 멀수록 0에 가까움
        dist_score = float(np.exp(-dist / DIST_SCALE))

        area = box_area(box)
        area_score = min(area, prev_area) / max(area, prev_area)

        score = (
            TRACK_IOU_WEIGHT * iou
            + TRACK_DIST_WEIGHT * dist_score
            + TRACK_AREA_WEIGHT * area_score
        )

        if score > best_score:
            best_score = score
            best = item
            best["iou_with_prev"] = iou
            best["track_score"] = score
            best["dist_score"] = dist_score
            best["area_score"] = area_score

    if best_score < TRACK_SCORE_THRES:
        return None

    return best


def draw_roi(frame):
    """
    확인용 영상에 ROI 영역을 파란 박스로 표시.
    """
    h, w = frame.shape[:2]

    if BATTER_ROI_MODE == "full":
        return frame

    x1 = int(w * ROI_X_MIN_RATIO)
    x2 = int(w * ROI_X_MAX_RATIO)
    y1 = int(h * ROI_Y_MIN_RATIO)
    y2 = int(h * ROI_Y_MAX_RATIO)

    vis = frame.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        vis,
        f"ROI: {BATTER_ROI_MODE}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return vis


def draw_box(frame, box, label):
    x1, y1, x2, y2 = map(int, box)
    vis = frame.copy()

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        vis,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return vis


def draw_candidate_boxes(frame, person_boxes):
    """
    디버깅용: ROI 안의 person 후보들을 노란색으로 표시.
    """
    vis = frame.copy()

    for idx, item in enumerate(person_boxes):
        x1, y1, x2, y2 = item["box"]
        conf = item["conf"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(
            vis,
            f"cand{idx} {conf:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


def make_video_writer(path, fourcc_text, fps, size):
    """
    OpenCV VideoWriter 생성 + 실패 여부 확인.
    실패하면 None 반환.
    """
    fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)

    if not writer.isOpened():
        print(f"[ERROR] VideoWriter 열기 실패: {path}")
        print(f"        fourcc={fourcc_text}, fps={fps}, size={size}")
        return None

    return writer


# =========================
# 영상 1개 처리
# =========================
def process_video(video_path, model):
    video_name = video_path.stem
    out_dir = OUTPUT_DIR / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox_vis_path = out_dir / f"{video_name}_batter_bbox_vis{OUTPUT_VIDEO_EXT}"
    crop_path = out_dir / f"{video_name}_batter_crop{OUTPUT_VIDEO_EXT}"
    csv_path = out_dir / f"{video_name}_batter_boxes.csv"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없음: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fps가 0, 음수, NaN이면 기본값 처리
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    print(f"\n[INFO] {video_path.name}")
    print(f"  fps         : {fps}")
    print(f"  total_frames: {total_frames}")
    print(f"  width       : {width}")
    print(f"  height      : {height}")
    print(f"  ROI mode    : {BATTER_ROI_MODE}")
    print(f"  ROI x range : {ROI_X_MIN_RATIO} ~ {ROI_X_MAX_RATIO}")
    print(f"  ROI y range : {ROI_Y_MIN_RATIO} ~ {ROI_Y_MAX_RATIO}")

    # 영상 크기 못 읽으면 VideoWriter가 거의 무조건 실패함
    if width <= 0 or height <= 0:
        print(f"[ERROR] 영상 크기를 읽지 못함: {video_path}")
        cap.release()
        return

    bbox_writer = make_video_writer(
        bbox_vis_path,
        OUTPUT_FOURCC,
        fps,
        (width, height),
    )

    if bbox_writer is None:
        cap.release()
        return

    crop_writer = make_video_writer(
        crop_path,
        OUTPUT_FOURCC,
        fps,
        (CROP_SIZE, CROP_SIZE),
    )

    if crop_writer is None:
        cap.release()
        bbox_writer.release()
        return

    prev_box = None
    bbox_queue = deque(maxlen=SMOOTH_WINDOW)
    missing_count = 0
    records = []

    pbar_total = total_frames if total_frames > 0 else None
    pbar = tqdm(total=pbar_total, desc=f"Processing {video_path.name}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model.predict(
            frame,
            conf=CONF_THRES,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]

        all_person_boxes = get_person_boxes(result, width, height)
        roi_person_boxes = filter_boxes_by_roi(all_person_boxes, width, height)

        # ROI 안 후보가 없을 때 전체 프레임 후보 사용 여부
        if len(roi_person_boxes) > 0:
            person_boxes = roi_person_boxes
            roi_used = 1
            fallback_used = 0
        else:
            if ALLOW_FALLBACK_FULL_FRAME:
                person_boxes = all_person_boxes
                roi_used = 0
                fallback_used = 1
            else:
                person_boxes = []
                roi_used = 1
                fallback_used = 0

        selected = None
        selected_reason = ""

        if prev_box is None:
            selected = choose_initial_batter(person_boxes, width, height)
            selected_reason = "initial_roi_largest"
        else:
            selected = choose_by_tracking_score(person_boxes, prev_box)
            selected_reason = "roi_tracking_score"

        if selected is not None:
            raw_box = selected["box"]
            bbox_queue.append(raw_box)

            smoothed_box = smooth_box(bbox_queue)
            smoothed_box = clip_box(smoothed_box, width, height)

            prev_box = smoothed_box
            missing_count = 0

            conf = selected.get("conf", None)
            iou_prev = selected.get("iou_with_prev", None)
            track_score = selected.get("track_score", None)
            dist_score = selected.get("dist_score", None)
            area_score = selected.get("area_score", None)

            # 시각화 영상
            vis_frame = frame.copy()
            vis_frame = draw_roi(vis_frame)
            vis_frame = draw_candidate_boxes(vis_frame, person_boxes)

            if track_score is not None:
                label = f"batter conf={conf:.2f} score={track_score:.2f}"
            else:
                label = f"batter conf={conf:.2f}" if conf is not None else "batter"

            vis_frame = draw_box(vis_frame, smoothed_box, label)

            # crop 영상
            crop_box = expand_box(smoothed_box, width, height, CROP_MARGIN)
            cx1, cy1, cx2, cy2 = crop_box
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                crop_resized = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            else:
                crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

            records.append({
                "frame": frame_idx,
                "detected": 1,
                "x1": smoothed_box[0],
                "y1": smoothed_box[1],
                "x2": smoothed_box[2],
                "y2": smoothed_box[3],
                "raw_x1": raw_box[0],
                "raw_y1": raw_box[1],
                "raw_x2": raw_box[2],
                "raw_y2": raw_box[3],
                "conf": conf,
                "iou_with_prev": iou_prev,
                "track_score": track_score,
                "dist_score": dist_score,
                "area_score": area_score,
                "reason": selected_reason,
                "all_person_count": len(all_person_boxes),
                "roi_person_count": len(roi_person_boxes),
                "roi_used": roi_used,
                "fallback_used": fallback_used,
                "crop_x1": cx1,
                "crop_y1": cy1,
                "crop_x2": cx2,
                "crop_y2": cy2,
            })

        else:
            missing_count += 1

            # 잠깐 검출 실패하면 이전 bbox 유지
            if prev_box is not None and missing_count <= MAX_MISSING_FRAMES:
                smoothed_box = prev_box

                vis_frame = frame.copy()
                vis_frame = draw_roi(vis_frame)
                vis_frame = draw_candidate_boxes(vis_frame, person_boxes)
                vis_frame = draw_box(
                    vis_frame,
                    smoothed_box,
                    f"batter missing_keep {missing_count}",
                )

                crop_box = expand_box(smoothed_box, width, height, CROP_MARGIN)
                cx1, cy1, cx2, cy2 = crop_box
                crop = frame[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    crop_resized = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
                else:
                    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

                records.append({
                    "frame": frame_idx,
                    "detected": 0,
                    "x1": smoothed_box[0],
                    "y1": smoothed_box[1],
                    "x2": smoothed_box[2],
                    "y2": smoothed_box[3],
                    "raw_x1": None,
                    "raw_y1": None,
                    "raw_x2": None,
                    "raw_y2": None,
                    "conf": None,
                    "iou_with_prev": None,
                    "track_score": None,
                    "dist_score": None,
                    "area_score": None,
                    "reason": "missing_keep_prev",
                    "all_person_count": len(all_person_boxes),
                    "roi_person_count": len(roi_person_boxes),
                    "roi_used": roi_used,
                    "fallback_used": fallback_used,
                    "crop_x1": cx1,
                    "crop_y1": cy1,
                    "crop_x2": cx2,
                    "crop_y2": cy2,
                })

            else:
                vis_frame = frame.copy()
                vis_frame = draw_roi(vis_frame)
                vis_frame = draw_candidate_boxes(vis_frame, person_boxes)

                crop_resized = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

                records.append({
                    "frame": frame_idx,
                    "detected": 0,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                    "raw_x1": None,
                    "raw_y1": None,
                    "raw_x2": None,
                    "raw_y2": None,
                    "conf": None,
                    "iou_with_prev": None,
                    "track_score": None,
                    "dist_score": None,
                    "area_score": None,
                    "reason": "missing_no_box",
                    "all_person_count": len(all_person_boxes),
                    "roi_person_count": len(roi_person_boxes),
                    "roi_used": roi_used,
                    "fallback_used": fallback_used,
                    "crop_x1": None,
                    "crop_y1": None,
                    "crop_x2": None,
                    "crop_y2": None,
                })

        bbox_writer.write(vis_frame)
        crop_writer.write(crop_resized)

        frame_idx += 1
        pbar.update(1)

    pbar.close()

    cap.release()
    bbox_writer.release()
    crop_writer.release()

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] {video_path.name}")
    print(f"  written frames: {frame_idx}")
    print(f"  bbox vis      : {bbox_vis_path}")
    print(f"  crop          : {crop_path}")
    print(f"  csv           : {csv_path}")

    if frame_idx == 0:
        print("[WARN] 읽힌 프레임이 0개임. 입력 영상 코덱 문제일 수 있음.")


# =========================
# 메인
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_paths = []
    for ext in VIDEO_EXTS:
        video_paths.extend(INPUT_DIR.glob(f"*{ext}"))

    video_paths = sorted(video_paths)

    if len(video_paths) == 0:
        print(f"[ERROR] 입력 영상이 없음: {INPUT_DIR}")
        return

    print(f"총 {len(video_paths)}개 영상 처리 시작")
    print(f"입력 폴더: {INPUT_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"ROI 모드: {BATTER_ROI_MODE}")

    model = YOLO(YOLO_MODEL)

    for video_path in video_paths:
        process_video(video_path, model)

    print("\n전체 처리 완료")


if __name__ == "__main__":
    main()