# Baseball Data Preprocessing

야구 데이터 전처리를 위한 파이프라인입니다. 하이라이트 영상에서 scene clip을 추출하고, 타석 화면을 분류한 뒤, 타석 클립에 대해 YOLO 객체 탐지를 수행합니다.

## 개요

이 프로젝트는 두 단계로 구성됩니다.

1. `clip_extractor`
   - 입력 영상(`data/raw/input/highlight.mp4`)을 장면 단위로 분할합니다.
   - 분할된 scene clip을 수동으로 라벨링합니다.
   - 라벨링된 클립에서 프레임 이미지를 추출해 뷰 분류 데이터셋을 만듭니다.
   - ResNet18 기반 분류 모델을 학습합니다.
   - 학습된 모델을 이용해 타석 장면만 골라 `data/clip_extractor/classified_atbat_clips`에 저장합니다.

2. `batter_tracking`
   - 검출된 타석 클립을 `data/raw/videos`로 복사합니다.
   - YOLO 모델로 각 영상에 대해 객체 탐지를 수행합니다.
   - 결과를 `outputs/batter_tracking/clips_result`에 저장합니다.

## 주요 경로

- 입력 영상: `data/raw/input/highlight.mp4`
- scene clip: `data/clip_extractor/scene_clips`
- 라벨링된 scene clip: `data/clip_extractor/labeled_scene_clips`
- 학습용 프레임: `data/clip_extractor/view_dataset`
- 분류된 타석 클립: `data/clip_extractor/classified_atbat_clips`
- 분류 결과 JSON: `data/clip_extractor/classified_atbat_clips/classified_atbat_results.json`
- 분류 모델: `runs/view_classifier/best_resnet18_atbat_view_cpu_light.pt`
- YOLO weight: `runs/yolo/batter_pitcher_yolov8n/weights/best.pt` 또는 `runs/yolo/batter_pitcher_yolov8n/weight/best.pt`
- YOLO 출력: `outputs/batter_tracking/clips_result`

## 설치

Python 환경에서 아래 패키지를 설치합니다.

```bash
pip install opencv-python torch torchvision tqdm imageio-ffmpeg ultralytics
```

## 실행

### 전체 파이프라인 실행

```bash
python pipeline.py
```

`pipeline.py`는 scene clip 추출, 타석 클립 분류, 타석 영상 복사, YOLO 추론을 순차적으로 실행합니다.

### 1) scene clip 생성

```bash
python clip_extractor/core/cut_editor.py
```

장면 전환 시점을 탐지해 `data/clip_extractor/scene_clips`에 clip을 저장합니다.

### 2) 클립 라벨링

생성된 `data/clip_extractor/scene_clips/*.mp4`를 확인한 뒤 아래 폴더로 분류합니다.

```text
data/clip_extractor/labeled_scene_clips/
├── 0_atbat_view/      # 타석 화면
└── 1_non_atbat_view/  # 타석 화면이 아닌 클립
```

### 3) 학습 데이터셋 생성

```bash
python clip_extractor/script/make_dataset.py
```

라벨링된 클립에서 20%, 30%, 40% 위치의 프레임을 추출해 `data/clip_extractor/view_dataset/train` 및 `val`에 저장합니다.

### 4) 뷰 분류 모델 학습

```bash
python clip_extractor/script/train.py
```

`data/clip_extractor/view_dataset`을 사용해 ResNet18 분류 모델을 학습합니다.

### 5) 타석 클립 분류

```bash
python clip_extractor/core/view_classifier.py
```

`data/clip_extractor/scene_clips`에서 타석 클립을 찾아 `data/clip_extractor/classified_atbat_clips`에 저장합니다.

### 6) YOLO 추론

`pipeline.py`을 사용하면 자동으로 실행되지만, 개별 실행도 가능합니다.

```bash
python batter_tracking/core/predict_video.py
```

`data/clip_extractor/classified_atbat_clips`에 있는 영상을 `outputs/batter_tracking/clips_result`로 탐지 결과를 저장합니다.

## 노트

- `clip_extractor/core/cut_editor.py`, `clip_extractor/script/make_dataset.py`, `clip_extractor/core/view_classifier.py`는 기본값으로 출력 디렉터리를 초기화합니다.
- `clip_extractor/core/view_classifier.py`는 `runs/view_classifier/best_resnet18_atbat_view_cpu_light.pt` 모델 파일을 필요로 합니다.
- `clip_extractor/script/make_dataset.py`는 `data/clip_extractor/labeled_scene_clips/0_atbat_view` 및 `1_non_atbat_view`에 mp4 파일이 있어야 합니다.
- `batter_tracking/core/predict_video.py`는 `runs/yolo/batter_pitcher_yolov8n/weights/best.pt` 또는 `runs/yolo/batter_pitcher_yolov8n/weight/best.pt`를 찾습니다.
- YOLO 모델이 없으면 infer 단계가 실패하므로 먼저 학습된 YOLO weight를 준비해야 합니다.
