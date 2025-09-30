# 실험 실행 가이드

## 완료된 작업 ✅

### 1. 모델 성능 심화 분석
- **제출 파일 통계**: 413개 이미지, 평균 72.4개 텍스트 박스
- **문제 케이스**: 19개 (4.6%) - 낮은/높은 탐지율
- **박스 크기 이상**: 655개 작은 박스, 14개 큰 박스
- **분석 파일**: `submission_analysis.png`, `checkpoint_analysis.png`, `error_analysis.png`

### 2. 오류 사례 분석
- **극단적 비율 박스**: 209개 (가로세로 비율 >20 또는 <0.1)
- **잠재적 노이즈**: 10x10 픽셀 박스들
- **대형 텍스트 영역**: 최대 118,038픽셀² 박스

### 3. 백본 모델 실험 계획
- **4개 백본 준비**: ResNet34, ResNet50, EfficientNet-B0, EfficientNet-B1
- **설정 파일 완성**: 각 백본별 최적화된 하이퍼파라미터
- **자동화 스크립트**: `run_backbone_experiments.py`

### 4. 하이퍼파라미터 실험 계획
- **4개 실험 영역**: 학습률, 배치/해상도, 정규화, 데이터 증강
- **5개 실험 설정**: 다양한 조합 준비
- **자동화 스크립트**: `run_hyperparameter_experiments.py`

## 실험 실행 방법 🚀

### 백본 모델 실험

#### Phase 1: ResNet 계열 (추천)
```bash
python run_backbone_experiments.py --phase 1
```
- ResNet34, ResNet50 실험
- 예상 시간: 10-16시간

#### Phase 2: EfficientNet 계열
```bash
python run_backbone_experiments.py --phase 2
```
- EfficientNet-B0, B1 실험
- 예상 시간: 10-16시간

#### 전체 백본 실험
```bash
python run_backbone_experiments.py --phase 3
```

### 하이퍼파라미터 실험

#### Phase 1: 학습률 최적화 (추천)
```bash
python run_hyperparameter_experiments.py --phase 1
```
- OneCycleLR, ReduceLROnPlateau 실험
- 예상 시간: 6-8시간

#### Phase 2: 배치 크기/해상도
```bash
python run_hyperparameter_experiments.py --phase 2
```
- 고해상도, 큰 배치 크기 실험
- 예상 시간: 6-8시간

#### 빠른 실험 (15 에포크)
```bash
python run_hyperparameter_experiments.py --phase 1 --quick
```

### 개별 실험 실행

#### 특정 백본 실험
```bash
# ResNet34 실험
python runners/train.py preset=experiment_resnet34

# EfficientNet-B0 실험
python runners/train.py preset=experiment_efficientnet_b0
```

#### 특정 하이퍼파라미터 실험
```bash
# OneCycleLR 실험
python runners/train.py preset=onecycle_lr_002

# 고해상도 실험
python runners/train.py preset=high_res_896
```

## 예상 성능 향상 🎯

### 백본 모델 실험
- **ResNet34**: F1 0.86-0.87 예상 (+1-2%)
- **ResNet50**: F1 0.87-0.88 예상 (+2-3%)
- **EfficientNet-B0**: F1 0.86-0.87 예상 (효율성 우수)
- **EfficientNet-B1**: F1 0.87-0.88 예상

### 하이퍼파라미터 실험
- **OneCycleLR**: 더 빠른 수렴, F1 +0.5-1.0% 예상
- **고해상도**: 더 세밀한 탐지, F1 +1-2% 예상
- **큰 배치**: 더 안정적인 훈련, F1 +0.5-1.0% 예상

## 추천 실행 순서 📋

### 1단계: 하이퍼파라미터 최적화 (우선)
```bash
python run_hyperparameter_experiments.py --phase 1
```
- 기존 ResNet18 기반으로 빠른 개선
- 시간 효율적 (6-8시간)

### 2단계: 백본 모델 업그레이드
```bash
python run_backbone_experiments.py --phase 1
```
- 더 강력한 백본으로 성능 향상
- 중간 시간 소요 (10-16시간)

### 3단계: 최적 조합 실험
최고 성능 하이퍼파라미터 + 최고 성능 백본 조합

## 모니터링 도구 📊

### 실시간 모니터링
```bash
# TensorBoard (포트 6006)
tensorboard --logdir outputs/

# 체크포인트 확인
ls -la outputs/*/checkpoints/
```

### 결과 분석
```bash
# 기존 분석 도구 재실행
python quick_analysis.py
python error_analysis.py

# 모델 비교
python compare_models.py
```

## 자원 관리 💾

### GPU 메모리 최적화
- **Mixed Precision**: 모든 실험에 `precision: 16` 적용
- **배치 크기 조정**: 메모리 부족 시 자동 감소
- **체크포인트 정리**: 디스크 공간 관리

### 실험 중단/재시작
```bash
# 특정 체크포인트부터 재시작
python runners/train.py preset=experiment_name resume_from_checkpoint=path/to/checkpoint.ckpt
```

## 성공 기준 ✨

### 단기 목표 (1-2주)
- **F1 Score > 0.87** (현재 0.851 대비 +2%)
- **안정적인 훈련** (수렴 성공률 >90%)

### 중기 목표 (2-4주)
- **F1 Score > 0.90** (+5% 이상 향상)
- **배포 가능한 모델** (효율성 + 성능)

현재 모든 실험 인프라가 준비되었으므로, 원하는 실험부터 시작하시면 됩니다! 🚀