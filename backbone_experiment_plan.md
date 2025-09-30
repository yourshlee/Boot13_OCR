# 백본 모델 실험 계획

## 현재 상태
- **현재 백본**: ResNet18 (11.7M parameters)
- **현재 성능**: F1 Score 0.851, Recall 0.849, Precision 0.859

## 실험 목표
1. **더 강력한 백본 모델**로 성능 향상 (F1 > 0.87 목표)
2. **다양한 아키텍처** 비교 분석
3. **계산 효율성** vs **성능** 트레이드오프 분석

## 실험 백본 모델 후보

### 1. CNN 기반 백본 (ResNet 계열)
- **ResNet34** - 더 깊은 ResNet (21.8M parameters)
- **ResNet50** - 표준 ResNet (25.6M parameters)
- **ResNet101** - 매우 깊은 ResNet (44.5M parameters)

### 2. 효율적인 CNN 백본
- **EfficientNet-B0** - 효율적인 아키텍처 (5.3M parameters)
- **EfficientNet-B1** - 더 큰 EfficientNet (7.8M parameters)
- **EfficientNet-B2** - 중간 크기 (9.1M parameters)

### 3. 최신 CNN 아키텍처
- **RegNetX-002** - Facebook AI 최신 아키텍처 (2.7M parameters)
- **RegNetY-002** - RegNet 변형 (3.2M parameters)
- **ConvNeXt-Tiny** - Vision Transformer에서 영감을 받은 CNN (28.6M parameters)

### 4. Vision Transformer 계열
- **DeiT-Tiny** - 작은 Vision Transformer (5.7M parameters)
- **DeiT-Small** - 중간 ViT (22.1M parameters)
- **Swin-Tiny** - 윈도우 기반 Transformer (28.3M parameters)

## 실험 우선순위

### Phase 1: 안전한 개선 (ResNet 계열)
1. **ResNet34** - ResNet18의 자연스러운 업그레이드
2. **ResNet50** - 표준적이고 안정적인 선택

### Phase 2: 효율성 탐구 (EfficientNet)
3. **EfficientNet-B0** - 매개변수는 적지만 성능이 좋음
4. **EfficientNet-B1** - 균형잡힌 선택

### Phase 3: 최신 아키텍처
5. **ConvNeXt-Tiny** - 최신 CNN 기술
6. **RegNetY-002** - 효율적인 최신 아키텍처

### Phase 4: Vision Transformer (실험적)
7. **DeiT-Tiny** - Transformer 아키텍처 탐구

## 실험 설정

### 공통 하이퍼파라미터
- **훈련 에포크**: 20-30 에포크
- **이미지 해상도**: 832x832 (현재와 동일)
- **배치 크기**: 16 (GPU 메모리에 따라 조정)
- **옵티마이저**: AdamW
- **스케줄러**: CosineAnnealingLR
- **데이터 증강**: 현재와 동일

### 백본별 특별 고려사항
- **ResNet 계열**: 기본 설정 유지
- **EfficientNet**: 더 작은 학습률 (0.0005) 고려
- **ConvNeXt**: Drop Path 추가 고려
- **Vision Transformer**: Warmup 스케줄러 고려

## 평가 지표
1. **F1 Score** (주요 지표)
2. **Recall** (놓치는 텍스트 최소화)
3. **Precision** (잘못된 탐지 최소화)
4. **훈련 시간** (효율성 측정)
5. **모델 크기** (배포 고려)

## 예상 결과
- **ResNet34/50**: F1 0.86-0.88 예상
- **EfficientNet**: F1 0.85-0.87 예상 (효율성 우수)
- **ConvNeXt**: F1 0.87-0.90 예상 (최고 성능)
- **ViT**: F1 0.83-0.86 예상 (텍스트 탐지에 적합성 검증)

## 실험 일정
- **각 모델당 훈련 시간**: 6-8시간 (30 에포크)
- **Phase 1**: 1-2일
- **Phase 2**: 1-2일
- **Phase 3**: 1-2일
- **Phase 4**: 1일 (선택적)

## 성공 기준
- **기본 목표**: F1 Score > 0.87 달성
- **우수 목표**: F1 Score > 0.90 달성
- **효율성 목표**: 현재 모델 대비 2배 이하 계산량으로 성능 향상

## 위험 요소 및 대응
- **GPU 메모리 부족**: 배치 크기 감소 또는 그래디언트 누적
- **훈련 시간 초과**: 에포크 수 감소 또는 조기 종료
- **성능 저하**: 하이퍼파라미터 튜닝 또는 백본 변경