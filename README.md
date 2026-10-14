# S5와 병변 존재 분류

제한된 라벨로 COVID CT 병변을 나누는 실험이다. 비교하는 모델은 네 개다.

| 모델 | 분할 | 병변 존재 헤드 |
|---|---|---|
| none | 원본 VAE. 디코더는 μ만 받는다. | 없음 |
| S5 | 같은 인코더에 B0–B3 concat skip. B4는 끈다. | 없음 |
| none+classification | none 마스크 | μ에서 P(병변 존재) |
| S5+classification | S5 마스크 | μ에서 P(병변 존재) |

`original/`의 BT-VAE 코드는 수정하지 않는다. 논문용 실행 코드는 `pipeline/`에만 있다. 디렉터리 설명은 [folder.md](folder.md)에 있다.

## 데이터

COVID는 `datasets/covid_8_1_1`이다. 학습 743장, 검증 92장, 테스트 94장이다. 테스트 94장 중 40장은 병변이 없는 슬라이스다.

라벨 비율 0.1, 0.3, 0.5, 0.7, 0.9는 학습 집합만 환자 단위로 줄인 비율이다. 검증과 테스트는 모든 비율에서 같다.

| 비율 | 학습 슬라이스 |
|---:|---:|
| 0.1 | 74 |
| 0.3 | 223 |
| 0.5 | 372 |
| 0.7 | 520 |
| 0.9 | 669 |

BP 정성 패널은 `datasets/bp_data/train_30_split`의 테스트 슬라이스를 쓴다.

## 학습

분할 손실은 원본 노트북과 같다. Dice와 BCE를 0.5씩 섞고 0.99를 곱한 뒤 KL 0.01을 더한다. 최적화는 AdamW, 학습률 1e-4, 배치 4이다. 검증 손실이 40 에폭 동안 줄지 않으면 멈추고, 검증 손실이 가장 낮은 가중치를 쓴다. 디코드는 μ로 고정한다.

분류 헤드는 로짓을 낸다. 손실은 `BCEWithLogits`이고 양성 가중치는 2이다. 손실 안에서 시그모이드가 적용된다. 보고하는 확률은 같은 로짓의 시그모이드이다.

```
p = sigmoid(Head(μ)) = P(병변 존재 | 슬라이스)
```

검증에서 Dice가 최대인 임계값 τ를 고른다. 테스트에서 p가 τ보다 낮으면 그 슬라이스 마스크를 비운다.

## 실행

저장소 루트에서 실행한다. Barlow Twins 인코더는 `checkpoints/covid/bt_encoder_best.pt`에 있어야 한다.

```
python -m pipeline segment --modes none,s5
python -m pipeline classify --ratios 0.1,0.3,0.5,0.7,0.9
python -m pipeline panel
python -m pipeline figures
```

1. `segment`는 none과 S5를 학습한다. 끝난 비율은 다시 돌리지 않는다.
2. `classify`는 각 분할기 위에 병변 확률 헤드를 학습한다.
3. `panel`은 BP 30% 패널을 `results/bp/panel.png`에 다시 그린다.
4. `figures`는 Dice 곡선과 표를 `results/covid/label_ratio/figures/`에 저장한다.

## 학습 결과

COVID 테스트 94장의 평균 Dice이다. 가로축은 학습에 쓴 라벨 비율이다.

![Dice score with labeled data ratio](results/covid/label_ratio/figures/dice_vs_ratio.png)

![Dice table](results/covid/label_ratio/figures/dice_vs_ratio_table.png)

BP 30% 테스트 네 장이다. 초록은 맞은 병변, 빨강은 오검출, 흰색은 놓친 병변이다.

![BP panel](results/bp/panel.png)

## 결과 위치

- 분할 보고서: `results/covid/label_ratio/rXX/{none,s5}/report.json`
- 분류 보고서: `results/covid/label_ratio/rXX/{none_cls,s5_cls}/report.json`
- Dice 그림: `results/covid/label_ratio/figures/dice_vs_ratio.png`
- Dice 표: `results/covid/label_ratio/figures/dice_vs_ratio_table.png`
- BP 패널: `results/bp/panel.png`

가중치(`*.pt`), `checkpoints/`, `datasets/`, `original/`, `old/wheels/`는 git에 포함하지 않는다. `old/`의 나머지 실험 코드는 포함한다.
