# BT-VAE-Based Medical Image Segmentation with Skip Connections and Lesion-Presence Classification

**스킵 연결과 분류를 결합한 BT-VAE 기반 제한된 레이블 환경의 의료영상 분할**

Code and results for the study above. The full pipeline (encoder, skip decoder, presence head) is walked through in [Main.ipynb](Main.ipynb). All figures are in [`results/`](results/).

[English](#english) · [한국어](#한국어)

---

## English

### Summary

BT-VAE pairs Barlow Twins pre-training with a variational autoencoder to segment medical images when few labels are available. Its decoder, however, receives spatial information only through the latent mean μ, so the position and boundary of small lesions are partly lost. We add multi-resolution skip connections to recover this detail. The skips introduce a new problem: the model starts to draw lesions on slices that contain none. To suppress these false positives, we train a lightweight lesion-presence head on the frozen μ and clear the whole mask of a slice whenever the predicted probability falls below a threshold τ chosen on the validation set.

On COVID-19 CT with 30% of the training labels, the combined model reaches an overall Dice of 0.801, against 0.745 for skip connections alone (+0.056, about 7.5% relative). The false-positive rate on lesion-free slices drops from 0.225 to 0.05 (about 77.8% relative).

### 1. Motivation

Pixel-level labels must be drawn by radiologists, so labeled data is usually a small fraction of what is available. BT-VAE addresses this by pre-training the encoder without labels and then fine-tuning it for segmentation. Two weaknesses remain.

- **Lost spatial detail.** The decoder works from μ alone. Small or irregular lesions lose position and edge information in the bottleneck.
- **False positives on empty slices.** Skip connections pass fine encoder features straight to the decoder. This helps on lesion slices, but normal anatomy or noise on a lesion-free slice can then be read as a lesion. The segmenter has no way to judge whether a lesion exists at all.

The two components therefore have separate jobs: skip connections restore the lesion shape, and the presence head decides whether the predicted mask should be kept.

### 2. Method

![Method overview](results/method.png)

*Figure 1. Barlow Twins pre-training, skip-connection segmentation, and lesion-presence gating.*

1. **Pre-training.** Two rotated views of each image go through a shared VAE encoder. The Barlow Twins objective on the latent means, together with a KL-based alignment term, makes the representation consistent across views and reduces redundancy between channels.
2. **Segmentation fine-tuning.** The pre-trained encoder is fine-tuned with a decoder. The skip variant concatenates encoder features of resolutions B0–B3 into the decoder (B4 is off). The decoder input is μ, not a sampled latent. The loss is Dice plus BCE, with a small KL regularizer.
3. **Presence gating.** With the segmenter frozen, a small head on μ (average and max pooling, then fully connected layers) outputs the probability that the slice contains a lesion. It is trained with a class-weighted BCE. At test time, if the probability is below τ, the entire mask is replaced by an empty one. The head never redraws boundaries.

Four models are compared:

| Paper name | Code tag | Description |
|---|---|---|
| BT-VAE | `none` | Decoder from μ only |
| Skip BT-VAE | `S5` | Adds B0–B3 concat skips |
| BT-VAE+CLS | `none+cls` | `none` mask, gated by the head |
| Skip BT-VAE+CLS | `S5+cls` | `S5` mask, gated by the head |

### 3. Experimental setup

- **COVID-19 CT.** 743 / 92 / 94 slices for train / validation / test. 40 of the 94 test slices contain no lesion. To mimic limited labels, the training set is reduced at the patient level to 10, 30, 50, 70, and 90% (74, 223, 372, 520, and 669 slices). Validation and test sets are identical across ratios.
- **BP ultrasound (brachial plexus).** 5,635 image–mask pairs split 8:1:1; only the 30% training condition is used, mainly for a qualitative check on a second modality.
- **Metrics.** Overall Dice over all test slices (an empty prediction on an empty slice counts as 1), lesion-slice Dice, the false-positive rate on lesion-free slices, and AUROC for the presence head.
- **Training.** AdamW throughout. Pre-training: 200 epochs, lr 3e-4. Segmentation: lr 1e-4, batch 4, up to 400 epochs with early stopping. Presence head: 40 epochs, lr 1e-3. τ is searched from 0.01 to 0.99 on the validation set to maximize overall Dice, with ties broken toward the lower false-positive rate.

### 4. Results

#### 4.1 Effect of the labeled-data ratio (COVID)

![Dice vs label ratio](results/covid/label_ratio/figures/dice_vs_ratio.png)

![Dice table](results/covid/label_ratio/figures/dice_vs_ratio_table.png)

*Figure 2. Overall Dice on the 94 test slices as the training labels are reduced.*

`S5+cls` has the highest overall Dice at every ratio. At 10%, overall Dice is 0.505 for BT-VAE, 0.663 for Skip BT-VAE, 0.693 for BT-VAE+CLS, and 0.768 for Skip BT-VAE+CLS. The gap between BT-VAE and Skip BT-VAE+CLS shrinks from 0.263 at 10% to 0.033 at 90%. Gating adds 0.105 to Skip BT-VAE at 10% and 0.025 at 90%. The benefit is therefore largest when labels are scarcest.

At 30%, the operating point of the study:

| Model | Overall Dice | Empty-slice FP rate |
|---|---:|---:|
| Skip BT-VAE | 0.745 | 0.225 |
| Skip BT-VAE+CLS | **0.801** | **0.050** |

Overall Dice includes empty slices, so part of the gain comes from suppressing false positives and not only from better lesion boundaries. Lesion-slice Dice does not rise with gating (it falls slightly, 0.723 to 0.691 in our run), because a few true lesions are removed together with the false positives.

**Qualitative comparison (30%).**

![COVID qualitative panel](results/covid_panel.png)

*Figure 5. Input, ground truth, and the four models on COVID test slices. Green: true positive. Red: false positive. White: missed lesion.*

- Case 1 and 3 (lesion slices): `S5` covers more of the lesion than `none`, with some false-positive and missed pixels remaining. `none+cls` removes the whole lesion in Case 1, while `S5+cls` keeps it.
- Case 2 (no lesion): the raw segmenters draw a false mask. Both gated models remove it.
- Case 4 (small lesions): `none+cls` returns an empty mask, while `S5+cls` keeps the small lesions.
- Where a mask is kept, residual false-positive pixels remain. Gating decides whether to keep a slice; it does not correct boundaries.

#### 4.2 Skip-connection ablation (full COVID training set)

![Skip ablation](results/test_dice_by_skip_setting.jpg)

*Figure 3. Test Dice for different skip ranges and fusion methods.*

| Setting | Test Dice |
|---|---:|
| No skip | 0.6544 |
| B0–B4, add | 0.7629 |
| B0–B3, add | 0.7848 |
| B0–B3, learnable fusion | 0.7928 |
| B0–B3, concat | **0.8110** |

B0–B3 concatenation is best on both test Dice and lesion-slice Dice (0.7890), and is the configuration used as Skip BT-VAE. Dropping B4 raises test Dice for the additive variants, but lesion-slice Dice is not higher (0.7829 vs 0.7796), so range and fusion method should be chosen together.

#### 4.3 Presence head versus a standalone CNN

![Presence AUROC](results/cls_auroc_cnn.png)

*Figure 4. Test AUROC of heads on μ and of a CNN trained from scratch.*

At 10% and 30%, the BT-VAE head scores highest (0.990 at both). At 30%, the Skip BT-VAE head reaches 0.987 and the CNN 0.973. At 50% and above the CNN is best (0.993 to 0.996). The Skip BT-VAE head stays above 0.94 at every ratio, which shows that μ, learned for segmentation, also carries enough information to tell whether a lesion is present. The head is used because it reuses the existing encoder and feeds the gate directly, not because it is the strongest classifier.

#### 4.4 Second modality: BP ultrasound (30%)

![BP qualitative panel](results/bp_panel.png)

*Figure 6. Qualitative comparison on BP test slices.*

On slices that contain the target (Case 1, 3, 4), `S5` recovers more of the region than `none`, and `S5+cls` keeps the mask after gating. On the empty slice (Case 2), `none`, `S5`, and `none+cls` all produce a false mask, and only `S5+cls` removes it. This is a qualitative check on a handful of slices, not a quantitative evaluation.

### 5. Limitations

- Gating is slice-level and all-or-nothing. If a lesion slice scores below τ, the whole lesion is lost. At 30% on COVID, gating reduces slice-level false positives from 9 to 2 but raises false negatives from 4 to 6 (TP 50→48, TN 31→38). A threshold rule that accounts for this trade-off is future work.
- Results are from a single run per setting. Repeated runs and quantitative evaluation on more datasets are needed to establish stability.
- The BP experiment is qualitative only. A head trained on COVID does not transfer to BP, so BP uses its own head and its own τ.

### 6. Reproduction

```bash
python -m pipeline segment --modes none,s5   # segmentation at each label ratio
python -m pipeline classify                  # presence heads and BP panel
python -m pipeline figures                   # Dice chart and table
python -m pipeline.stage_classify --covid-panel --covid-ratio 0.3   # COVID qualitative panel
```

---

## 한국어

### 요약

BT-VAE는 Barlow Twins 사전학습과 VAE를 결합해 라벨이 적은 환경에서 의료영상 분할을 수행한다. 다만 디코더가 공간 정보를 잠재 평균 μ를 통해서만 받으므로, 작은 병변의 위치와 경계가 일부 손실된다. 본 연구는 이를 보완하려고 다중 해상도 스킵 연결을 추가했다. 그런데 스킵 연결을 쓰면 병변이 없는 슬라이스에도 병변을 그리는 오검출이 생긴다. 이를 억제하기 위해 고정한 μ 위에 경량 병변 존재 헤드를 학습하고, 예측 확률이 검증 데이터에서 정한 임계값 τ보다 낮으면 해당 슬라이스의 마스크 전체를 제거한다.

COVID-19 CT에서 학습 라벨 30%만 사용했을 때 두 구성을 결합한 모델의 전체 Dice는 0.801로, 스킵 연결만 쓴 0.745보다 0.056(상대 약 7.5%) 높았다. 병변 없는 슬라이스의 오검출률은 0.225에서 0.05로 줄었다(상대 약 77.8% 감소).

### 1. 연구 배경

의료영상의 픽셀 단위 라벨은 전문가가 직접 작성해야 해서, 실제로 쓸 수 있는 라벨은 전체 데이터의 일부에 그친다. BT-VAE는 라벨 없이 인코더를 사전학습한 뒤 분할에 미세조정하는 방식으로 이 문제에 대응한다. 그래도 두 가지 한계가 남는다.

- **공간 정보 손실.** 디코더가 μ만 받으면 병목에서 작거나 불규칙한 병변의 위치와 경계 정보가 줄어든다.
- **빈 슬라이스 오검출.** 스킵 연결은 인코더의 세부 특징을 디코더에 바로 전달한다. 병변 슬라이스에는 도움이 되지만, 병변이 없는 슬라이스에서는 정상 구조나 잡음이 병변으로 읽힐 수 있다. 분할기 자체는 병변이 있는지 판단하지 못한다.

그래서 역할을 나눴다. 스킵 연결은 병변 형태의 복원을, 병변 존재 헤드는 예측 마스크를 유지할지 결정하는 일을 맡는다.

### 2. 방법

![방법 개요](results/method.png)

*그림 1. Barlow Twins 사전학습, 스킵 연결 분할, 병변 존재 게이팅.*

1. **사전학습.** 한 영상에서 회전 증강 두 뷰를 만들어 공유 VAE 인코더에 넣는다. 잠재 평균에 대한 Barlow Twins 손실과 KL 기반 정렬 항으로 뷰 간 일관성을 높이고 채널 간 중복을 줄인다.
2. **분할 미세조정.** 사전학습된 인코더를 디코더와 함께 미세조정한다. 스킵 모델은 해상도 B0–B3의 인코더 특징을 디코더에 concat으로 전달한다(B4는 사용하지 않음). 디코더 입력은 샘플링한 잠재변수가 아니라 μ이다. 손실은 Dice와 BCE의 결합에 작은 KL 정규화를 더한다.
3. **존재 게이팅.** 분할기를 고정한 채 μ에 평균·최대 풀링과 완전연결층으로 구성한 작은 헤드를 붙여 슬라이스의 병변 존재 확률을 구한다. 클래스 가중 BCE로 학습한다. 테스트에서 확률이 τ보다 낮으면 마스크 전체를 빈 마스크로 바꾼다. 헤드는 경계를 다시 그리지 않는다.

비교 모델은 네 가지다.

| 논문 표기 | 코드 | 설명 |
|---|---|---|
| BT-VAE | `none` | μ만으로 디코드 |
| Skip BT-VAE | `S5` | B0–B3 concat 스킵 추가 |
| BT-VAE+CLS | `none+cls` | `none` 마스크에 게이팅 |
| Skip BT-VAE+CLS | `S5+cls` | `S5` 마스크에 게이팅 |

### 3. 실험 설정

- **COVID-19 CT.** 학습 743장, 검증 92장, 테스트 94장이며 테스트 중 40장은 병변이 없다. 제한된 라벨 환경을 모사하려고 학습 집합만 환자 단위로 10, 30, 50, 70, 90%(74, 223, 372, 520, 669장)로 줄였다. 검증·테스트 집합은 모든 비율에서 같다.
- **BP 초음파(상완 신경총).** 이미지–마스크 5,635쌍을 8:1:1로 나눴고, 학습 데이터 30% 조건만 사용한다. 다른 영상 종류에서의 정성 확인이 목적이다.
- **평가 지표.** 테스트 전체 Dice(빈 슬라이스에서 빈 예측이면 1), 병변 슬라이스 Dice, 병변 없는 슬라이스의 거짓 양성 비율, 존재 헤드의 AUROC.
- **학습.** 모든 단계 AdamW. 사전학습은 200 에폭, 학습률 3e-4. 분할은 학습률 1e-4, 배치 4, 최대 400 에폭에 조기종료. 존재 헤드는 40 에폭, 학습률 1e-3. τ는 검증 집합에서 0.01~0.99를 탐색해 전체 Dice가 최대인 값으로 정하고, 동률이면 거짓 양성 비율이 낮은 값을 택한다.

### 4. 결과

#### 4.1 레이블 비율에 따른 성능 (COVID)

![레이블 비율별 Dice](results/covid/label_ratio/figures/dice_vs_ratio.png)

![Dice 표](results/covid/label_ratio/figures/dice_vs_ratio_table.png)

*그림 2. 학습 라벨을 줄였을 때 테스트 94장의 전체 Dice.*

모든 비율에서 `S5+cls`의 전체 Dice가 가장 높았다. 10%에서 BT-VAE 0.505, Skip BT-VAE 0.663, BT-VAE+CLS 0.693, Skip BT-VAE+CLS 0.768이다. BT-VAE와 Skip BT-VAE+CLS의 차이는 10%의 0.263에서 90%의 0.033으로 줄고, Skip BT-VAE에 게이팅을 더한 향상 폭도 10%에서 0.105, 90%에서 0.025이다. 라벨이 적을수록 효과가 크다.

본 연구의 기준 조건인 30%는 다음과 같다.

| 모델 | 전체 Dice | 빈 슬라이스 오검출 |
|---|---:|---:|
| Skip BT-VAE | 0.745 | 0.225 |
| Skip BT-VAE+CLS | **0.801** | **0.050** |

전체 Dice에는 빈 슬라이스가 포함되므로, 향상의 일부는 경계 복원이 아니라 오검출 억제에서 온다. 병변 슬라이스 Dice는 게이팅으로 오르지 않고 오히려 조금 낮아졌다(본 실험에서 0.723 → 0.691). 오검출과 함께 일부 실제 병변도 지워지기 때문이다.

**정성 비교 (30%).**

![COVID 정성 비교](results/covid_panel.png)

*그림 5. COVID 테스트 슬라이스의 입력, 정답, 네 모델 결과. 초록: 정답과 일치, 빨강: 거짓 양성, 흰색: 미검출.*

- Case 1, 3(병변 슬라이스): `S5`가 `none`보다 병변을 넓게 덮고, 거짓 양성과 미검출은 일부 남는다. Case 1에서 `none+cls`는 병변 전체를 지우고 `S5+cls`는 유지한다.
- Case 2(병변 없음): 게이팅 전 모델은 가짜 마스크를 그리고, 게이팅을 적용한 두 모델은 이를 제거한다.
- Case 4(작은 병변): `none+cls`는 빈 마스크를 내고 `S5+cls`는 작은 병변을 유지한다.
- 마스크가 유지된 슬라이스에는 거짓 양성 픽셀이 남는다. 게이팅은 경계 보정이 아니라 슬라이스를 남길지 정하는 단계다.

#### 4.2 스킵 연결 소거 실험 (COVID 전체 학습 집합)

![스킵 소거](results/test_dice_by_skip_setting.jpg)

*그림 3. 스킵 범위와 결합 방식에 따른 테스트 Dice.*

| 설정 | 테스트 Dice |
|---|---:|
| 스킵 없음 | 0.6544 |
| B0–B4, add | 0.7629 |
| B0–B3, add | 0.7848 |
| B0–B3, 학습형 결합 | 0.7928 |
| B0–B3, concat | **0.8110** |

B0–B3 concat이 테스트 Dice와 병변 슬라이스 Dice(0.7890) 모두 가장 높아 Skip BT-VAE의 구성으로 채택했다. add 방식에서 B4를 빼면 테스트 Dice는 오르지만 병변 슬라이스 Dice는 오르지 않아(0.7829 대 0.7796), 연결 범위와 결합 방식을 함께 정해야 한다.

#### 4.3 존재 헤드와 독립 CNN 비교

![존재 AUROC](results/cls_auroc_cnn.png)

*그림 4. μ 위의 헤드와 처음부터 학습한 CNN의 테스트 AUROC.*

10%와 30%에서는 BT-VAE 헤드가 0.990으로 가장 높았다. 30%에서 Skip BT-VAE 헤드는 0.987, CNN은 0.973이다. 50% 이상에서는 CNN이 0.993~0.996으로 가장 높다. Skip BT-VAE 헤드는 모든 비율에서 0.94 이상을 유지해, 분할용으로 학습된 μ만으로도 병변 존재를 판별할 수 있음을 보인다. 이 헤드를 쓰는 이유는 분류 성능 자체보다, 기존 인코더를 재사용하고 그 출력이 곧바로 게이팅에 쓰이기 때문이다.

#### 4.4 다른 영상 종류: BP 초음파 (30%)

![BP 정성 비교](results/bp_panel.png)

*그림 6. BP 테스트 슬라이스의 정성 비교.*

대상 영역이 있는 Case 1, 3, 4에서 `S5`는 `none`보다 영역을 더 넓게 복원하고, `S5+cls`는 게이팅 후에도 마스크를 유지한다. 대상이 없는 Case 2에서는 `none`, `S5`, `none+cls`가 모두 가짜 마스크를 내고 `S5+cls`만 이를 제거한다. 소수 슬라이스에 대한 정성 확인이며 정량 평가는 아니다.

### 5. 한계

- 게이팅은 슬라이스 단위의 전부-아니면-전무 방식이다. 병변 슬라이스가 τ 미만이면 병변 전체를 잃는다. COVID 30%에서 슬라이스 단위 거짓 양성은 9장에서 2장으로 줄었지만 거짓 음성은 4장에서 6장으로 늘었다(TP 50→48, TN 31→38). 이 균형을 반영한 임계값 선정은 향후 과제이다.
- 설정당 한 번의 실행 결과이다. 안정성을 보이려면 반복 실험과 더 많은 데이터셋의 정량 평가가 필요하다.
- BP 실험은 정성 평가에 그친다. COVID에서 학습한 헤드는 BP에 그대로 전이되지 않아, BP는 자체 헤드와 τ를 따로 학습했다.

### 6. 재현

```bash
python -m pipeline segment --modes none,s5   # 레이블 비율별 분할 학습
python -m pipeline classify                  # 존재 헤드와 BP 패널
python -m pipeline figures                   # Dice 그래프와 표
python -m pipeline.stage_classify --covid-panel --covid-ratio 0.3   # COVID 정성 패널
```

---

*This document was drafted with Cursor and will be revised. 이 문서는 Cursor로 초안을 작성했으며 이후 수정할 예정입니다.*
