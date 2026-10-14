# Pipeline

네 모델만 이 폴더에 있다. `original/`의 BT-VAE 코드는 수정하지 않는다.

| 모델 | 하는 일 |
|---|---|
| none | 원본 VAE. 디코더 입력은 μ뿐. |
| S5 | 같은 인코더에 B0–B3 concat skip. B4는 끈다. |
| none+classification | 학습이 끝난 none의 μ로 병변 존재 확률을 낸다. |
| S5+classification | 학습이 끝난 S5의 μ로 같은 확률을 낸다. |

분류 헤드는 로짓을 내고, 확률은 `p = sigmoid(로짓)`이다. 손실은 `BCEWithLogits`, 양성 가중치 2이다. 슬라이스 마스크를 비울 임계값 τ는 검증 Dice로 고른다.

COVID 학습 집합만 0.1, 0.3, 0.5, 0.7, 0.9로 줄인다. 검증 92장과 테스트 94장은 고정이다.

## 순서

저장소 루트에서 실행한다.

```
python -m pipeline segment --modes none,s5
python -m pipeline classify --ratios 0.1,0.3,0.5,0.7,0.9
python -m pipeline panel
python -m pipeline figures
```

1. `stage_segment.py` — none과 S5 분할 학습. 결과는 `results/covid/label_ratio/rXX/{none,s5}/`.
2. `stage_classify.py` — 각 분할기 위에 병변 확률 헤드. 결과는 `rXX/none_cls`, `rXX/s5_cls`.
3. `stage_panel.py` — BP 30% 정성 패널 `results/bp/panel.png`.
4. `stage_figures.py` — Dice 곡선과 표. `results/covid/label_ratio/figures/`.

Barlow Twins 인코더 가중치는 `checkpoints/covid/`에 둔다. 분할 가중치 `*.pt`와 `old/`, `datasets/`는 git에 올리지 않는다.

공용 조각은 `config.py`, `data.py`, `engine.py`, `models_skip.py`, `models_presence.py`, `metrics.py`이다.
