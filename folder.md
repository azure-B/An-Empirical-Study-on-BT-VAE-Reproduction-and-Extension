# 폴더

```
BT_VAE/
├── pipeline/          논문 실험 코드
├── original/          원본 BT-VAE. 수정하지 않음
├── checkpoints/       Barlow Twins 인코더 가중치
│   └── covid/
├── results/
│   ├── covid/         COVID 라벨 비율 결과
│   └── bp/            BP 30% 정성 패널
├── datasets/          covid_8_1_1, bp_data
├── docs/              참고 논문 PDF
├── old/               사용하지 않는 실험
├── README.md
└── folder.md
```

## pipeline

| 파일 | 역할 |
|---|---|
| `__main__.py` | `python -m pipeline` 진입점. segment, classify, panel, figures |
| `config.py` | 체크포인트와 결과 경로, 데이터 이름 |
| `data.py` | COVID·BP 로더. 슬라이스는 디스크에서 읽는다 |
| `engine.py` | 원본 VAE 로드, 분할 손실, Dice 계산. `original/models1.py`를 경로로 불러온다 |
| `models_skip.py` | S5. B0–B3 concat skip, B4는 끔 |
| `models_presence.py` | 병변 존재 헤드. μ를 GAP·GMP한 뒤 로짓 1개 |
| `metrics.py` | AUROC와, τ로 슬라이스를 비운 뒤의 Dice |
| `stage_segment.py` | none, S5 학습. 학습 라벨만 비율로 줄인다 |
| `stage_classify.py` | none+classification, S5+classification 헤드 학습 |
| `stage_panel.py` | BP 패널. 열은 Input, GT, none, S5, none+cls, S5+cls |
| `stage_figures.py` | Dice 곡선과 표 PNG, CSV |
| `cases.py` | 초록 TP, 빨강 FP, 흰색 FN 오버레이 |

## results/covid/label_ratio

비율마다 폴더가 있다. `r10`은 학습 라벨 10%이다.

| 하위 폴더 | 내용 |
|---|---|
| `none/`, `s5/` | 게이트를 끈 분할 가중치와 `report.json` |
| `none_cls/`, `s5_cls/` | 병변 확률 헤드, τ, AUROC, 게이트 적용 Dice |
| `figures/` | `dice_vs_ratio.png`, `dice_vs_ratio_table.png`, `dice_vs_ratio.csv` |
| `cases/` | COVID 테스트 슬라이스 오버레이 |

`curve.json`과 `cls_metrics.json`은 같은 테스트의 요약이다.

## results/bp

`panel.png`는 테스트 네 장이다. 케이스는 `cases_none_cls.json`에 파일명과 Dice가 있다. 분할 가중치는 `old/parked/results/bp_original/`의 none, S5 체크포인트를 읽고, 헤드는 이 폴더의 `head_best.pt`, `none_head_best.pt`이다.

## original

원본 모델 `models1.py`와 BT-VAE 노트북이 있다. 파이프라인은 이 디렉터리를 고치지 않고 인코더·디코더만 불러온다.

## git에 올리지 않는 것

