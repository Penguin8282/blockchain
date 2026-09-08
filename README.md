# 이더리움 지갑 주소 사기 여부 분류 모델

Kaggle **Ethereum Fraud Detection Dataset** 을 이용해, 신용평가 모형을 만들듯이
**해석 가능한(interpretable) 사기 주소 분류 모델**을 만드는 프로젝트입니다.

- 데이터 한 행 = 지갑 주소 하나
- 목표 변수 `FLAG` : `1` = 사기(피싱·스캠) 주소, `0` = 정상 주소
- 나머지 열 = 거래 횟수 / 금액 / 시간 간격 / ERC20 토큰 활동 등 집계 특징

---

## ⚠️ 이 데이터셋의 중요한 한계 (단계 2에서 발견)

이 데이터에는 **정답이 새어 나오는 구조적 문제(data leakage)** 가 있습니다.

단계 2·3에서 **정답이 새는 경로를 최소 4군데** 찾았습니다.

| # | 경로 | 규모 | 사기 비율 | 조치 |
|---|---|---|---|---|
| 1 | `Total ERC20 tnxs` 가 **비어 있음** | 829행 | **100%** | 0으로 채워 차단 |
| 2 | **모든 특징이 0인 '껍데기 행'** | 274행 | **100%** | 남아 있음 |
| 3 | 토큰 이름이 `Blockwell say NOTSAFU` 등 | 1,970행 | **100%** | 문자열 열을 모델에서 제외 |
| 4 | `total ether received` = **정확히 101.0** | 1,476행 | **0%** | 남아 있음 |

추가 증거:

- 원본 CSV 행 순서: 앞 7,000행 사기 0%, 8,000행 이후 사기 100% → **두 파일을 이어붙인 흔적**
- **41개 특징이 전부 IV > 0.5** — 실무에서 IV 1.0을 넘는 변수는 거의 예외 없이 데이터 문제를 의심합니다
- 1,476개 주소가 **정확히 101.0 ETH**를 받았습니다. 진짜 거래라면 금액이 소수점까지
  제각각이어야 하므로, 정상 주소 집단 상당수가 인위적으로 만들어졌을 가능성이 있습니다 (**확인 필요**)

원인은 **정상 주소와 사기 주소를 서로 다른 방법으로 수집해 합쳤기 때문**으로 보입니다.
따라서 이 데이터로 만든 모델의 AUC가 0.99여도 **현실 성능이 아닙니다.**

자세한 근거: `notebooks/01_eda.ipynb` 의 "2-B. 데이터 누수 정밀 점검",
`notebooks/02_woe_iv.ipynb` 의 "3-B. 오염 경로 카탈로그".

### 단계 4에서 확인된 결과

| 지표 | 현실의 좋은 사기 탐지 모형 | 이 데이터 (LightGBM, no_leak) |
|---|---|---|
| AUC | 0.75 ~ 0.85 | **0.9938** |
| KS | 0.30 ~ 0.50 | **0.9255** |

**파이프라인 버그는 아닙니다.** 정답(FLAG)을 무작위로 섞은 대조군 실험에서
AUC 가 **0.4901** (동전 던지기)로 나왔습니다. 즉 교차검증 코드는 정상이고,
높은 성능은 **데이터 자체의 성질**입니다.

**그래도 프로젝트를 계속하는 이유**: WoE·IV·SHAP·임계값 분석이라는 방법론을 익히는 것이
목표이고, "성능 숫자를 의심하고 원인을 찾아내는 것"이 분석가의 핵심 역량이기 때문입니다.
단계 4부터는 **누수 포함(full) / 누수 최소 차단(no_leak) 두 트랙**을 나란히 비교합니다.

---

## 문서 지도

| 경로 | 무엇을 하는 파일인가 | 상태 |
|---|---|---|
| `README.md` | 프로젝트 소개, 문서 지도, 설치·실행 방법 | 완료 |
| `requirements.txt` | 필요한 파이썬 패키지 목록 (`pip install -r requirements.txt`) | 완료 |
| `.env.example` | 설정값·비밀값의 **예시** 파일. 복사해서 `.env` 로 쓴다 | 완료 |
| `.env` | 실제 설정값·API 키. **깃에 올라가지 않음** (`.gitignore` 처리) | 사용자가 생성 |
| `.gitignore` | 깃에 올리면 안 되는 것들(비밀값·데이터·캐시) 목록 | 완료 |
| `src/config.py` | 경로·난수 시드·목표 열 이름 등 **모든 설정을 한곳에서** 정의 | 완료 |
| `src/check_environment.py` | 패키지·폴더·`.env`·원본 CSV 가 준비됐는지 점검 | 완료 |
| `src/download_dataset.py` | Kaggle 데이터셋 내려받기 (**이미 받았으면 재호출 안 함 = 캐시**) | 완료 |
| `src/data_loader.py` | 원본 CSV 읽기, 열 이름 정리, 특징 4묶음 정의 | 완료 |
| `src/plot_style.py` | 그래프 색·한글 글꼴을 프로젝트 전체에서 통일 | 완료 |
| `src/build_data_dictionary.py` | `reports/data_dictionary.md` 를 자동 생성 | 완료 |
| `src/woe_iv.py` | WoE·IV 직접 구현 (fit/transform 구조, 구간 자동 분할) | 완료 |
| `src/preprocessing.py` | 두 트랙(full / no_leak)의 모델링 데이터 생성 | 완료 |
| `src/modeling.py` | 모델 3종 학습 + 층화 5-fold 교차검증 + AUC·KS·F1 | 완료 |
| `src/explain.py` | SHAP 값 추출·가법성 검산·전역/개별 설명 표 생성 | 완료 |
| `src/threshold_analysis.py` | 비용 기반 임계값(cut-off) 분석 + 처리 용량 제약 | 완료 |
| `notebooks/01_eda.ipynb` | 탐색적 데이터 분석 + **데이터 누수 점검** (실행 결과 포함) | 완료 |
| `reports/data_dictionary.md` | 51개 열의 한국어 설명 + 확신 수준(확실/추정/확인 필요) | 완료 |
| `reports/feature_groups.md` | 특징 47개의 4묶음 분류표 | 완료 |
| `reports/iv_ranking.md` | IV 순위표 + 오염 경로 카탈로그 | 완료 |
| `reports/model_comparison.md` | 모델 3종 성능 비교 + 진단 2종 (껍데기 행 제거·대조군) | 완료 |
| `reports/linkedin_summary.md` | 링크드인용 5줄 요약 (국문·영문) | 완료 |
| `reports/figures/01_class_balance.png` | 클래스 분포 그림 | 완료 |
| `reports/figures/02_feature_distributions.png` | 묶음별 대표 특징 분포 비교 | 완료 |
| `notebooks/02_woe_iv.ipynb` | WoE·IV 계산 + 오염 경로 카탈로그 (실행 결과 포함) | 완료 |
| `reports/figures/03_woe_by_bin.png` | IV 상위 6개 특징의 구간별 WoE | 완료 |
| `notebooks/03_model_comparison.ipynb` | 모델 3종 비교 + 파이프라인 검증 (실행 결과 포함) | 완료 |
| `reports/figures/04_model_comparison.png` | 모델 3종 × 조건 3가지 성능 비교 | 완료 |
| `reports/figures/05_fold_variation.png` | fold 5개의 AUC 흩어진 정도 | 완료 |
| `notebooks/04_shap_explain.ipynb` | SHAP 전역·개별 설명 + 누수 의존 검증 (실행 결과 포함) | 완료 |
| `reports/shap_summary.md` | SHAP 설명 보고서 + 3문장 요약 | 완료 |
| `reports/figures/06_shap_global_importance.png` | SHAP 전역 중요도 상위 15개 | 완료 |
| `reports/figures/07_shap_beeswarm.png` | 특징 값 높낮이가 미는 방향 (beeswarm) | 완료 |
| `reports/figures/08_shap_individual.png` | 사기 1건·정상 1건 개별 설명 | 완료 |
| `notebooks/05_threshold_analysis.ipynb` | 임계값 비용 분석 (실행 결과 포함) | 완료 |
| `reports/threshold_analysis.md` | 비용비별 최적 임계값 + 금액 시나리오 + 용량 제약 | 완료 |
| `reports/figures/09_threshold_tradeoff.png` | 임계값에 따른 정밀도·재현율·F1 | 완료 |
| `reports/figures/10_cost_curves.png` | 비용비 5가지의 총비용 곡선 | 완료 |
| `data/raw/` | **원본 그대로** 두는 폴더. 절대 수정하지 않음 (깃 제외) | 완료 |
| `data/processed/` | 전처리·특징 가공 결과 저장 폴더 (깃 제외) | 완료 |

> `data/raw/` 와 `data/processed/` 는 용량이 크고 코드로 재생성 가능하므로 깃에 올리지 않습니다.
> 폴더 구조 자체는 `.gitkeep` 빈 파일로 유지됩니다.
> 반면 `reports/figures/*.png` 는 보고서의 일부이므로 깃에 포함합니다.

---

## 진행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 프로젝트 뼈대 (폴더, requirements, README, .env, .gitignore) | ✅ 완료 |
| 2 | EDA 노트북 + 데이터 사전 | ✅ 완료 |
| 3 | WoE / IV 계산 및 IV 상위 15개 특징 | ✅ 완료 |
| 4 | 모델 3종 비교 (로지스틱회귀·LightGBM·랜덤포레스트) | ✅ 완료 |
| 5 | SHAP 설명 (전역 중요도 + 개별 사례 2건) | ✅ 완료 |
| 6 | 비용 기반 임계값(cut-off) 분석 | ✅ 완료 |
| 7 | 링크드인용 요약 5줄 | ✅ 완료 |

**전체 7단계 완료.**

---

## 최종 결과 한눈에 보기

### 모델 성능 (층화 5-fold 교차검증, no_leak 트랙)

| 모델 | AUC | KS | F1(사기) |
|---|---|---|---|
| **LightGBM** | 0.9938 ± 0.0009 | 0.9255 ± 0.0094 | 0.9301 ± 0.0038 |
| 랜덤포레스트 | 0.9907 ± 0.0007 | 0.9102 ± 0.0073 | 0.9035 ± 0.0051 |
| 로지스틱회귀 (WoE) | 0.9763 ± 0.0022 | 0.8571 ± 0.0072 | 0.8752 ± 0.0142 |

> 이 숫자들은 **현실 성능이 아닙니다.** 위 '데이터셋의 중요한 한계' 절을 반드시 함께 보세요.

### 검증한 것들

| 검증 | 방법 | 결과 |
|---|---|---|
| 교차검증 파이프라인이 올바른가 | 정답을 무작위로 섞어 학습 | **AUC 0.4901** → 정상 |
| WoE·IV 공식이 맞게 구현됐나 | 노트북에서 손계산으로 검산 | 일치 |
| KS 계산이 맞나 | 완벽 분리/무작위 두 극단으로 확인 | 1.0 / 0.02 |
| SHAP 가법성이 성립하나 | 기준값 + SHAP 합계 = 예측 로짓 | 오차 6.2×10⁻¹² |
| 비용 계산이 맞나 | 비용비 1:1 에서 오류 최소 지점 확인 | 곡선이 평평함을 확인 |

### 비용에 따른 최적 임계값

| 비용비 (미탐:오탐) | 임계값 | 재현율 | 정밀도 |
|---|---|---|---|
| 1 : 1 | 0.62 | 89.5% | 97.0% |
| 10 : 1 | 0.05 | 97.5% | 82.1% |
| 50 : 1 | 0.01 | 98.8% | 68.2% |

---

## 노트북 실행 순서

```
notebooks/01_eda.ipynb              탐색적 분석 + 데이터 누수 발견
notebooks/02_woe_iv.ipynb           WoE·IV 계산 + 오염 경로 카탈로그
notebooks/03_model_comparison.ipynb 모델 3종 비교 + 파이프라인 검증
notebooks/04_shap_explain.ipynb     SHAP 전역·개별 설명
notebooks/05_threshold_analysis.ipynb 비용 기반 임계값 분석
```

모든 노트북은 **실행 결과가 포함된 상태**로 저장돼 있어, 실행하지 않고도 GitHub 에서
바로 읽을 수 있습니다.

---

## 설치 및 실행

### 1) 패키지 설치

```bash
pip install -r requirements.txt
```

### 2) 설정 파일 준비

```bash
cp .env.example .env
```

`.env` 는 깃에 올라가지 않으므로 API 키를 넣어도 안전합니다.
지금 단계에서는 API 키가 없어도 모든 코드가 정상 동작합니다.

### 3) 원본 데이터 내려받기

```bash
python -m src.download_dataset
```

- 내려받은 CSV 를 `data/raw/transaction_dataset.csv` 로 저장합니다.
- **이미 파일이 있으면 네트워크를 쓰지 않고 그냥 넘어갑니다(캐시).**
  강제로 다시 받으려면 `--force` 를 붙이세요.
- Kaggle 인증이 필요할 수 있습니다. Kaggle > Settings > API > *Create New Token* 으로
  받은 `kaggle.json` 을 `~/.kaggle/kaggle.json` 에 두거나, `.env` 에
  `KAGGLE_USERNAME` / `KAGGLE_KEY` 를 넣어 두면 됩니다.

자동 내려받기가 막힌 환경이라면, 브라우저로 직접 받아
`data/raw/transaction_dataset.csv` 에 두어도 똑같이 동작합니다.
파일 이름이 다르면 `.env` 의 `RAW_DATASET_FILENAME` 값만 바꾸면 됩니다.

### 4) 환경 점검

```bash
python -m src.check_environment
```

패키지 / 폴더 / `.env` / 원본 CSV 네 가지를 순서대로 확인하고 요약을 출력합니다.

---

## 설계 원칙

1. **비밀값은 코드에 쓰지 않는다.** 모든 설정은 `.env` → `src/config.py` 를 거쳐 들어옵니다.
2. **설정은 한곳에만.** 경로나 시드를 바꿀 일이 생기면 `src/config.py` 한 파일만 고칩니다.
3. **원본 데이터는 건드리지 않는다.** `data/raw/` 는 읽기 전용처럼 다루고,
   가공 결과는 `data/processed/` 에 따로 저장합니다.
4. **외부 API 결과는 캐시한다.** 같은 요청을 다시 하지 않도록 `data/raw/` 에 저장해 두고
   재실행 시에는 저장된 파일을 읽습니다. *(현재 파이프라인은 외부 API 를 호출하지 않습니다.)*
5. **재현 가능하게.** 난수 시드를 고정해 몇 번을 돌려도 같은 결과가 나오게 합니다.

---

## 확인 필요 항목

- [x] `data/raw/transaction_dataset.csv` 확보 완료 (2.75 MB, 51개 열, `FLAG` 열 존재 확인).
      단, 이 파일은 `.gitignore` 로 깃에서 제외되므로 **다른 PC 에서 작업을 이어받을 때는**
      `python -m src.download_dataset` 을 한 번 실행해야 합니다.
      *(참고: 개발용 원격 컨테이너에서는 `api.kaggle.com` 이 방화벽 정책으로 차단되어 있어
      자동 내려받기가 되지 않습니다. 로컬 PC 에서는 정상 동작합니다.)*
- [x] 열 51개의 의미를 `reports/data_dictionary.md` 에 정리했습니다.
- [ ] **의미가 확정되지 않은 열 3개** — 추측하지 않고 남겨 두었습니다.
      원 데이터 제작자의 설명이나 Etherscan API 문서 확인이 필요합니다.
  - `Index` — 9,841행인데 고유값이 4,729개뿐이라 단순 행 번호가 아님
  - `ERC20 uniq sent addr.1` — 원본 CSV 에 같은 이름의 열이 두 개 있어 pandas 가 붙인 이름.
    앞 열과 값이 83%만 일치하고 분포가 전혀 달라 별개의 의미로 보임
  - `ERC20 avg time between rec 2 tnx` — `rec 2` 가 무엇을 뜻하는지 알 수 없음 (전 행이 0)
- [ ] ERC20 금액 열들의 **단위** — 이름에는 `Ether` 가 붙어 있지만 실제로는 토큰 수량으로 보입니다.
      토큰마다 소수점 자릿수가 다르므로, 서로 다른 토큰의 수량을 그냥 더한 값이라면
      해석에 주의가 필요합니다.
