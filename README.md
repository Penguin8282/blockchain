# 이더리움 지갑 주소 사기 여부 분류 모델

Kaggle **Ethereum Fraud Detection Dataset** 을 이용해, 신용평가 모형을 만들듯이
**해석 가능한(interpretable) 사기 주소 분류 모델**을 만드는 프로젝트입니다.

- 데이터 한 행 = 지갑 주소 하나
- 목표 변수 `FLAG` : `1` = 사기(피싱·스캠) 주소, `0` = 정상 주소
- 나머지 열 = 거래 횟수 / 금액 / 시간 간격 / ERC20 토큰 활동 등 집계 특징

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
| `src/data_loader.py` | 원본 CSV 읽기 + 외부 API 결과 캐싱 | 단계 2 예정 |
| `src/woe_iv.py` | WoE(증거가중치)·IV(정보가치) 직접 구현 | 단계 3 예정 |
| `src/modeling.py` | 모델 3종 학습 + 층화 5-fold 교차검증 + AUC·KS·F1 | 단계 4 예정 |
| `src/explain.py` | SHAP 기반 전역·개별 설명 그림 생성 | 단계 5 예정 |
| `src/threshold_analysis.py` | 비용 기반 임계값(cut-off) 분석 | 단계 6 예정 |
| `notebooks/01_eda.ipynb` | 탐색적 데이터 분석(클래스 불균형, 결측·상수 열, 특징 4묶음) | 단계 2 예정 |
| `reports/data_dictionary.md` | 각 열의 의미를 한국어 한 줄로 정리한 데이터 사전 | 단계 2 예정 |
| `reports/model_comparison.md` | 모델 3종 성능 비교 표 | 단계 4 예정 |
| `reports/linkedin_summary.md` | 링크드인용 5줄 요약 | 단계 7 예정 |
| `reports/figures/` | SHAP 그림 등 이미지 산출물 | 단계 5 예정 |
| `data/raw/` | **원본 그대로** 두는 폴더. 절대 수정하지 않음 (깃 제외) | 완료 |
| `data/processed/` | 전처리·특징 가공 결과 저장 폴더 (깃 제외) | 완료 |

> 데이터 폴더와 `reports/figures/` 는 용량이 크고 코드로 재생성 가능하므로 깃에 올리지 않습니다.
> 폴더 구조 자체는 `.gitkeep` 빈 파일로 유지됩니다.

---

## 진행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 프로젝트 뼈대 (폴더, requirements, README, .env, .gitignore) | ✅ 완료 |
| 2 | EDA 노트북 + 데이터 사전 | ⬜ 대기 |
| 3 | WoE / IV 계산 및 IV 상위 15개 특징 | ⬜ 대기 |
| 4 | 모델 3종 비교 (로지스틱회귀·LightGBM·랜덤포레스트) | ⬜ 대기 |
| 5 | SHAP 설명 (전역 중요도 + 개별 사례 2건) | ⬜ 대기 |
| 6 | 비용 기반 임계값(cut-off) 분석 | ⬜ 대기 |
| 7 | 링크드인용 요약 5줄 | ⬜ 대기 |

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
- [ ] 열(column) 이름과 의미는 실제 CSV 를 읽어 본 뒤 단계 2에서 확정합니다.
      의미가 불확실한 열은 `reports/data_dictionary.md` 에 **'확인 필요'** 로 표시합니다.
