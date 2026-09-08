# 링크드인용 요약

프로젝트: **이더리움 지갑 주소 사기 탐지 — 해석 가능한 모델 만들기**
저장소: https://github.com/Penguin8282/blockchain

---

## 📌 5줄 요약 (본문용)

> **1. 문제** — 이더리움 지갑 주소가 피싱·스캠 계정인지 판별하는 문제를 다뤘습니다.
> 단순히 잘 맞히는 모델이 아니라, 신용평가 모형처럼 **"왜 그렇게 판단했는지 설명할 수 있는"**
> 모델을 목표로 했습니다.
>
> **2. 데이터** — Kaggle Ethereum Fraud Detection Dataset(지갑 9,841개 × 51개 열, 사기 22.1%)의
> 집계 특징 47개를 **거래 빈도 / 금액 / 시간 간격 / 토큰 활동** 4묶음으로 분류하고,
> 의미가 불확실한 열 3개는 추측하지 않고 '확인 필요'로 남겼습니다.
>
> **3. 방법** — 신용평가에서 쓰는 **WoE(증거가중치)·IV(정보가치)를 직접 구현**하고,
> 로지스틱회귀(WoE 입력)·LightGBM·랜덤포레스트를 층화 5-fold 교차검증으로 비교했습니다.
> SHAP으로 전역·개별 설명을 만들고, 미탐/오탐 비용이 다를 때의 **임계값 최적화**까지 이어갔습니다.
>
> **4. 결과** — LightGBM이 **AUC 0.9938 / KS 0.9255**로 가장 좋았지만,
> 저는 이 숫자를 **믿지 않고 검증했습니다.** 정답을 무작위로 섞은 대조군에서 **AUC 0.4901**이 나와
> 파이프라인은 정상임을 확인했고, 대신 데이터에서 **정답이 새는 경로 4가지**를 찾아냈습니다 —
> ERC20 결측 829행(100% 사기), 모든 값이 0인 행 274행(100% 사기),
> 특정 토큰 이름 1,970행(100% 사기), `total ether received = 정확히 101.0`인 1,476행(0% 사기).
>
> **5. 배운 점** — **"성능이 너무 좋으면 먼저 의심하라"**가 가장 큰 배움이었습니다.
> 41개 특징이 전부 IV 0.5를 넘었을 때(실무에선 1.0만 넘어도 데이터 문제를 의심합니다)
> 멈춰서 원인을 파고든 것이, 좋은 점수를 보고하는 것보다 훨씬 가치 있는 작업이었습니다.
> AUC를 0.99로 만드는 법보다 **그 0.99를 의심하는 법**을 배운 프로젝트였습니다.

---

## 📊 곁들일 만한 숫자들

| 항목 | 값 |
|---|---|
| 데이터 규모 | 지갑 9,841개 × 51개 열 |
| 클래스 불균형 | 사기 22.1% (정상:사기 = 3.52 : 1) |
| 최종 특징 수 | 38개 (상수 열 7개·고차원 문자열 2개 제외) |
| 최고 성능 (LightGBM) | AUC 0.9938 ± 0.0009 / KS 0.9255 / F1 0.9301 |
| 대조군(라벨 셔플) AUC | **0.4901** ← 파이프라인 정상 확인 |
| 발견한 누수 경로 | **4가지** (총 4,549행에 영향) |
| SHAP 가법성 검산 오차 | 6.2 × 10⁻¹² |
| 최적 임계값 (비용비 1:1 → 10:1) | 0.62 → 0.05 |

---

## 🖼 함께 올리면 좋은 이미지

1. `reports/figures/04_model_comparison.png` — 모델 3종 × 조건 3가지 성능 비교
2. `reports/figures/08_shap_individual.png` — 사기/정상 주소 개별 설명
3. `reports/figures/10_cost_curves.png` — 비용비별 최적 임계값이 왼쪽으로 이동하는 모습

---

## 🔖 해시태그 후보

`#데이터분석` `#머신러닝` `#신용평가모형` `#WoE` `#IV` `#SHAP` `#설명가능한AI`
`#블록체인` `#이상거래탐지` `#FDS` `#데이터누수` `#DataLeakage` `#LightGBM`

---

## 🌐 영문 버전 (선택)

> **1. Problem** — Classifying Ethereum wallet addresses as phishing/scam accounts.
> The goal wasn't just accuracy, but an **interpretable** model in the spirit of credit scoring.
>
> **2. Data** — Kaggle's Ethereum Fraud Detection Dataset (9,841 wallets × 51 columns, 22.1% fraud).
> I grouped the 47 aggregate features into four families — transaction frequency, monetary value,
> time intervals, and token activity — and flagged 3 columns whose meaning I could not verify.
>
> **3. Method** — Implemented **Weight of Evidence and Information Value from scratch**,
> compared logistic regression (WoE input), LightGBM, and random forest under stratified 5-fold CV,
> then added SHAP explanations and cost-sensitive threshold optimization.
>
> **4. Result** — LightGBM reached **AUC 0.9938 / KS 0.9255** — and I didn't trust it.
> A label-permutation control returned **AUC 0.4901**, confirming the pipeline was sound,
> which pointed the finger at the data: I identified **four distinct leakage channels**,
> including 829 rows whose missing ERC20 fields were 100% fraud, and 1,476 rows
> that received *exactly* 101.0 ETH and were 100% legitimate.
>
> **5. Takeaway** — **When performance looks too good, investigate before you celebrate.**
> All 41 features exceeded IV 0.5 (practitioners get suspicious above 1.0). Stopping to find out why
> was worth far more than reporting a high score. I learned less about pushing AUC to 0.99
> and more about **doubting it**.

---

## ✍️ 작성 팁

- **4번(결과)을 가장 앞에 끌어오는 편집**도 좋습니다. "AUC 0.99를 만들었는데, 믿지 않았습니다"로
  시작하면 훅이 강해집니다.
- 채용 담당자는 **"높은 점수"보다 "검증 습관"**을 봅니다. 대조군 실험(AUC 0.4901)은
  꼭 남기세요 — 이 프로젝트에서 가장 차별화되는 지점입니다.
- 데이터셋을 비판할 때는 **제작자를 탓하는 뉘앙스를 피하고**, "이 데이터의 수집 방식이 남긴 흔적"
  정도로 사실만 적는 게 안전합니다.
