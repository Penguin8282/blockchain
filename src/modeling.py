"""
모델 3종을 층화 5-fold 교차검증으로 비교하는 모듈.

■ 비교하는 모델
  1. 로지스틱회귀 (WoE 변환 입력) — 신용평가 표준. 계수를 그대로 해석할 수 있다.
  2. LightGBM                    — 표 형식 데이터에서 성능이 가장 좋은 편인 부스팅 모델
  3. 랜덤포레스트                — 결정트리 여러 개의 다수결. 튼튼하고 과적합에 강하다.

■ 왜 '층화(stratified)' 5-fold 인가
  데이터를 5조각으로 나눠 4조각으로 학습하고 1조각으로 평가하기를 5번 반복합니다.
  '층화'는 **각 조각의 사기 비율을 원본과 똑같이 맞춘다**는 뜻입니다.
  이걸 안 하면 어떤 조각에 사기가 몰리거나 거의 없어서 성적이 들쭉날쭉해집니다.

  단계 2에서 원본 CSV 의 앞부분은 전부 정상, 뒷부분은 전부 사기인 것을 확인했습니다.
  그래서 **반드시 무작위로 섞어야(shuffle=True)** 합니다. 안 섞으면 어떤 fold 는
  정상만, 어떤 fold 는 사기만 갖게 되어 평가 자체가 불가능해집니다.

■ WoE 를 언제 계산하는가 (매우 중요)
  WoE 계산에는 정답(FLAG)이 들어갑니다. 그래서 **학습 fold 안에서만 fit** 하고
  검증 fold 에는 transform 만 적용합니다. 전체 데이터로 미리 계산해 두면
  검증 fold 의 정답이 학습에 새어 들어가 성능이 부풀려집니다.

■ 클래스 가중치를 쓰지 않는 이유
  사기 비율이 22%로 심한 불균형이 아니고, 무엇보다 단계 6의 **비용 기반 임계값 분석**에서
  모델이 내놓는 확률값이 실제 확률에 가까워야 합니다. 가중치를 주면 확률이 위로 밀려
  "이 주소가 사기일 확률 30%" 같은 값을 그대로 믿을 수 없게 됩니다.
  대신 임계값 조정으로 사기 탐지율을 올리는 방식(단계 6)을 씁니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from src import config
from src.woe_iv import WeightOfEvidenceEncoder

# 모델 이름 상수 (표에 그대로 쓰인다)
MODEL_LOGISTIC_WOE = "로지스틱회귀 (WoE)"
MODEL_LIGHTGBM = "LightGBM"
MODEL_RANDOM_FOREST = "랜덤포레스트"

ALL_MODEL_NAMES = [MODEL_LOGISTIC_WOE, MODEL_LIGHTGBM, MODEL_RANDOM_FOREST]

# 기본 판정 임계값. "확률이 이 값을 넘으면 사기로 판정한다"
# 단계 6에서 이 값을 비용에 맞게 다시 고릅니다.
DEFAULT_DECISION_THRESHOLD = 0.5


def compute_ks_statistic(true_labels: np.ndarray, predicted_scores: np.ndarray) -> float:
    """
    KS 통계량(Kolmogorov-Smirnov)을 계산한다.

    ■ KS 가 뭘 재는가
      "사기 집단의 점수 분포"와 "정상 집단의 점수 분포"가 **얼마나 멀리 떨어져 있는가**를
      0~1 사이 숫자로 나타낸 것입니다. 신용평가에서 AUC 와 함께 가장 많이 쓰입니다.

          KS = max_t | TPR(t) − FPR(t) |

      기호 뜻:
          t      : 임계값 (이 값을 넘으면 사기로 판정)
          TPR(t) : 진짜 사기 중에서 사기라고 맞힌 비율 (True Positive Rate, 재현율)
          FPR(t) : 진짜 정상 중에서 사기라고 잘못 찍은 비율 (False Positive Rate)
          max_t  : 모든 임계값 t 중에서 가장 큰 값을 고른다

      즉 **"사기는 많이 잡으면서 정상은 적게 건드리는 지점"의 최대 격차**입니다.

    ■ 읽는 법
          KS = 0    두 분포가 완전히 겹침 → 모델이 전혀 구분 못 함
          KS = 1    두 분포가 완전히 분리 → 완벽한 구분 (현실에선 데이터 누수 의심)
      신용평가 실무에서는 KS 0.3~0.5 면 쓸 만한 모형으로 봅니다.

    ■ AUC 와 무엇이 다른가
      AUC 는 모든 임계값에서의 성능을 '평균'낸 값이고,
      KS 는 '가장 잘 갈리는 한 지점'의 성능입니다.
      그래서 KS 가 최대가 되는 지점은 임계값 후보로도 자주 쓰입니다(단계 6 참고).
    """
    # roc_curve 는 여러 임계값에서의 FPR, TPR 을 한 번에 계산해 준다.
    false_positive_rates, true_positive_rates, _ = roc_curve(true_labels, predicted_scores)
    return float(np.max(true_positive_rates - false_positive_rates))


def evaluate_predictions(
    true_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict[str, float]:
    """
    예측 결과를 여러 지표로 평가한다.

    AUC 와 KS 는 확률값 자체로 계산하므로 임계값과 무관합니다.
    F1·정밀도·재현율은 "사기/정상" 판정이 필요하므로 임계값이 있어야 합니다.

    지표 뜻 (모두 '사기 클래스' 기준):
        AUC      : 무작위로 뽑은 사기 1개와 정상 1개를 비교했을 때,
                   사기의 점수가 더 높을 확률. 0.5 = 동전 던지기, 1.0 = 완벽
        KS       : 위 함수 설명 참고
        F1       : 정밀도와 재현율의 조화평균. 둘 다 높아야 커진다
        정밀도    : 사기라고 찍은 것 중 진짜 사기의 비율 (헛다리 안 짚기)
        재현율    : 진짜 사기 중 잡아낸 비율 (놓치지 않기)
    """
    # 확률이 임계값을 넘으면 1(사기), 아니면 0(정상)
    predicted_labels = (predicted_probabilities >= decision_threshold).astype(int)

    return {
        "AUC": roc_auc_score(true_labels, predicted_probabilities),
        "KS": compute_ks_statistic(true_labels, predicted_probabilities),
        "F1(사기)": f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0),
        "정밀도(사기)": precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0),
        "재현율(사기)": recall_score(true_labels, predicted_labels, pos_label=1, zero_division=0),
    }


def create_model(model_name: str, random_seed: int = config.RANDOM_SEED):
    """
    모델 이름을 받아 '아직 학습되지 않은' 새 모델 객체를 만들어 돌려준다.

    fold 마다 새 모델을 만들어야 합니다. 같은 객체를 재사용하면
    이전 fold 의 학습 내용이 남아 결과가 오염됩니다.
    """
    if model_name == MODEL_LOGISTIC_WOE:
        # max_iter=2000 : 기본값(100)으로는 수렴하지 않아 경고가 뜨므로 넉넉히 준다
        return LogisticRegression(max_iter=2000, random_state=random_seed)

    if model_name == MODEL_LIGHTGBM:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=300,      # 트리 300그루
            learning_rate=0.05,    # 한 그루가 결과를 얼마나 크게 바꿀지 (작을수록 신중)
            num_leaves=31,         # 트리 하나의 최대 잎 개수 (클수록 복잡한 규칙)
            random_state=random_seed,
            verbose=-1,            # 학습 로그를 끈다
        )

    if model_name == MODEL_RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=300,      # 트리 300그루의 다수결
            min_samples_leaf=2,    # 잎 하나에 최소 2개 샘플 (과적합 완화)
            random_state=random_seed,
            n_jobs=-1,             # CPU 코어를 모두 사용
        )

    raise ValueError(f"모르는 모델 이름입니다: {model_name}")


def run_cross_validation(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
    fold_count: int = config.CROSS_VALIDATION_FOLDS,
    random_seed: int = config.RANDOM_SEED,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> pd.DataFrame:
    """
    한 모델을 층화 K-fold 교차검증으로 평가하고, fold 별 지표를 표로 돌려준다.

    돌려주는 표의 각 행 = fold 하나의 성적
    """
    # shuffle=True 가 필수입니다. 원본 CSV 는 정상/사기가 순서대로 정렬돼 있습니다.
    cross_validator = StratifiedKFold(
        n_splits=fold_count, shuffle=True, random_state=random_seed
    )

    fold_results = []

    for fold_index, (train_indices, validation_indices) in enumerate(
        cross_validator.split(features, target), start=1
    ):
        train_features = features.iloc[train_indices]
        train_target = target.iloc[train_indices]
        validation_features = features.iloc[validation_indices]
        validation_target = target.iloc[validation_indices]

        if model_name == MODEL_LOGISTIC_WOE:
            # ★ WoE 는 반드시 학습 fold 로만 fit 한다 ★
            #   여기서 전체 데이터로 fit 하면 검증 fold 의 정답이 새어 들어간다.
            woe_encoder = WeightOfEvidenceEncoder(max_bin_count=10, separate_zero_bin=True)
            woe_encoder.fit(train_features, train_target)

            train_input = woe_encoder.transform(train_features)
            validation_input = woe_encoder.transform(validation_features)
        else:
            # 트리 계열 모델은 원본 값을 그대로 쓴다.
            # 트리는 "값이 X보다 큰가?"로 나누기 때문에 스케일 변환이 필요 없다.
            train_input = train_features
            validation_input = validation_features

        model = create_model(model_name, random_seed=random_seed)
        model.fit(train_input, train_target)

        # predict_proba 는 [정상일 확률, 사기일 확률] 을 돌려준다. [:, 1] 이 사기 확률.
        validation_probabilities = model.predict_proba(validation_input)[:, 1]

        metrics = evaluate_predictions(
            validation_target.to_numpy(), validation_probabilities, decision_threshold
        )
        metrics["fold"] = fold_index
        metrics["모델"] = model_name
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)


def summarize_fold_results(fold_results: pd.DataFrame) -> pd.Series:
    """
    fold 별 성적을 '평균 ± 표준편차' 형태로 요약한다.

    표준편차를 함께 보는 이유: 평균이 같아도 fold 마다 들쭉날쭉한 모델은
    새 데이터에서 성능을 예측하기 어렵습니다. 안정성도 성능의 일부입니다.
    """
    metric_names = ["AUC", "KS", "F1(사기)", "정밀도(사기)", "재현율(사기)"]
    summary = {"모델": fold_results["모델"].iloc[0]}

    for metric_name in metric_names:
        mean_value = fold_results[metric_name].mean()
        std_value = fold_results[metric_name].std()
        summary[metric_name] = f"{mean_value:.4f} ± {std_value:.4f}"
        summary[f"_{metric_name}_평균"] = mean_value  # 정렬용 숫자값

    return pd.Series(summary)


def compare_all_models(
    features: pd.DataFrame,
    target: pd.Series,
    model_names: list[str] | None = None,
    fold_count: int = config.CROSS_VALIDATION_FOLDS,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    모델 여러 개를 같은 조건으로 비교한다.

    돌려주는 값: (요약표, fold별 상세표)
    """
    if model_names is None:
        model_names = ALL_MODEL_NAMES

    all_fold_results = []
    summary_rows = []

    for model_name in model_names:
        fold_results = run_cross_validation(
            features, target, model_name, fold_count=fold_count, random_seed=random_seed
        )
        all_fold_results.append(fold_results)
        summary_rows.append(summarize_fold_results(fold_results))

    summary_table = pd.DataFrame(summary_rows).reset_index(drop=True)
    detailed_table = pd.concat(all_fold_results, ignore_index=True)

    return summary_table, detailed_table


def generate_out_of_fold_probabilities(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = MODEL_LIGHTGBM,
    fold_count: int = config.CROSS_VALIDATION_FOLDS,
    random_seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """
    '폴드 밖(out-of-fold)' 예측 확률을 만든다.

    ■ 이게 왜 필요한가
      임계값을 고르려면 "이 모델이 처음 보는 데이터에 어떤 확률을 줄까"를 알아야 합니다.
      학습에 쓴 데이터로 확률을 뽑으면 모델이 답을 외운 상태라 확률이 지나치게 확신에 차고,
      그걸로 임계값을 정하면 실제 서비스에서 엉뚱하게 동작합니다.

    ■ 방법
      교차검증과 똑같이 5조각으로 나눈 뒤, 각 조각을 '검증 fold' 로 쓸 때의 예측만 모읍니다.
      그러면 **모든 행이 "학습에 쓰이지 않은 상태에서 받은 확률"** 을 한 번씩 갖게 됩니다.
      단일 분할(80/20)보다 데이터를 다 쓸 수 있어 임계값 추정이 안정적입니다.

    돌려주는 값: 입력과 같은 길이의 확률 배열 (i번째 = i번째 행의 사기 확률)
    """
    cross_validator = StratifiedKFold(
        n_splits=fold_count, shuffle=True, random_state=random_seed
    )

    # 결과를 담을 빈 배열을 먼저 만들고, fold 마다 해당 위치에 채워 넣는다
    out_of_fold_probabilities = np.zeros(len(features), dtype=float)

    for train_indices, validation_indices in cross_validator.split(features, target):
        train_features = features.iloc[train_indices]
        train_target = target.iloc[train_indices]
        validation_features = features.iloc[validation_indices]

        if model_name == MODEL_LOGISTIC_WOE:
            woe_encoder = WeightOfEvidenceEncoder(max_bin_count=10, separate_zero_bin=True)
            woe_encoder.fit(train_features, train_target)
            train_input = woe_encoder.transform(train_features)
            validation_input = woe_encoder.transform(validation_features)
        else:
            train_input = train_features
            validation_input = validation_features

        model = create_model(model_name, random_seed=random_seed)
        model.fit(train_input, train_target)

        out_of_fold_probabilities[validation_indices] = model.predict_proba(validation_input)[:, 1]

    return out_of_fold_probabilities
