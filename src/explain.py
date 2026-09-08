"""
SHAP 으로 "모델이 왜 그렇게 판단했는지" 설명하는 모듈.

──────────────────────────────────────────────────────────────────────
■ SHAP 이란? (SHapley Additive exPlanations)

원래는 게임 이론에서 나온 개념입니다.
"여러 명이 함께 일해서 성과를 냈을 때, 각자의 공로를 어떻게 공정하게 나눌까?"

여기서 '선수'는 **특징(feature)** 이고, '성과'는 **모델의 예측값** 입니다.
SHAP 값은 **각 특징이 예측을 평균에서 얼마나 밀어 올렸는지/내렸는지**를 나타냅니다.

  예측 로짓 = 기준값(base value) + SHAP₁ + SHAP₂ + ... + SHAP₃₈

  기호 뜻:
    기준값   : 아무 정보도 없을 때의 예측. 학습 데이터 전체의 평균 로짓
    SHAPⱼ    : j번째 특징이 이 예측을 얼마나 밀었는지 (양수=사기 쪽, 음수=정상 쪽)

이 등식이 **정확히** 성립합니다(가법성, additivity). 그래서
"이 주소가 사기로 판정된 이유는 A 때문에 +2.1, B 때문에 +1.4..." 처럼
숫자로 딱 떨어지게 설명할 수 있습니다.

■ 로짓(logit)이 뭔가요?
  확률 p 를 −∞ ~ +∞ 범위로 펼친 값입니다.

      로짓 = ln( p / (1 − p) )

  왜 이렇게 하나? 확률은 0~1 사이에 갇혀 있어서 덧셈이 어색합니다
  (0.9 + 0.2 = 1.1 은 확률이 될 수 없죠). 로짓으로 펼치면 자유롭게 더할 수 있고,
  다 더한 뒤에 다시 확률로 되돌리면 됩니다.

      확률 = 1 / (1 + e^(−로짓))     ← 시그모이드 함수

  로짓 0 = 확률 50%, 로짓 +2 ≈ 88%, 로짓 −2 ≈ 12%

■ 전역 설명 vs 개별 설명
  전역(global) : "모델이 전반적으로 어떤 특징을 중요하게 보는가"
                 → 모든 샘플의 |SHAP| 을 평균낸 값으로 순위를 매긴다
  개별(local)  : "이 주소 하나를 왜 그렇게 판단했는가"
                 → 그 샘플의 SHAP 값을 하나씩 뜯어본다

  신용평가에서 '왜 대출이 거절됐는지' 고객에게 설명해야 할 때 개별 설명을 씁니다.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def logit_to_probability(logit_value: float) -> float:
    """
    로짓을 확률로 되돌린다 (시그모이드 함수).

        확률 = 1 / (1 + e^(−로짓))
    """
    return float(1.0 / (1.0 + np.exp(-logit_value)))


def extract_shap_matrix(shap_explanation) -> tuple[np.ndarray, np.ndarray]:
    """
    SHAP 결과에서 '사기 클래스' 기준의 값 행렬과 기준값을 꺼낸다.

    왜 이런 함수가 필요한가?
      shap 라이브러리는 버전과 모델 종류에 따라 결과 모양이 다릅니다.
      - 2차원 (샘플수, 특징수)          : 이진 분류에서 사기 클래스 값만 주는 경우
      - 3차원 (샘플수, 특징수, 클래스수) : 클래스별로 따로 주는 경우
      어느 쪽이 오든 똑같이 동작하도록 여기서 한 번에 처리합니다.
    """
    shap_values = shap_explanation.values
    base_values = np.asarray(shap_explanation.base_values)

    if shap_values.ndim == 3:
        # 마지막 축이 클래스. 인덱스 1 이 '사기(FLAG=1)' 클래스
        shap_values = shap_values[:, :, 1]
        base_values = base_values[:, 1]

    return shap_values, np.ravel(base_values)


def verify_shap_additivity(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    predicted_probabilities: np.ndarray,
) -> dict:
    """
    "기준값 + SHAP 합계 = 예측 로짓" 이 실제로 성립하는지 검산한다.

    SHAP 을 쓸 때 반드시 한 번은 확인해야 하는 성질입니다.
    이게 안 맞으면 설명 자체를 믿을 수 없습니다.
    """
    # 예측 확률을 로짓으로 되돌린다. clip 은 0 이나 1 로 나누기를 막기 위한 안전장치.
    clipped_probabilities = np.clip(predicted_probabilities, 1e-12, 1 - 1e-12)
    actual_logits = np.log(clipped_probabilities / (1 - clipped_probabilities))

    reconstructed_logits = shap_values.sum(axis=1) + base_values
    absolute_errors = np.abs(reconstructed_logits - actual_logits)

    return {
        "최대 오차": float(absolute_errors.max()),
        "평균 오차": float(absolute_errors.mean()),
        "가법성 성립": bool(absolute_errors.max() < 1e-6),
    }


def build_global_importance_table(
    shap_values: np.ndarray, feature_names: list[str], top_count: int | None = None
) -> pd.DataFrame:
    """
    전역 중요도 표를 만든다.

    중요도 = |SHAP| 의 평균 (표에서는 '평균 SHAP 크기' 열)
      절댓값을 쓰는 이유: 어떤 특징은 사기 쪽으로(+), 어떤 샘플에서는 정상 쪽으로(−)
      밀 수 있습니다. 그냥 더하면 서로 상쇄되어 0에 가까워집니다.
      "얼마나 크게 영향을 주는가"를 보려면 방향을 무시하고 크기만 봐야 합니다.

    평균 SHAP(부호 포함)도 함께 넣어 '주로 어느 쪽으로 미는지'를 볼 수 있게 했습니다.
    """
    importance_table = pd.DataFrame({
        "특징": feature_names,
        "평균 SHAP 크기": np.abs(shap_values).mean(axis=0),
        "평균 SHAP(부호)": shap_values.mean(axis=0),
    })
    importance_table = importance_table.sort_values("평균 SHAP 크기", ascending=False)
    importance_table = importance_table.reset_index(drop=True)
    importance_table.insert(0, "순위", range(1, len(importance_table) + 1))

    if top_count is not None:
        importance_table = importance_table.head(top_count)

    return importance_table


def build_single_explanation_table(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    feature_frame: pd.DataFrame,
    row_position: int,
    top_count: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    샘플 하나의 설명 표를 만든다.

    돌려주는 값:
      (표, 요약정보)
      표      : 영향이 큰 순서로 정렬한 특징별 기여도
      요약정보: 기준값, SHAP 합계, 최종 로짓, 최종 확률
    """
    sample_shap_values = shap_values[row_position]
    sample_feature_values = feature_frame.iloc[row_position]

    explanation_table = pd.DataFrame({
        "특징": feature_frame.columns,
        "이 주소의 값": sample_feature_values.to_numpy(),
        "SHAP 기여": sample_shap_values,
    })
    # 밀어 올린 방향을 한국어로 붙여 준다
    explanation_table["방향"] = np.where(
        explanation_table["SHAP 기여"] > 0, "사기 쪽 ↑", "정상 쪽 ↓"
    )
    # 영향의 '크기'가 큰 순서로 정렬 (부호와 무관하게)
    explanation_table = explanation_table.reindex(
        explanation_table["SHAP 기여"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    base_value = float(base_values[row_position])
    shap_sum = float(sample_shap_values.sum())
    final_logit = base_value + shap_sum

    summary = {
        "기준값(base value)": base_value,
        "SHAP 합계": shap_sum,
        "최종 로짓": final_logit,
        "최종 사기 확률": logit_to_probability(final_logit),
    }

    return explanation_table.head(top_count), summary


def pick_representative_samples(
    predicted_probabilities: np.ndarray, true_labels: np.ndarray
) -> dict[str, int]:
    """
    개별 설명에 쓸 대표 샘플 2개를 고른다.

    고르는 기준: **모델이 자신 있게, 그리고 정확하게 맞힌** 사례
      - 사기 대표 : 실제 사기이면서 사기 확률이 가장 높은 주소
      - 정상 대표 : 실제 정상이면서 사기 확률이 가장 낮은 주소

    왜 이렇게 고르나? 애매한 사례는 설명도 애매합니다.
    "모델이 생각하는 전형적인 사기/정상"이 무엇인지 보려면 확신이 강한 사례가 좋습니다.
    """
    true_labels = np.asarray(true_labels)

    # np.where 로 조건에 맞는 행 위치를 찾고, 그 안에서 확률이 극단인 것을 고른다
    fraud_positions = np.where(true_labels == 1)[0]
    normal_positions = np.where(true_labels == 0)[0]

    most_confident_fraud = fraud_positions[np.argmax(predicted_probabilities[fraud_positions])]
    most_confident_normal = normal_positions[np.argmin(predicted_probabilities[normal_positions])]

    return {
        "사기 대표": int(most_confident_fraud),
        "정상 대표": int(most_confident_normal),
    }
