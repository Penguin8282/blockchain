"""
임계값(cut-off)을 비용 관점에서 고르는 모듈 — FDS(이상거래탐지) 실무 관점.

──────────────────────────────────────────────────────────────────────
■ 문제: 모델은 확률을 주는데, 우리는 '예/아니오'를 결정해야 한다

모델은 "이 주소가 사기일 확률 0.37" 같은 숫자를 줍니다.
하지만 실제로는 **차단할지 말지**를 정해야 합니다.
그 경계선이 **임계값(threshold, cut-off)** 입니다.

    확률 >= 임계값  ->  사기로 판정 (차단·심사)
    확률 <  임계값  ->  정상으로 통과

기본값 0.5 를 쓰는 경우가 많지만, **0.5 에는 아무 근거가 없습니다.**
"틀리는 두 가지 방식의 비용이 같다"고 가정할 때만 타당한 값입니다.

■ 틀리는 방식은 두 가지고, 비용이 전혀 다르다

                        실제 사기        실제 정상
    사기로 판정        TP (잘 잡음)    FP (오탐) ← 정상 고객이 막힘
    정상으로 통과      FN (놓침) ←피해   TN (잘 통과)

    FN (False Negative, 미탐) : 사기를 놓침 → **금전 피해가 그대로 발생**
    FP (False Positive, 오탐) : 정상을 막음 → **고객 불편 + 심사 인건비**

보통 FN 이 FP 보다 훨씬 비쌉니다. 사기 1건을 놓치면 수백만 원이 날아가지만,
정상 고객 1명을 잘못 막으면 심사 몇 분과 사과 한 번으로 끝나는 경우가 많으니까요.

■ 그래서 총비용을 최소화하는 임계값을 찾는다

    총비용(t) = FN(t) × C_FN  +  FP(t) × C_FP

    기호 뜻:
      t      : 임계값 (0~1)
      FN(t)  : 임계값을 t 로 뒀을 때 놓친 사기 건수
      FP(t)  : 임계값을 t 로 뒀을 때 잘못 막은 정상 건수
      C_FN   : 사기 1건을 놓쳤을 때의 비용 (원)
      C_FP   : 정상 1건을 잘못 막았을 때의 비용 (원)

  임계값을 낮추면 → 사기를 많이 잡지만(FN↓) 오탐이 늘어난다(FP↑)
  임계값을 높이면 → 오탐은 줄지만(FP↓) 사기를 많이 놓친다(FN↑)
  둘의 합이 최소가 되는 지점이 최적 임계값입니다.

  중요한 것은 **C_FN 과 C_FP 의 절대값이 아니라 비율**입니다.
  둘 다 10배 해도 최적 임계값은 그대로니까요.

■ 비용만으로 정할 수 없는 현실 제약: 처리 용량
  임계값을 아주 낮추면 사기를 다 잡지만, 심사팀에 하루 수천 건이 쏟아집니다.
  실무에서는 "하루에 볼 수 있는 건수"가 정해져 있어서,
  **경보율(alert rate)** 을 함께 봐야 합니다.

      경보율 = 사기로 판정한 건수 / 전체 건수
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 비교해 볼 비용 비율들 (C_FN : C_FP)
# 1:1 은 "두 실수의 비용이 같다"는 뜻이고, 이때 최적 임계값이 0.5 근처로 나오는지
# 확인하면 계산이 맞게 되었는지 검산할 수 있다.
DEFAULT_COST_RATIOS = [1, 5, 10, 20, 50]


def build_threshold_table(
    true_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    threshold_count: int = 99,
) -> pd.DataFrame:
    """
    여러 임계값에서의 혼동행렬과 지표를 한 표로 만든다.

    돌려주는 표의 각 행 = 임계값 하나에서의 결과
      TP : 사기를 사기로 맞힘        FP : 정상을 사기로 잘못 찍음(오탐)
      FN : 사기를 정상으로 놓침(미탐)  TN : 정상을 정상으로 맞힘
    """
    true_labels = np.asarray(true_labels)
    predicted_probabilities = np.asarray(predicted_probabilities)

    # 0.01 부터 0.99 까지 균등하게 나눈 임계값 후보들
    thresholds = np.linspace(0.01, 0.99, threshold_count)

    total_fraud_count = int((true_labels == 1).sum())
    total_normal_count = int((true_labels == 0).sum())
    total_count = len(true_labels)

    rows = []
    for threshold in thresholds:
        predicted_labels = (predicted_probabilities >= threshold).astype(int)

        true_positive = int(((predicted_labels == 1) & (true_labels == 1)).sum())
        false_positive = int(((predicted_labels == 1) & (true_labels == 0)).sum())
        false_negative = int(((predicted_labels == 0) & (true_labels == 1)).sum())
        true_negative = int(((predicted_labels == 0) & (true_labels == 0)).sum())

        # 0으로 나누는 것을 막기 위해 분모가 0이면 0을 넣는다
        alert_count = true_positive + false_positive
        precision = true_positive / alert_count if alert_count > 0 else 0.0
        recall = true_positive / total_fraud_count if total_fraud_count > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        rows.append({
            "임계값": round(float(threshold), 3),
            "TP(사기 적발)": true_positive,
            "FP(정상 오탐)": false_positive,
            "FN(사기 미탐)": false_negative,
            "TN(정상 통과)": true_negative,
            "정밀도": precision,
            "재현율": recall,
            "F1": f1_score,
            "경보율": alert_count / total_count,
            "미탐율": false_negative / total_fraud_count if total_fraud_count > 0 else 0.0,
            "오탐율": false_positive / total_normal_count if total_normal_count > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def add_cost_column(
    threshold_table: pd.DataFrame,
    cost_of_missed_fraud: float,
    cost_of_false_alarm: float,
    column_name: str = "총비용",
) -> pd.DataFrame:
    """
    비용 열을 추가한다.

        총비용 = FN × C_FN + FP × C_FP

    cost_of_missed_fraud : C_FN — 사기 1건을 놓쳤을 때의 비용
    cost_of_false_alarm  : C_FP — 정상 1건을 잘못 막았을 때의 비용
    """
    result_table = threshold_table.copy()
    result_table[column_name] = (
        result_table["FN(사기 미탐)"] * cost_of_missed_fraud
        + result_table["FP(정상 오탐)"] * cost_of_false_alarm
    )
    return result_table


def find_minimum_cost_threshold(
    threshold_table: pd.DataFrame,
    cost_of_missed_fraud: float,
    cost_of_false_alarm: float,
) -> dict:
    """
    총비용이 가장 작아지는 임계값과 그때의 상세 지표를 돌려준다.
    """
    table_with_cost = add_cost_column(
        threshold_table, cost_of_missed_fraud, cost_of_false_alarm
    )
    # idxmin 은 값이 가장 작은 행의 인덱스를 돌려준다
    best_row = table_with_cost.loc[table_with_cost["총비용"].idxmin()]

    return {
        "비용비(C_FN:C_FP)": f"{cost_of_missed_fraud / cost_of_false_alarm:.0f} : 1",
        "최적 임계값": float(best_row["임계값"]),
        "TP(사기 적발)": int(best_row["TP(사기 적발)"]),
        "FP(정상 오탐)": int(best_row["FP(정상 오탐)"]),
        "FN(사기 미탐)": int(best_row["FN(사기 미탐)"]),
        "재현율": float(best_row["재현율"]),
        "정밀도": float(best_row["정밀도"]),
        "F1": float(best_row["F1"]),
        "경보율": float(best_row["경보율"]),
        "총비용": float(best_row["총비용"]),
    }


def build_cost_ratio_comparison(
    threshold_table: pd.DataFrame,
    cost_of_false_alarm: float,
    cost_ratios: list[float] | None = None,
) -> pd.DataFrame:
    """
    여러 비용 비율에 대해 최적 임계값을 각각 구해 한 표로 모은다.

    C_FP 는 고정하고 C_FN = 비율 × C_FP 로 두어 비교합니다.
    (최적 임계값은 비율에만 좌우되므로 이렇게 해도 일반성을 잃지 않습니다)
    """
    if cost_ratios is None:
        cost_ratios = DEFAULT_COST_RATIOS

    rows = []
    for ratio in cost_ratios:
        result = find_minimum_cost_threshold(
            threshold_table,
            cost_of_missed_fraud=ratio * cost_of_false_alarm,
            cost_of_false_alarm=cost_of_false_alarm,
        )
        rows.append(result)

    return pd.DataFrame(rows)


def find_reference_thresholds(threshold_table: pd.DataFrame) -> pd.DataFrame:
    """
    비용을 모를 때 흔히 쓰는 '참고용 임계값' 후보들을 모아 준다.

    - F1 최대   : 정밀도와 재현율의 균형점. 비용 정보가 전혀 없을 때의 기본 선택
    - KS 최대   : 사기 분포와 정상 분포가 가장 크게 갈리는 지점
                  (KS = 재현율 − 오탐율. 단계 4의 KS 정의와 같은 식)
    - 기본값 0.5 : 아무 근거 없이 흔히 쓰이는 값. 비교 기준으로만 둔다
    """
    table = threshold_table.copy()
    # KS 는 '재현율(TPR) − 오탐율(FPR)' 이다. 단계 4에서 쓴 정의와 동일하다.
    table["KS"] = table["재현율"] - table["오탐율"]

    candidates = {
        "F1 최대": table.loc[table["F1"].idxmax()],
        "KS 최대": table.loc[table["KS"].idxmax()],
        # 0.5 에 가장 가까운 임계값 행을 고른다
        "기본값 0.5": table.loc[(table["임계값"] - 0.5).abs().idxmin()],
    }

    rows = []
    for label, row in candidates.items():
        rows.append({
            "기준": label,
            "임계값": float(row["임계값"]),
            "재현율": float(row["재현율"]),
            "정밀도": float(row["정밀도"]),
            "F1": float(row["F1"]),
            "경보율": float(row["경보율"]),
            "FN(사기 미탐)": int(row["FN(사기 미탐)"]),
            "FP(정상 오탐)": int(row["FP(정상 오탐)"]),
        })

    return pd.DataFrame(rows)


def apply_capacity_constraint(
    threshold_table: pd.DataFrame, maximum_alert_rate: float
) -> dict:
    """
    처리 용량 제약이 있을 때 쓸 수 있는 가장 낮은 임계값을 찾는다.

    현실의 심사팀은 하루에 볼 수 있는 건수가 정해져 있습니다.
    "경보율이 5% 를 넘으면 감당 못 한다"면, 그 조건을 만족하는 것 중
    **가장 낮은 임계값**(= 사기를 가장 많이 잡는 임계값)을 골라야 합니다.
    """
    feasible_rows = threshold_table[threshold_table["경보율"] <= maximum_alert_rate]

    if len(feasible_rows) == 0:
        return {"가능 여부": False, "사유": f"경보율 {maximum_alert_rate:.1%} 이하가 불가능"}

    # 조건을 만족하는 것 중 임계값이 가장 낮은 행 = 사기를 가장 많이 잡는 선택
    best_row = feasible_rows.loc[feasible_rows["임계값"].idxmin()]

    return {
        "가능 여부": True,
        "최대 경보율": maximum_alert_rate,
        "임계값": float(best_row["임계값"]),
        "실제 경보율": float(best_row["경보율"]),
        "재현율": float(best_row["재현율"]),
        "정밀도": float(best_row["정밀도"]),
        "FN(사기 미탐)": int(best_row["FN(사기 미탐)"]),
        "FP(정상 오탐)": int(best_row["FP(정상 오탐)"]),
    }
