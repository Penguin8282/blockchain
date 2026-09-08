"""
원본 데이터를 '모델에 넣을 수 있는 형태'로 다듬는 모듈.

단계 2에서 발견한 **데이터 누수** 때문에, 이 모듈은 서로 다른 두 벌의 데이터를 만듭니다.

  트랙 "full"     : 누수를 그대로 둔 데이터 (Kaggle 공개 노트북들이 쓰는 방식)
                    → 'ERC20 이력 없음' 표시 열을 특징으로 넣습니다.
                      이 열 하나가 829행을 100% 사기로 찍어 버립니다.
  트랙 "no_leak"  : 누수 경로를 최소한으로 차단한 데이터
                    → 표시 열을 만들지 않고, ERC20 결측을 그냥 0으로 채웁니다.
                      그러면 '결측이라 100% 사기'였던 829행이
                      '원래 값이 0이라 100% 정상'이던 4,399행과 같은 칸에 섞여
                      완벽한 분리가 사라집니다 (사기 비율 15.9%).

두 데이터로 같은 모델을 돌려 성능을 비교하면, 누수가 성능을 얼마나 부풀렸는지
숫자로 보여줄 수 있습니다.

공통으로 하는 정리:
  1) 중복 주소 행 제거          (같은 주소가 학습셋과 검증셋에 나뉘어 들어가면 성능이 부풀려짐)
  2) 상수 열 7개 제거           (모든 값이 같으면 정보가 0이고, WoE 계산도 깨짐)
  3) 식별자 열 3개 제거         (주소·행번호는 예측 재료가 아님)
  4) 문자열 열 2개 제거         (아래 '설계 결정' 참고)
  5) ERC20 결측을 0으로 채우기  ("토큰 거래 이력 없음"을 0으로 표현)

■ 설계 결정: 문자열 열 2개를 왜 모델에서 빼는가
  `ERC20 most sent token type`(304종), `ERC20_most_rec_token_type`(466종)은
  토큰 '이름' 문자열입니다. 종류가 너무 많아서 모델에 넣으려면 목표값 기반 인코딩
  (target encoding)이 필요한데, 그 인코딩 자체가 교차검증 안에서 새로운 누수를
  만들기 쉽습니다. 게다가 이 열들의 정보 상당 부분은
  `ERC20 uniq sent/rec token name`(토큰 종류 개수)에 이미 들어 있습니다.
  그래서 **모델 입력에서는 빼되, 단계 3의 IV 순위표에는 포함**해
  "얼마나 유용했을지"는 확인할 수 있게 했습니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_loader

# 누수를 만드는 표시 열의 이름
ERC20_MISSING_INDICATOR_NAME = "ERC20 이력 없음(표시)"

# 이 열의 결측 여부가 곧 누수의 통로였다
LEAKAGE_SOURCE_COLUMN = "Total ERC20 tnxs"

TRACK_FULL = "full"
TRACK_NO_LEAK = "no_leak"


def get_string_feature_names(dataframe: pd.DataFrame) -> list[str]:
    """특징 중 문자열(범주형) 열 이름을 돌려준다."""
    return [
        column_name
        for column_name in data_loader.get_feature_column_names()
        if not pd.api.types.is_numeric_dtype(dataframe[column_name])
    ]


def build_modeling_dataset(
    track: str = TRACK_NO_LEAK,
    include_string_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    모델에 넣을 특징 행렬 X 와 정답 y 를 만들어 돌려준다.

    매개변수
      track : "full"(누수 포함) 또는 "no_leak"(누수 최소 차단)
      include_string_features : True 면 문자열 열 2개도 포함한다.
                                단계 3의 IV 계산에서만 True 로 쓴다.

    돌려주는 값
      (X, y, info)  — info 에는 몇 개가 어떻게 제거됐는지 기록이 담긴다.
    """
    if track not in (TRACK_FULL, TRACK_NO_LEAK):
        raise ValueError(f"track 은 '{TRACK_FULL}' 또는 '{TRACK_NO_LEAK}' 여야 합니다: {track}")

    dataframe = data_loader.load_raw_dataset()
    processing_log: dict = {"트랙": track, "시작 행 수": len(dataframe)}

    # ── 1) 중복 주소 제거 ────────────────────────────────────────────
    # keep="first": 같은 주소가 여러 번 나오면 첫 번째만 남긴다.
    before_row_count = len(dataframe)
    dataframe = dataframe.drop_duplicates(subset=["Address"], keep="first").reset_index(drop=True)
    processing_log["중복 주소 제거"] = before_row_count - len(dataframe)

    # ── 2) 누수 표시 열은 '제거하기 전에' 만들어 둔다 ─────────────────
    # (결측을 0으로 채우고 나면 '원래 비어 있었다'는 정보가 사라지기 때문)
    erc20_history_missing = dataframe[LEAKAGE_SOURCE_COLUMN].isna()

    # ── 3) 특징 열 고르기 ───────────────────────────────────────────
    constant_column_names = data_loader.find_constant_columns(dataframe)
    string_column_names = get_string_feature_names(dataframe)

    selected_feature_names = []
    for feature_name in data_loader.get_feature_column_names():
        if feature_name in constant_column_names:
            continue  # 상수 열 제외
        if feature_name in string_column_names and not include_string_features:
            continue  # 문자열 열 제외 (기본값)
        selected_feature_names.append(feature_name)

    processing_log["상수 열 제거"] = len(constant_column_names)
    processing_log["문자열 열 제거"] = (
        0 if include_string_features else len(string_column_names)
    )

    feature_frame = dataframe[selected_feature_names].copy()

    # ── 4) ERC20 결측을 0으로 채우기 ────────────────────────────────
    # "토큰 거래 이력이 없다" = "토큰 거래량이 0이다" 로 표현한다.
    # 평균으로 채우면 '거래한 적 없음'이 '평균만큼 거래함'으로 바뀌므로 절대 안 된다.
    numeric_feature_names = [
        name for name in selected_feature_names
        if pd.api.types.is_numeric_dtype(feature_frame[name])
    ]
    filled_cell_count = int(feature_frame[numeric_feature_names].isna().sum().sum())
    feature_frame[numeric_feature_names] = feature_frame[numeric_feature_names].fillna(0.0)
    processing_log["0으로 채운 칸 수"] = filled_cell_count

    # ── 5) 트랙에 따라 누수 표시 열을 넣거나 뺀다 ────────────────────
    if track == TRACK_FULL:
        # 누수를 '일부러' 남기는 트랙. astype(int) 로 True/False 를 1/0 으로 바꾼다.
        feature_frame[ERC20_MISSING_INDICATOR_NAME] = erc20_history_missing.astype(int).to_numpy()
        processing_log["누수 표시 열"] = "포함"
    else:
        processing_log["누수 표시 열"] = "제외"

    target_series = dataframe[data_loader.TARGET_COLUMN].reset_index(drop=True)

    processing_log["최종 행 수"] = len(feature_frame)
    processing_log["최종 특징 수"] = feature_frame.shape[1]
    processing_log["사기 비율(%)"] = round(float(target_series.mean()) * 100, 2)

    return feature_frame, target_series, processing_log


def describe_residual_leakage(feature_frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """
    '최소 제거' 후에도 남아 있는 누수 위험을 점검한다.

    확인하는 것: 모든 특징이 0인 '껍데기 행'이 있는지, 있다면 그 행들의 사기 비율은 얼마인지.
    이런 행이 100% 사기라면, 모델은 "값이 다 0이면 사기"라는 규칙을 외울 수 있다.
    """
    numeric_columns = [
        name for name in feature_frame.columns
        if pd.api.types.is_numeric_dtype(feature_frame[name])
    ]
    all_zero_rows = (feature_frame[numeric_columns] == 0).all(axis=1)

    rows = [
        {
            "구분": "모든 특징이 0인 행",
            "행 수": int(all_zero_rows.sum()),
            "사기 수": int(target[all_zero_rows].sum()),
            "사기 비율(%)": round(float(target[all_zero_rows].mean()) * 100, 1)
            if all_zero_rows.any() else np.nan,
        },
        {
            "구분": "그 외 행",
            "행 수": int((~all_zero_rows).sum()),
            "사기 수": int(target[~all_zero_rows].sum()),
            "사기 비율(%)": round(float(target[~all_zero_rows].mean()) * 100, 1),
        },
    ]
    return pd.DataFrame(rows)
