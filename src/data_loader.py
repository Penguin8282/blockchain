"""
원본 CSV 를 읽어 들이고, 분석하기 좋은 형태로 정리해 주는 모듈.

여기서 하는 정리 작업은 딱 세 가지뿐이다(모델링에 영향을 주는 가공은 하지 않는다):
  1) 열 이름 앞뒤의 불필요한 공백 제거   예) ' ERC20 min val rec' -> 'ERC20 min val rec'
  2) 식별자 열(Unnamed: 0, Index, Address)과 목표 열(FLAG)을 구분해 두기
  3) 47개 특징을 '거래 빈도 / 금액 / 시간 간격 / 토큰 활동' 4묶음으로 분류해 두기

왜 열 이름 공백을 지우나?
  원본 CSV 에는 ' ERC20 min val rec' 처럼 앞에 공백이 붙은 열이 26개 있다.
  이런 이름은 df['ERC20 min val rec'] 로 접근하면 KeyError 가 나서, 나중에
  "분명 있는 열인데 왜 없다고 하지?" 하며 시간을 버리게 된다. 그래서 미리 정리한다.
"""

from __future__ import annotations

import pandas as pd

from src import config

# ------------------------------------------------------------
# 1) 식별자 열과 목표 열
# ------------------------------------------------------------
# 아래 세 열은 '지갑을 구분하는 이름표'일 뿐, 사기 여부를 예측하는 재료가 아니다.
# 모델에 넣으면 안 된다(넣으면 의미 없는 번호로 정답을 외워버릴 수 있다).
IDENTIFIER_COLUMNS = [
    "Unnamed: 0",  # CSV 저장할 때 딸려온 행 번호
    "Index",       # 출처 불명의 번호 (data_dictionary.md 에 '확인 필요'로 표시)
    "Address",     # 지갑 주소 문자열
]

# 목표(정답) 열: 1 = 사기, 0 = 정상
TARGET_COLUMN = config.TARGET_COLUMN_NAME  # "FLAG"

# ------------------------------------------------------------
# 2) 특징 4묶음 정의
# ------------------------------------------------------------
# 아래 이름들은 '공백을 제거한 뒤'의 이름 기준이다.
#
# 분류 기준:
#   거래 빈도 : "몇 번" 했는가 (횟수·상대방 수)  -> 단위가 '개수'
#   금액      : "얼마" 주고받았는가              -> 단위가 'Ether'
#   시간 간격 : "얼마나 자주/오래"               -> 단위가 '분(minutes)'
#   토큰 활동 : ERC20 토큰 관련 전부              -> 이더 자체가 아닌 '토큰' 세계의 행동
#
# 주의: ERC20 열 중에도 시간·금액 성격인 것이 있지만, 그것들의 정체성은
#       "토큰 활동"이므로 토큰 묶음에 넣었다. 대신 아래 FEATURE_MEASUREMENT_AXIS 에
#       "이 열이 실제로 재는 것이 시간인지 금액인지"를 따로 기록해 둔다.

TRANSACTION_FREQUENCY_FEATURES = [
    "Sent tnx",                                            # 보낸 거래 횟수
    "Received Tnx",                                        # 받은 거래 횟수
    "Number of Created Contracts",                         # 생성한 컨트랙트 개수
    "Unique Received From Addresses",                      # 서로 다른 '보낸 사람' 수
    "Unique Sent To Addresses",                            # 서로 다른 '받는 사람' 수
    "total transactions (including tnx to create contract", # 전체 거래 횟수
]

MONETARY_VALUE_FEATURES = [
    "min value received",          # 받은 금액 중 최솟값
    "max value received",          # 받은 금액 중 최댓값
    "avg val received",            # 받은 금액 평균
    "min val sent",                # 보낸 금액 중 최솟값
    "max val sent",                # 보낸 금액 중 최댓값
    "avg val sent",                # 보낸 금액 평균
    "min value sent to contract",  # 컨트랙트로 보낸 금액 최솟값
    "max val sent to contract",    # 컨트랙트로 보낸 금액 최댓값
    "avg value sent to contract",  # 컨트랙트로 보낸 금액 평균
    "total Ether sent",            # 총 보낸 이더
    "total ether received",        # 총 받은 이더
    "total ether sent contracts",  # 컨트랙트로 보낸 총 이더
    "total ether balance",         # 잔액 (받은 총액 - 보낸 총액)
]

TIME_INTERVAL_FEATURES = [
    "Avg min between sent tnx",                # 보낸 거래 사이 평균 간격(분)
    "Avg min between received tnx",            # 받은 거래 사이 평균 간격(분)
    "Time Diff between first and last (Mins)", # 첫 거래~마지막 거래 총 기간(분) = 지갑 수명
]

TOKEN_ACTIVITY_FEATURES = [
    "Total ERC20 tnxs",
    "ERC20 total Ether received",
    "ERC20 total ether sent",
    "ERC20 total Ether sent contract",
    "ERC20 uniq sent addr",
    "ERC20 uniq rec addr",
    "ERC20 uniq sent addr.1",
    "ERC20 uniq rec contract addr",
    "ERC20 avg time between sent tnx",
    "ERC20 avg time between rec tnx",
    "ERC20 avg time between rec 2 tnx",
    "ERC20 avg time between contract tnx",
    "ERC20 min val rec",
    "ERC20 max val rec",
    "ERC20 avg val rec",
    "ERC20 min val sent",
    "ERC20 max val sent",
    "ERC20 avg val sent",
    "ERC20 min val sent contract",
    "ERC20 max val sent contract",
    "ERC20 avg val sent contract",
    "ERC20 uniq sent token name",
    "ERC20 uniq rec token name",
    "ERC20 most sent token type",
    "ERC20_most_rec_token_type",
]

# 묶음 이름 -> 열 목록 (표를 만들 때 이 사전을 순회한다)
FEATURE_GROUPS: dict[str, list[str]] = {
    "거래 빈도": TRANSACTION_FREQUENCY_FEATURES,
    "금액": MONETARY_VALUE_FEATURES,
    "시간 간격": TIME_INTERVAL_FEATURES,
    "토큰 활동": TOKEN_ACTIVITY_FEATURES,
}

# ERC20 열이 '실제로 재는 것'이 무엇인지 보조 기록.
# (묶음은 '토큰 활동'이지만, 성격은 시간/금액/빈도/범주 중 하나다)
FEATURE_MEASUREMENT_AXIS: dict[str, str] = {
    "Total ERC20 tnxs": "빈도",
    "ERC20 total Ether received": "금액",
    "ERC20 total ether sent": "금액",
    "ERC20 total Ether sent contract": "금액",
    "ERC20 uniq sent addr": "빈도",
    "ERC20 uniq rec addr": "빈도",
    "ERC20 uniq sent addr.1": "빈도",
    "ERC20 uniq rec contract addr": "빈도",
    "ERC20 avg time between sent tnx": "시간",
    "ERC20 avg time between rec tnx": "시간",
    "ERC20 avg time between rec 2 tnx": "시간",
    "ERC20 avg time between contract tnx": "시간",
    "ERC20 min val rec": "금액",
    "ERC20 max val rec": "금액",
    "ERC20 avg val rec": "금액",
    "ERC20 min val sent": "금액",
    "ERC20 max val sent": "금액",
    "ERC20 avg val sent": "금액",
    "ERC20 min val sent contract": "금액",
    "ERC20 max val sent contract": "금액",
    "ERC20 avg val sent contract": "금액",
    "ERC20 uniq sent token name": "빈도",
    "ERC20 uniq rec token name": "빈도",
    "ERC20 most sent token type": "범주(문자열)",
    "ERC20_most_rec_token_type": "범주(문자열)",
}

# 의미가 불확실해 '확인 필요'로 표시한 열들.
# 추측해서 잘못된 해석을 퍼뜨리지 않기 위해 코드에도 명시해 둔다.
COLUMNS_NEEDING_VERIFICATION = [
    "Index",
    "ERC20 uniq sent addr.1",
    "ERC20 avg time between rec 2 tnx",
]


def load_raw_dataset(csv_path=None) -> pd.DataFrame:
    """
    원본 CSV 를 읽어 열 이름의 앞뒤 공백만 제거한 DataFrame 을 돌려준다.

    csv_path 를 주지 않으면 config.RAW_DATASET_PATH (data/raw/transaction_dataset.csv)를 쓴다.
    """
    dataset_path = csv_path if csv_path is not None else config.RAW_DATASET_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"원본 데이터를 찾을 수 없습니다: {dataset_path}\n"
            "먼저 `python -m src.download_dataset` 을 실행하세요."
        )

    dataframe = pd.read_csv(dataset_path)

    # str.strip() 은 문자열 앞뒤의 공백을 제거한다. 열 이름 전체에 한 번에 적용.
    dataframe.columns = [column_name.strip() for column_name in dataframe.columns]

    # 공백을 지운 뒤 이름이 겹치는 열이 생기면 조용히 덮어써질 위험이 있으므로 확인한다.
    duplicated_names = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicated_names:
        raise ValueError(f"공백 제거 후 열 이름이 중복됩니다: {duplicated_names}")

    return dataframe


def get_feature_column_names() -> list[str]:
    """4개 묶음에 속한 특징 열 이름을 하나의 리스트로 합쳐 돌려준다(총 47개)."""
    feature_names: list[str] = []
    for column_names in FEATURE_GROUPS.values():
        feature_names.extend(column_names)
    return feature_names


def find_constant_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    '상수 열'(모든 행이 같은 값인 열)을 찾아 이름 목록을 돌려준다.

    상수 열은 정보가 0이다. 사기든 정상이든 값이 똑같으니 구분에 아무 도움이 안 된다.
    (비유: 모든 학생의 시험 점수가 100점이면, 그 점수로는 누가 잘했는지 알 수 없다)
    nunique(dropna=True) 는 결측을 뺀 고유값 개수를 센다.
    """
    return [
        column_name
        for column_name in dataframe.columns
        if dataframe[column_name].nunique(dropna=True) <= 1
    ]


def build_missing_value_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    열별 결측 개수와 비율을 표로 만들어 돌려준다(결측이 있는 열만).
    """
    missing_counts = dataframe.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    return pd.DataFrame(
        {
            "결측 개수": missing_counts,
            "결측 비율(%)": (missing_counts / len(dataframe) * 100).round(2),
        }
    )


def summarize_class_balance(dataframe: pd.DataFrame) -> dict:
    """
    목표 열(FLAG)의 클래스 불균형 정도를 계산해 사전(dict)으로 돌려준다.

    불균형 비율 = 정상 개수 / 사기 개수
      -> 이 값이 크면 "정상이 압도적으로 많다"는 뜻이고,
         모델이 무조건 '정상'이라고만 답해도 정확도가 높게 나오는 함정이 생긴다.
         그래서 정확도(accuracy) 대신 AUC·F1 같은 지표를 쓴다.
    """
    target_series = dataframe[TARGET_COLUMN]
    normal_count = int((target_series == 0).sum())
    fraud_count = int((target_series == 1).sum())

    return {
        "전체 행 수": len(dataframe),
        "정상(FLAG=0)": normal_count,
        "사기(FLAG=1)": fraud_count,
        "사기 비율(%)": round(fraud_count / len(dataframe) * 100, 2),
        "불균형 비율(정상:사기)": round(normal_count / fraud_count, 2),
    }
