"""
reports/data_dictionary.md (데이터 사전)를 자동으로 만들어 주는 스크립트.

왜 손으로 안 쓰고 스크립트로 만드나?
  - 결측 개수·고유값 수 같은 숫자는 데이터가 바뀌면 같이 바뀌어야 한다.
  - 손으로 쓰면 데이터와 문서가 어긋나기 쉽다. 그래서 '설명 문장'만 사람이 쓰고,
    '숫자'는 코드가 매번 다시 계산하게 한다.

실행:  python -m src.build_data_dictionary
"""

import sys

import pandas as pd

from src import config
from src import data_loader

# ------------------------------------------------------------
# 열 이름 -> (한국어 설명, 단위, 확신 수준)
# ------------------------------------------------------------
# 확신 수준:
#   확실   : 열 이름과 실제 값 분포가 일치해 해석에 이견이 없음
#   추정   : 이름상 의미는 분명하나 계산 방식(예: 0값 포함 여부)까지는 확인 못 함
#   확인 필요 : 이름만으로 의미를 알 수 없음. 추측하지 않고 남겨 둠
COLUMN_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    # ---------- 식별자 / 목표 ----------
    "Unnamed: 0": ("CSV 를 저장할 때 딸려 들어온 행 번호. 분석에 쓰지 않음", "-", "확실"),
    "Index": ("출처 불명의 번호. 9841행인데 고유값이 4729개뿐이라 단순 행 번호가 아님", "-", "확인 필요"),
    "Address": ("이더리움 지갑 주소(0x로 시작하는 문자열). 행을 구분하는 이름표", "-", "확실"),
    "FLAG": ("목표 변수. 1 = 사기(피싱·스캠) 주소, 0 = 정상 주소", "0/1", "확실"),

    # ---------- 거래 빈도 ----------
    "Sent tnx": ("이 지갑이 이더를 '보낸' 거래의 총 횟수", "건", "확실"),
    "Received Tnx": ("이 지갑이 이더를 '받은' 거래의 총 횟수", "건", "확실"),
    "Number of Created Contracts": ("이 지갑이 새로 만든 스마트 컨트랙트의 개수", "개", "확실"),
    "Unique Received From Addresses": ("이 지갑에 이더를 보내 준 '서로 다른' 주소의 수", "개", "확실"),
    "Unique Sent To Addresses": ("이 지갑이 이더를 보낸 '서로 다른' 주소의 수", "개", "확실"),
    "total transactions (including tnx to create contract": (
        "보낸 거래 + 받은 거래 + 컨트랙트 생성 거래를 모두 합한 총 거래 수", "건", "추정"),

    # ---------- 금액 ----------
    "min value received": ("받은 거래들 중 가장 작은 금액", "Ether", "추정"),
    "max value received": ("받은 거래들 중 가장 큰 금액", "Ether", "추정"),
    "avg val received": ("받은 거래 1건당 평균 금액", "Ether", "추정"),
    "min val sent": ("보낸 거래들 중 가장 작은 금액", "Ether", "추정"),
    "max val sent": ("보낸 거래들 중 가장 큰 금액", "Ether", "추정"),
    "avg val sent": ("보낸 거래 1건당 평균 금액", "Ether", "추정"),
    "min value sent to contract": ("스마트 컨트랙트로 보낸 거래 중 가장 작은 금액", "Ether", "추정"),
    "max val sent to contract": ("스마트 컨트랙트로 보낸 거래 중 가장 큰 금액", "Ether", "추정"),
    "avg value sent to contract": ("스마트 컨트랙트로 보낸 거래 1건당 평균 금액", "Ether", "추정"),
    "total Ether sent": ("이 지갑이 보낸 이더의 총합", "Ether", "확실"),
    "total ether received": ("이 지갑이 받은 이더의 총합", "Ether", "확실"),
    "total ether sent contracts": ("스마트 컨트랙트로 보낸 이더의 총합", "Ether", "확실"),
    "total ether balance": ("잔액. '받은 총액 - 보낸 총액'으로 계산된 것으로 보임(음수도 존재)", "Ether", "추정"),

    # ---------- 시간 간격 ----------
    "Avg min between sent tnx": ("보낸 거래와 그 다음 보낸 거래 사이의 평균 시간 간격", "분", "확실"),
    "Avg min between received tnx": ("받은 거래와 그 다음 받은 거래 사이의 평균 시간 간격", "분", "확실"),
    "Time Diff between first and last (Mins)": (
        "첫 거래부터 마지막 거래까지의 총 기간. 사실상 '지갑의 활동 수명'", "분", "확실"),

    # ---------- 토큰 활동 (ERC20) ----------
    "Total ERC20 tnxs": ("ERC20 토큰 거래의 총 횟수(보낸 것 + 받은 것)", "건", "추정"),
    "ERC20 total Ether received": (
        "받은 ERC20 토큰 수량의 총합. 이름에 'Ether'가 있지만 실제로는 토큰 수량으로 보임", "토큰 수량", "확인 필요"),
    "ERC20 total ether sent": (
        "보낸 ERC20 토큰 수량의 총합. 위와 같은 이유로 단위가 불명확", "토큰 수량", "확인 필요"),
    "ERC20 total Ether sent contract": ("컨트랙트로 보낸 ERC20 토큰 수량의 총합", "토큰 수량", "확인 필요"),
    "ERC20 uniq sent addr": ("ERC20 토큰을 보낸 '서로 다른' 상대 주소의 수", "개", "추정"),
    "ERC20 uniq rec addr": ("ERC20 토큰을 받은 '서로 다른' 상대 주소의 수", "개", "추정"),
    "ERC20 uniq sent addr.1": (
        "원본 CSV 에 같은 이름의 열이 두 개 있어 pandas 가 '.1'을 붙인 열. "
        "앞의 'ERC20 uniq sent addr' 와 값이 83%만 일치하고 분포도 전혀 달라 별개의 의미로 추정됨", "개", "확인 필요"),
    "ERC20 uniq rec contract addr": ("ERC20 토큰을 받은 '서로 다른' 컨트랙트 주소의 수", "개", "추정"),
    "ERC20 avg time between sent tnx": ("ERC20 토큰을 보낸 거래 사이의 평균 간격. 전 행이 0", "분", "확인 필요"),
    "ERC20 avg time between rec tnx": ("ERC20 토큰을 받은 거래 사이의 평균 간격. 전 행이 0", "분", "확인 필요"),
    "ERC20 avg time between rec 2 tnx": (
        "이름의 'rec 2'가 무엇을 뜻하는지 알 수 없음. 전 행이 0", "분", "확인 필요"),
    "ERC20 avg time between contract tnx": ("ERC20 컨트랙트 거래 사이의 평균 간격. 전 행이 0", "분", "확인 필요"),
    "ERC20 min val rec": ("받은 ERC20 토큰 거래 중 가장 작은 수량", "토큰 수량", "추정"),
    "ERC20 max val rec": ("받은 ERC20 토큰 거래 중 가장 큰 수량", "토큰 수량", "추정"),
    "ERC20 avg val rec": ("받은 ERC20 토큰 거래 1건당 평균 수량", "토큰 수량", "추정"),
    "ERC20 min val sent": ("보낸 ERC20 토큰 거래 중 가장 작은 수량", "토큰 수량", "추정"),
    "ERC20 max val sent": ("보낸 ERC20 토큰 거래 중 가장 큰 수량", "토큰 수량", "추정"),
    "ERC20 avg val sent": ("보낸 ERC20 토큰 거래 1건당 평균 수량", "토큰 수량", "추정"),
    "ERC20 min val sent contract": ("컨트랙트로 보낸 토큰 최소 수량. 전 행이 0", "토큰 수량", "확인 필요"),
    "ERC20 max val sent contract": ("컨트랙트로 보낸 토큰 최대 수량. 전 행이 0", "토큰 수량", "확인 필요"),
    "ERC20 avg val sent contract": ("컨트랙트로 보낸 토큰 평균 수량. 전 행이 0", "토큰 수량", "확인 필요"),
    "ERC20 uniq sent token name": ("보낸 적 있는 '서로 다른' 토큰 종류의 수", "개", "확실"),
    "ERC20 uniq rec token name": ("받은 적 있는 '서로 다른' 토큰 종류의 수", "개", "확실"),
    "ERC20 most sent token type": ("가장 많이 보낸 토큰의 이름(문자열)", "문자열", "확실"),
    "ERC20_most_rec_token_type": ("가장 많이 받은 토큰의 이름(문자열)", "문자열", "확실"),
}


def determine_group_name(column_name: str) -> str:
    """열이 어느 묶음(거래 빈도/금액/시간 간격/토큰 활동)에 속하는지 알려준다."""
    if column_name in data_loader.IDENTIFIER_COLUMNS:
        return "식별자"
    if column_name == data_loader.TARGET_COLUMN:
        return "목표 변수"
    for group_name, column_names in data_loader.FEATURE_GROUPS.items():
        if column_name in column_names:
            return group_name
    return "미분류"


def build_markdown(dataframe: pd.DataFrame) -> str:
    """DataFrame 을 훑어 데이터 사전 마크다운 문자열을 만든다."""
    total_row_count = len(dataframe)
    lines: list[str] = []

    lines.append("# 데이터 사전 (Data Dictionary)")
    lines.append("")
    lines.append("`data/raw/transaction_dataset.csv` 의 모든 열을 한국어 한 줄로 정리한 문서입니다.")
    lines.append("")
    lines.append(f"- 전체 크기: **{total_row_count:,}행 × {dataframe.shape[1]}열**")
    lines.append("- 이 문서는 `python -m src.build_data_dictionary` 로 다시 생성됩니다.")
    lines.append("  (설명 문장은 사람이 쓰고, 결측·고유값 숫자는 매번 코드가 다시 계산합니다.)")
    lines.append("")
    lines.append("### 확신 수준 표기")
    lines.append("")
    lines.append("| 표기 | 뜻 |")
    lines.append("|---|---|")
    lines.append("| 확실 | 열 이름과 실제 값 분포가 일치해 해석에 이견이 없음 |")
    lines.append("| 추정 | 이름상 의미는 분명하나 세부 계산 방식(0값 포함 여부 등)까지는 확인 못 함 |")
    lines.append("| **확인 필요** | 이름만으로 의미를 알 수 없음. **추측하지 않고 남겨 둠** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 묶음별로 표를 나눠서 출력한다(한 표에 51행을 넣으면 읽기 힘들다).
    group_order = ["식별자", "목표 변수", "거래 빈도", "금액", "시간 간격", "토큰 활동"]

    for group_name in group_order:
        columns_in_group = [
            column_name
            for column_name in dataframe.columns
            if determine_group_name(column_name) == group_name
        ]
        if not columns_in_group:
            continue

        lines.append(f"## {group_name} ({len(columns_in_group)}개)")
        lines.append("")
        lines.append("| 열 이름 | 한국어 설명 | 단위 | 결측 | 고유값 | 확신 |")
        lines.append("|---|---|---|---|---|---|")

        for column_name in columns_in_group:
            description, unit, confidence = COLUMN_DESCRIPTIONS.get(
                column_name, ("(설명 미작성)", "-", "확인 필요")
            )
            missing_count = int(dataframe[column_name].isna().sum())
            missing_percent = missing_count / total_row_count * 100
            unique_count = int(dataframe[column_name].nunique(dropna=True))

            missing_text = "0" if missing_count == 0 else f"{missing_count:,} ({missing_percent:.1f}%)"
            # 확인 필요는 눈에 띄게 굵게 표시
            confidence_text = f"**{confidence}**" if confidence == "확인 필요" else confidence

            lines.append(
                f"| `{column_name}` | {description} | {unit} | {missing_text} "
                f"| {unique_count:,} | {confidence_text} |"
            )

        lines.append("")

    # 확인 필요 열만 따로 모아 마지막에 다시 정리한다.
    needs_verification = [
        column_name
        for column_name in dataframe.columns
        if COLUMN_DESCRIPTIONS.get(column_name, ("", "", "확인 필요"))[2] == "확인 필요"
    ]
    lines.append("---")
    lines.append("")
    lines.append(f"## 확인 필요 열 정리 ({len(needs_verification)}개)")
    lines.append("")
    lines.append("아래 열들은 **의미를 추측하지 않았습니다.** 원 데이터 제작자의 설명이나")
    lines.append("Etherscan API 문서를 확인한 뒤 채워 넣어야 합니다.")
    lines.append("")
    for column_name in needs_verification:
        description = COLUMN_DESCRIPTIONS[column_name][0]
        lines.append(f"- `{column_name}` — {description}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    dataframe = data_loader.load_raw_dataset()

    # 설명을 빠뜨린 열이 없는지 먼저 확인한다(데이터가 바뀌면 알려주기 위해).
    columns_without_description = [
        column_name for column_name in dataframe.columns if column_name not in COLUMN_DESCRIPTIONS
    ]
    if columns_without_description:
        print(f"[경고] 설명이 없는 열이 있습니다: {columns_without_description}")

    markdown_text = build_markdown(dataframe)

    config.ensure_directories_exist()
    output_path = config.REPORTS_DIRECTORY / "data_dictionary.md"
    output_path.write_text(markdown_text, encoding="utf-8")

    print(f"[저장] {output_path}")
    print(f"       열 {dataframe.shape[1]}개 정리 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
