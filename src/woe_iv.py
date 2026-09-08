"""
WoE(증거가중치)와 IV(정보가치)를 직접 구현한 모듈.

──────────────────────────────────────────────────────────────────────
■ WoE 란 무엇인가 (Weight of Evidence, 증거가중치)

신용평가에서 쓰는 변환입니다. 어떤 특징(예: '보낸 거래 횟수')을 몇 개의 구간(bin)으로
나눈 뒤, **각 구간이 얼마나 '사기 쪽 냄새'가 나는지를 하나의 숫자로 바꾼 것**입니다.

    WoE_i = ln( (사기_i / 사기_전체) / (정상_i / 정상_전체) )

기호 뜻:
    i          : 구간 번호 (예: '거래 0~2회' 구간)
    사기_i     : i 구간 안에 있는 사기 주소 개수
    사기_전체  : 데이터 전체의 사기 주소 개수 (여기서는 2,179)
    정상_i     : i 구간 안에 있는 정상 주소 개수
    정상_전체  : 데이터 전체의 정상 주소 개수 (여기서는 7,662)
    ln         : 자연로그 (밑이 e ≈ 2.718 인 로그)

읽는 법:
    WoE > 0  →  이 구간에는 사기가 '평균보다 많이' 모여 있다 (위험 신호)
    WoE = 0  →  이 구간의 사기 비율이 전체 평균과 같다 (정보 없음)
    WoE < 0  →  이 구간에는 정상이 '평균보다 많이' 모여 있다 (안전 신호)

비유: 반 전체 평균이 70점인데, 어떤 문제를 맞힌 학생들의 평균이 90점이라면
      "그 문제를 맞혔다"는 사실은 실력의 증거가 됩니다. WoE 는 그 '증거의 세기'입니다.

※ 부호 규약 주의
   전통적인 신용평가 교과서는 WoE = ln(정상비율 / 사기비율) 로 정의해,
   값이 클수록 '우량'을 뜻하게 만듭니다. 이 모듈은 사기 탐지에 맞춰
   **반대 부호(= ln(사기비율 / 정상비율))** 를 씁니다. 값이 클수록 위험합니다.
   부호만 뒤집힌 것이라 IV 값과 모델 성능은 규약과 무관하게 동일합니다.

──────────────────────────────────────────────────────────────────────
■ IV 란 무엇인가 (Information Value, 정보가치)

WoE 는 '구간 하나'의 증거 세기입니다. 이걸 특징 전체에 대해 합치면
"이 특징이 사기/정상 구분에 얼마나 도움이 되는가"를 숫자 하나로 요약할 수 있습니다.

    IV = Σ_i ( 사기_i/사기_전체 − 정상_i/정상_전체 ) × WoE_i

기호 뜻:
    Σ_i   : 모든 구간 i 에 대해 더하라는 뜻 (시그마)
    앞의 괄호 = 두 집단의 '비중 차이'
    뒤의 WoE_i = 그 구간의 증거 세기

즉 **비중 차이 × 증거 세기** 를 모든 구간에서 더한 값입니다.
차이도 크고 증거도 센 구간이 많을수록 IV 가 커집니다. IV 는 항상 0 이상입니다.

해석 기준 (신용평가 실무에서 널리 쓰이는 Siddiqi 기준):
    IV < 0.02        거의 쓸모없음
    0.02 ~ 0.1       약한 예측력
    0.1  ~ 0.3       중간 예측력
    0.3  ~ 0.5       강한 예측력
    IV > 0.5         "너무 좋다" → 데이터 누수를 의심해 봐야 함

마지막 줄이 중요합니다. IV 가 비정상적으로 높으면 실력이 아니라
**정답이 새어 들어온 것**일 가능성이 큽니다.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 구간을 나눌 때 쓰는 특별 이름표
MISSING_BIN_LABEL = "결측"
ZERO_BIN_LABEL = "0"
OTHER_CATEGORY_LABEL = "기타"

# IV 해석 구간 (아래 경계값, 설명)
IV_INTERPRETATION_THRESHOLDS = [
    (0.02, "거의 쓸모없음"),
    (0.10, "약함"),
    (0.30, "중간"),
    (0.50, "강함"),
    (float("inf"), "매우 강함 (누수 의심)"),
]


def _parse_bin_sort_key(bin_label: str) -> float:
    """
    구간 이름표에서 '정렬용 숫자'를 뽑아낸다.

    왜 필요한가?
      pandas 가 만든 구간 이름은 "(-inf, 0.844]", "(100.695, 101.0]" 같은 문자열이다.
      이걸 그냥 문자열로 정렬하면 "(100.695..." 가 "(2.0..." 보다 앞에 온다.
      ('1' 이 '2' 보다 작은 글자이기 때문. 사전 순 정렬의 함정)
      그러면 "값이 커질수록 WoE 가 어떻게 변하는가"를 볼 수 없다.
      그래서 구간의 '왼쪽 경계 숫자'를 뽑아 그 값으로 정렬한다.

    규칙:
      "결측"           -> +무한대 (맨 뒤에 둔다)
      "0"              -> 0.0     (숫자 0의 제자리)
      "(a, b]"         -> a       (왼쪽 경계)
      그 외            -> 0.0
    """
    if bin_label == MISSING_BIN_LABEL:
        return float("inf")
    if bin_label == ZERO_BIN_LABEL:
        return 0.0

    # "(-inf, 0.844]" 처럼 생겼는지 확인한 뒤 왼쪽 숫자를 떼어낸다
    if bin_label.startswith(("(", "[")) and bin_label.endswith(("]", ")")):
        inner_text = bin_label[1:-1]           # 괄호 제거
        left_text = inner_text.split(",")[0]   # 쉼표 앞부분이 왼쪽 경계
        try:
            return float(left_text)
        except ValueError:
            return 0.0

    return 0.0


def interpret_information_value(information_value: float) -> str:
    """IV 숫자를 한국어 해석 문구로 바꿔 준다."""
    for upper_bound, description in IV_INTERPRETATION_THRESHOLDS:
        if information_value < upper_bound:
            return description
    return "매우 강함 (누수 의심)"


class WeightOfEvidenceEncoder:
    """
    특징들을 구간으로 나누고 WoE 값으로 변환해 주는 변환기.

    scikit-learn 과 같은 방식(fit -> transform)으로 만들었습니다.
    이렇게 만든 이유가 중요합니다:

      교차검증에서 WoE 를 **전체 데이터로 한 번에 계산하면 안 됩니다.**
      WoE 계산에는 정답(FLAG)이 들어가기 때문에, 검증셋의 정답 정보가
      학습셋으로 흘러 들어가 성능이 부풀려집니다.
      반드시 **학습 fold 로만 fit 하고, 검증 fold 에는 transform 만** 해야 합니다.

    주요 매개변수
      max_bin_count       : 특징 하나를 최대 몇 구간으로 나눌지 (기본 10)
      separate_zero_bin   : 값이 0인 행을 별도 구간으로 뺄지 (기본 True)
                            이 데이터는 0이 몰려 있는 열이 많아서 켜 두는 게 유리하다.
      min_zero_fraction   : 0의 비율이 이 값 이상일 때만 0을 따로 뺀다 (기본 0.05 = 5%)
      min_category_fraction : 범주형에서 이 비율보다 드문 값은 '기타'로 묶는다 (기본 0.01)
      smoothing_count     : 구간에 사기 또는 정상이 0건일 때 로그가 무한대로 튀는 것을
                            막기 위해 양쪽에 더해 주는 작은 수 (기본 0.5)
    """

    def __init__(
        self,
        max_bin_count: int = 10,
        separate_zero_bin: bool = True,
        min_zero_fraction: float = 0.05,
        min_category_fraction: float = 0.01,
        smoothing_count: float = 0.5,
    ):
        self.max_bin_count = max_bin_count
        self.separate_zero_bin = separate_zero_bin
        self.min_zero_fraction = min_zero_fraction
        self.min_category_fraction = min_category_fraction
        self.smoothing_count = smoothing_count

        # fit 이후에 채워지는 결과들
        self.bin_specifications_: dict[str, dict] = {}   # 열 -> 구간 정의
        self.woe_mappings_: dict[str, dict[str, float]] = {}  # 열 -> {구간이름: WoE}
        self.information_values_: dict[str, float] = {}  # 열 -> IV
        self.binning_tables_: dict[str, pd.DataFrame] = {}  # 열 -> 구간별 상세표
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------
    # 1단계: 구간 나누기 (binning)
    # ------------------------------------------------------------
    def _build_bin_specification(self, feature_series: pd.Series) -> dict:
        """한 열을 어떻게 구간으로 나눌지 '설계도'를 만든다."""
        if not pd.api.types.is_numeric_dtype(feature_series):
            # ── 범주형(문자열) ──
            # 너무 드문 범주는 '기타'로 묶는다. 한 구간에 몇 건 없으면
            # 그 구간의 사기 비율이 우연에 크게 흔들리기 때문이다.
            value_fractions = feature_series.value_counts(normalize=True, dropna=True)
            kept_categories = set(
                value_fractions[value_fractions >= self.min_category_fraction].index
            )
            return {"kind": "categorical", "categories": kept_categories}

        # ── 수치형 ──
        non_missing_values = feature_series.dropna()

        # 0을 따로 뺄지 결정한다. 0이 충분히 많을 때만 뺀다.
        zero_fraction = (
            float((non_missing_values == 0).mean()) if len(non_missing_values) else 0.0
        )
        use_zero_bin = self.separate_zero_bin and zero_fraction >= self.min_zero_fraction

        # 구간 경계를 계산할 대상: 0을 따로 뺐다면 0이 아닌 값들만 대상으로 한다.
        values_for_edges = (
            non_missing_values[non_missing_values != 0] if use_zero_bin else non_missing_values
        )

        if len(values_for_edges) == 0:
            return {"kind": "numeric", "edges": np.array([]), "use_zero_bin": use_zero_bin}

        # 분위수(quantile)로 경계를 잡는다 → 각 구간에 비슷한 '개수'가 들어간다.
        # np.linspace(0, 1, n+1) 은 0, 0.1, 0.2, ... 1 처럼 균등한 분위 지점을 만든다.
        quantile_points = np.linspace(0, 1, self.max_bin_count + 1)
        raw_edges = np.quantile(values_for_edges, quantile_points)

        # 같은 값이 반복되면 경계가 겹친다(예: 0.1분위와 0.2분위가 둘 다 1.0).
        # np.unique 로 중복을 없애면 구간 수가 줄어들지만, 그게 올바른 동작이다.
        unique_edges = np.unique(raw_edges)

        # 바깥쪽 경계를 무한대로 열어 둔다.
        # 나중에 새 데이터에 학습 때보다 크거나 작은 값이 와도 구간에 들어가게 하기 위해서다.
        if len(unique_edges) >= 2:
            unique_edges[0] = -np.inf
            unique_edges[-1] = np.inf

        return {"kind": "numeric", "edges": unique_edges, "use_zero_bin": use_zero_bin}

    def _assign_bin_labels(self, feature_series: pd.Series, bin_specification: dict) -> pd.Series:
        """설계도에 따라 각 행에 '구간 이름표'를 붙인다."""
        if bin_specification["kind"] == "categorical":
            kept_categories = bin_specification["categories"]
            # 결측이면 '결측', 남겨둔 범주면 그대로, 나머지는 '기타'
            return feature_series.map(
                lambda value: MISSING_BIN_LABEL
                if pd.isna(value)
                else (str(value) if value in kept_categories else OTHER_CATEGORY_LABEL)
            ).astype(str)

        edges = bin_specification["edges"]
        use_zero_bin = bin_specification["use_zero_bin"]

        # 기본값은 '결측'으로 두고, 조건에 맞는 행만 덮어쓴다.
        bin_labels = pd.Series(MISSING_BIN_LABEL, index=feature_series.index, dtype=object)
        is_missing = feature_series.isna()

        if use_zero_bin:
            is_zero = (~is_missing) & (feature_series == 0)
            bin_labels[is_zero] = ZERO_BIN_LABEL
            rows_to_cut = (~is_missing) & (feature_series != 0)
        else:
            rows_to_cut = ~is_missing

        if len(edges) >= 2 and rows_to_cut.any():
            # pd.cut 은 값을 경계에 따라 구간으로 나눠 준다.
            cut_result = pd.cut(feature_series[rows_to_cut], bins=edges, include_lowest=True)
            bin_labels[rows_to_cut] = cut_result.astype(str)
        elif rows_to_cut.any():
            # 경계를 만들 수 없을 만큼 값 종류가 적으면 한 구간으로 둔다.
            bin_labels[rows_to_cut] = "전체"

        return bin_labels.astype(str)

    # ------------------------------------------------------------
    # 2단계: 구간별 WoE 와 IV 계산
    # ------------------------------------------------------------
    def _compute_woe_table(
        self, bin_labels: pd.Series, target: pd.Series, bin_kind: str = "numeric"
    ) -> pd.DataFrame:
        """
        구간별로 WoE 와 IV 기여도를 계산한 표를 만든다.

        표의 각 열이 뜻하는 것:
            건수        : 그 구간에 속한 주소 수
            사기 건수   : 그중 FLAG=1 인 수      (공식의 사기_i)
            정상 건수   : 그중 FLAG=0 인 수      (공식의 정상_i)
            사기 비율   : 그 구간 안에서의 사기 비율 (= 사기_i / 건수)
            사기 분포   : 전체 사기 중 이 구간이 차지하는 몫 (= 사기_i / 사기_전체)
            정상 분포   : 전체 정상 중 이 구간이 차지하는 몫 (= 정상_i / 정상_전체)
            WoE         : ln(사기 분포 / 정상 분포)
            IV 기여     : (사기 분포 − 정상 분포) × WoE
        """
        grouped = pd.DataFrame({"bin": bin_labels, "target": target}).groupby("bin", observed=True)

        counts_table = grouped["target"].agg(["count", "sum"])
        counts_table.columns = ["건수", "사기 건수"]
        counts_table["정상 건수"] = counts_table["건수"] - counts_table["사기 건수"]

        # 스무딩(smoothing): 어떤 구간에 사기가 0건이면 분자가 0이 되어 ln(0) = -무한대가 된다.
        # 그래서 양쪽에 아주 작은 수(기본 0.5)를 더해 계산이 깨지지 않게 한다.
        # 이 보정은 건수가 많은 구간에는 거의 영향이 없고, 극단적인 구간만 부드럽게 만든다.
        smoothed_fraud = counts_table["사기 건수"] + self.smoothing_count
        smoothed_normal = counts_table["정상 건수"] + self.smoothing_count

        total_smoothed_fraud = smoothed_fraud.sum()
        total_smoothed_normal = smoothed_normal.sum()

        fraud_distribution = smoothed_fraud / total_smoothed_fraud   # 사기_i / 사기_전체
        normal_distribution = smoothed_normal / total_smoothed_normal  # 정상_i / 정상_전체

        # ★ WoE 공식 ★
        weight_of_evidence = np.log(fraud_distribution / normal_distribution)

        # ★ IV 기여도 공식 ★
        iv_contribution = (fraud_distribution - normal_distribution) * weight_of_evidence

        counts_table["사기 비율"] = counts_table["사기 건수"] / counts_table["건수"]
        counts_table["사기 분포"] = fraud_distribution
        counts_table["정상 분포"] = normal_distribution
        counts_table["WoE"] = weight_of_evidence
        counts_table["IV 기여"] = iv_contribution

        counts_table = counts_table.reset_index()

        # 구간을 '읽기 좋은 순서'로 정렬한다.
        #   수치형 : 값이 작은 구간 -> 큰 구간 순서 (그래야 WoE 의 추세가 보인다)
        #   범주형 : WoE 가 낮은(안전) 쪽 -> 높은(위험) 쪽 순서
        if bin_kind == "numeric":
            sort_keys = counts_table["bin"].map(_parse_bin_sort_key)
            counts_table = counts_table.assign(_sort=sort_keys).sort_values("_sort")
        else:
            counts_table = counts_table.sort_values("WoE")

        return counts_table.drop(columns="_sort", errors="ignore").reset_index(drop=True)

    # ------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------
    def fit(self, feature_frame: pd.DataFrame, target: pd.Series) -> "WeightOfEvidenceEncoder":
        """구간을 나누고 각 구간의 WoE·IV 를 학습한다."""
        target = pd.Series(target).reset_index(drop=True)
        feature_frame = feature_frame.reset_index(drop=True)

        self.feature_names_ = list(feature_frame.columns)
        self.bin_specifications_.clear()
        self.woe_mappings_.clear()
        self.information_values_.clear()
        self.binning_tables_.clear()

        for feature_name in self.feature_names_:
            feature_series = feature_frame[feature_name]

            bin_specification = self._build_bin_specification(feature_series)
            bin_labels = self._assign_bin_labels(feature_series, bin_specification)
            woe_table = self._compute_woe_table(
                bin_labels, target, bin_kind=bin_specification["kind"]
            )

            self.bin_specifications_[feature_name] = bin_specification
            self.woe_mappings_[feature_name] = dict(zip(woe_table["bin"], woe_table["WoE"]))
            self.information_values_[feature_name] = float(woe_table["IV 기여"].sum())
            self.binning_tables_[feature_name] = woe_table

        return self

    def transform(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """
        각 값을 그 값이 속한 구간의 WoE 숫자로 바꾼다.

        학습 때 못 본 구간이 나오면 WoE 를 0(= 정보 없음, 중립)으로 둔다.
        """
        transformed_columns = {}

        for feature_name in self.feature_names_:
            bin_specification = self.bin_specifications_[feature_name]
            bin_labels = self._assign_bin_labels(feature_frame[feature_name], bin_specification)
            woe_mapping = self.woe_mappings_[feature_name]

            transformed_columns[feature_name] = (
                bin_labels.map(woe_mapping).astype(float).fillna(0.0).to_numpy()
            )

        return pd.DataFrame(transformed_columns, index=feature_frame.index)

    def fit_transform(self, feature_frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """fit 과 transform 을 한 번에 (학습 데이터에만 쓸 것)."""
        return self.fit(feature_frame, target).transform(feature_frame)

    def get_information_value_ranking(self) -> pd.DataFrame:
        """IV 가 높은 순으로 정렬한 표를 돌려준다."""
        ranking = pd.DataFrame(
            {
                "특징": list(self.information_values_.keys()),
                "IV": list(self.information_values_.values()),
            }
        )
        ranking["해석"] = ranking["IV"].apply(interpret_information_value)
        ranking["구간 수"] = [
            len(self.binning_tables_[name]) for name in ranking["특징"]
        ]
        ranking = ranking.sort_values("IV", ascending=False).reset_index(drop=True)
        ranking.insert(0, "순위", range(1, len(ranking) + 1))
        return ranking
