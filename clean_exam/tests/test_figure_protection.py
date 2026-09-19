"""인쇄된 그래프·도형은 지키고, 학생이 손으로 그린 그림은 강조하지 않는지 검증한다.

왜 따로 테스트하나: 시험지를 풀다 보면 학생이 여백에 좌표축을 직접 그린다.
규칙 엔진이 "긴 직선이 있으니 인쇄된 그래프다"라고 보호해 버리면, 지우기는커녕
**새까맣게 굵어져 더 눈에 띈다.** 가장 보기 나쁜 실패라서 따로 지킨다.

핵심 설계: **"보호"와 "강조"를 분리한다.**
  · figure  → 지우지 않고 굵게 + 새까맣게 (확실한 인쇄 그래프만)
  · printed → 지우지만 않음 (인쇄인지 손그림인지 헷갈리는 가는 선)
헷갈리는 선을 printed 로 두면, 잘못 판단해도 학생 선이 도드라지지는 않는다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.clean import clean_image
from engine.color_filter import remove_color_pen
from engine.preprocess import preprocess_photo
from engine.settings import load_config
from engine.stroke_filter import binarize_ink, analyze_components
from tests.synthetic_exam import make_synthetic_page

# 회귀 방지선 (실측: 손그림 강조 2.1%, 인쇄 도형 보호 84.6%)
MAXIMUM_STUDENT_GRAPH_EMPHASIS_RATIO = 0.06
MINIMUM_PRINTED_FIGURE_PROTECTION_RATIO = 0.70

SAMPLE_PHOTO_PATH = Path(__file__).resolve().parent / "samples" / "01_기하_0071.jpg"


def measure_figure_handling(seed: int) -> tuple[float, float]:
    """(학생 손그림이 도형으로 강조된 비율, 인쇄 도형이 보호된 비율)을 잰다."""
    config = load_config()
    page = make_synthetic_page(seed=seed, difficulty="보통")
    result = clean_image(page.written_page_image, config, api_key=None, debug=True)
    if result["cleaned_image"].shape[:2] != page.clean_page_image.shape[:2]:
        pytest.skip("보정 과정에서 크기가 바뀌어 픽셀 대조를 할 수 없다.")

    strengthened_figure_mask = result["debug_images"]["07_그래프"] > 0
    written_gray = cv2.cvtColor(page.written_page_image, cv2.COLOR_BGR2GRAY)

    student_graph_ink = (page.student_graph_mask > 0) & (written_gray < 215)
    printed_figure_ink = (page.figure_mask > 0) & (page.handwriting_mask == 0) & (written_gray < 180)

    student_emphasis = float(strengthened_figure_mask[student_graph_ink].mean()) \
        if student_graph_ink.any() else 0.0
    printed_protection = float(strengthened_figure_mask[printed_figure_ink].mean()) \
        if printed_figure_ink.any() else 1.0
    return student_emphasis, printed_protection


def test_student_drawn_graph_is_not_emphasised() -> None:
    """학생이 그린 좌표축·포물선이 인쇄 그래프로 오인돼 새까맣게 칠해지면 안 된다."""
    measurements = [measure_figure_handling(seed) for seed in range(6)]
    average_emphasis = float(np.mean([emphasis for emphasis, _ in measurements]))
    average_protection = float(np.mean([protection for _, protection in measurements]))

    print(f"\n학생 손그림이 도형으로 강조된 비율 {average_emphasis*100:.1f}% / "
          f"인쇄 도형이 보호된 비율 {average_protection*100:.1f}%")

    assert average_emphasis <= MAXIMUM_STUDENT_GRAPH_EMPHASIS_RATIO, (
        "학생이 손으로 그린 그림이 인쇄 그래프로 오인돼 강조되고 있다. "
        "지우는 것보다 나쁜 실패다."
    )
    assert average_protection >= MINIMUM_PRINTED_FIGURE_PROTECTION_RATIO, (
        "인쇄된 그래프·도형이 충분히 보호되지 않는다."
    )


@pytest.mark.skipif(not SAMPLE_PHOTO_PATH.exists(), reason="실제 샘플 사진이 없다")
def test_printed_fraction_bars_survive_on_real_photo() -> None:
    """실제 사진의 보기 ①~⑤ **분수 가로줄**이 살아남아야 한다.

    가는 인쇄선은 가장자리가 흐려져 평균 진하기가 떨어지기 때문에, 규칙이 조금만
    공격적이어도 통째로 지워진다. 지워지면 분수가 정수로 바뀌어 문제가 망가진다.
    """
    config = load_config()
    original_image = cv2.imread(str(SAMPLE_PHOTO_PATH))
    preprocess_result = preprocess_photo(original_image, config)
    gray_image, _, color_print_mask = remove_color_pen(
        preprocess_result.normalized_gray_image,
        preprocess_result.corrected_color_image,
        config["color_filter"],
    )
    ink_mask = binarize_ink(gray_image, config["stroke_filter"])
    _, components = analyze_components(
        gray_image, ink_mask, config["stroke_filter"],
        exclude_from_reference_mask=color_print_mask,
    )

    # 보기 줄 부근의 가로로 긴 요소 = 분수 가로줄
    fraction_bars = [
        component for component in components
        if (1000 <= component.bounding_box[1] <= 1105
            and component.bounding_box[0] < 900
            and component.bounding_box[2] > 55)
    ]
    assert len(fraction_bars) >= 5, f"분수 가로줄을 찾지 못했다({len(fraction_bars)}개)."
    assert all(component.label in ("printed_text", "figure") for component in fraction_bars), (
        f"분수 가로줄이 필기로 판정됐다: {[component.label for component in fraction_bars]}"
    )

    # 결과 이미지에도 실제로 남아 있어야 한다
    cleaned_image = clean_image(original_image, config, api_key=None)["cleaned_image"]
    for left, top, width, height in (component.bounding_box for component in fraction_bars):
        remaining_dark_ratio = float(
            (cleaned_image[top:top + height, left:left + width] < 200).mean()
        )
        assert remaining_dark_ratio > 0.2, (
            f"분수 가로줄 ({left},{top}) 이 결과에서 거의 사라졌다({remaining_dark_ratio*100:.0f}%)."
        )
