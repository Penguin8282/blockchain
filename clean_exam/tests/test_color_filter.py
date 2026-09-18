"""컬러 인쇄를 색펜으로 오인하지 않는지 검증한다.

왜 따로 테스트하나: 예전 가정("인쇄물의 잉크는 무채색이다")이 요즘 문제집에는 맞지 않는다.
단원 제목 띠, 유형 아이콘, 색으로 인쇄된 문제 번호가 흔하고, 채도만 보고 지우면
이것들이 통째로 사라진다. 실측으로 컬러 인쇄 픽셀의 88% 가 지워지고 있었다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.clean import clean_image
from engine.color_filter import classify_color_components
from engine.preprocess import preprocess_photo
from engine.settings import load_config
from tests.synthetic_exam import make_synthetic_page

# 회귀 방지선 (실측: 컬러 인쇄 손실 평균 9.4%, 색펜 제거 평균 86.4%)
MAXIMUM_COLOR_PRINT_LOSS_RATIO = 0.20
MINIMUM_COLOR_PEN_REMOVAL_RATIO = 0.80

SAMPLE_PHOTO_PATH = Path(__file__).resolve().parent / "samples" / "01_기하_0071.jpg"


def measure_color_handling(seed: int) -> tuple[float, float]:
    """합성 시험지 한 장에서 (컬러 인쇄 손실률, 색펜 제거율)을 잰다."""
    config = load_config()
    page = make_synthetic_page(seed=seed, difficulty="보통")
    cleaned_image = clean_image(page.written_page_image, config, api_key=None)["cleaned_image"]
    if cleaned_image.shape[:2] != page.clean_page_image.shape[:2]:
        pytest.skip("보정 과정에서 크기가 바뀌어 픽셀 대조를 할 수 없다.")

    written_gray = cv2.cvtColor(page.written_page_image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(page.written_page_image, cv2.COLOR_BGR2HSV)[:, :, 1]
    became_white = cleaned_image >= 250

    # 정답은 "실제로 색이 찍힌 픽셀"로만 센다.
    # (제목 띠 안의 흰 글자는 원래 흰색이므로 손실로 세면 안 된다.)
    color_print_ink = (page.color_print_mask > 0) & (page.handwriting_mask == 0) & (saturation > 60)
    color_pen_ink = (page.handwriting_mask > 0) & (saturation > 60) & (written_gray < 215)

    color_print_loss = float(became_white[color_print_ink].mean()) if color_print_ink.any() else 0.0
    color_pen_removal = float(became_white[color_pen_ink].mean()) if color_pen_ink.any() else 1.0
    return color_print_loss, color_pen_removal


def test_color_printing_survives_while_color_pen_is_removed() -> None:
    """컬러 인쇄는 지키고 색펜은 지워야 한다."""
    measurements = [measure_color_handling(seed) for seed in range(6)]
    average_print_loss = float(np.mean([loss for loss, _ in measurements]))
    average_pen_removal = float(np.mean([removal for _, removal in measurements]))

    print(f"\n컬러 인쇄 손실 {average_print_loss*100:.1f}% / 색펜 제거 {average_pen_removal*100:.1f}%")

    assert average_print_loss <= MAXIMUM_COLOR_PRINT_LOSS_RATIO, (
        "컬러 인쇄(제목 띠·아이콘·색 문제 번호)가 지워지고 있다."
    )
    assert average_pen_removal >= MINIMUM_COLOR_PEN_REMOVAL_RATIO


def test_solid_banner_is_kept_and_pen_circle_is_removed() -> None:
    """손으로 만든 최소 예제: 꽉 찬 색 띠는 지키고, 색 동그라미는 지운다."""
    config = load_config()
    test_image = np.full((400, 600, 3), 250, dtype=np.uint8)

    # 인쇄된 제목 띠: 넓고 납작하지만 축에 나란한 반듯한 직사각형
    cv2.rectangle(test_image, (40, 30), (430, 78), (150, 110, 60), -1)
    # 색펜 동그라미: 가운데가 크게 비어 있다
    cv2.circle(test_image, (300, 250), 90, (45, 45, 215), 6)

    components, _ = classify_color_components(test_image, config["color_filter"])
    banner = max(components, key=lambda component: component.area_px)
    circle = max((component for component in components if component is not banner),
                 key=lambda component: component.area_px)

    assert banner.is_printed, f"제목 띠가 색펜으로 판정됐다: {banner.reason}"
    assert not circle.is_printed, f"색펜 동그라미가 인쇄로 판정됐다: {circle.reason}"


@pytest.mark.skipif(not SAMPLE_PHOTO_PATH.exists(), reason="실제 샘플 사진이 없다")
def test_blue_printed_problem_number_survives_on_real_photo() -> None:
    """실제 시험지 사진의 **파란색으로 인쇄된 문제 번호**가 살아남아야 한다.

    이 사진에서 문제 번호 0071·0074 는 파란색 인쇄다. 예전 규칙은 이것을 색펜으로 보고
    통째로 지웠다. 합성이 아니라 실제 사진으로 확인하는 회귀 테스트다.
    """
    config = load_config()
    original_image = cv2.imread(str(SAMPLE_PHOTO_PATH))
    corrected_image = preprocess_photo(original_image, config).corrected_color_image

    components, _ = classify_color_components(corrected_image, config["color_filter"])

    # 오른쪽 단의 문제 번호 0074 가 있는 자리 (보정 후 좌표)
    number_left, number_top, number_right, number_bottom = 920, 790, 1010, 840
    # 숫자 한 자 크기(실측 18x30, 넓이 200 안팎)인 덩어리만 본다.
    # 같은 사각형 안에 빨간펜이 남긴 몇 픽셀짜리 부스러기도 들어오는데, 그건 지우는 게 맞다.
    number_components = [
        component for component in components
        if (number_left <= component.bounding_box[0] <= number_right
            and number_top <= component.bounding_box[1] <= number_bottom
            and component.area_px >= 150)
    ]
    assert len(number_components) >= 3, (
        f"문제 번호 자리에서 숫자 크기의 색 덩어리를 찾지 못했다({len(number_components)}개)."
    )
    assert all(component.is_printed for component in number_components), (
        "파란색으로 인쇄된 문제 번호가 색펜으로 판정됐다: "
        + str([component.reason for component in number_components])
    )

    # 큰 빨간펜 채점 표시는 여전히 지워져야 한다
    large_pen_marks = [
        component for component in components
        if component.area_px > 2000 and min(component.bounding_box[2:]) > 100
    ]
    assert large_pen_marks, "큰 빨간펜 표시를 찾지 못했다."
    assert not any(component.is_printed for component in large_pen_marks), (
        "큰 빨간펜 채점 표시가 인쇄로 판정됐다."
    )
