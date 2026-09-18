"""엔진 전체(1~6단계)를 합성 시험지로 검증한다.

합성 시험지는 "어느 픽셀이 필기/인쇄/도형인지"를 알고 만들기 때문에 정확도를 수치로 잴 수 있다.
다만 실제 시험지 사진보다는 쉬운 문제이므로, 여기 숫자가 곧 실제 성능은 아니다.
실제 사진의 성능은 tests/samples/ 에 사진을 넣고 사람이 눈으로 확인해야 한다.

아래 기준값은 "지금 이 정도는 나온다"는 **회귀 방지선**이다(실측값에 여유를 둔 값).
4단계(자체 모델)로 올라가면 이 숫자를 함께 올린다.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from engine.clean import clean_image
from engine.settings import load_config
from tests.synthetic_exam import make_synthetic_page

# ── 회귀 방지선 (2026-09 기준 실측값과 그 여유) ────────────────────────────
# 실측: 연필/검은볼펜 제거 33~39%, 색펜 제거 94~95%, 인쇄 손실 1.0~2.3%, 도형 손실 0.0~2.0%
MINIMUM_PENCIL_REMOVAL_RATIO = 0.28    # 연필·검은 볼펜 (키 없이 규칙만. 가장 약한 부분이다)
MINIMUM_COLOR_PEN_REMOVAL_RATIO = 0.85 # 빨간펜 채점 표시
MAXIMUM_PRINTED_LOSS_RATIO = 0.05      # 인쇄 글자를 잘못 지운 비율
MAXIMUM_FIGURE_LOSS_RATIO = 0.06       # 그래프·도형을 잘못 지운 비율 (가장 치명적인 실패)
MAXIMUM_SECONDS_PER_PAGE = 3.0


def measure_accuracy(seed: int, difficulty: str) -> dict[str, float]:
    """합성 시험지 한 장을 처리하고 종류별 제거율·손실률을 잰다.

    "잉크가 실제로 찍힌 픽셀"만 센다. 마스크 가장자리(잉크가 거의 없는 곳)까지 세면
    숫자가 흐려지기 때문이다.
    """
    config = load_config()
    page = make_synthetic_page(seed=seed, difficulty=difficulty)

    started_at = time.time()
    result = clean_image(page.written_page_image, config, api_key=None)
    elapsed_seconds = time.time() - started_at

    cleaned_image = result["cleaned_image"]
    # 기울지 않은 입력이므로 크기가 그대로여야 한다(회전이 걸리면 캔버스가 커진다).
    assert cleaned_image.shape[:2] == page.clean_page_image.shape[:2], (
        f"결과 크기 {cleaned_image.shape[:2]} 가 원본 {page.clean_page_image.shape[:2]} 와 다르다. "
        "페이지 배치는 절대 바뀌면 안 된다."
    )

    written_gray = cv2.cvtColor(page.written_page_image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(page.written_page_image, cv2.COLOR_BGR2HSV)[:, :, 1]
    became_white = cleaned_image >= 250

    handwriting_ink = (page.handwriting_mask > 0) & (written_gray < 215)
    pencil_ink = handwriting_ink & (saturation <= 60)      # 연필 · 검은 볼펜
    color_pen_ink = handwriting_ink & (saturation > 60)    # 빨간펜
    printed_ink = (page.printed_text_mask > 0) & (page.handwriting_mask == 0) & (written_gray < 180)
    figure_ink = (page.figure_mask > 0) & (page.handwriting_mask == 0) & (written_gray < 180)

    def white_ratio(mask: np.ndarray) -> float:
        return float(became_white[mask].mean()) if mask.any() else float("nan")

    return {
        "pencil_removal_ratio": white_ratio(pencil_ink),
        "color_pen_removal_ratio": white_ratio(color_pen_ink),
        "printed_loss_ratio": white_ratio(printed_ink),
        "figure_loss_ratio": white_ratio(figure_ink),
        "elapsed_seconds": elapsed_seconds,
    }


@pytest.mark.parametrize("difficulty", ["쉬움", "보통", "어려움"])
def test_pipeline_accuracy_by_difficulty(difficulty: str) -> None:
    """난이도별로 필기는 지우고 인쇄·도형은 남기는지 확인한다.

    "어려움" 에는 인쇄만큼 진한 검은 볼펜 필기가 섞인다 — 규칙으로 가장 어려운 경우다.
    """
    measurements = [measure_accuracy(seed, difficulty) for seed in range(4)]

    def average(key: str) -> float:
        return float(np.nanmean([measurement[key] for measurement in measurements]))

    pencil_removal = average("pencil_removal_ratio")
    color_pen_removal = average("color_pen_removal_ratio")
    printed_loss = average("printed_loss_ratio")
    figure_loss = average("figure_loss_ratio")
    slowest_seconds = max(measurement["elapsed_seconds"] for measurement in measurements)

    print(f"\n[{difficulty}] 연필·볼펜 제거 {pencil_removal*100:.0f}% / "
          f"색펜 제거 {color_pen_removal*100:.0f}% / 인쇄 손실 {printed_loss*100:.1f}% / "
          f"도형 손실 {figure_loss*100:.1f}% / 최대 {slowest_seconds:.2f}초")

    assert pencil_removal >= MINIMUM_PENCIL_REMOVAL_RATIO
    assert printed_loss <= MAXIMUM_PRINTED_LOSS_RATIO
    assert figure_loss <= MAXIMUM_FIGURE_LOSS_RATIO, (
        "그래프·도형이 지워졌다. 시험지에서 가장 치명적인 실패다."
    )
    assert slowest_seconds <= MAXIMUM_SECONDS_PER_PAGE
    if difficulty != "쉬움":
        assert color_pen_removal >= MINIMUM_COLOR_PEN_REMOVAL_RATIO


def test_result_reports_api_skipped_without_key() -> None:
    """키 없이 처리하면 실패하지 않고 api_skipped 와 한국어 안내가 담겨야 한다."""
    config = load_config()
    page = make_synthetic_page(seed=5)

    result = clean_image(page.written_page_image, config, api_key=None)

    assert result["api_skipped"] is True
    assert result["api_call_count"] == 0
    assert result["api_cost_usd"] == 0.0
    assert "키" in result["api_message"]


def test_comparison_image_is_produced() -> None:
    """원본/결과 비교 이미지가 만들어져야 한다(강사가 눈으로 확인하는 핵심 산출물)."""
    config = load_config()
    page = make_synthetic_page(seed=4)

    result = clean_image(page.written_page_image, config, api_key=None)
    compare_image = result["compare_image"]

    assert compare_image.ndim == 3
    # 좌우로 붙였으므로 너비가 원본의 2배 남짓이어야 한다
    assert compare_image.shape[1] > page.written_page_image.shape[1] * 1.9


def test_synthetic_masks_are_disjoint_and_nonempty() -> None:
    """합성 시험지의 정답 마스크 자체가 올바른지 확인한다(테스트의 테스트).

    정답이 틀리면 위의 모든 숫자가 의미를 잃는다.
    """
    page = make_synthetic_page(seed=2)

    assert page.printed_text_mask.sum() > 5000, "인쇄 글자가 너무 적다."
    assert page.figure_mask.sum() > 500, "도형이 그려지지 않았다."
    assert page.handwriting_mask.sum() > 5000, "필기가 너무 적다."
    # 인쇄 글자 마스크와 도형 마스크는 겹치지 않아야 한다
    assert (page.printed_text_mask & page.figure_mask).sum() == 0
    # 마스크 값은 0 또는 1 이어야 한다
    for mask in (page.printed_text_mask, page.figure_mask, page.handwriting_mask):
        assert set(np.unique(mask)).issubset({0, 1})
