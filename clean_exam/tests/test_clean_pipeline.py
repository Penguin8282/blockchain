"""엔진 전체(1~6단계)를 합성 시험지로 검증한다.

합성 시험지는 "어느 픽셀이 필기인지"를 알고 만들기 때문에 정확도를 수치로 잴 수 있다.
실제 시험지 사진보다 쉬운 문제이므로, 여기 숫자가 좋다고 실제 성능이 그만큼 나오는 것은 아니다.
(실제 사진의 성능은 tests/samples/ 로 사람이 눈으로 확인한다.)
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from engine.clean import clean_image
from engine.settings import load_config
from tests.synthetic_exam import add_student_handwriting, make_clean_exam_page

# 아래 기준값은 "지금 이 정도는 나온다"는 기록이자 회귀 방지선이다.
# 4단계(자체 모델)로 올라가면 이 숫자를 함께 올린다.
MINIMUM_HANDWRITING_REMOVAL_RATIO = 0.55   # 필기 픽셀 중 지워진 비율
MAXIMUM_PRINTED_LOSS_RATIO = 0.12          # 인쇄 픽셀 중 잘못 지워진 비율
MAXIMUM_SECONDS_PER_PAGE = 3.0


def measure_accuracy(seed: int) -> dict[str, float]:
    """합성 시험지 한 장을 처리하고 필기 제거율·인쇄 손실률을 잰다."""
    config = load_config()
    clean_page = make_clean_exam_page(seed=seed)
    written_page, handwriting_mask = add_student_handwriting(clean_page, seed=seed)

    started_at = time.time()
    result = clean_image(written_page, config, api_key=None)
    elapsed_seconds = time.time() - started_at

    cleaned_image = result["cleaned_image"]
    # 기울지 않은 입력이므로 크기가 그대로여야 한다(회전이 걸리면 캔버스가 커진다).
    assert cleaned_image.shape[:2] == clean_page.shape[:2], (
        f"결과 크기 {cleaned_image.shape[:2]} 가 원본 {clean_page.shape[:2]} 와 다르다. "
        "페이지 배치는 절대 바뀌면 안 된다."
    )

    # 정답: 필기가 실제로 그려진 곳 중 잉크가 찍힌 픽셀
    written_gray = cv2.cvtColor(written_page, cv2.COLOR_BGR2GRAY)
    handwriting_ink = (handwriting_mask > 0) & (written_gray < 200)
    # 정답: 원래 인쇄 잉크가 있던 곳 (필기가 덮어쓴 자리는 제외 — 거기는 판단 불가)
    printed_ink = (clean_page < 120) & (handwriting_mask == 0)

    became_white = cleaned_image >= 250
    handwriting_removal_ratio = float(became_white[handwriting_ink].mean()) if handwriting_ink.any() else 1.0
    printed_loss_ratio = float(became_white[printed_ink].mean()) if printed_ink.any() else 0.0

    return {
        "handwriting_removal_ratio": handwriting_removal_ratio,
        "printed_loss_ratio": printed_loss_ratio,
        "elapsed_seconds": elapsed_seconds,
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pipeline_removes_handwriting_and_keeps_printed(seed: int) -> None:
    """필기는 충분히 지우고 인쇄는 거의 남겨야 하며, 한 장을 3초 안에 처리해야 한다."""
    measurements = measure_accuracy(seed)
    print(f"\n[seed {seed}] 필기 제거율 {measurements['handwriting_removal_ratio']*100:.0f}% / "
          f"인쇄 손실률 {measurements['printed_loss_ratio']*100:.1f}% / "
          f"{measurements['elapsed_seconds']:.2f}초")

    assert measurements["handwriting_removal_ratio"] >= MINIMUM_HANDWRITING_REMOVAL_RATIO
    assert measurements["printed_loss_ratio"] <= MAXIMUM_PRINTED_LOSS_RATIO
    assert measurements["elapsed_seconds"] <= MAXIMUM_SECONDS_PER_PAGE


def test_result_reports_api_skipped_without_key() -> None:
    """키 없이 처리하면 실패하지 않고 api_skipped 와 한국어 안내가 담겨야 한다."""
    config = load_config()
    clean_page = make_clean_exam_page(seed=5)
    written_page, _ = add_student_handwriting(clean_page, seed=5)

    result = clean_image(written_page, config, api_key=None)

    assert result["api_skipped"] is True
    assert result["api_call_count"] == 0
    assert result["api_cost_usd"] == 0.0
    assert "키" in result["api_message"]


def test_color_pen_is_removed() -> None:
    """빨간펜 채점 표시는 규칙만으로도(키 없이) 지워져야 한다."""
    config = load_config()
    clean_page = make_clean_exam_page(seed=9)
    written_page, _ = add_student_handwriting(clean_page, seed=9)

    # 빨간 픽셀(채도가 높은 픽셀)이 결과에서 얼마나 남았는지 본다
    hsv_before = cv2.cvtColor(written_page, cv2.COLOR_BGR2HSV)
    red_pen_mask = (hsv_before[:, :, 1] > config["color_filter"]["saturation_threshold"])
    assert red_pen_mask.sum() > 1000, "테스트 사진에 빨간펜이 충분히 그려지지 않았다."

    result = clean_image(written_page, config, api_key=None)
    remaining_dark_ratio = float((result["cleaned_image"][red_pen_mask] < 200).mean())
    print(f"\n빨간펜 자리에 남은 어두운 픽셀 비율: {remaining_dark_ratio*100:.1f}%")
    assert remaining_dark_ratio < 0.15, "빨간펜이 충분히 지워지지 않았다."


def test_comparison_image_is_produced() -> None:
    """원본/결과 비교 이미지가 만들어져야 한다(강사가 눈으로 확인하는 핵심 산출물)."""
    config = load_config()
    clean_page = make_clean_exam_page(seed=4)
    written_page, _ = add_student_handwriting(clean_page, seed=4)

    result = clean_image(written_page, config, api_key=None)
    compare_image = result["compare_image"]

    assert compare_image.ndim == 3
    # 좌우로 붙였으므로 너비가 원본의 2배 남짓이어야 한다
    assert compare_image.shape[1] > written_page.shape[1] * 1.9
