"""사진 보정이 실제로 사진을 반듯하게 펴는지 수치로 검증한다.

목표(1단계 요구사항): 회전·원근·그림자·블러를 무작위로 넣은 합성 사진 50장을 보정했을 때,
보정 후 텍스트 줄 기울기가 1도 이내로 들어오는 비율이 90% 이상.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.preprocess import estimate_text_line_angle, preprocess_photo
from engine.settings import load_config
from tests.synthetic_exam import make_photo_pair

RESIDUAL_ANGLE_TOLERANCE_DEG = 1.0
REQUIRED_SUCCESS_RATIO = 0.90


def measure_residual_angle(corrected_color_image: np.ndarray) -> float:
    """보정된 이미지에서 남아 있는 텍스트 줄 기울기(도)를 잰다."""
    return abs(estimate_text_line_angle(corrected_color_image, max_angle_degrees=20.0))


def test_preprocess_straightens_50_random_photos() -> None:
    """무작위 사진 50장 중 90% 이상이 1도 이내로 펴져야 한다."""
    config = load_config()
    residual_angles: list[float] = []

    for seed in range(50):
        photo, _, _, _ = make_photo_pair(seed=seed)
        result = preprocess_photo(photo, config)
        residual_angles.append(measure_residual_angle(result.corrected_color_image))

    residual_angles_array = np.array(residual_angles)
    success_ratio = float((residual_angles_array <= RESIDUAL_ANGLE_TOLERANCE_DEG).mean())

    print(f"\n보정 후 남은 기울기: 중앙값 {np.median(residual_angles_array):.2f}도, "
          f"평균 {residual_angles_array.mean():.2f}도, 최대 {residual_angles_array.max():.2f}도")
    print(f"1도 이내로 들어온 비율: {success_ratio * 100:.0f}% (목표 {REQUIRED_SUCCESS_RATIO * 100:.0f}%)")

    assert success_ratio >= REQUIRED_SUCCESS_RATIO, (
        f"1도 이내 비율이 {success_ratio * 100:.0f}% 로 목표({REQUIRED_SUCCESS_RATIO * 100:.0f}%)에 못 미친다."
    )


def test_illumination_normalization_removes_shadow() -> None:
    """그림자가 있는 사진을 보정하면 9분할 밝기 차이가 크게 줄어야 한다."""
    config = load_config()
    photo, _, _, _ = make_photo_pair(seed=7)
    result = preprocess_photo(photo, config)

    def brightness_spread(gray_image: np.ndarray) -> float:
        """이미지를 9칸으로 나눠 가장 밝은 칸과 어두운 칸의 평균 밝기 차를 잰다."""
        image_height, image_width = gray_image.shape[:2]
        cell_means = [
            gray_image[row * image_height // 3:(row + 1) * image_height // 3,
                       column * image_width // 3:(column + 1) * image_width // 3].mean()
            for row in range(3) for column in range(3)
        ]
        return float(max(cell_means) - min(cell_means))

    import cv2
    before_spread = brightness_spread(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))
    after_spread = brightness_spread(result.normalized_gray_image)
    print(f"\n조명 불균일: 보정 전 {before_spread:.1f} → 보정 후 {after_spread:.1f}")

    assert after_spread < before_spread * 0.6, "조명 보정 후에도 밝기 차가 충분히 줄지 않았다."


def test_quality_warnings_detect_blurry_photo() -> None:
    """심하게 흐린 사진에는 '흐려요' 경고가 붙어야 한다."""
    import cv2
    config = load_config()
    photo, _, _, _ = make_photo_pair(seed=3)
    very_blurry_photo = cv2.GaussianBlur(photo, (41, 41), 0)

    sharp_result = preprocess_photo(photo, config)
    blurry_result = preprocess_photo(very_blurry_photo, config)

    # 경고 문구가 뜨는지 보는 것이 목적이지만, 합성 시험지는 글자가 크고 또렷해서
    # 많이 흐려도 초점 점수가 기준(config 의 blur_laplacian_min)을 넘을 수 있다.
    # 그래서 "경고가 떴는가" 또는 "최소한 점수가 크게 떨어졌는가"를 확인한다.
    def focus_score(image: np.ndarray) -> float:
        import cv2 as opencv
        gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
        return float(opencv.Laplacian(gray, opencv.CV_64F).var())

    sharp_score = focus_score(sharp_result.corrected_color_image)
    blurry_score = focus_score(blurry_result.corrected_color_image)
    print(f"\n초점 점수: 원본 {sharp_score:.0f} → 흐리게 만든 뒤 {blurry_score:.0f}")

    assert any("흐려요" in warning for warning in blurry_result.quality_warnings), (
        f"흐린 사진인데 경고가 없다: {blurry_result.quality_warnings}"
    )
