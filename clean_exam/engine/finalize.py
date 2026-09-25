"""6) 최종 마무리 — 필기를 실제로 지우고, 인쇄 글자와 그래프를 진하게 만든다.

절대 지켜야 할 것: **원본과 같은 해상도, 같은 위치.** 페이지 배치가 바뀌면 안 된다.
그래서 여기서는 픽셀 값만 바꾸고, 자르거나 옮기거나 크기를 바꾸는 일은 하지 않는다.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def erase_handwriting(gray_image: np.ndarray, handwriting_mask: np.ndarray,
                      inpaint_radius: int = 3) -> np.ndarray:
    """필기로 판정된 픽셀을 지우고 그 자리를 주변 종이 색으로 메운다."""
    if handwriting_mask.sum() == 0:
        return gray_image
    return cv2.inpaint(gray_image, handwriting_mask * 255, inpaint_radius, cv2.INPAINT_TELEA)


def apply_final_contrast(gray_image: np.ndarray, figure_mask: np.ndarray,
                         finalize_config: dict[str, Any],
                         color_print_mask: np.ndarray | None = None,
                         corrected_color_image: np.ndarray | None = None,
                         thin_line_mask: np.ndarray | None = None) -> np.ndarray:
    """인쇄 글자는 또렷하게, 그래프는 완전한 검정으로, 배경은 완전한 흰색으로 만든다.

    thin_line_mask 는 분수 가로줄·표 선처럼 "지키되 강조하지는 않는" 가는 인쇄선이다.
    이 자리는 배경 흰색 처리와 뒷장 비침 제거에서 빼 준다. 가는 인쇄선은 가장자리가 흐려져
    밝기가 높게 나오는 탓에, 빼 주지 않으면 통째로 하얗게 날아간다
    (실측: 그래프로 강조하던 것을 인쇄로 바꾸자 도형 손실이 0.4% → 12.2% 로 뛰었다).

    color_print_mask 가 주어지면 그 자리는 **원본 컬러 사진의 밝기로 되살린다.**
    왜 필요한가: 단원 제목 띠처럼 크고 진한 색 면은 조명 정규화가 "이 동네 종이는 원래
    어둡구나"로 착각해 배경으로 추정해 버린다. 그래서 정규화 뒤에는 하얗게 날아가 있다
    (실측: 컬러 인쇄 손실이 9% → 27% 로 뛰었다).
    통계가 이미 다 끝난 **맨 마지막**에 되살려야 다른 단계의 판단을 망치지 않는다.
    """
    working_image = gray_image.astype(np.float32)

    if (color_print_mask is not None and corrected_color_image is not None
            and color_print_mask.sum() > 0):
        original_luminance = cv2.cvtColor(corrected_color_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        selected = color_print_mask > 0
        working_image[selected] = np.minimum(working_image[selected], original_luminance[selected])

    # 배경을 먼저 하얗게 만든다.
    # 순서가 중요하다: 감마 보정은 밝은 값도 조금 어둡게 만들기 때문에, 감마 뒤에만
    # 잘라 내면 종이가 회색으로 남는다(실측: 결과 이미지의 순백 비율이 16% 에 그쳤다).
    # 그래서 감마 앞뒤로 두 번 잘라 낸다.
    protected = np.zeros(working_image.shape, dtype=bool)
    if color_print_mask is not None:
        protected |= color_print_mask > 0
    if thin_line_mask is not None:
        protected |= thin_line_mask > 0

    paper_cutoff = finalize_config["background_white_cutoff"]
    working_image[(working_image >= paper_cutoff) & ~protected] = 255.0

    # 뒷장 비침(종이 뒤쪽 인쇄가 비쳐 보이는 것)을 지운다.
    # 인쇄 잉크는 훨씬 진하므로(실측 밝기 ~127) 이 기준보다 밝은 것만 지우면 안전하다.
    bleed_cutoff = finalize_config["bleed_through_cutoff"]
    working_image[(working_image >= bleed_cutoff) & ~protected] = 255.0

    # 언샤프 마스크: 원본에서 흐린 버전을 빼면 경계가 강조된다
    blur_radius = finalize_config["unsharp_radius_px"]
    kernel_size = 2 * blur_radius + 1
    blurred_image = cv2.GaussianBlur(working_image, (kernel_size, kernel_size), 0)
    sharpened = working_image + finalize_config["unsharp_amount"] * (working_image - blurred_image)
    sharpened = np.clip(sharpened, 0, 255)

    # 감마 보정으로 글자를 더 진하게 (감마 < 1 이면 어두운 쪽이 더 어두워진다)
    gamma = finalize_config["text_darken_gamma"]
    darkened = 255.0 * np.power(sharpened / 255.0, 1.0 / max(gamma, 1e-6))

    # 배경을 완전한 흰색으로
    white_cutoff = finalize_config["background_white_cutoff"]
    darkened[(darkened >= white_cutoff) & ~protected] = 255.0

    # 그래프는 완전한 검정으로 (강화된 마스크 자리를 그대로 0 으로 찍는다)
    result_image = darkened.astype(np.uint8)
    result_image[figure_mask > 0] = 0
    return result_image


def make_comparison_image(original_color_image: np.ndarray, cleaned_gray_image: np.ndarray) -> np.ndarray:
    """원본과 결과를 좌우로 나란히 붙인 비교 이미지를 만든다.

    강사(그리고 개발 중인 나)가 한눈에 "뭐가 지워졌고 뭐가 잘못 지워졌는지" 보라고 만드는 것이다.
    """
    cleaned_color_image = cv2.cvtColor(cleaned_gray_image, cv2.COLOR_GRAY2BGR)

    # 두 이미지의 높이를 맞춘다(보정 과정에서 크기가 달라졌을 수 있다)
    target_height = max(original_color_image.shape[0], cleaned_color_image.shape[0])

    def pad_to_height(image: np.ndarray) -> np.ndarray:
        pad_amount = target_height - image.shape[0]
        if pad_amount <= 0:
            return image
        return cv2.copyMakeBorder(image, 0, pad_amount, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    left_panel = pad_to_height(original_color_image)
    right_panel = pad_to_height(cleaned_color_image)
    divider = np.full((target_height, 8, 3), 200, dtype=np.uint8)
    return np.hstack([left_panel, divider, right_panel])
