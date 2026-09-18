"""2) 색 기반 필기 제거 — 빨강·파랑·초록 펜과 채점 표시를 지운다.

가정: 인쇄물의 잉크는 무채색(검정)이다. 그래서 **채도(saturation)가 높은 픽셀**은
학생이나 강사가 색펜으로 쓴 것으로 본다. 이 가정은 흑백 인쇄 시험지에서 잘 맞는다.
(컬러 인쇄가 섞인 교재라면 config 의 saturation_threshold 를 올려야 한다.)

한계(반드시 알고 있어야 할 것): 빨간 동그라미가 인쇄된 문제 번호 위에 겹쳐 그어져 있으면,
그 자리의 인쇄 글자는 사진상 이미 빨강으로 덮여 있다. 지운 뒤 주변 색으로 메우는(inpaint)
방법으로는 덮인 글자를 되살릴 수 없다. 3단계 방법 B(같은 시험지 여러 장 겹치기)만이
이 문제를 실제로 푼다.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def build_color_pen_mask(corrected_color_image: np.ndarray, color_config: dict[str, Any]) -> np.ndarray:
    """색펜으로 쓴 곳을 1, 나머지를 0 으로 표시한 마스크(uint8)를 만든다.

    인자:
        corrected_color_image: 보정이 끝난 BGR 컬러 이미지
        color_config: config.yaml 의 color_filter 절
    """
    hsv_image = cv2.cvtColor(corrected_color_image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]

    # 채도가 높으면서(색이 있고) 너무 어둡지는 않은 픽셀 = 색펜
    color_pen_mask = (
        (saturation_channel > color_config["saturation_threshold"])
        & (value_channel > color_config["value_min"])
    ).astype(np.uint8)

    # 점점이 흩어진 노이즈(JPEG 색 번짐 등)를 없앤다
    color_pen_mask = cv2.morphologyEx(color_pen_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 너무 작은 얼룩은 버린다
    component_count, labeled_image, statistics, _ = cv2.connectedComponentsWithStats(color_pen_mask, connectivity=8)
    minimum_area = color_config["min_blob_area_px"]
    cleaned_mask = np.zeros_like(color_pen_mask)
    for component_index in range(1, component_count):
        if statistics[component_index, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned_mask[labeled_image == component_index] = 1

    # 테두리 잔상이 남지 않게 조금 부풀린다
    dilate_pixels = color_config["dilate_px"]
    if dilate_pixels > 0:
        kernel_size = 2 * dilate_pixels + 1
        cleaned_mask = cv2.dilate(cleaned_mask, np.ones((kernel_size, kernel_size), np.uint8))

    return cleaned_mask


def remove_color_pen(normalized_gray_image: np.ndarray, corrected_color_image: np.ndarray,
                     color_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """색펜을 지우고 그 자리를 주변 종이 색으로 메운 그레이스케일 이미지를 돌려준다.

    돌려주는 값: (색펜을 지운 이미지, 지운 자리 마스크)
    """
    if not color_config["enabled"]:
        return normalized_gray_image, np.zeros(normalized_gray_image.shape[:2], dtype=np.uint8)

    color_pen_mask = build_color_pen_mask(corrected_color_image, color_config)
    if color_pen_mask.sum() == 0:
        return normalized_gray_image, color_pen_mask

    inpainted_image = cv2.inpaint(
        normalized_gray_image,
        color_pen_mask * 255,
        color_config["inpaint_radius"],
        cv2.INPAINT_TELEA,
    )
    return inpainted_image, color_pen_mask
