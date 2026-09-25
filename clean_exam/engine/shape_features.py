"""덩어리(연결 요소)의 모양을 재는 공용 함수들.

색 필터와 획 필터가 같은 측정을 쓰므로 한 곳에 둔다 (서로 import 하면 순환이 생긴다).
"""

from __future__ import annotations

import cv2
import numpy as np


def measure_internal_holes(component_pixels: np.ndarray) -> tuple[float, float]:
    """덩어리 안의 "구멍"을 재서 (메웠을 때의 꽉참, 가장 큰 구멍의 비율)을 돌려준다.

    · 색으로 인쇄된 제목 띠 → 흰 글자가 **자잘한 구멍 여러 개**를 뚫는다.
    · 펜 동그라미           → 가운데에 **큰 구멍 하나**가 뚫려 있다.
    단순 꽉참은 둘이 겹치고(띠 0.41 / 동그라미 0.30), 구멍을 다 메우면 동그라미도 원반이 된다.
    "구멍이 자잘한가, 하나가 큰가"가 답이다.
    """
    component_height, component_width = component_pixels.shape
    box_area = float(max(component_height * component_width, 1))

    background = (~component_pixels).astype(np.uint8)
    hole_count, hole_labels, hole_statistics, _ = cv2.connectedComponentsWithStats(background, 4)

    # 사각형 테두리에 닿은 배경 덩어리는 "바깥"이지 구멍이 아니다
    border_labels = set(hole_labels[0, :]) | set(hole_labels[-1, :]) \
        | set(hole_labels[:, 0]) | set(hole_labels[:, -1])

    total_hole_area = 0.0
    largest_hole_area = 0.0
    for hole_index in range(1, hole_count):
        if hole_index in border_labels:
            continue
        hole_area = float(hole_statistics[hole_index, cv2.CC_STAT_AREA])
        total_hole_area += hole_area
        largest_hole_area = max(largest_hole_area, hole_area)

    filled_area = float(component_pixels.sum()) + total_hole_area
    return filled_area / box_area, largest_hole_area / box_area
