"""2) 색 기반 필기 제거 — 색펜은 지우고, **컬러 인쇄는 지키는** 단계.

예전 가정: "인쇄물의 잉크는 무채색이다. 그러니 채도가 높으면 전부 색펜이다."
→ **틀렸다.** 요즘 문제집에는 단원 제목 띠, 동그란 유형 아이콘, 색으로 인쇄된 문제 번호,
  색 강조 박스가 흔하다. 채도만 보고 지우면 이것들이 통째로 사라진다
  (합성 시험지로 재 보니 컬러 인쇄 픽셀의 88% 가 지워졌다).

그래서 픽셀 하나하나가 아니라 **색 덩어리(연결 요소) 하나하나를 보고** 판정한다.
실제 시험지 사진에서 잰 값이 근거다:

                     획 최대 두께      크기           꽉참        채도
  파란 인쇄 문제번호    5.7~6.6 px     18x30 px      0.40~0.63    78~91
  빨간 볼펜 동그라미·별  13~42 px     139~474 px    0.06~0.73   101~124

"컬러 인쇄"로 보고 지키는 경우는 두 가지다.
  (A) 글자형: 인쇄 텍스트 줄 위에 앉아 있고, 글자 높이가 맞고, 획이 가늘다
              → 색으로 인쇄된 문제 번호·강조 글자
  (B) 면형  : 사각형을 빽빽하게 채우고, 짧은 변도 충분히 크고, 지나치게 길쭉하지 않다
              → 제목 띠, 원형 아이콘, 강조 박스
그 밖의 모든 색 덩어리는 색펜으로 보고 지운다.

여전히 남는 한계: 색펜이 인쇄 글자 **위에 겹쳐** 그어져 있으면, 그 자리의 인쇄 글자는
사진상 이미 색에 덮여 있다. 지운 뒤 주변 색으로 메우는 방법으로는 되살릴 수 없다.
3단계 방법 B(같은 시험지 여러 장 겹치기)만이 이 문제를 실제로 푼다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from engine.shape_features import measure_internal_holes
from engine.stroke_filter import find_printed_text_lines


@dataclass
class ColorComponent:
    """색이 있는 덩어리 하나에 대해 알아낸 것들."""

    component_index: int
    bounding_box: tuple[int, int, int, int]   # (x, y, 너비, 높이)
    area_px: int
    max_thickness_px: float                   # 획의 가장 두꺼운 곳
    fill_ratio: float                         # 넓이 ÷ 사각형 넓이 (구멍은 뺀 값)
    filled_fill_ratio: float                  # 내부 구멍을 메웠을 때의 꽉참
    closed_fill_ratio: float                  # 작은 닫힘 연산으로 글자 틈을 메웠을 때의 꽉참
    largest_hole_ratio: float                 # 가장 큰 내부 구멍 ÷ 사각형 넓이
    mean_saturation: float
    is_printed: bool = False                  # True = 컬러 인쇄(지키기), False = 색펜(지우기)
    reason: str = ""                          # 왜 그렇게 판정했는지 (디버그용)


def determine_saturation_threshold(corrected_color_image: np.ndarray,
                                   color_config: dict[str, Any]) -> int:
    """이 사진에서 "색이 있다"고 볼 채도 기준을 사진 스스로에게서 정한다.

    왜 고정값을 못 쓰나: 사진마다 색이 살아 있는 정도가 천차만별이다.
    실측한 다섯 장의 잉크 픽셀 채도 분포(중앙값 → 90분위):
        선명한 사진      11 → 100     (고정 60 으로 잘 잡힌다)
        빛바랜 사진       8 →  20     (고정 60 으로는 빨간펜이 **한 픽셀도** 안 잡혔다)
        화면 캡처       255 → 255     (전부 원색)
    빛바랜 사진에서는 빨간 동그라미가 그대로 살아남아 결과가 엉망이 됐다.

    방법: 잉크(어두운) 픽셀의 채도만 모아 Otsu 로 두 무리(무채색 잉크 / 색 잉크)로 가른다.
    두 무리의 평균 차이가 충분하지 않으면(= 색펜이 없는 흑백 시험지면) 설정값을 그대로 쓴다.
    """
    fallback_threshold = int(color_config["saturation_threshold"])
    gray_image = cv2.cvtColor(corrected_color_image, cv2.COLOR_BGR2GRAY)
    saturation_channel = cv2.cvtColor(corrected_color_image, cv2.COLOR_BGR2HSV)[:, :, 1]

    ink_mask = cv2.adaptiveThreshold(gray_image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 35, 12)
    ink_saturation_values = saturation_channel[ink_mask > 0]
    if ink_saturation_values.size < 500:
        return fallback_threshold

    otsu_threshold, _ = cv2.threshold(ink_saturation_values, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_group = ink_saturation_values[ink_saturation_values <= otsu_threshold]
    high_group = ink_saturation_values[ink_saturation_values > otsu_threshold]
    if low_group.size < 100 or high_group.size < 100:
        return fallback_threshold

    group_gap = float(high_group.mean() - low_group.mean())
    colored_group_mean = float(high_group.mean())
    achromatic_group_mean = float(low_group.mean())

    # 갈라진 두 무리 중 **낮은 쪽까지 이미 색이 진하다**면, 이 페이지에는 무채색 잉크가
    # 거의 없고 전부 색이라는 뜻이다. 그럴 때 Otsu 는 "색 vs 무색"이 아니라
    # "덜 진한 색 vs 더 진한 색"을 가르게 되어, 연한 색으로 인쇄된 제목 띠가
    # 통째로 "색이 아님"으로 빠져 버린다. 그런 경우에는 하한을 그대로 쓴다.
    if achromatic_group_mean > color_config["saturation_colored_group_min"]:
        return int(color_config["saturation_threshold_floor"])
    if (group_gap < color_config["saturation_split_min_gap"]
            or colored_group_mean < color_config["saturation_colored_group_min"]):
        # 두 무리로 갈리지 않거나, 갈려도 "색이 있는 쪽"이 충분히 색답지 않다
        # = 색펜이 없는 흑백 시험지다. Otsu 는 흑백 이미지에서도 잡음을 두 무리로 가르므로
        #   이 두 조건이 없으면 종이 얼룩을 색펜으로 오인한다(실제로 겪었다).
        return fallback_threshold

    return max(int(otsu_threshold), int(color_config["saturation_threshold_floor"]))


def build_colored_pixel_mask(corrected_color_image: np.ndarray,
                             color_config: dict[str, Any]) -> np.ndarray:
    """채도가 높은 픽셀(색이 있는 곳)을 1로 표시한 마스크를 만든다.

    여기서는 색펜인지 컬러 인쇄인지 아직 가리지 않는다. "색이 있다"까지만 본다.
    """
    hsv_image = cv2.cvtColor(corrected_color_image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]

    saturation_threshold = determine_saturation_threshold(corrected_color_image, color_config)
    colored_mask = (
        (saturation_channel > saturation_threshold)
        & (value_channel > color_config["value_min"])
    ).astype(np.uint8)
    # 점점이 흩어진 노이즈(JPEG 색 번짐 등)를 없앤다
    return cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# 색상(hue) 계열. OpenCV 의 hue 는 0~179 이고 빨강은 0 과 180 양쪽으로 걸쳐 있다.
HUE_FAMILIES: list[tuple[str, list[tuple[int, int]]]] = [
    ("빨강", [(0, 12), (162, 179)]),
    ("주황·노랑", [(13, 38)]),
    ("초록", [(39, 85)]),
    ("청록·파랑", [(86, 135)]),
    ("보라·자주", [(136, 161)]),
]


def split_colored_mask_by_hue(colored_mask: np.ndarray, hue_channel: np.ndarray,
                              saturation_channel: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """색 마스크를 색상 계열별로 쪼갠다.

    왜 쪼개나: 빨간 볼펜 동그라미가 파란 제목 띠를 가로지르면 둘이 **하나의 덩어리**로
    이어져 버린다. 그러면 크기·꽉참·구멍 같은 근거가 전부 뒤섞여 판정이 불가능해진다
    (실측: 536x217 크기에 채도 145 — 인쇄 110 과 펜 200 의 중간값이 나왔고, 제목 띠가
     통째로 색펜으로 판정되어 지워졌다).
    색상이 다르면 애초에 섞이지 않게 나눠서 각각 덩어리를 찾는다.
    """
    masks_by_family: list[tuple[str, np.ndarray]] = []
    for family_name, hue_ranges in HUE_FAMILIES:
        family_mask = np.zeros_like(colored_mask)
        for hue_low, hue_high in hue_ranges:
            family_mask |= ((colored_mask > 0)
                            & (hue_channel >= hue_low)
                            & (hue_channel <= hue_high)).astype(np.uint8)
        if family_mask.sum() == 0:
            continue
        masks_by_family.extend(
            (f"{family_name}/{band_name}", band_mask)
            for band_name, band_mask in split_mask_by_saturation(family_mask, saturation_channel)
        )
    return masks_by_family


def split_mask_by_saturation(family_mask: np.ndarray, saturation_channel: np.ndarray,
                             minimum_gap: int = 25) -> list[tuple[str, np.ndarray]]:
    """색상이 같은 마스크를 채도 높낮이로 한 번 더 쪼갠다(갈릴 때만).

    왜 채도로 또 나누나: 색상이 비슷한 인쇄와 펜은 색상만으로 안 갈린다
    (예: 팥죽색으로 인쇄된 제목 띠 위에 빨간 볼펜으로 동그라미를 친 경우).
    물리적으로 인쇄 색은 망점을 섞어 만들어 채도가 낮고, 펜 잉크는 염료라 채도가 높다.
    실제 시험지 사진에서 잰 값: 파란 인쇄 채도 78~91 / 빨간 볼펜 채도 101~124.

    Otsu 로 두 무리로 가르되, 두 무리의 평균 차이가 minimum_gap 보다 작으면
    (= 사실은 한 무리면) 나누지 않는다.
    """
    saturation_values = saturation_channel[family_mask > 0]
    if saturation_values.size < 200:
        return [("", family_mask)]

    otsu_threshold, _ = cv2.threshold(saturation_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_band = saturation_values[saturation_values <= otsu_threshold]
    high_band = saturation_values[saturation_values > otsu_threshold]
    if low_band.size < 100 or high_band.size < 100:
        return [("", family_mask)]
    if float(high_band.mean() - low_band.mean()) < minimum_gap:
        return [("", family_mask)]

    low_mask = ((family_mask > 0) & (saturation_channel <= otsu_threshold)).astype(np.uint8)
    high_mask = ((family_mask > 0) & (saturation_channel > otsu_threshold)).astype(np.uint8)
    return [("연한", low_mask), ("진한", high_mask)]


def find_achromatic_text_lines(corrected_color_image: np.ndarray,
                               color_config: dict[str, Any]) -> tuple[list[tuple[int, int]], float]:
    """검은 인쇄 글자만 보고 텍스트 줄의 위/아래 경계를 찾는다.

    색 덩어리가 "인쇄 글자 줄 위에 앉아 있는지" 판단하려면 줄 위치가 필요하다.
    색 있는 픽셀은 빼고 재야 색펜 낙서에 줄 위치가 휘둘리지 않는다.

    돌려주는 값: ([(줄 시작 y, 줄 끝 y), ...], 대표 줄 높이)
    """
    gray_image = cv2.cvtColor(corrected_color_image, cv2.COLOR_BGR2GRAY)
    saturation_channel = cv2.cvtColor(corrected_color_image, cv2.COLOR_BGR2HSV)[:, :, 1]

    ink_mask = cv2.adaptiveThreshold(gray_image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 35, 12)
    ink_mask[saturation_channel > determine_saturation_threshold(corrected_color_image, color_config)] = 0

    ink_gray_values = gray_image[ink_mask > 0]
    if ink_gray_values.size < 100:
        return [], 0.0
    dark_cutoff = float(np.percentile(ink_gray_values, 50))
    dark_ink_mask = ((ink_mask > 0) & (gray_image <= dark_cutoff)).astype(np.uint8)

    text_lines = find_printed_text_lines(ink_mask, dark_ink_mask, minimum_gap_px=4)
    line_heights = [bottom - top for top, bottom in text_lines]
    typical_line_height = float(np.median(line_heights)) if line_heights else 0.0
    return text_lines, typical_line_height


def measure_closed_fill(component_pixels: np.ndarray, max_thickness_px: float) -> float:
    """획 두께의 1/3 크기 닫힘(closing) 연산으로 **글자 틈만** 메운 뒤의 꽉참을 잰다.

    왜 필요한가: 저해상도 사진을 키우면 제목 띠의 흰 글자가 띠 가장자리까지 번져서
    "닫힌 구멍"이 아니라 "열린 통로"가 된다. 그러면 구멍 메우기로는 꽉참이 안 올라간다
    (실측: 고해상도 띠 1.00 → 저해상도 띠 0.63, 그래서 색펜으로 오인돼 지워졌다).
    띠 두께의 1/3 짜리 작은 닫힘은 글자 틈(몇 px)은 메우지만, 비스듬히 그은 굵은 펜 획의
    큰 빈 모서리나 동그라미의 큰 구멍은 못 메운다. 그래서 띠만 골라 복원된다.
    """
    kernel_size = int(max(3, min(15, round(max_thickness_px / 3.0))))
    kernel_size |= 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(component_pixels.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return float(closed.sum()) / float(max(component_pixels.size, 1))


def classify_color_components(corrected_color_image: np.ndarray,
                              color_config: dict[str, Any]) -> tuple[list[ColorComponent], np.ndarray]:
    """색 덩어리를 하나씩 보고 "컬러 인쇄"와 "색펜"으로 나눈다.

    색상 계열(빨강/파랑/초록…)별로 따로 덩어리를 찾는다. 서로 다른 색이 겹쳐 그어져도
    한 덩어리로 뭉쳐지지 않게 하기 위해서다.

    돌려주는 값: (덩어리 목록, 덩어리 번호가 칠해진 라벨 이미지)
    """
    colored_mask = build_colored_pixel_mask(corrected_color_image, color_config)
    hsv_image = cv2.cvtColor(corrected_color_image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    saturation_channel = hsv_image[:, :, 1]

    combined_labels = np.zeros(colored_mask.shape, dtype=np.int32)
    text_lines, typical_line_height = find_achromatic_text_lines(corrected_color_image, color_config)

    minimum_area = color_config["min_blob_area_px"]
    maximum_print_thickness = color_config["print_text_max_thickness_px"]
    minimum_solid_fill = color_config["print_solid_min_fill"]
    minimum_solid_side = color_config["print_solid_min_side_px"]
    maximum_solid_aspect = color_config["print_solid_max_aspect"]
    maximum_hole_ratio = color_config["print_solid_max_hole_ratio"]

    components: list[ColorComponent] = []
    next_label = 1
    for family_name, family_mask in split_colored_mask_by_hue(
            colored_mask, hue_channel, saturation_channel):
        component_count, labeled_image, statistics, _ = cv2.connectedComponentsWithStats(family_mask, 8)
        if component_count <= 1:
            continue
        distance_map = cv2.distanceTransform(family_mask, cv2.DIST_L2, 3)

        for component_index in range(1, component_count):
            area_px = int(statistics[component_index, cv2.CC_STAT_AREA])
            if area_px < minimum_area:
                continue
            left = int(statistics[component_index, cv2.CC_STAT_LEFT])
            top = int(statistics[component_index, cv2.CC_STAT_TOP])
            width = int(statistics[component_index, cv2.CC_STAT_WIDTH])
            height = int(statistics[component_index, cv2.CC_STAT_HEIGHT])

            box_slice = (slice(top, top + height), slice(left, left + width))
            component_pixels = labeled_image[box_slice] == component_index
            max_thickness = float(2.0 * distance_map[box_slice][component_pixels].max())
            fill_ratio = area_px / float(max(width * height, 1))
            filled_fill_ratio, largest_hole_ratio = measure_internal_holes(component_pixels)
            closed_fill_ratio = measure_closed_fill(component_pixels, max_thickness)
            mean_saturation = float(saturation_channel[box_slice][component_pixels].mean())

            # 계열별 라벨 번호가 겹치지 않도록 전역 번호를 새로 붙인다
            combined_labels[box_slice][component_pixels] = next_label
            component = ColorComponent(
                component_index=next_label,
                bounding_box=(left, top, width, height),
                area_px=area_px,
                max_thickness_px=max_thickness,
                fill_ratio=fill_ratio,
                filled_fill_ratio=filled_fill_ratio,
                closed_fill_ratio=closed_fill_ratio,
                largest_hole_ratio=largest_hole_ratio,
                mean_saturation=mean_saturation,
            )
            next_label += 1

            # (A) 글자형 컬러 인쇄: 인쇄 줄 위에 앉아 있고, 글자 높이가 맞고, 획이 가늘다
            center_y = top + height / 2.0
            sits_on_text_line = any(line_top <= center_y <= line_bottom
                                    for line_top, line_bottom in text_lines)
            height_fits_line = (
                typical_line_height > 0 and 0.25 <= height / typical_line_height <= 1.40
            )
            thickness_limit = maximum_print_thickness
            if typical_line_height > 0:
                # 글자가 클수록 획도 굵다. 줄 높이의 절반까지는 인쇄 글자로 인정한다.
                thickness_limit = max(thickness_limit, typical_line_height * 0.5)
            # 손글씨는 크게 휘갈겨 사각형을 성기게 채우고(실측 0.30), 인쇄 글자는
            # 빽빽하다(실측 0.42). 색 볼펜으로 쓴 글씨가 마침 인쇄 줄 위에 얹혔을 때
            # 인쇄로 오인되는 것을 이 조건으로 걸러낸다.
            is_compact_like_print = fill_ratio >= color_config["print_text_min_fill"]
            if (sits_on_text_line and height_fits_line
                    and max_thickness <= thickness_limit and is_compact_like_print):
                component.is_printed = True
                component.reason = f"인쇄 줄 위의 가는 색 글자({family_name})"

            # (B) 면형 컬러 인쇄: 구멍이 자잘하고 빽빽하게 채워진 큼직한 덩어리
            longer_side = max(width, height)
            shorter_side = min(width, height)
            aspect_ratio = longer_side / max(shorter_side, 1)

            # 이미지 테두리에 닿는 색 면은 인쇄 띠·아이콘이 아니라 **사진에 들어온 책상·물건**이다.
            # 실제 사진에서 종이 위쪽에 보이던 초록 책상 면(1101x144, 10만 픽셀)이 "가로 띠"로
            # 잡혀 지켜지는 바람에 순백 비율이 81% → 73% 로 떨어졌다. 인쇄 띠는 페이지 안쪽에 있다.
            image_height, image_width = colored_mask.shape[:2]
            touches_image_border = (left == 0 or top == 0
                                    or left + width >= image_width or top + height >= image_height)

            # 길쭉한 덩어리는 두 가지일 수 있다: 넓고 납작한 **제목 띠**(인쇄)와
            # 길게 그은 **굵은 펜 획**. 띠는 축에 나란한 반듯한 직사각형이라 사각형을
            # 거의 100% 채우지만, 비스듬히 그은 펜 획은 사각형 모서리가 비어 0.7 언저리다.
            # (실측: 제목 띠 484x39 → 1.00 / 실제 사진의 굵은 빨간 획 22x195 → 0.73)
            required_fill = minimum_solid_fill
            if aspect_ratio > maximum_solid_aspect:
                required_fill = color_config["print_solid_elongated_min_fill"]

            # 꽉참은 "구멍 메운 값"과 "글자 틈 닫은 값" 중 큰 쪽을 쓴다(둘 다 띠를 복원하는 방법이고,
            # 동그라미의 큰 구멍은 어느 쪽으로도 안 메워진다)
            effective_fill_ratio = max(filled_fill_ratio, closed_fill_ratio)
            if (effective_fill_ratio >= required_fill
                    and largest_hole_ratio <= maximum_hole_ratio
                    and shorter_side >= minimum_solid_side
                    and not touches_image_border):
                component.is_printed = True
                component.reason = f"빽빽하게 채워진 색 면({family_name})"

            # (C) 가로로 놓인 굵은 띠: 저해상도에서는 흰 글자가 넓게 번져 띠의 꽉참이 0.6 대로
            # 떨어지지만, **최대 두께가 띠의 높이와 같다**는 성질은 흔들리지 않는다
            # (실측: 두께 36/높이 39, 두께 31/높이 32). 펜 획은 아무리 굵어도 두께가 획 굵기다.
            # 실제 사진에서 이 조건을 거의 통과한 유일한 펜 자국(굵은 빨간 획 22x195)은
            # **세로**로 놓여 있었다. 인쇄 띠는 가로로 놓이므로 가로 방향만 인정한다.
            # 대가: 가로로 길게 그은 굵은 형광펜 자국은 인쇄로 봐서 안 지운다(지우면 밑의 글자도
            #       같이 사라지므로 안 지우는 쪽이 오히려 안전하다).
            is_horizontal_band = (
                width >= height * color_config["band_min_aspect"]
                and height >= minimum_solid_side
                and max_thickness >= height * color_config["band_thickness_to_height_min"]
                and largest_hole_ratio <= maximum_hole_ratio
                and effective_fill_ratio >= color_config["band_min_fill"]
                and not touches_image_border
            )
            if is_horizontal_band and not component.is_printed:
                component.is_printed = True
                component.reason = f"가로로 놓인 굵은 인쇄 띠({family_name})"

            if not component.is_printed:
                component.reason = f"색펜으로 판단({family_name})"
            components.append(component)

    return components, combined_labels


def build_pen_and_print_masks(components: list[ColorComponent], labeled_image: np.ndarray,
                              image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """판정 결과를 "지울 색펜" 마스크와 "지킬 컬러 인쇄" 마스크 두 장으로 만든다."""
    pen_mask = np.zeros(image_shape, dtype=np.uint8)
    print_mask = np.zeros(image_shape, dtype=np.uint8)
    for component in components:
        left, top, width, height = component.bounding_box
        box_slice = (slice(top, top + height), slice(left, left + width))
        component_pixels = labeled_image[box_slice] == component.component_index
        target_mask = print_mask if component.is_printed else pen_mask
        target_mask[box_slice][component_pixels] = 1
    return pen_mask, print_mask


def remove_color_pen(normalized_gray_image: np.ndarray, corrected_color_image: np.ndarray,
                     color_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """색펜을 지우고 컬러 인쇄는 지킨 그레이스케일 이미지를 돌려준다.

    돌려주는 값: (색펜을 지운 이미지, 지운 자리 마스크, 인쇄로 확정한 색 영역 마스크)

    인쇄로 확정한 영역은 **픽셀 값을 건드리지 않고** 마스크로만 알려 준다.
    한때 여기서 "인쇄니까 진하게" 하려고 어둡게 눌러 봤는데, 그러면 페이지 전체의
    "인쇄 잉크는 이만큼 진하다"는 기준이 올라가서 **멀쩡한 검은 인쇄 글자가 상대적으로
    흐려 보이게 되고, 필기로 오인되어 지워졌다**(실측: 인쇄 손실이 7.7% → 16.0% 로 뛰었다).
    값을 바꾸는 대신 다음 단계에 "이건 인쇄다"라고 알려 주는 편이 안전하다.
    """
    empty_mask = np.zeros(normalized_gray_image.shape[:2], dtype=np.uint8)
    if not color_config["enabled"]:
        return normalized_gray_image, empty_mask, empty_mask

    components, labeled_image = classify_color_components(corrected_color_image, color_config)
    if not components:
        return normalized_gray_image, empty_mask, empty_mask

    pen_mask, print_mask = build_pen_and_print_masks(
        components, labeled_image, normalized_gray_image.shape[:2]
    )

    # 지울 영역만 조금 부풀린다(테두리 잔상 방지). 지킬 영역은 절대 건드리지 않는다.
    dilate_pixels = color_config["dilate_px"]
    if dilate_pixels > 0:
        kernel_size = 2 * dilate_pixels + 1
        pen_mask = cv2.dilate(pen_mask, np.ones((kernel_size, kernel_size), np.uint8))
    pen_mask[print_mask > 0] = 0   # 부풀린 영역이 컬러 인쇄를 침범하지 않게 한다

    result_image = normalized_gray_image
    if pen_mask.sum() > 0:
        result_image = cv2.inpaint(result_image, pen_mask * 255,
                                   color_config["inpaint_radius"], cv2.INPAINT_TELEA)

    return result_image, pen_mask, print_mask
