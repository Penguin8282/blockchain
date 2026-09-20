"""5) 그래프·도형 보호 및 강화.

시험지에서 가장 지우면 안 되는 것이 함수 그래프와 도형이다. 학생이 그 위에 보조선을
그어 놓는 일이 많아, 획 하나하나만 보면 필기와 구별이 어렵다. 그래서 "큰 그림"을 본다:

  · 길이가 긴 직선(좌표축, 표 선, 삼각형 변) → Hough 변환으로 찾는다
  · 크면서 획 두께가 일정한 덩어리(곡선, 도형)  → 연결 요소의 크기와 두께 변동계수로 찾는다
  · 판정관이 figure 라고 한 요소                → 무조건 유지

보호하기로 한 것은 끊긴 곳을 잇고(closing), 조금 굵게 만들어(dilate) 완전한 검정으로 올린다.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from engine.preprocess import reshape_hough_lines
from engine.stroke_filter import (
    LABEL_FIGURE,
    LABEL_HANDWRITING,
    LABEL_PRINTED,
    StrokeComponent,
)


def find_long_straight_lines(ink_mask: np.ndarray, graph_config: dict[str, Any]) -> np.ndarray:
    """좌표축·격자·표 선처럼 긴 직선이 지나가는 자리를 1로 칠한 마스크를 만든다."""
    line_mask = np.zeros(ink_mask.shape[:2], dtype=np.uint8)
    detected_lines = cv2.HoughLinesP(
        ink_mask * 255, 1, np.pi / 180,
        threshold=60,
        minLineLength=graph_config["hough_min_line_length"],
        maxLineGap=graph_config["hough_max_line_gap"],
    )
    if detected_lines is None or len(detected_lines) == 0:
        return line_mask
    for start_x, start_y, end_x, end_y in reshape_hough_lines(detected_lines):
        cv2.line(line_mask, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 1, thickness=3)
    return line_mask


def mark_figure_components(labeled_image: np.ndarray, components: list[StrokeComponent],
                           ink_mask: np.ndarray, graph_config: dict[str, Any]) -> list[StrokeComponent]:
    """그래프·도형으로 보이는 요소의 label 을 figure 로 바꾼다(제자리 수정).

    판단 기준 두 가지:
      · 크고, 획 두께가 일정하고, **성긴** 요소 (도형은 속이 빈 윤곽선이라 성글다)
      · 긴 직선이 지나가는 요소

    승격시키지 않는 경우(실제로 겪은 실패를 막기 위한 안전장치):
      · **확신을 갖고** 필기로 판정된 요소(점수 >= rescue_score_ceiling)
        → 학생이 자로 그은 듯한 긴 연필선이 "긴 직선"으로 잡혀 보호되고 새까맣게
          칠해지는 사고가 났다. 확신 있는 필기 판정은 뒤집지 않는다.
      · 빽빽한 요소(fill_ratio 가 큰 것) → 한글 글자 여러 개가 붙어 한 덩어리가 되면
        넓이가 커서 "큰 요소"로 잡히는데, 이것을 도형으로 칠하면 본문 일부만
        새까맣게 굵어진다("민아는", "향으로" 가 실제로 그렇게 됐다).

    반대로 **구제하는 경우**: 필기로 분류됐지만 점수가 rescue_score_ceiling 미만이면서
    긴 직선 위에 있는 요소는 도형으로 되돌린다. 제본 쪽으로 흐려진 좌표축·표 선은
    "흐리고 가늘고 줄 밖"이라 손글씨 점수가 높게 나오는데, 구제 경로가 없으면 그대로
    지워진다(합성 시험지 6장 중 2장에서 도형 픽셀의 36%·68% 가 사라졌다).
    긴 직선을 찾을 때도 확신 있는 필기 획은 빼고 찾는다.
    """
    minimum_figure_area = graph_config["min_figure_area_px"]
    maximum_thickness_cv = graph_config["thickness_cv_max"]
    maximum_fill_ratio = graph_config["figure_max_fill_ratio"]
    rescue_score_ceiling = graph_config["rescue_score_ceiling"]
    minimum_line_coverage = graph_config["line_coverage_min"]
    rescue_line_coverage = graph_config["rescue_line_coverage_min"]
    minimum_figure_darkness_ratio = graph_config["figure_min_darkness_ratio"]

    # 확신 있는 필기 획을 뺀 잉크 마스크에서만 긴 직선을 찾는다
    ink_without_confident_handwriting = ink_mask.copy()
    for component in components:
        if component.label != LABEL_HANDWRITING or component.handwriting_score < rescue_score_ceiling:
            continue
        left, top, width, height = component.bounding_box
        box_slice = (slice(top, top + height), slice(left, left + width))
        ink_without_confident_handwriting[box_slice][
            labeled_image[box_slice] == component.component_index] = 0
    long_line_mask = find_long_straight_lines(ink_without_confident_handwriting, graph_config)

    for component in components:
        if component.label == LABEL_FIGURE:
            continue
        # 회색으로 면을 채운 인쇄 부분은 도형으로 올리지 않는다.
        # 올리면 dilate + 완전 검정으로 칠해져 색칠된 반원이 새까만 덩어리가 된다.
        if component.is_shaded_area:
            continue
        # 아주 확신 있는 필기(점수가 천장을 넘는 것)만 손대지 않는다.
        # 천장을 높게 둔 이유는 위의 직선 덮임 근거가 훨씬 믿을 만하기 때문이다.
        if component.label == LABEL_HANDWRITING and component.handwriting_score >= rescue_score_ceiling:
            continue
        left, top, width, height = component.bounding_box
        box_slice = (slice(top, top + height), slice(left, left + width))
        component_pixels = labeled_image[box_slice] == component.component_index

        # 요소 픽셀 중 얼마나 많은 부분이 "검출된 곧은 직선" 위에 놓여 있는가.
        # 이것이 인쇄된 선 그림과 학생이 손으로 그린 그림을 가르는 가장 강한 근거다.
        # 실측(합성 시험지 6장): 인쇄 도형 요소 0.91~0.95 / 학생이 그린 좌표축·포물선 0.00.
        # 사람이 그은 선은 미세하게 흔들려서 Hough 변환이 곧은 직선으로 아예 못 잡는다.
        overlapping_line_ratio = float(
            (long_line_mask[box_slice][component_pixels] > 0).mean()
        ) if component_pixels.any() else 0.0
        lies_on_long_line = overlapping_line_ratio >= minimum_line_coverage

        # "크고 성기고 두께가 일정하다"만으로는 부족하다. 학생이 그린 좌표축+포물선도
        # 크고 성기고 두께가 일정하기 때문이다(실측: 이 규칙 때문에 손그림의 12% 가
        # 도형으로 보호되어 **새까맣게 강조**됐다. 지우기는커녕 더 눈에 띄게 만든 셈이다).
        # 그래서 인쇄 잉크다운 진하기를 함께 요구한다
        # (실측 진하기비: 인쇄 도형선 1.56 / 학생 손그림선 0.88).
        is_dark_like_print = (
            component.dark_core_darkness
            >= component.local_darkness_reference * minimum_figure_darkness_ratio
        )
        is_large_sparse_and_uniform = (
            component.area_px >= minimum_figure_area
            and component.thickness_cv <= maximum_thickness_cv
            and component.fill_ratio <= maximum_fill_ratio
            and is_dark_like_print
        )

        if component.label == LABEL_HANDWRITING:
            # 필기로 분류된 것을 되돌리는 것은 "거의 전부가 곧은 직선 위에 있다"는
            # 아주 확실한 근거가 있을 때만. 크기·성김만으로 되돌리면 학생의 큰 낙서가
            # 도형으로 살아난다.
            # 기준을 필기 점수가 아니라 **직선 덮임**으로 잡는 이유:
            # 실측에서 인쇄 도형의 직선 덮임은 0.89~0.95, 학생이 손으로 그린 선은 0.00 이라
            # 완전히 갈린다. 반면 필기 점수는 인쇄 도형도 0.81 까지 나와(제본 쪽으로 흐려진
            # 경우) 점수로 자르면 멀쩡한 인쇄 그래프의 절반이 지워졌다.
            if overlapping_line_ratio >= rescue_line_coverage:
                component.label = LABEL_FIGURE
        elif is_large_sparse_and_uniform or lies_on_long_line:
            component.label = LABEL_FIGURE
    return components


def build_label_masks(labeled_image: np.ndarray, components: list[StrokeComponent],
                      image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """분류 결과를 "지울 곳 / 그래프 / 인쇄 글자" 세 장의 마스크로 만든다."""
    masks = {
        LABEL_HANDWRITING: np.zeros(image_shape, dtype=np.uint8),
        LABEL_FIGURE: np.zeros(image_shape, dtype=np.uint8),
        LABEL_PRINTED: np.zeros(image_shape, dtype=np.uint8),
    }
    for component in components:
        target_mask = masks.get(component.label)
        if target_mask is None:
            continue   # unsure 는 어느 마스크에도 넣지 않는다(= 건드리지 않음 = 안전)
        left, top, width, height = component.bounding_box
        box_slice = (slice(top, top + height), slice(left, left + width))
        component_pixels = labeled_image[box_slice] == component.component_index
        target_mask[box_slice][component_pixels] = 1
    return masks


def strengthen_figures(figure_mask: np.ndarray, graph_config: dict[str, Any]) -> np.ndarray:
    """그래프 마스크의 끊긴 선을 잇고 조금 굵게 만든다."""
    strengthened = figure_mask.copy()

    closing_pixels = graph_config["closing_px"]
    if closing_pixels > 0:
        closing_kernel = np.ones((closing_pixels, closing_pixels), np.uint8)
        strengthened = cv2.morphologyEx(strengthened, cv2.MORPH_CLOSE, closing_kernel)

    dilate_pixels = graph_config["dilate_px"]
    if dilate_pixels > 0:
        dilate_kernel = np.ones((2 * dilate_pixels + 1, 2 * dilate_pixels + 1), np.uint8)
        strengthened = cv2.dilate(strengthened, dilate_kernel)

    return strengthened
