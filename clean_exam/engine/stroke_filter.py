"""3) 연필/검정 필기 점수 매기기.

색펜은 채도로 쉽게 걸러지지만, 연필과 검은 볼펜은 인쇄 글자와 색이 같다.
그래서 연결 요소(획 덩어리) 하나하나에 대해 **세 가지 근거**로 "손글씨 점수"(0~1)를 매긴다.

  a) 진하기   : 인쇄는 진하고 균일, 연필은 흐리고 들쭉날쭉하다
  b) 획 두께  : 인쇄는 두께가 일정, 손글씨는 굵었다 얇았다 한다
  c) 위치·정렬: 인쇄 글자는 가지런한 가로 줄 위에 앉아 있고 높이도 고르다.
                줄 밖에 있거나 글자 높이가 안 맞으면 필기일 가능성이 높다

세 근거를 config 의 가중치로 섞어 점수를 만들고,
  점수 >= remove_threshold → "필기"(지움)
  점수 <= keep_threshold   → "인쇄"(안 지움)
  그 사이                  → "애매"  ← 이것만 4단계 클로드 판정관에게 물어본다
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.morphology import skeletonize

# 분류 이름 (문자열을 코드 여기저기에 흩어 놓지 않기 위해 한곳에 모아 둔다)
LABEL_HANDWRITING = "handwriting"   # 학생 필기 → 지운다
LABEL_PRINTED = "printed_text"      # 인쇄된 문제 글자 → 남긴다
LABEL_FIGURE = "figure"             # 인쇄된 그래프·도형·표 → 남기고 더 진하게
LABEL_UNSURE = "unsure"             # 애매 → 판정관에게 물어본다


@dataclass
class StrokeComponent:
    """획 덩어리(연결 요소) 하나에 대해 알아낸 것들."""

    component_index: int
    bounding_box: tuple[int, int, int, int]   # (x, y, 너비, 높이)
    area_px: int
    mean_darkness: float                      # 0(흰색)~255(완전 검정)
    dark_core_darkness: float                 # 획 심지의 진하기(75분위). 가는 선 판별에 쓴다
    darkness_std: float
    thickness_mean_px: float
    thickness_cv: float                       # 두께 변동계수 = 표준편차 / 평균
    fill_ratio: float                         # 사각형을 채운 비율 (인쇄 글자는 빽빽, 손글씨는 성글다)
    local_darkness_reference: float           # 이 요소 주변에서 본 "인쇄 잉크의 진하기" 기준
    density_score: float                      # a) 진하기 근거 (1에 가까울수록 필기스럽다)
    thickness_score: float                    # b) 두께 근거
    fill_score: float                         # b') 획 꽉참 근거
    alignment_score: float                    # c) 정렬 근거
    handwriting_score: float                  # 세 근거의 가중 평균
    label: str = LABEL_UNSURE
    is_protected_thin_line: bool = False       # 분수 가로줄·표 선처럼 '지키되 강조는 안 하는' 선


def binarize_ink(gray_image: np.ndarray, stroke_config: dict[str, Any]) -> np.ndarray:
    """종이에서 잉크(글자·필기) 픽셀만 1로 표시한 마스크를 만든다.

    적응형 이진화를 쓰는 이유: 조명을 이미 고르게 만들었어도 연필은 인쇄보다 훨씬 흐려서,
    페이지 전체에 하나의 밝기 기준을 쓰면 연필이 통째로 날아간다.
    """
    block_size = int(stroke_config["binarize_block_size"])
    if block_size % 2 == 0:
        block_size += 1
    ink_mask = cv2.adaptiveThreshold(
        gray_image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, stroke_config["binarize_offset"],
    )
    # 한 픽셀짜리 먼지 제거
    return cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def find_printed_text_lines(ink_mask: np.ndarray, dark_ink_mask: np.ndarray,
                            minimum_gap_px: int) -> list[tuple[int, int]]:
    """가로 투영으로 인쇄 텍스트 줄의 위/아래 경계 목록을 찾는다.

    인쇄 글자는 '진한 잉크'이므로 dark_ink_mask(진한 픽셀만)로 투영해야 연필 필기에
    휘둘리지 않는다. 돌려주는 값은 [(줄 시작 y, 줄 끝 y), ...].
    """
    row_ink_counts = dark_ink_mask.sum(axis=1).astype(np.float32)
    if row_ink_counts.max() <= 0:
        return []
    # 살짝 부드럽게 만들어 글자 사이 빈 줄에 흔들리지 않게 한다
    smoothing_kernel = np.ones(5, dtype=np.float32) / 5.0
    row_ink_counts = np.convolve(row_ink_counts, smoothing_kernel, mode="same")

    line_threshold = row_ink_counts.max() * 0.12
    is_text_row = row_ink_counts > line_threshold

    text_lines: list[tuple[int, int]] = []
    line_start: int | None = None
    gap_length = 0
    for row_index, row_has_text in enumerate(is_text_row):
        if row_has_text:
            if line_start is None:
                line_start = row_index
            gap_length = 0
        else:
            if line_start is not None:
                gap_length += 1
                if gap_length >= minimum_gap_px:
                    text_lines.append((line_start, row_index - gap_length))
                    line_start = None
                    gap_length = 0
    if line_start is not None:
        text_lines.append((line_start, len(is_text_row) - 1))

    # 너무 얇은 줄(가로선 한 줄짜리)은 텍스트 줄이 아니다
    return [(top, bottom) for top, bottom in text_lines if bottom - top >= 6]


def find_text_columns(dark_ink_mask: np.ndarray) -> list[tuple[int, int]]:
    """세로 투영으로 "단(column)"의 좌/우 경계를 찾는다.

    왜 필요한가: 문제집은 2단 편집이 흔하다. 페이지 전체를 한 번에 가로 투영해 텍스트 줄을
    찾으면, 오른쪽 단의 글자 줄은 왼쪽 단의 줄과 높이가 달라 "줄 밖"으로 잘못 판정된다.
    (실제 샘플에서 오른쪽 단 인쇄 글자가 전부 정렬 점수 1.0 을 받아 필기로 오인되었다.)
    그래서 단을 먼저 나누고, 줄은 단 안에서 따로 찾는다.

    돌려주는 값: [(단 시작 x, 단 끝 x), ...]. 단을 못 나누면 페이지 전체를 한 단으로 본다.
    """
    image_height, image_width = dark_ink_mask.shape[:2]
    column_ink_counts = dark_ink_mask.sum(axis=0).astype(np.float32)
    if column_ink_counts.max() <= 0:
        return [(0, image_width - 1)]

    smoothing_width = max(3, image_width // 100)
    smoothing_kernel = np.ones(smoothing_width, dtype=np.float32) / smoothing_width
    column_ink_counts = np.convolve(column_ink_counts, smoothing_kernel, mode="same")

    column_threshold = column_ink_counts.max() * 0.10
    has_text = column_ink_counts > column_threshold

    # 글자가 있는 구간(run)들을 모은다
    runs: list[list[int]] = []
    for x_position, column_has_text in enumerate(has_text):
        if column_has_text:
            if runs and x_position - runs[-1][1] <= 1:
                runs[-1][1] = x_position
            else:
                runs.append([x_position, x_position])
    if not runs:
        return [(0, image_width - 1)]

    # 좁은 틈(단 사이 구분선 근처 등)은 이어 붙인다
    merge_gap = max(6, int(image_width * 0.02))
    merged_runs = [runs[0]]
    for run_start, run_end in runs[1:]:
        if run_start - merged_runs[-1][1] <= merge_gap:
            merged_runs[-1][1] = run_end
        else:
            merged_runs.append([run_start, run_end])

    # 너무 좁은 구간은 단이 아니다(세로 구분선 하나 같은 것)
    minimum_column_width = int(image_width * 0.08)
    columns = [(start, end) for start, end in merged_runs if end - start >= minimum_column_width]
    return columns if columns else [(0, image_width - 1)]


def build_local_darkness_reference(raw_measurements: list[dict[str, Any]],
                                   image_shape: tuple[int, int],
                                   global_reference: float, tile_count: int = 6) -> np.ndarray:
    """위치마다 다른 "인쇄 잉크의 진하기 기준"을 담은 지도를 만든다.

    왜 필요한가: 한 장 안에서도 위치에 따라 인쇄가 다르게 찍힌다. 실제 샘플에서는 제본 쪽으로
    휘어 들어간 오른쪽 단의 인쇄 글자가 왼쪽 단보다 27% 흐리게 나왔고(128 대 94),
    페이지에 기준을 하나만 쓰면 그 인쇄 글자가 전부 "흐리니까 연필"로 잘못 판정된다.

    방법: 페이지를 tile_count × tile_count 칸으로 나누고, 칸마다 그 안에 중심이 있는 요소들의
    평균 진하기를 넓이로 가중해 85분위를 구한다. 전체 기준과 **같은 척도**로 재야 하므로
    픽셀 단위가 아니라 반드시 요소 평균 단위로 잰다.
    연필만 있는 칸에서 기준이 너무 낮아지지 않도록 전체 기준의 70% 를 바닥값으로 깐다.
    """
    image_height, image_width = image_shape
    reference_floor = global_reference * 0.70
    tile_reference_grid = np.full((tile_count, tile_count), global_reference, dtype=np.float32)

    # 요소를 자기 칸으로 나눠 담는다
    measurements_by_tile: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for measurement in raw_measurements:
        tile_row = min(int(measurement["center_y"] * tile_count / image_height), tile_count - 1)
        tile_column = min(int(measurement["center_x"] * tile_count / image_width), tile_count - 1)
        measurements_by_tile.setdefault((tile_row, tile_column), []).append(measurement)

    for (tile_row, tile_column), tile_measurements in measurements_by_tile.items():
        if len(tile_measurements) < 8:   # 표본이 너무 적으면 전체 기준을 그대로 쓴다
            continue
        tile_reference_grid[tile_row, tile_column] = max(
            weighted_darkness_percentile(tile_measurements, 0.85), reference_floor
        )

    return cv2.resize(tile_reference_grid, (image_width, image_height), interpolation=cv2.INTER_LINEAR)


def weighted_darkness_percentile(measurements: list[dict[str, Any]], percentile_ratio: float) -> float:
    """요소들의 평균 진하기를 넓이로 가중해 분위수를 구한다.

    넓이로 가중하는 이유: 작은 점 하나와 큰 글자 덩어리를 똑같이 한 표로 세면
    먼지 같은 작은 요소가 기준을 좌우한다. 페이지의 대표 잉크는 넓이를 많이 차지하는 쪽이다.
    """
    darkness_values = np.array([m["mean_darkness"] for m in measurements])
    area_weights = np.array([m["area_px"] for m in measurements], dtype=np.float64)
    sorted_order = np.argsort(darkness_values)
    cumulative_weight = np.cumsum(area_weights[sorted_order]) / area_weights.sum()
    position = min(int(np.searchsorted(cumulative_weight, percentile_ratio)), len(sorted_order) - 1)
    return max(float(darkness_values[sorted_order[position]]), 1.0)


def is_long_thin_printed_line(measurement: dict[str, Any], local_reference: float,
                              minimum_length_px: int) -> bool:
    """분수 가로줄·좌표축·표 선처럼 "길고 가늘고 진한" 요소인지 판단한다.

    이런 요소는 손글씨 점수 계산에 넣으면 거의 항상 "줄 밖 + 높이 안 맞음"이 되어 필기로
    오인된다(실제 샘플에서 보기 ①~⑤ 의 분수 가로줄이 전부 지워질 뻔했다).
    그래서 점수를 매기기 전에 먼저 걸러 그래프·도형으로 보호한다.
    흐린 연필 밑줄까지 보호하지 않도록 "진하기가 인쇄 기준의 80% 이상"이라는 조건을 함께 건다.
    """
    _, _, width, height = measurement["bounding_box"]
    longer_side = max(width, height)
    shorter_side = max(min(width, height), 1)
    aspect_ratio = longer_side / shorter_side

    # 진하기는 평균이 아니라 "획 심지"(진하기 75분위)로 잰다.
    # 가는 인쇄선은 가장자리가 흐려져 평균이 크게 떨어지지만 심지는 진하다.
    # 실측: 분수 가로줄 심지 105~149 vs 연필 가로획 심지 30~55 로 깨끗이 갈렸다.
    # (평균으로 재면 분수 가로줄 74~91 이라 연필과 섞여 전부 지워질 뻔했다.)
    is_dark_enough = measurement["dark_core_darkness"] >= local_reference * 0.80
    return aspect_ratio >= 8.0 and longer_side >= minimum_length_px and is_dark_enough


def measure_stroke_thickness(ink_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """획 두께를 재기 위한 거리 변환과 뼈대(skeleton)를 만든다.

    거리 변환은 "이 픽셀에서 가장 가까운 배경까지의 거리"다. 획 한가운데(뼈대)에서
    이 값의 2배가 그 자리의 획 두께다. 그래서 뼈대 위의 값만 모으면 두께 분포가 나온다.
    """
    distance_map = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 3)
    skeleton_mask = skeletonize(ink_mask.astype(bool)).astype(np.uint8)
    return distance_map, skeleton_mask


def analyze_components(gray_image: np.ndarray, ink_mask: np.ndarray,
                       stroke_config: dict[str, Any],
                       exclude_from_reference_mask: np.ndarray | None = None
                       ) -> tuple[np.ndarray, list[StrokeComponent]]:
    """모든 연결 요소에 대해 근거들을 재고 손글씨 점수를 매긴다.

    exclude_from_reference_mask: "인쇄 잉크는 이만큼 진하다"는 기준을 정할 때 빼고 셀 영역.
      단원 제목 띠처럼 크고 새까만 색 면이 들어오면, 넓이로 가중한 기준이 그쪽으로 끌려가
      **멀쩡한 검은 본문 글자가 상대적으로 흐려 보이게 되고 필기로 오인되어 지워진다**
      (실측: 제목 띠가 있는 페이지에서 인쇄 글자 손실이 2.2% → 8.0% 로 뛰었다).
      이미 "컬러 인쇄"라고 확정한 영역은 기준 계산에서 빼는 것이 맞다.

    돌려주는 값: (요소 번호가 칠해진 라벨 이미지, StrokeComponent 목록)
    """
    component_count, labeled_image, statistics, centroids = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8
    )
    distance_map, skeleton_mask = measure_stroke_thickness(ink_mask)
    darkness_image = (255.0 - gray_image.astype(np.float32))

    # 인쇄 글자를 찾기 위한 "진한 잉크" 마스크: 페이지에서 특히 어두운 픽셀만
    ink_darkness_values = darkness_image[ink_mask > 0]
    if ink_darkness_values.size == 0:
        return labeled_image, []
    dark_ink_cutoff = float(np.percentile(ink_darkness_values, 75))
    dark_ink_mask = ((ink_mask > 0) & (darkness_image >= dark_ink_cutoff)).astype(np.uint8)

    # 단을 먼저 나누고, 각 단 안에서 텍스트 줄을 찾는다 (2단 편집 대응)
    text_columns = find_text_columns(dark_ink_mask)
    text_lines_by_column: list[list[tuple[int, int]]] = []
    typical_line_height_by_column: list[float] = []
    for column_left, column_right in text_columns:
        column_dark_ink = dark_ink_mask[:, column_left:column_right + 1]
        column_ink = ink_mask[:, column_left:column_right + 1]
        column_text_lines = find_printed_text_lines(
            column_ink, column_dark_ink, stroke_config["text_line_min_gap_px"]
        )
        line_heights = [bottom - top for top, bottom in column_text_lines]
        text_lines_by_column.append(column_text_lines)
        typical_line_height_by_column.append(float(np.median(line_heights)) if line_heights else 0.0)

    minimum_area = stroke_config["min_component_area_px"]

    # ── 1차 통과: 각 요소의 원시 측정값을 모은다 ──────────────────────────
    raw_measurements: list[dict[str, Any]] = []
    for component_index in range(1, component_count):
        area_px = int(statistics[component_index, cv2.CC_STAT_AREA])
        if area_px < minimum_area:
            continue
        left = int(statistics[component_index, cv2.CC_STAT_LEFT])
        top = int(statistics[component_index, cv2.CC_STAT_TOP])
        width = int(statistics[component_index, cv2.CC_STAT_WIDTH])
        height = int(statistics[component_index, cv2.CC_STAT_HEIGHT])

        # 성능을 위해 요소의 사각형 범위 안에서만 계산한다
        box_slice = (slice(top, top + height), slice(left, left + width))
        component_pixels = labeled_image[box_slice] == component_index

        component_darkness = darkness_image[box_slice][component_pixels]
        skeleton_pixels = component_pixels & (skeleton_mask[box_slice] > 0)
        thickness_samples = 2.0 * distance_map[box_slice][skeleton_pixels]
        if thickness_samples.size < 2:
            thickness_samples = 2.0 * distance_map[box_slice][component_pixels]

        thickness_mean = float(thickness_samples.mean()) if thickness_samples.size else 1.0
        thickness_std = float(thickness_samples.std()) if thickness_samples.size else 0.0

        raw_measurements.append({
            "component_index": component_index,
            "bounding_box": (left, top, width, height),
            "area_px": area_px,
            "mean_darkness": float(component_darkness.mean()),
            "dark_core_darkness": float(np.percentile(component_darkness, 75)),
            "darkness_std": float(component_darkness.std()),
            "thickness_mean_px": thickness_mean,
            "thickness_cv": thickness_std / max(thickness_mean, 1e-6),
            "fill_ratio": area_px / float(max(width * height, 1)),
            "center_x": float(centroids[component_index][0]),
            "center_y": float(centroids[component_index][1]),
        })

    if not raw_measurements:
        return labeled_image, []

    # ── 페이지 전체의 기준 진하기 → 위치별 기준 지도 ──────────────────────
    # (사진마다 노출이 다르고, 한 장 안에서도 위치마다 인쇄 진하기가 다르므로
    #  고정 숫자를 쓰지 않고 페이지 스스로에서 기준을 정한다.
    #  면적이 큰 요소일수록 페이지의 대표 잉크이므로 면적으로 가중한다.)
    measurements_for_reference = raw_measurements
    if exclude_from_reference_mask is not None and exclude_from_reference_mask.sum() > 0:
        kept: list[dict[str, Any]] = []
        for measurement in raw_measurements:
            left, top, width, height = measurement["bounding_box"]
            box_slice = (slice(top, top + height), slice(left, left + width))
            component_pixels = labeled_image[box_slice] == measurement["component_index"]
            overlap_ratio = float(
                (exclude_from_reference_mask[box_slice][component_pixels] > 0).mean()
            ) if component_pixels.any() else 0.0
            if overlap_ratio < 0.5:
                kept.append(measurement)
        if len(kept) >= 10:     # 표본이 너무 줄면 오히려 불안정해진다
            measurements_for_reference = kept

    global_darkness_reference = weighted_darkness_percentile(measurements_for_reference, 0.85)
    local_reference_map = build_local_darkness_reference(
        measurements_for_reference, gray_image.shape[:2], global_darkness_reference
    )

    # ── 2차 통과: 근거들을 0~1 점수로 바꾸고 합친다 ───────────────────────
    weight_density = stroke_config["weight_density"]
    weight_thickness = stroke_config["weight_thickness"]
    weight_fill = stroke_config["weight_fill"]
    weight_alignment = stroke_config["weight_alignment"]
    remove_threshold = stroke_config["remove_threshold"]
    keep_threshold = stroke_config["keep_threshold"]
    minimum_line_length = stroke_config["protect_line_min_length_px"]

    components: list[StrokeComponent] = []
    for measurement in raw_measurements:
        center_x = int(np.clip(measurement["center_x"], 0, local_reference_map.shape[1] - 1))
        center_y = int(np.clip(measurement["center_y"], 0, local_reference_map.shape[0] - 1))
        local_reference = float(local_reference_map[center_y, center_x])

        # a) 진하기: 그 자리의 인쇄 기준보다 얼마나 흐린가. 기준의 60% 이하면 만점(1.0).
        darkness_ratio = measurement["mean_darkness"] / max(local_reference, 1.0)
        density_score = float(np.clip((1.0 - darkness_ratio) / 0.40, 0.0, 1.0))

        # b) 두께 변동계수. ※ 실제 시험지 사진에서 인쇄 0.262 / 연필 0.264 로 거의 차이가 없어
        #    판별력이 낮다는 것을 측정으로 확인했다. config 의 기본 가중치를 낮게 둔 이유다.
        thickness_score = float(np.clip((measurement["thickness_cv"] - 0.22) / 0.20, 0.0, 1.0))

        # b') 획 꽉참: 인쇄 글자는 네모를 빽빽이 채우고(0.42), 손글씨는 성글다(0.30).
        #     같은 사진에서 실제로 잰 값이며 두께보다 훨씬 잘 갈린다.
        fill_score = float(np.clip((0.42 - measurement["fill_ratio"]) / 0.16, 0.0, 1.0))

        # c) 정렬: 자기 단 안의 인쇄 텍스트 줄에 맞으면 0점, 아니면 1점
        alignment_score = compute_alignment_score(
            measurement, text_columns, text_lines_by_column, typical_line_height_by_column
        )

        handwriting_score = (
            weight_density * density_score
            + weight_thickness * thickness_score
            + weight_fill * fill_score
            + weight_alignment * alignment_score
        )

        # 분수 가로줄·좌표축·표 선은 점수와 무관하게 먼저 **보호**한다.
        # 단, "그래프(figure)"가 아니라 "인쇄(printed)"로 둔다. 둘의 차이:
        #   figure  → 지우지 않고 **굵게 + 새까맣게** 강조한다
        #   printed → 지우지만 않는다
        # 이 규칙은 실제 인쇄 분수 가로줄(진하기비 0.91~1.03)과 학생이 연필로 그은
        # 좌표축 조각(0.83~1.05)을 구별하지 못한다. 재 봐도 겹친다.
        # 그래서 "보호"까지만 하고 "강조"는 하지 않는다. 잘못 보호해도 학생 선이
        # 새까맣게 도드라지지는 않는다(실측: 손그림이 도형으로 강조되는 비율 7.7%).
        # 진짜 인쇄 그래프는 5단계에서 Hough 직선 덮임(인쇄 0.91~0.95 / 손그림 0.00)으로
        # 다시 확인해 figure 로 올린다.
        is_protected_thin_line = is_long_thin_printed_line(
            measurement, local_reference, minimum_line_length)
        if is_protected_thin_line:
            label = LABEL_PRINTED
        elif handwriting_score >= remove_threshold:
            label = LABEL_HANDWRITING
        elif handwriting_score <= keep_threshold:
            label = LABEL_PRINTED
        else:
            label = LABEL_UNSURE

        components.append(StrokeComponent(
            component_index=measurement["component_index"],
            bounding_box=measurement["bounding_box"],
            area_px=measurement["area_px"],
            mean_darkness=measurement["mean_darkness"],
            dark_core_darkness=measurement["dark_core_darkness"],
            darkness_std=measurement["darkness_std"],
            thickness_mean_px=measurement["thickness_mean_px"],
            thickness_cv=measurement["thickness_cv"],
            fill_ratio=measurement["fill_ratio"],
            local_darkness_reference=local_reference,
            density_score=density_score,
            thickness_score=thickness_score,
            fill_score=fill_score,
            alignment_score=alignment_score,
            handwriting_score=float(handwriting_score),
            label=label,
            is_protected_thin_line=is_protected_thin_line,
        ))

    return labeled_image, components


def compute_alignment_score(measurement: dict[str, Any], text_columns: list[tuple[int, int]],
                            text_lines_by_column: list[list[tuple[int, int]]],
                            typical_line_height_by_column: list[float]) -> float:
    """요소가 "자기 단"의 인쇄 텍스트 줄에 얼마나 잘 맞는지 0(딱 맞음)~1(전혀 안 맞음)로 점수를 낸다."""
    center_x = measurement["center_x"]
    center_y = measurement["center_y"]

    # 이 요소가 어느 단에 속하는지 찾는다(어디에도 안 들어가면 가장 가까운 단)
    column_position = None
    for position, (column_left, column_right) in enumerate(text_columns):
        if column_left <= center_x <= column_right:
            column_position = position
            break
    if column_position is None:
        column_position = int(np.argmin([
            min(abs(center_x - left), abs(center_x - right)) for left, right in text_columns
        ]))

    text_lines = text_lines_by_column[column_position]
    typical_line_height = typical_line_height_by_column[column_position]
    if not text_lines or typical_line_height <= 0:
        return 0.5   # 줄을 못 찾았으면 판단 근거가 없으니 중립

    _, _, _, height = measurement["bounding_box"]
    inside_line = any(line_top <= center_y <= line_bottom for line_top, line_bottom in text_lines)
    if not inside_line:
        return 1.0   # 줄과 줄 사이 빈 공간에 쓴 것 = 필기일 가능성이 높다

    # 줄 안에 있어도 글자 높이가 줄 높이와 크게 다르면 필기 쪽으로 기운다
    height_ratio = height / typical_line_height
    if 0.35 <= height_ratio <= 1.30:
        return 0.0
    if height_ratio > 2.0 or height_ratio < 0.20:
        return 1.0
    return 0.5


def render_classification_debug_image(gray_image: np.ndarray, labeled_image: np.ndarray,
                                      components: list[StrokeComponent]) -> np.ndarray:
    """--debug 용: 빨강=필기, 초록=인쇄, 노랑=애매, 파랑=그래프로 칠한 이미지를 만든다."""
    debug_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    label_colors = {
        LABEL_HANDWRITING: (0, 0, 255),    # 빨강 (BGR)
        LABEL_PRINTED: (0, 170, 0),        # 초록
        LABEL_UNSURE: (0, 210, 235),       # 노랑
        LABEL_FIGURE: (255, 120, 0),       # 파랑
    }
    for component in components:
        left, top, width, height = component.bounding_box
        box_slice = (slice(top, top + height), slice(left, left + width))
        component_pixels = labeled_image[box_slice] == component.component_index
        debug_image[box_slice][component_pixels] = label_colors[component.label]
    return debug_image
