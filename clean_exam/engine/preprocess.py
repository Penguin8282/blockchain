"""1) 사진 보정 — 스마트폰으로 대충 찍은 사진을 반듯하고 깨끗하게 만든다.

강사는 책상 위에서 비스듬히, 그림자 지게, 흔들리게 찍는다. 그래서 다음을 차례로 한다.
  a) 방향 판단        : 90/180/270도 돌아간 사진을 세로로 되돌린다
  b) 시험지 영역 찾기 : 세 가지 방법을 차례로 시도(윤곽 → 기울기 보정 → 원본 그대로)
  c) 원근 보정        : 네 모서리가 잡히면 정면 직사각형으로 편다
  d) 휘어진 종이 펴기 : 제본된 문제집처럼 줄이 곡선이면 가로 띠별로 회전(선택)
  e) 조명·그림자 제거 : 배경을 추정해 나눗셈 정규화 → 흰 배경 + 검은 잉크
  f) 품질 검사        : 흐림·어두움·너무 작게 찍힘을 수치로 재서 한국어 경고를 남긴다

각 단계는 실패해도 예외를 던지지 않고 "이 단계는 못 했다"고 기록한 뒤 다음으로 넘어간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# 기울기를 잴 때 쓰는 축소본의 최대 한 변(px). 각도는 배율과 무관하므로 정확도 손해가 없다.
ANGLE_MEASUREMENT_MAX_SIDE_PX = 1200
# 기울기를 잴 때 이 채도를 넘는 픽셀(색펜)은 빼고 잰다. 인쇄 잉크는 무채색이기 때문이다.
COLOR_PEN_SATURATION_FOR_ANGLE = 60
# 기울기를 잴 때 쓸 '진한 잉크'의 기준 분위수. 낮출수록 인쇄만 남지만 표본이 적어진다.
DARK_INK_PERCENTILE_FOR_ANGLE = 35
# 조명(배경) 추정에서 제외할 색 영역의 채도 기준. 컬러 인쇄를 종이로 오인하지 않기 위해서다.
COLOR_AREA_SATURATION_FOR_BACKGROUND = 60


@dataclass
class PreprocessResult:
    """사진 보정 결과를 한 덩어리로 담는 상자."""

    corrected_color_image: np.ndarray          # 보정된 컬러 이미지 (BGR)
    normalized_gray_image: np.ndarray          # 조명까지 고른 그레이스케일 (흰 배경 + 검은 잉크)
    rectify_method: str = "none"               # 어떤 방법으로 폈는지: contour / deskew / none
    rotation_applied_degrees: float = 0.0      # 실제로 돌린 각도
    paper_area_ratio: float = 1.0              # 시험지가 화면에서 차지하는 비율
    quality_warnings: list[str] = field(default_factory=list)   # 강사에게 보여줄 한국어 경고
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)  # --debug 용 중간 이미지


# ---------------------------------------------------------------------------
# a) 방향 판단
# ---------------------------------------------------------------------------
def detect_and_fix_orientation(color_image: np.ndarray) -> tuple[np.ndarray, float]:
    """사진이 90/180/270도 돌아가 있으면 글자 줄 방향을 보고 세로로 되돌린다.

    원리: 인쇄된 글자는 가로 줄로 늘어서 있다. 그래서 잉크 픽셀을 가로로 합치면
    (줄이 있는 행은 값이 크고, 줄 사이 빈 행은 0) 들쭉날쭉한 신호가 나온다.
    반대로 세로로 합치면 비교적 평평하다. 이 '들쭉날쭉함(분산)'이 큰 쪽이 글자 줄 방향이다.

    돌려주는 값: (바로 세운 이미지, 돌린 각도)
    ※ 180도 뒤집힘은 글자 모양을 봐야 알 수 있어 여기서는 다루지 않는다(0도로 둔다).
    """
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray_image, (400, 400), interpolation=cv2.INTER_AREA)
    ink_mask = (small_gray < np.percentile(small_gray, 25)).astype(np.float32)

    horizontal_projection = ink_mask.sum(axis=1)   # 행마다 잉크 픽셀 수
    vertical_projection = ink_mask.sum(axis=0)     # 열마다 잉크 픽셀 수

    # 정규화한 분산으로 비교한다(합계 크기의 영향을 없애기 위해)
    def normalized_variance(projection: np.ndarray) -> float:
        if projection.mean() <= 1e-6:
            return 0.0
        return float(projection.var() / (projection.mean() ** 2))

    image_height, image_width = color_image.shape[:2]
    is_landscape_photo = image_width > image_height

    projection_says_rotated = (
        normalized_variance(vertical_projection) > normalized_variance(horizontal_projection) * 1.3
    )

    # 투영 근거와 "가로로 누운 사진" 근거가 **둘 다** 맞을 때만 90도를 돌린다.
    # 투영 근거만 믿었더니, 필기가 사방으로 흩어진 세로 사진 50장 중 5장을 잘못 눕혔다.
    # 세로로 찍힌 사진은 이미 바로 서 있을 가능성이 압도적으로 높으므로 이 조건이 안전하다.
    if projection_says_rotated and is_landscape_photo:
        return cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE), 90.0
    return color_image, 0.0


# ---------------------------------------------------------------------------
# b) 시험지 영역 찾기 + c) 원근 보정
# ---------------------------------------------------------------------------
def find_paper_quadrilateral(color_image: np.ndarray, preprocess_config: dict[str, Any]) -> np.ndarray | None:
    """밝은 종이와 어두운 배경의 경계에서 가장 큰 사각형(네 모서리)을 찾는다.

    못 찾으면 None 을 돌려준다. 다음 경우에 흔히 실패한다:
      · 종이가 화면을 꽉 채워 경계가 안 보일 때
      · 배경(책상)도 밝아서 대비가 없을 때
      · 제본된 문제집의 한 쪽만 찍어 종이 테두리가 잘렸을 때
    """
    paper_config = preprocess_config["paper_detect"]
    if not paper_config["enabled"]:
        return None

    image_height, image_width = color_image.shape[:2]
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edge_image = cv2.Canny(blurred_image, paper_config["canny_low"], paper_config["canny_high"])
    # 끊긴 테두리를 이어 준다
    edge_image = cv2.dilate(edge_image, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    total_image_area = float(image_height * image_width)
    minimum_area = total_image_area * paper_config["min_area_ratio"]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        contour_area = cv2.contourArea(contour)
        if contour_area < minimum_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approximated = cv2.approxPolyDP(contour, paper_config["approx_epsilon_ratio"] * perimeter, True)
        if len(approximated) == 4 and cv2.isContourConvex(approximated):
            # 화면 전체와 거의 같으면(= 그냥 사진 테두리를 잡은 것) 쓸모가 없다
            if contour_area > total_image_area * 0.98:
                continue
            return approximated.reshape(4, 2).astype(np.float32)
    return None


def order_quadrilateral_corners(corner_points: np.ndarray) -> np.ndarray:
    """네 점을 좌상-우상-우하-좌하 순서로 정렬한다.

    원리: x+y 가 가장 작은 점이 좌상, 가장 큰 점이 우하.
          x-y 가 가장 작은 점이 우상, 가장 큰 점이 좌하.
    """
    ordered_points = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = corner_points.sum(axis=1)
    coordinate_difference = np.diff(corner_points, axis=1).ravel()
    ordered_points[0] = corner_points[np.argmin(coordinate_sum)]     # 좌상
    ordered_points[2] = corner_points[np.argmax(coordinate_sum)]     # 우하
    ordered_points[1] = corner_points[np.argmin(coordinate_difference)]  # 우상
    ordered_points[3] = corner_points[np.argmax(coordinate_difference)]  # 좌하
    return ordered_points


def warp_to_rectangle(color_image: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    """네 모서리를 정면 직사각형으로 펴고(호모그래피), 종이 비율(A4/B4/B5)에 스냅한다."""
    ordered_points = order_quadrilateral_corners(corner_points)
    top_left, top_right, bottom_right, bottom_left = ordered_points

    measured_width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    measured_height = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    if measured_width < 10 or measured_height < 10:
        return color_image

    # 측정된 비율에 가장 가까운 표준 종이 비율(세로/가로)로 스냅한다
    measured_ratio = measured_height / measured_width
    standard_paper_ratios = {"A4": 297 / 210, "B4": 364 / 257, "B5": 257 / 182}
    closest_ratio = min(standard_paper_ratios.values(), key=lambda ratio: abs(ratio - measured_ratio))
    if abs(closest_ratio - measured_ratio) < 0.15:   # 너무 동떨어지면 스냅하지 않는다
        measured_ratio = closest_ratio

    output_width = int(round(measured_width))
    output_height = int(round(measured_width * measured_ratio))
    destination_points = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32,
    )
    homography_matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)
    return cv2.warpPerspective(color_image, homography_matrix, (output_width, output_height),
                               flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """가중 중앙값 — 가중치 합의 절반을 넘기는 지점의 값.

    평균과 달리 몇 개의 엉뚱한 값(이상치)에 흔들리지 않는다.
    """
    if values.size == 0:
        return 0.0
    sorted_order = np.argsort(values)
    sorted_values = values[sorted_order]
    cumulative_weights = np.cumsum(weights[sorted_order])
    if cumulative_weights[-1] <= 0:
        return float(np.median(values))
    half_weight = cumulative_weights[-1] / 2.0
    return float(sorted_values[int(np.searchsorted(cumulative_weights, half_weight))])


def build_paper_region_mask(gray_image: np.ndarray) -> np.ndarray:
    """밝은 종이 영역만 1로 표시한 마스크를 만든다(어두운 책상 배경을 제외하기 위해).

    Otsu 로 밝기 경계를 자동으로 찾은 뒤, 가장 큰 밝은 덩어리만 남기고 살짝 깎아
    테두리 근처를 버린다. 종이가 화면을 꽉 채운 사진에서는 전체가 종이로 나온다.
    """
    _, bright_mask = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_mask = bright_mask.astype(np.uint8)

    # 구멍(글자)을 메워 종이를 하나의 덩어리로 만든다
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, fill_kernel)

    component_count, labeled, statistics, _ = cv2.connectedComponentsWithStats(bright_mask, 8)
    if component_count <= 1:
        return np.ones_like(bright_mask)
    largest_index = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    paper_mask = (labeled == largest_index).astype(np.uint8)

    # 밝은 영역이 화면 대부분이면(종이가 꽉 찬 사진) 그냥 전체를 쓴다
    if paper_mask.mean() > 0.92:
        return np.ones_like(paper_mask)

    # 테두리 근처는 버린다(종이 가장자리 그림자·접힘이 직선으로 잡히는 것을 막는다)
    shrink_pixels = max(3, int(min(gray_image.shape[:2]) * 0.01))
    shrink_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * shrink_pixels + 1, 2 * shrink_pixels + 1))
    return cv2.erode(paper_mask, shrink_kernel)


def estimate_text_line_angle(color_image: np.ndarray, max_angle_degrees: float) -> float:
    """인쇄 텍스트 줄들의 기울기를 Hough 변환으로 구한다(단위: 도).

    종이 윤곽을 못 찾았을 때 쓰는 두 번째 방법이다. 글자를 가로로 뭉개서 '줄 덩어리'로
    만든 뒤, 그 덩어리들이 이루는 직선의 각도를 재서 중앙값을 쓴다.
    """
    # 각도는 크기를 줄여도 그대로이므로, 속도를 위해 작은 사본에서 잰다.
    # (원본 그대로 재면 큰 사진에서 이 함수 하나가 0.5초 넘게 걸린다.)
    measurement_image = resize_to_max_side(color_image, ANGLE_MEASUREMENT_MAX_SIDE_PX)
    gray_image = cv2.cvtColor(measurement_image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 31, 10)

    # **종이 밖(책상)을 뺀다.** 책상은 종이보다 어두워서 적응형 이진화가 종이 테두리를
    # 긴 직선으로 잡아낸다. 그 테두리는 사진이 기운 각도 그대로 누워 있어서, 보정을 하면
    # 테두리 각도도 같이 따라 돌고 다음 회차에서 다시 측정된다 → 부호가 뒤집히며 진동한다
    # (실측: -13.4 → +13.0 → -11.5 → +11.4 로 영원히 수렴하지 않고 캔버스만 1600→2874px 로 부풀었다).
    paper_mask = build_paper_region_mask(gray_image)
    binary_image[paper_mask == 0] = 0

    # **색펜을 뺀다.** 인쇄 잉크는 무채색이라는 것이 이 프로젝트의 기본 가정이다.
    # 빨간 동그라미·별 같은 채점 표시는 크고 진해서 각도 측정을 크게 끌어당긴다
    # (실측: 빨간펜이 많은 합성 시험지에서 곧은 페이지가 1도 가까이 잘못 돌아갔다).
    saturation_channel = cv2.cvtColor(measurement_image, cv2.COLOR_BGR2HSV)[:, :, 1]
    binary_image[saturation_channel > COLOR_PEN_SATURATION_FOR_ANGLE] = 0

    # **진한 잉크만 남긴다.** 기울기는 인쇄 텍스트 줄에서 재야 한다.
    # 연필 필기는 사방으로 제각각 기울어 있어서 같이 재면 측정이 끌려간다
    # (실측: 곧게 놓인 합성 시험지에서 필기 때문에 0.9도가 잘못 측정됐다).
    # 인쇄는 연필보다 확실히 진하므로, 잉크 픽셀 중 가장 어두운 35% 만 쓴다.
    # (실측으로 50/40/35/25 분위를 비교해 35 를 골랐다: 곧은 페이지 12장의 측정 오차가
    #  50분위 평균 0.20도·최대 0.94도 → 35분위 평균 0.08도·최대 0.51도로 줄었다.)
    ink_gray_values = gray_image[binary_image > 0]
    if ink_gray_values.size > 100:
        dark_ink_cutoff = float(np.percentile(ink_gray_values, DARK_INK_PERCENTILE_FOR_ANGLE))
        binary_image = ((binary_image > 0) & (gray_image <= dark_ink_cutoff)).astype(np.uint8) * 255

    # 가로로 긴 커널로 팽창시켜 글자들을 한 줄로 이어 붙인다
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    line_blobs = cv2.dilate(binary_image, line_kernel, iterations=2)

    detected_lines = cv2.HoughLinesP(line_blobs, 1, np.pi / 360, threshold=120,
                                     minLineLength=max(60, gray_image.shape[1] // 8), maxLineGap=20)
    if detected_lines is None or len(detected_lines) == 0:
        return 0.0

    measured_angles: list[float] = []
    line_lengths: list[float] = []
    for line in reshape_hough_lines(detected_lines):
        start_x, start_y, end_x, end_y = line
        angle_degrees = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))
        if abs(angle_degrees) <= max_angle_degrees:   # 세로선·대각선은 글자 줄이 아니다
            measured_angles.append(float(angle_degrees))
            line_lengths.append(float(np.hypot(end_x - start_x, end_y - start_y)))
    if not measured_angles:
        return 0.0

    # **길이로 가중한 중앙값**을 쓴다.
    # 인쇄 텍스트 줄은 가로로 이어 붙이면 길고, 필기 조각이나 잡음에서 나온 선은 짧다.
    # 길이로 가중하면 믿을 만한 긴 선이 결과를 결정한다.
    return weighted_median(np.array(measured_angles), np.array(line_lengths))


def reshape_hough_lines(detected_lines: np.ndarray) -> np.ndarray:
    """HoughLinesP 결과를 항상 (선 개수, 4) 모양으로 맞춰 준다.

    OpenCV 버전에 따라 (N, 1, 4) 로 주기도 하고 (N, 4) 로 주기도 해서,
    호출하는 쪽이 버전 걱정을 안 하도록 여기서 한 번에 정리한다.
    """
    lines_array = np.asarray(detected_lines)
    return lines_array.reshape(-1, 4)


def rotate_image_by_angle(color_image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """이미지를 주어진 각도만큼 돌린다(잘리지 않게 캔버스를 넓히고 여백은 흰색)."""
    image_height, image_width = color_image.shape[:2]
    image_center = (image_width / 2.0, image_height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle_degrees, 1.0)

    cosine_value = abs(rotation_matrix[0, 0])
    sine_value = abs(rotation_matrix[0, 1])
    new_width = int(image_height * sine_value + image_width * cosine_value)
    new_height = int(image_height * cosine_value + image_width * sine_value)
    rotation_matrix[0, 2] += new_width / 2.0 - image_center[0]
    rotation_matrix[1, 2] += new_height / 2.0 - image_center[1]

    return cv2.warpAffine(color_image, rotation_matrix, (new_width, new_height),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def refine_deskew(color_image: np.ndarray, max_angle_degrees: float,
                  maximum_passes: int = 3, tolerance_degrees: float = 0.8) -> tuple[np.ndarray, float]:
    """텍스트 줄이 수평이 될 때까지 기울기 보정을 반복하되, **원본에서 한 번만** 돌린다.

    왜 반복하나: 크게 기울어진 사진에서는 기울기 측정 자체가 조금 빗나간다.
    실측으로 12도 기울인 사진의 측정값이 10.84도였고, 한 번만 보정하면 1.3도가 남았다.

    왜 '원본에서 한 번만' 인가: 회전할 때마다 캔버스가 커지고 화질이 뭉개진다.
    누적 각도를 계산해 마지막에 한 번만 돌리면 캔버스도 한 번만 커지고 화질 손해도 한 번뿐이다.
    (그전에는 회차마다 돌려서 캔버스가 1600 → 2874px 까지 부푼 적이 있다.)

    왜 진동을 감시하나: 측정이 수렴하지 않고 부호를 뒤집으며 튀는 사진이 있다.
    그럴 때는 더 돌리지 말고 **지금까지 중 가장 반듯했던 각도**를 쓰는 것이 안전하다.

    돌려주는 값: (펴진 이미지, 총 회전량)
    """
    accumulated_angle = 0.0
    best_angle = 0.0
    smallest_measured_angle = abs(estimate_text_line_angle(color_image, max_angle_degrees))
    if smallest_measured_angle <= tolerance_degrees:
        return color_image, 0.0

    for _ in range(maximum_passes):
        candidate_image = rotate_image_by_angle(color_image, accumulated_angle) \
            if accumulated_angle != 0.0 else color_image
        measured_angle = estimate_text_line_angle(candidate_image, max_angle_degrees)

        if abs(measured_angle) < smallest_measured_angle:
            smallest_measured_angle = abs(measured_angle)
            best_angle = accumulated_angle
        elif accumulated_angle != 0.0:
            # 더 나아지지 않는다 = 수렴하지 않고 튀는 중이다. 여기서 멈춘다.
            break

        if abs(measured_angle) <= tolerance_degrees:
            best_angle = accumulated_angle
            break
        accumulated_angle += measured_angle

    if abs(best_angle) < 1e-6:
        return color_image, 0.0
    return rotate_image_by_angle(color_image, best_angle), best_angle


# ---------------------------------------------------------------------------
# d) 휘어진 종이 펴기
# ---------------------------------------------------------------------------
def dewarp_by_horizontal_bands(color_image: np.ndarray, band_count: int, max_angle_degrees: float) -> np.ndarray:
    """제본된 문제집처럼 휘어 텍스트 줄이 곡선일 때, 가로 띠마다 따로 회전해 편다.

    완벽할 필요는 없고, 줄이 대체로 수평이 되면 된다.
    """
    image_height, image_width = color_image.shape[:2]
    band_height = max(1, image_height // band_count)
    corrected_bands: list[np.ndarray] = []

    for band_index in range(band_count):
        top = band_index * band_height
        bottom = image_height if band_index == band_count - 1 else (band_index + 1) * band_height
        band_image = color_image[top:bottom]
        if band_image.shape[0] < 20:
            corrected_bands.append(band_image)
            continue
        band_angle = estimate_text_line_angle(band_image, max_angle_degrees)
        if abs(band_angle) < 0.3:
            corrected_bands.append(band_image)
            continue
        # 띠 높이가 바뀌지 않도록 캔버스를 넓히지 않고 제자리 회전한다
        band_center = (band_image.shape[1] / 2.0, band_image.shape[0] / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(band_center, band_angle, 1.0)
        rotated_band = cv2.warpAffine(band_image, rotation_matrix,
                                      (band_image.shape[1], band_image.shape[0]),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        corrected_bands.append(rotated_band)

    return np.vstack(corrected_bands)


# ---------------------------------------------------------------------------
# e) 조명 불균일·그림자 제거
# ---------------------------------------------------------------------------
def normalize_illumination(color_image: np.ndarray, illumination_config: dict[str, Any]) -> np.ndarray:
    """손 그림자, 창가 밝기 차이를 없애고 "흰 배경 + 검은 잉크" 그레이스케일을 만든다.

    원리: 글자보다 훨씬 큰 커널로 모폴로지 닫기(closing)를 하면 글자가 뭉개져 사라지고
    '종이 밝기 지도'(배경)만 남는다. 원본을 이 배경으로 나누면 조명 차이가 사라진다.
    (나눗셈이라 어두운 쪽이 밝게 끌어올려지고, 잉크는 배경보다 훨씬 어두우므로 그대로 남는다.)
    """
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # **색이 칠해진 넓은 영역은 배경 추정에서 뺀다.**
    # 단원 제목 띠처럼 크고 진한 색 면은 모폴로지 커널만큼 커서 "이 동네 종이는 원래
    # 어둡구나"로 추정되고, 그러면 그 주변 글자까지 나눗셈에서 하얗게 날아간다
    # (실측: 제목 띠가 있는 페이지에서 인쇄 글자 손실이 2.2% → 7.2% 로 뛰었다).
    # 색 부분을 주변 종이 밝기로 임시로 메운 뒤 배경을 추정한다. 원본은 그대로 둔다.
    saturation_channel = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)[:, :, 1]
    colored_area_mask = (saturation_channel > COLOR_AREA_SATURATION_FOR_BACKGROUND).astype(np.uint8)
    background_source_image = gray_image
    if colored_area_mask.mean() > 0.001:
        paper_brightness = float(np.percentile(gray_image, 80))
        background_source_image = gray_image.copy()
        background_source_image[colored_area_mask > 0] = paper_brightness

    kernel_size = int(illumination_config["background_kernel_px"])
    if kernel_size % 2 == 0:
        kernel_size += 1
    background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    estimated_background = cv2.morphologyEx(background_source_image, cv2.MORPH_CLOSE, background_kernel)
    estimated_background = cv2.GaussianBlur(estimated_background, (kernel_size, kernel_size), 0)
    estimated_background = np.maximum(estimated_background, 1)   # 0으로 나누기 방지

    normalized = gray_image.astype(np.float32) / estimated_background.astype(np.float32)
    # 밝은 쪽 분위수를 흰색(255)에 맞춘다
    white_reference = np.percentile(normalized, illumination_config["clip_percentile"])
    normalized = np.clip(normalized / max(white_reference, 1e-6) * 255.0, 0, 255)
    return normalized.astype(np.uint8)


# ---------------------------------------------------------------------------
# f) 품질 검사
# ---------------------------------------------------------------------------
def check_photo_quality(color_image: np.ndarray, paper_area_ratio: float,
                        quality_config: dict[str, Any]) -> list[str]:
    """흐림·어두움·너무 작게 찍힘을 수치로 재서 한국어 경고 문구 목록을 돌려준다."""
    warnings: list[str] = []
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 초점은 **가장자리를 잘라 낸 안쪽**에서만 잰다.
    # 원근·기울기 보정을 하면 테두리에 흰 여백과 종이 경계가 생기는데, 그 경계는
    # 아주 날카로워서 초점 점수를 떠받친다. 실측: 41px 로 뭉갠 사진의 점수가 118 로
    # 나와 흐림 기준(100)을 넘겨 버렸고, 경고가 뜨지 않았다.
    image_height, image_width = gray_image.shape[:2]
    margin_y = int(image_height * 0.10)
    margin_x = int(image_width * 0.10)
    interior_image = gray_image[margin_y:image_height - margin_y, margin_x:image_width - margin_x]
    if interior_image.size < 100:
        interior_image = gray_image

    focus_score = float(cv2.Laplacian(interior_image, cv2.CV_64F).var())
    if focus_score < quality_config["blur_laplacian_min"]:
        warnings.append(f"사진이 흐려요(초점 점수 {focus_score:.0f}). 다시 찍으면 더 좋아져요.")

    mean_brightness = float(gray_image.mean())
    if mean_brightness < quality_config["dark_mean_min"]:
        warnings.append(f"사진이 어두워요(평균 밝기 {mean_brightness:.0f}). 밝은 곳에서 다시 찍어 보세요.")

    if paper_area_ratio < quality_config["paper_area_ratio_min"]:
        warnings.append(f"시험지가 화면의 {paper_area_ratio*100:.0f}% 밖에 안 돼요. 더 가까이서 찍어 주세요.")

    return warnings


# ---------------------------------------------------------------------------
# 전체를 순서대로 실행
# ---------------------------------------------------------------------------
def preprocess_photo(original_color_image: np.ndarray, config: dict[str, Any]) -> PreprocessResult:
    """사진 보정 전체를 순서대로 실행하고 결과를 PreprocessResult 로 돌려준다."""
    preprocess_config = config["preprocess"]
    debug_images: dict[str, np.ndarray] = {}

    # 처리용으로 크기를 줄인다(속도 때문에). 원본 비율은 유지한다.
    working_image = resize_to_max_side(original_color_image, preprocess_config["work_max_side_px"])
    debug_images["00_입력"] = working_image.copy()

    # a) 방향
    rotation_applied = 0.0
    if preprocess_config["auto_orient"]:
        working_image, rotation_applied = detect_and_fix_orientation(working_image)

    # b)+c) 시험지 영역 → 원근 보정
    rectify_method = "none"
    paper_area_ratio = 1.0
    quadrilateral = find_paper_quadrilateral(working_image, preprocess_config)
    if quadrilateral is not None:
        image_area = float(working_image.shape[0] * working_image.shape[1])
        paper_area_ratio = float(cv2.contourArea(quadrilateral.astype(np.int32)) / image_area)
        working_image = warp_to_rectangle(working_image, quadrilateral)
        rectify_method = "contour"

    # 어느 경로로 왔든 마지막에 기울기를 반복 보정한다.
    # 윤곽으로 편 뒤에도 모서리 추정이 미세하게 어긋나 1~2도가 남는 일이 흔하다(실측).
    if preprocess_config["deskew"]["enabled"]:
        working_image, refine_rotation = refine_deskew(
            working_image,
            preprocess_config["deskew"]["max_angle_deg"],
            maximum_passes=preprocess_config["deskew"]["max_passes"],
            tolerance_degrees=preprocess_config["deskew"]["min_angle_deg"],
        )
        rotation_applied += refine_rotation
        if rectify_method == "none":
            rectify_method = "deskew"
    debug_images["01_보정후"] = working_image.copy()

    # d) 휘어진 종이
    if preprocess_config["dewarp"]["enabled"]:
        working_image = dewarp_by_horizontal_bands(
            working_image,
            preprocess_config["dewarp"]["band_count"],
            preprocess_config["deskew"]["max_angle_deg"],
        )
        debug_images["02_휨보정"] = working_image.copy()

    # e) 조명
    normalized_gray = normalize_illumination(working_image, preprocess_config["illumination"])
    debug_images["03_조명정규화"] = normalized_gray.copy()

    # f) 품질
    quality_warnings = check_photo_quality(working_image, paper_area_ratio, preprocess_config["quality"])

    return PreprocessResult(
        corrected_color_image=working_image,
        normalized_gray_image=normalized_gray,
        rectify_method=rectify_method,
        rotation_applied_degrees=rotation_applied,
        paper_area_ratio=paper_area_ratio,
        quality_warnings=quality_warnings,
        debug_images=debug_images,
    )


def resize_to_max_side(image: np.ndarray, max_side_pixels: int) -> np.ndarray:
    """긴 변이 max_side_pixels 를 넘지 않게 비율을 유지하며 줄인다(작으면 그대로)."""
    image_height, image_width = image.shape[:2]
    longest_side = max(image_height, image_width)
    if longest_side <= max_side_pixels:
        return image
    scale = max_side_pixels / longest_side
    new_size = (int(round(image_width * scale)), int(round(image_height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
