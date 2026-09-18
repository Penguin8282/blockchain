"""테스트용 합성 시험지 만들기.

실제 시험지 사진이 없어도 파이프라인 전체를 자동으로 검증할 수 있게,
"깨끗한 인쇄 시험지"와 "그 위에 학생이 쓴 필기"를 따로 만들어 준다.
정답(어느 픽셀이 필기인지)을 알고 만들기 때문에 정확도를 수치로 잴 수 있다.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 한글이 나오는 글꼴을 찾는다. 없으면 기본 글꼴로 영문만 그린다.
CANDIDATE_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/etc/alternatives/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

PRINTED_LINES = [
    "0071  오른쪽 그림과 같이 세 도로가 만",
    "나는 지점을 각각 A, B, C라 할 때,",
    "∠BAC = 90° 이고 AB = AC = 2√2 km,",
    "BC = 4 km 이다. 민아는 A 지점을 출발",
    "하여 C 지점 방향으로 시속 √2 km 의",
    "속력으로 걷고, 승우는 C 지점을 출발",
    "하여 B 지점 방향으로 걷는다.",
    "두 사람 사이의 거리의 최솟값은?",
]
CHOICE_LINE = "①  3√10/10 km    ②  2√10/5 km    ③  √10/2 km"


def find_usable_font(font_size: int) -> ImageFont.FreeTypeFont:
    """쓸 수 있는 글꼴을 찾아 돌려준다."""
    for font_path in CANDIDATE_FONT_PATHS:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_clean_exam_page(width: int = 1000, height: int = 1400, seed: int = 0) -> np.ndarray:
    """필기가 하나도 없는 "깨끗한 인쇄 시험지"를 만든다(그레이스케일).

    인쇄물답게 진하고(잉크 값 30 안팎) 굵기가 일정한 글자와, 좌표축·삼각형 도형을 그린다.
    """
    random_generator = random.Random(seed)
    page_image = Image.new("L", (width, height), color=255)
    drawing = ImageDraw.Draw(page_image)

    body_font = find_usable_font(26)
    small_font = find_usable_font(20)

    # 본문 (왼쪽 단)
    text_top = 260
    for line_index, line_text in enumerate(PRINTED_LINES):
        drawing.text((70, text_top + line_index * 52), line_text, font=body_font, fill=30)
    drawing.text((90, text_top + len(PRINTED_LINES) * 52 + 30), CHOICE_LINE, font=body_font, fill=30)

    # 오른쪽 단 (2단 편집을 흉내 낸다 — 단 분리 로직을 시험하기 위해)
    drawing.line([(width - 230, 40), (width - 230, height - 40)], fill=60, width=2)
    for line_index, line_text in enumerate(["수를 n, k의", "(단, O는 원", "작다.)", "① 7+√3", "④ 9+2√3"]):
        drawing.text((width - 210, 60 + line_index * 56), line_text, font=body_font, fill=30)

    # 도형: 좌표축 + 삼각형 (그래프 보호를 시험하기 위해)
    axis_origin_x, axis_origin_y = 600, 200
    drawing.line([(axis_origin_x - 20, axis_origin_y + 160), (axis_origin_x + 260, axis_origin_y + 160)], fill=20, width=3)
    drawing.line([(axis_origin_x + 20, axis_origin_y - 20), (axis_origin_x + 20, axis_origin_y + 190)], fill=20, width=3)
    triangle_points = [(axis_origin_x + 120, axis_origin_y), (axis_origin_x + 40, axis_origin_y + 150),
                       (axis_origin_x + 220, axis_origin_y + 150)]
    drawing.line(triangle_points + [triangle_points[0]], fill=20, width=3, joint="curve")
    drawing.text((axis_origin_x + 110, axis_origin_y - 34), "A", font=small_font, fill=30)

    # 쪽 번호
    drawing.text((80, height - 60), "18   I. 도형의 방정식", font=small_font, fill=40)

    page_array = np.array(page_image, dtype=np.uint8)
    # 인쇄물다운 아주 약한 질감
    noise = random_generator.gauss
    texture = np.array([[noise(0, 2) for _ in range(0)]])  # (자리만 표시) 실제 잡음은 아래에서
    page_array = cv2.GaussianBlur(page_array, (3, 3), 0)
    return page_array


def add_student_handwriting(clean_page: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """깨끗한 시험지 위에 학생 필기를 그린다.

    돌려주는 값: (필기가 그려진 컬러 이미지 BGR, 필기 픽셀 마스크)
    필기는 인쇄보다 흐리게(잉크 값 110~170), 굵기가 들쭉날쭉하게 그려서
    실제 연필 필기의 성질을 흉내 낸다.
    """
    random_generator = random.Random(seed)
    page_height, page_width = clean_page.shape[:2]
    color_page = cv2.cvtColor(clean_page, cv2.COLOR_GRAY2BGR)
    handwriting_mask = np.zeros((page_height, page_width), dtype=np.uint8)

    # 1) 연필 낙서 곡선 (베지어 흉내: 임의의 점들을 잇는 부드러운 선)
    for _ in range(18):
        start_x = random_generator.randint(40, page_width - 120)
        start_y = random_generator.randint(40, page_height - 60)
        points = [(start_x, start_y)]
        for _ in range(random_generator.randint(3, 7)):
            points.append((points[-1][0] + random_generator.randint(-70, 90),
                           points[-1][1] + random_generator.randint(-40, 40)))
        pencil_gray = random_generator.randint(110, 170)
        for point_index in range(len(points) - 1):
            thickness = random_generator.randint(1, 3)   # 두께가 일정하지 않다
            cv2.line(color_page, points[point_index], points[point_index + 1],
                     (pencil_gray,) * 3, thickness, cv2.LINE_AA)
            cv2.line(handwriting_mask, points[point_index], points[point_index + 1], 1, thickness + 1)

    # 2) 손으로 쓴 숫자·글자 흉내 (짧은 획 여러 개)
    for _ in range(40):
        center_x = random_generator.randint(40, page_width - 60)
        center_y = random_generator.randint(40, page_height - 40)
        pencil_gray = random_generator.randint(110, 165)
        for _ in range(random_generator.randint(2, 4)):
            offset_x = random_generator.randint(-14, 14)
            offset_y = random_generator.randint(-16, 16)
            cv2.line(color_page, (center_x, center_y), (center_x + offset_x, center_y + offset_y),
                     (pencil_gray,) * 3, random_generator.randint(1, 2), cv2.LINE_AA)
            cv2.line(handwriting_mask, (center_x, center_y), (center_x + offset_x, center_y + offset_y),
                     1, 2)

    # 3) 빨간펜 채점 표시 (동그라미 · 별 · 체크)
    for _ in range(4):
        center_x = random_generator.randint(120, page_width - 120)
        center_y = random_generator.randint(120, page_height - 120)
        radius = random_generator.randint(50, 110)
        cv2.circle(color_page, (center_x, center_y), radius, (40, 40, 220), 5, cv2.LINE_AA)
        cv2.circle(handwriting_mask, (center_x, center_y), radius, 1, 7)
    for _ in range(3):
        center_x = random_generator.randint(100, page_width - 100)
        center_y = random_generator.randint(100, page_height - 100)
        size = random_generator.randint(30, 60)
        for angle_index in range(5):   # 오각별
            angle_a = math.radians(90 + angle_index * 144)
            angle_b = math.radians(90 + (angle_index + 1) * 144)
            point_a = (int(center_x + size * math.cos(angle_a)), int(center_y - size * math.sin(angle_a)))
            point_b = (int(center_x + size * math.cos(angle_b)), int(center_y - size * math.sin(angle_b)))
            cv2.line(color_page, point_a, point_b, (40, 40, 220), 4, cv2.LINE_AA)
            cv2.line(handwriting_mask, point_a, point_b, 1, 6)

    return color_page, handwriting_mask


def simulate_camera(color_page: np.ndarray, seed: int = 0,
                    rotation_degrees: float | None = None) -> tuple[np.ndarray, float]:
    """스마트폰으로 찍은 것처럼 회전·원근·그림자·블러·노이즈·JPEG 를 입힌다.

    돌려주는 값: (사진처럼 만든 이미지, 실제로 넣은 회전 각도)
    """
    random_generator = random.Random(seed)
    page_height, page_width = color_page.shape[:2]

    # 어두운 책상 위에 시험지를 올린 것처럼 여백을 준다
    margin = int(min(page_height, page_width) * 0.10)
    desk_color = random_generator.randint(70, 120)
    photo = cv2.copyMakeBorder(color_page, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=(desk_color,) * 3)

    # 회전
    if rotation_degrees is None:
        rotation_degrees = random_generator.uniform(-15.0, 15.0)
    photo_center = (photo.shape[1] / 2.0, photo.shape[0] / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(photo_center, rotation_degrees, 1.0)
    photo = cv2.warpAffine(photo, rotation_matrix, (photo.shape[1], photo.shape[0]),
                           borderValue=(desk_color,) * 3)

    # 원근 왜곡 (비스듬히 찍기)
    warp_strength = random_generator.uniform(0.0, 0.03)
    height_now, width_now = photo.shape[:2]
    source_points = np.float32([[0, 0], [width_now, 0], [width_now, height_now], [0, height_now]])
    destination_points = source_points + np.float32([
        [width_now * warp_strength, height_now * warp_strength],
        [-width_now * warp_strength, height_now * warp_strength * 0.5],
        [-width_now * warp_strength * 0.5, -height_now * warp_strength],
        [width_now * warp_strength * 0.5, -height_now * warp_strength * 0.5],
    ])
    perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    photo = cv2.warpPerspective(photo, perspective_matrix, (width_now, height_now),
                                borderValue=(desk_color,) * 3)

    # 그림자 (한쪽이 어두워지는 밝기 기울기)
    gradient_x = np.linspace(random_generator.uniform(0.55, 0.9), 1.0, photo.shape[1], dtype=np.float32)
    gradient_y = np.linspace(random_generator.uniform(0.65, 1.0), 1.0, photo.shape[0], dtype=np.float32)
    shadow_map = np.outer(gradient_y, gradient_x)[:, :, None]
    photo = np.clip(photo.astype(np.float32) * shadow_map, 0, 255).astype(np.uint8)

    # 블러 + 노이즈
    blur_size = random_generator.choice([1, 3, 3, 5])
    if blur_size > 1:
        photo = cv2.GaussianBlur(photo, (blur_size, blur_size), 0)
    noise = np.random.default_rng(seed).normal(0, 3.0, photo.shape)
    photo = np.clip(photo.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG 압축 흔적
    encode_succeeded, encoded = cv2.imencode(".jpg", photo, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if encode_succeeded:
        photo = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return photo, rotation_degrees


def make_photo_pair(seed: int = 0, rotation_degrees: float | None = None):
    """테스트에서 가장 많이 쓰는 조합을 한 번에 만들어 준다.

    돌려주는 값: (사진, 필기 마스크, 깨끗한 원본, 넣은 회전 각도)
    """
    clean_page = make_clean_exam_page(seed=seed)
    written_page, handwriting_mask = add_student_handwriting(clean_page, seed=seed)
    photo, applied_rotation = simulate_camera(written_page, seed=seed, rotation_degrees=rotation_degrees)
    return photo, handwriting_mask, clean_page, applied_rotation
