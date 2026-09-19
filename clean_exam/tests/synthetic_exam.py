"""테스트·학습용 합성 시험지 만들기.

왜 합성인가:
  · 실제 문제집을 인터넷에서 수집하는 것은 저작권 문제가 있다(한국에는 AI 학습용 TDM
    면책 조항이 아직 없고, 수능 기출조차 공공누리 제4유형이라 상업적 이용·변경이 금지된다).
  · 합성은 **어느 픽셀이 필기인지 정답을 알고** 만들기 때문에 정확도를 수치로 잴 수 있다.
    실제 사진은 사람이 일일이 칠해야 정답이 생긴다(3단계에서 할 일).
  · 무한히 만들 수 있고, 난이도를 조절할 수 있다.

실제 시험지 사진을 재 보고 맞춘 것들:
  · 인쇄 잉크 진하기 ~128 / 연필 ~77 (255 = 완전 검정 기준의 '진하기')
  · 제본 쪽으로 갈수록 인쇄가 흐려짐 (한 장 안에서 27% 차이)
  · 뒷장 비침이 배경에 희미하게 보임
  · 연필은 사각사각한 질감이라 획 안에서 농도가 들쭉날쭉함
  · 검은 볼펜 필기는 인쇄만큼 진해서 가장 어려운 경우다
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 글꼴 찾기
# ---------------------------------------------------------------------------
def find_korean_font_paths() -> list[str]:
    """쓸 수 있는 한글 글꼴 경로를 찾는다.

    나눔글꼴은 OFL 라이선스라 재배포·상업적 이용·변경이 자유롭다.
    pip 로 설치되는 koreanize-matplotlib 안에 들어 있어서 그걸 먼저 찾아본다.
    """
    font_paths: list[str] = []
    try:
        import koreanize_matplotlib

        bundled_directory = Path(koreanize_matplotlib.__file__).parent
        font_paths.extend(
            str(path) for path in sorted(bundled_directory.rglob("Nanum*.ttf"))
        )
    except ImportError:
        pass

    for fallback_path in (
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/etc/alternatives/fonts-japanese-gothic.ttf",
    ):
        if Path(fallback_path).exists():
            font_paths.append(fallback_path)
    return font_paths


def load_font(font_size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """본문용 글꼴을 연다(한글이 되는 것 우선)."""
    available_paths = find_korean_font_paths()
    if bold:
        bold_paths = [path for path in available_paths if "Bold" in path]
        if bold_paths:
            available_paths = bold_paths + available_paths
    for font_path in available_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 시험지 본문 재료
# ---------------------------------------------------------------------------
PROBLEM_BODIES = [
    ["오른쪽 그림과 같이 세 도로가 만", "나는 지점을 각각 A, B, C라 할 때,",
     "∠BAC=90°이고 AB=AC=2√2 km,", "BC=4 km이다. 민아는 A 지점을 출발",
     "하여 C 지점 방향으로 시속 √2 km의", "속력으로 걷고, 승우는 C 지점을 출발",
     "하여 B 지점 방향으로 시속 2 km의", "속력으로 걷는다. 두 사람 사이의",
     "직선 거리의 최솟값은?"],
    ["좌표평면 위의 두 점 A(1, 3), B(5, -1)", "에 대하여 선분 AB를 2 : 1로 내분하는",
     "점 P의 좌표를 (a, b)라 할 때, a+b의", "값을 구하시오."],
    ["이차함수 y=x²-4x+k의 그래프가 x축과", "서로 다른 두 점에서 만나도록 하는 정수 k의",
     "최댓값은?"],
    ["원 x²+y²=25 위의 점 (3, 4)에서의 접선이", "x축, y축과 만나는 점을 각각 A, B라 할 때,",
     "삼각형 OAB의 넓이는? (단, O는 원점이다.)"],
    ["두 직선 y=2x+1과 y=-x+4의 교점을 지나고", "기울기가 3인 직선의 y절편을 구하시오."],
]
CHOICE_SETS = [
    ["① 3√10/10", "② 2√10/5", "③ √10/2", "④ 3√10/5", "⑤ 7√10/10"],
    ["① 7+√3", "② 8+√3", "③ 9+2√3", "④ 10+2√3", "⑤ 11+3√3"],
    ["① 1", "② 2", "③ 3", "④ 4", "⑤ 5"],
]
SIDE_COLUMN_LINES = [
    "수를 n, k의", "(단, O는 원", "작다.)", "삼각형 ABC에서", "선분 DC를",
    "위에 있고", "형 FBD의", "때 k의 값은?",
]
CHAPTER_FOOTERS = ["I. 도형의 방정식", "II. 집합과 명제", "III. 함수와 그래프", "IV. 경우의 수"]

# 학생이 쓸 법한 필기 내용 (실제 사진에서 보이는 것들을 흉내)
HANDWRITING_SNIPPETS = [
    "2t-√2t", "(2-2√2t)²", "y=-x+4", "9t²-12t+4", "B₁(2,4)", "10+6", "=16",
    "(t, -t+2)", "√5/5", "2√2t", "4(1-√2)t²", "x=2", "답 ③", "2√10/5",
    "a+b=7", "k≤3", "O(0,0)", "d=|2t-4|",
]


@dataclass
class SyntheticPage:
    """합성 시험지 한 장에 대해 알고 있는 모든 것."""

    photo_image: np.ndarray                  # 카메라로 찍은 것처럼 만든 BGR 이미지
    clean_page_image: np.ndarray             # 필기가 없는 깨끗한 원본 (그레이스케일)
    written_page_image: np.ndarray           # 필기가 있는, 카메라 왜곡 전의 BGR 이미지
    handwriting_mask: np.ndarray             # 필기 픽셀 = 1
    printed_text_mask: np.ndarray            # 인쇄 글자 픽셀 = 1
    figure_mask: np.ndarray                  # 인쇄된 그래프·도형 픽셀 = 1
    color_print_mask: np.ndarray | None = None   # 컬러로 인쇄된 부분(제목 띠·아이콘·번호)
    student_graph_mask: np.ndarray | None = None # 학생이 손으로 그린 좌표축·곡선(지워야 함)
    applied_rotation_degrees: float = 0.0
    notes: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1) 깨끗한 인쇄 시험지
# ---------------------------------------------------------------------------
def make_clean_exam_page(width: int = 1000, height: int = 1400, seed: int = 0,
                         two_column: bool | None = None) -> np.ndarray:
    """필기가 하나도 없는 "깨끗한 인쇄 시험지"를 만든다(그레이스케일).

    하위 호환을 위해 그레이스케일 한 장만 돌려준다.
    인쇄 글자/도형을 따로 구분한 마스크가 필요하면 make_clean_exam_page_with_masks 를 써라.
    """
    return make_clean_exam_page_with_masks(width, height, seed, two_column)[1]


def make_clean_exam_page_with_masks(
    width: int = 1000, height: int = 1400, seed: int = 0, two_column: bool | None = None,
    with_color_printing: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """깨끗한 시험지와 마스크들을 만든다.

    요즘 문제집은 흑백이 아니다. 단원 제목 띠, 동그란 유형 아이콘, 색으로 인쇄된 문제 번호,
    색 강조 박스가 흔하다. 이것들을 색펜으로 오인해 지우면 시험지가 망가지므로,
    합성에도 반드시 넣어야 한다.

    돌려주는 값: (컬러 시험지 BGR, 그레이스케일, 인쇄 글자 마스크, 도형 마스크, 컬러 인쇄 마스크)
    """
    random_generator = random.Random(seed)
    if two_column is None:
        two_column = random_generator.random() < 0.5

    page_image = Image.new("L", (width, height), color=255)
    text_mask_image = Image.new("L", (width, height), color=0)
    figure_mask_image = Image.new("L", (width, height), color=0)
    page_drawing = ImageDraw.Draw(page_image)
    text_mask_drawing = ImageDraw.Draw(text_mask_image)
    figure_mask_drawing = ImageDraw.Draw(figure_mask_image)

    body_font_size = random_generator.choice([24, 26, 28])
    body_font = load_font(body_font_size)
    number_font = load_font(body_font_size + 2, bold=True)
    small_font = load_font(body_font_size - 5)

    printed_ink_value = random_generator.randint(25, 45)   # 인쇄는 진하다
    line_height = int(body_font_size * 1.95)

    def draw_text_both(position, text, font, ink_value=printed_ink_value):
        """본문과 '인쇄 글자 마스크'에 동시에 그린다."""
        page_drawing.text(position, text, font=font, fill=ink_value)
        text_mask_drawing.text(position, text, font=font, fill=255)

    def draw_line_both(points, ink_value, width_px, is_figure=True):
        """본문과 마스크(도형/글자)에 동시에 선을 그린다."""
        page_drawing.line(points, fill=ink_value, width=width_px)
        target = figure_mask_drawing if is_figure else text_mask_drawing
        target.line(points, fill=255, width=width_px)

    main_column_right = width - 240 if two_column else width - 60
    left_margin = 70

    # ── 왼쪽(주) 단: 문제 번호 + 본문 + 보기 ──────────────────────────────
    current_y = random_generator.randint(120, 260)
    for problem_position in range(2):
        problem_number = f"{random_generator.randint(1, 99):04d}"
        draw_text_both((left_margin, current_y), problem_number, number_font)
        current_y += line_height

        body_lines = random_generator.choice(PROBLEM_BODIES)
        for line_text in body_lines:
            draw_text_both((left_margin + 10, current_y), line_text, body_font)
            current_y += line_height
        current_y += int(line_height * 0.4)

        # 보기: 분수 가로줄을 진짜로 그린다(가는 인쇄선 보호를 시험하기 위해)
        choices = random_generator.choice(CHOICE_SETS)
        choice_x = left_margin + 20
        for choice_position, choice_text in enumerate(choices[:3]):
            draw_text_both((choice_x, current_y), choice_text, body_font)
            if "/" in choice_text:
                bar_left = choice_x + 26
                bar_right = bar_left + 62
                bar_y = current_y + body_font_size // 2 + 2
                draw_line_both([(bar_left, bar_y), (bar_right, bar_y)], printed_ink_value, 2)
            choice_x += (main_column_right - left_margin) // 3
            if choice_x > main_column_right - 120:
                break
        current_y += line_height * 2

        if current_y > height - 400:
            break

    # ── 도형: 좌표축 / 삼각형 / 원 중 하나 ────────────────────────────────
    figure_left = main_column_right - 300
    figure_top = random_generator.randint(140, 220)
    figure_kind = random_generator.choice(["좌표축", "삼각형", "원", "표"])
    figure_ink = max(15, printed_ink_value - 10)

    if figure_kind in ("좌표축", "원"):
        origin_x, origin_y = figure_left + 40, figure_top + 180
        draw_line_both([(origin_x - 30, origin_y), (origin_x + 220, origin_y)], figure_ink, 2)
        draw_line_both([(origin_x, origin_y + 30), (origin_x, origin_y - 190)], figure_ink, 2)
        if figure_kind == "원":
            page_drawing.ellipse([origin_x - 60, origin_y - 130, origin_x + 130, origin_y + 60],
                                 outline=figure_ink, width=2)
            figure_mask_drawing.ellipse([origin_x - 60, origin_y - 130, origin_x + 130, origin_y + 60],
                                        outline=255, width=2)
        else:
            curve_points = [(origin_x + x, origin_y - int(0.006 * (x - 110) ** 2) + 20)
                            for x in range(0, 220, 4)]
            draw_line_both(curve_points, figure_ink, 2)
        draw_text_both((origin_x - 22, origin_y + 6), "O", small_font)
    elif figure_kind == "삼각형":
        apex = (figure_left + 130, figure_top + 20)
        left_point = (figure_left + 40, figure_top + 180)
        right_point = (figure_left + 230, figure_top + 180)
        draw_line_both([apex, left_point, right_point, apex], figure_ink, 3)
        draw_line_both([apex, ((left_point[0] + right_point[0]) // 2, left_point[1])], figure_ink, 2)
        for label, point in (("A", apex), ("B", left_point), ("C", right_point)):
            draw_text_both((point[0] - 6, point[1] - 30), label, small_font)
    else:   # 표
        table_left, table_top = figure_left + 30, figure_top + 30
        cell_width, cell_height, column_count, row_count = 62, 44, 3, 3
        for row_position in range(row_count + 1):
            y_position = table_top + row_position * cell_height
            draw_line_both([(table_left, y_position),
                            (table_left + column_count * cell_width, y_position)], figure_ink, 2)
        for column_position in range(column_count + 1):
            x_position = table_left + column_position * cell_width
            draw_line_both([(x_position, table_top),
                            (x_position, table_top + row_count * cell_height)], figure_ink, 2)

    # ── 오른쪽 단 (2단 편집일 때만) ───────────────────────────────────────
    if two_column:
        divider_x = width - 220
        draw_line_both([(divider_x, 40), (divider_x, height - 40)], printed_ink_value + 20, 2)
        side_y = 60
        for line_text in random_generator.sample(SIDE_COLUMN_LINES, 6):
            draw_text_both((divider_x + 20, side_y), line_text, body_font)
            side_y += line_height

    # ── 쪽 번호 ─────────────────────────────────────────────────────────
    draw_text_both((left_margin, height - 58),
                   f"{random_generator.randint(10, 250)}   {random_generator.choice(CHAPTER_FOOTERS)}",
                   small_font)

    page_array = np.array(page_image, dtype=np.uint8)
    printed_text_mask = (np.array(text_mask_image) > 100).astype(np.uint8)
    figure_mask = (np.array(figure_mask_image) > 100).astype(np.uint8)
    # 글자와 도형이 겹친 자리는 도형 쪽을 우선한다(선이 더 중요하다)
    printed_text_mask[figure_mask > 0] = 0

    # 제본 쪽으로 갈수록 인쇄가 흐려진다 (실제 사진에서 한 장 안 27% 차이를 쟀다)
    binding_on_right = random_generator.random() < 0.5
    fade_profile = np.linspace(1.0, random_generator.uniform(0.62, 0.85), width, dtype=np.float32)
    if not binding_on_right:
        fade_profile = fade_profile[::-1]
    ink_amount = (255.0 - page_array.astype(np.float32)) * fade_profile[None, :]
    page_array = np.clip(255.0 - ink_amount, 0, 255).astype(np.uint8)

    page_array = cv2.GaussianBlur(page_array, (3, 3), 0)

    # ── 컬러 인쇄 ───────────────────────────────────────────────────────
    color_page = cv2.cvtColor(page_array, cv2.COLOR_GRAY2BGR)
    color_print_mask = np.zeros((height, width), dtype=np.uint8)
    if with_color_printing:
        color_page, color_print_mask = draw_color_printing(
            color_page, color_print_mask, printed_text_mask, random_generator, two_column
        )
    # 제목 띠에 덮인 글자·도형은 사진에 보이지 않으므로 정답에서 뺀다.
    # (빼지 않으면 "보이지도 않는 글자가 지워졌다"고 세게 되어 점수가 엉터리가 된다.)
    printed_text_mask[color_print_mask > 0] = 0
    figure_mask[color_print_mask > 0] = 0
    return color_page, page_array, printed_text_mask, figure_mask, color_print_mask


# 문제집에서 실제로 쓰이는 색들 (BGR). 인쇄이므로 채도가 펜보다 살짝 낮다.
PRINT_COLORS_BGR = [
    (150, 110, 60),    # 남색·청록 계열 헤더
    (170, 120, 70),    # 파랑
    (90, 140, 80),     # 초록
    (110, 90, 190),    # 팥죽색·자주
]


def draw_color_printing(color_page: np.ndarray, color_print_mask: np.ndarray,
                        printed_text_mask: np.ndarray, random_generator: random.Random,
                        two_column: bool) -> tuple[np.ndarray, np.ndarray]:
    """단원 제목 띠 · 유형 아이콘 · 색 문제 번호 같은 "컬러 인쇄"를 그린다.

    이것들은 인쇄물이므로 **지우면 안 된다.** 색펜과 구별하는 것이 2단계 색 필터의 숙제다.
    실제 사진에서 잰 값에 맞춰 그린다: 색 인쇄 글자는 획 두께 5~7px, 채도 78~91.
    (같은 사진의 빨간 볼펜은 두께 13~42px, 채도 101~124 였다.)
    """
    page_height, page_width = color_page.shape[:2]
    print_color = random_generator.choice(PRINT_COLORS_BGR)

    # (1) 단원 제목 띠 — 넓고 꽉 찬 사각형
    banner_height = random_generator.randint(34, 52)
    banner_width = random_generator.randint(int(page_width * 0.35), int(page_width * 0.55))
    banner_left = random_generator.randint(50, 90)
    banner_top = random_generator.randint(30, 70)
    cv2.rectangle(color_page, (banner_left, banner_top),
                  (banner_left + banner_width, banner_top + banner_height), print_color, -1)
    cv2.rectangle(color_print_mask, (banner_left, banner_top),
                  (banner_left + banner_width, banner_top + banner_height), 1, -1)
    banner_font = load_font(banner_height - 16, bold=True)
    banner_image = Image.fromarray(cv2.cvtColor(color_page, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(banner_image).text((banner_left + 14, banner_top + 6),
                                      "중단원 마무리문제", font=banner_font, fill=(255, 255, 255))
    color_page[:] = cv2.cvtColor(np.array(banner_image), cv2.COLOR_RGB2BGR)

    # (2) 동그란 유형 아이콘 — 꽉 찬 원
    icon_center = (page_width - random_generator.randint(90, 160), banner_top + banner_height // 2)
    icon_radius = random_generator.randint(16, 24)
    cv2.circle(color_page, icon_center, icon_radius, print_color, -1)
    cv2.circle(color_print_mask, icon_center, icon_radius, 1, -1)

    # (3) 색으로 인쇄된 문제 번호 — 가는 글자. 인쇄 글자 마스크에도 넣는다.
    number_font = load_font(random_generator.randint(26, 32), bold=True)
    overlay_image = Image.fromarray(cv2.cvtColor(color_page, cv2.COLOR_BGR2RGB))
    overlay_drawing = ImageDraw.Draw(overlay_image)
    number_mask_image = Image.fromarray(color_print_mask * 255)
    number_mask_drawing = ImageDraw.Draw(number_mask_image)
    rgb_color = (print_color[2], print_color[1], print_color[0])
    for number_position in range(random_generator.randint(2, 4)):
        number_x = random_generator.randint(60, 110)
        number_y = random_generator.randint(180, page_height - 260)
        number_text = f"{random_generator.randint(1, 999):04d}"
        overlay_drawing.text((number_x, number_y), number_text, font=number_font, fill=rgb_color)
        number_mask_drawing.text((number_x, number_y), number_text, font=number_font, fill=255)
    color_page[:] = cv2.cvtColor(np.array(overlay_image), cv2.COLOR_RGB2BGR)
    color_print_mask[:] = (np.array(number_mask_image) > 100).astype(np.uint8)

    return color_page, color_print_mask


def add_bleed_through(page_array: np.ndarray, seed: int) -> np.ndarray:
    """뒷장 인쇄가 비쳐 보이는 효과를 넣는다(좌우로 뒤집은 아주 흐린 글자)."""
    random_generator = random.Random(seed + 9999)
    _, reverse_page, _, _, _ = make_clean_exam_page_with_masks(
        page_array.shape[1], page_array.shape[0], seed=seed + 500, with_color_printing=False
    )
    mirrored = cv2.flip(reverse_page, 1)
    strength = random_generator.uniform(0.06, 0.16)
    reverse_ink = (255.0 - mirrored.astype(np.float32)) * strength
    return np.clip(page_array.astype(np.float32) - reverse_ink, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2) 학생 필기
# ---------------------------------------------------------------------------
def draw_text_into_coverage(coverage_layer: np.ndarray, text: str, position: tuple[int, int],
                            font_size: int, rotation_degrees: float) -> None:
    """글자를 기울여 "덮임 정도(coverage)" 층에 그린다.

    coverage 는 0~1 값으로 "이 픽셀이 잉크에 얼마나 덮였나"를 뜻한다.
    나중에 여기에 질감과 색을 입힌다. 손글씨는 기울고 크기가 제각각이므로 회전을 넣는다.
    """
    font = load_font(font_size)
    text_bbox = font.getbbox(text)
    text_width = max(text_bbox[2] - text_bbox[0], 1) + 10
    text_height = max(text_bbox[3] - text_bbox[1], 1) + 10

    stamp_image = Image.new("L", (text_width, text_height), color=0)
    ImageDraw.Draw(stamp_image).text((5 - text_bbox[0], 5 - text_bbox[1]), text, font=font, fill=255)
    stamp_image = stamp_image.rotate(rotation_degrees, expand=True, resample=Image.BILINEAR)

    stamp_array = np.array(stamp_image, dtype=np.float32) / 255.0
    left, top = position
    stamp_height, stamp_width = stamp_array.shape
    layer_height, layer_width = coverage_layer.shape

    left = max(0, min(left, layer_width - stamp_width - 1))
    top = max(0, min(top, layer_height - stamp_height - 1))
    if stamp_width >= layer_width or stamp_height >= layer_height:
        return
    target = coverage_layer[top:top + stamp_height, left:left + stamp_width]
    np.maximum(target, stamp_array, out=target)


def draw_student_drawn_graph(coverage_layer: np.ndarray, random_generator: random.Random) -> tuple[int, int, int, int]:
    """학생이 연필로 직접 그린 좌표축과 곡선을 그린다.

    왜 필요한가: 시험지를 풀다 보면 학생이 여백에 좌표축을 직접 그리고 포물선을 그린다.
    이것은 **지워야 할 필기**인데, 규칙 엔진이 "긴 직선이 있으니 인쇄된 그래프다"라고
    보호해 버리면 오히려 새까맣게 굵어진다. 그 실패를 재려면 합성에도 있어야 한다.

    사람이 그은 선은 자로 댄 듯 곧지 않고 미세하게 흔들린다. 그 흔들림을 넣는 것이 핵심이다.

    돌려주는 값: 그린 영역의 사각형 (x, y, 너비, 높이)
    """
    layer_height, layer_width = coverage_layer.shape
    origin_x = random_generator.randint(120, max(121, layer_width - 320))
    origin_y = random_generator.randint(300, max(301, layer_height - 260))
    axis_length = random_generator.randint(150, 230)

    def draw_wobbly_line(start_point: tuple[int, int], end_point: tuple[int, int]) -> None:
        """두 점을 잇되 손으로 그은 것처럼 미세하게 흔들리는 선을 그린다."""
        segment_count = 12
        previous_point = start_point
        for segment_index in range(1, segment_count + 1):
            ratio = segment_index / segment_count
            target_x = start_point[0] + (end_point[0] - start_point[0]) * ratio
            target_y = start_point[1] + (end_point[1] - start_point[1]) * ratio
            # 진행 방향과 상관없이 좌우로 몇 픽셀씩 흔들린다
            target_x += random_generator.uniform(-2.5, 2.5)
            target_y += random_generator.uniform(-2.5, 2.5)
            current_point = (int(target_x), int(target_y))
            cv2.line(coverage_layer, previous_point, current_point, 1.0,
                     random_generator.randint(1, 2), cv2.LINE_AA)
            previous_point = current_point

    # 가로축·세로축
    draw_wobbly_line((origin_x - 30, origin_y), (origin_x + axis_length, origin_y))
    draw_wobbly_line((origin_x, origin_y + 40), (origin_x, origin_y - axis_length))

    # 포물선
    curve_width = axis_length - 20
    vertex_x = origin_x + curve_width // 2
    curve_points: list[tuple[int, int]] = []
    for offset_x in range(-curve_width // 2, curve_width // 2, 6):
        curve_y = origin_y - 20 - int(0.010 * offset_x * offset_x) + random_generator.randint(-2, 2)
        curve_points.append((vertex_x + offset_x, curve_y))
    for point_index in range(len(curve_points) - 1):
        cv2.line(coverage_layer, curve_points[point_index], curve_points[point_index + 1],
                 1.0, random_generator.randint(1, 2), cv2.LINE_AA)

    left = max(0, origin_x - 40)
    top = max(0, origin_y - axis_length - 20)
    return (left, top, min(axis_length + 80, layer_width - left),
            min(axis_length + 80, layer_height - top))


def add_student_handwriting(clean_page: np.ndarray, seed: int = 0, difficulty: str = "보통",
                            return_graph_boxes: bool = False):
    """깨끗한 시험지 위에 학생 필기를 그린다.

    인자:
        difficulty: "쉬움"(연필만) / "보통"(연필 + 색펜) / "어려움"(검은 볼펜 포함)
                    검은 볼펜은 인쇄만큼 진해서 규칙으로 가장 구별하기 어렵다.

    돌려주는 값: (필기가 그려진 컬러 이미지 BGR, 필기 픽셀 마스크)
    """
    random_generator = random.Random(seed)
    numpy_generator = np.random.default_rng(seed)
    page_height, page_width = clean_page.shape[:2]

    # 펜 종류별로 따로 층을 만든다(질감과 색이 다르기 때문)
    pencil_coverage = np.zeros((page_height, page_width), dtype=np.float32)
    ballpoint_coverage = np.zeros((page_height, page_width), dtype=np.float32)
    red_coverage = np.zeros((page_height, page_width), dtype=np.float32)
    blue_pen_coverage = np.zeros((page_height, page_width), dtype=np.float32)

    # ── 손으로 쓴 수식·숫자 (실제 글자 모양이라야 현실적이다) ────────────
    snippet_count = {"쉬움": 14, "보통": 26, "어려움": 34}[difficulty]
    for _ in range(snippet_count):
        snippet_text = random_generator.choice(HANDWRITING_SNIPPETS)
        font_size = random_generator.randint(22, 40)
        rotation = random_generator.uniform(-12, 12)
        position = (random_generator.randint(20, max(21, page_width - 220)),
                    random_generator.randint(20, max(21, page_height - 90)))
        # 검은 볼펜은 "어려움"에서만 나온다
        pen_choice = random_generator.random()
        if difficulty == "어려움" and pen_choice < 0.30:
            target_layer = ballpoint_coverage      # 검은 볼펜 (인쇄만큼 진하다)
        elif pen_choice > 0.85:
            target_layer = blue_pen_coverage       # 파란 볼펜
        else:
            target_layer = pencil_coverage
        draw_text_into_coverage(target_layer, snippet_text, position, font_size, rotation)

    # ── 낙서 곡선 (풀이 과정에서 그은 보조선·화살표) ─────────────────────
    for _ in range(random_generator.randint(10, 18)):
        points = [(random_generator.randint(30, page_width - 60),
                   random_generator.randint(30, page_height - 40))]
        for _ in range(random_generator.randint(3, 7)):
            points.append((int(np.clip(points[-1][0] + random_generator.randint(-80, 100), 0, page_width - 1)),
                           int(np.clip(points[-1][1] + random_generator.randint(-45, 45), 0, page_height - 1))))
        target_layer = pencil_coverage if random_generator.random() > 0.25 else ballpoint_coverage
        for point_index in range(len(points) - 1):
            cv2.line(target_layer, points[point_index], points[point_index + 1],
                     1.0, random_generator.randint(1, 3), cv2.LINE_AA)

    # ── 학생이 직접 그린 좌표축·포물선 (지워야 할 필기다) ───────────────
    student_graph_boxes: list[tuple[int, int, int, int]] = []
    for _ in range(random_generator.randint(1, 2)):
        student_graph_boxes.append(draw_student_drawn_graph(pencil_coverage, random_generator))

    # ── 빨간펜 채점 표시 (동그라미 · 별 · 체크) ──────────────────────────
    if difficulty != "쉬움":
        for _ in range(random_generator.randint(2, 4)):
            center = (random_generator.randint(120, page_width - 120),
                      random_generator.randint(120, page_height - 120))
            cv2.circle(red_coverage, center, random_generator.randint(50, 115), 1.0, 5, cv2.LINE_AA)
        for _ in range(random_generator.randint(1, 3)):
            center_x = random_generator.randint(100, page_width - 100)
            center_y = random_generator.randint(100, page_height - 100)
            size = random_generator.randint(30, 60)
            for angle_index in range(5):
                angle_a = math.radians(90 + angle_index * 144)
                angle_b = math.radians(90 + (angle_index + 1) * 144)
                point_a = (int(center_x + size * math.cos(angle_a)), int(center_y - size * math.sin(angle_a)))
                point_b = (int(center_x + size * math.cos(angle_b)), int(center_y - size * math.sin(angle_b)))
                cv2.line(red_coverage, point_a, point_b, 1.0, 4, cv2.LINE_AA)

    # ── 연필 질감: 획 안에서 농도가 들쭉날쭉하다 ─────────────────────────
    # (실제 연필은 종이 결에 따라 사각사각 얹혀서 균일하지 않다.
    #  이 질감이 없으면 합성이 실제보다 훨씬 쉬워져 테스트 점수가 부풀려진다.)
    grain_noise = numpy_generator.uniform(0.40, 1.0, size=pencil_coverage.shape).astype(np.float32)
    grain_noise = cv2.GaussianBlur(grain_noise, (3, 3), 0)
    pencil_coverage = pencil_coverage * grain_noise
    pencil_coverage = cv2.GaussianBlur(pencil_coverage, (3, 3), 0.6)

    # ── 합치기: 층마다 다른 색·진하기로 종이에 얹는다 ────────────────────
    if clean_page.ndim == 3:
        color_page = clean_page.astype(np.float32)
    else:
        color_page = cv2.cvtColor(clean_page, cv2.COLOR_GRAY2BGR).astype(np.float32)

    def composite(coverage: np.ndarray, bgr_color: tuple[int, int, int]) -> None:
        """coverage(0~1) 만큼 종이에 잉크를 얹는다."""
        alpha = np.clip(coverage, 0.0, 1.0)[:, :, None]
        ink = np.array(bgr_color, dtype=np.float32)[None, None, :]
        color_page[:] = color_page * (1.0 - alpha) + ink * alpha

    pencil_gray = random_generator.randint(115, 175)        # 연필은 흐리다 (실측 ~178 밝기)
    ballpoint_gray = random_generator.randint(35, 70)       # 검은 볼펜은 인쇄만큼 진하다
    composite(pencil_coverage, (pencil_gray,) * 3)
    composite(ballpoint_coverage, (ballpoint_gray,) * 3)
    composite(red_coverage, (45, 45, 215))
    composite(blue_pen_coverage, (200, 70, 40))   # 파란 볼펜 — 컬러 인쇄와 가장 헷갈리는 상대

    written_page = np.clip(color_page, 0, 255).astype(np.uint8)

    handwriting_mask = (
        (pencil_coverage > 0.12) | (ballpoint_coverage > 0.12)
        | (red_coverage > 0.12) | (blue_pen_coverage > 0.12)
    ).astype(np.uint8)

    if return_graph_boxes:
        # 학생이 그린 그래프만 따로 표시한 마스크도 만들어 준다
        student_graph_mask = np.zeros_like(handwriting_mask)
        for box_left, box_top, box_width, box_height in student_graph_boxes:
            box_slice = (slice(box_top, box_top + box_height), slice(box_left, box_left + box_width))
            student_graph_mask[box_slice] = handwriting_mask[box_slice]
        return written_page, handwriting_mask, student_graph_mask
    return written_page, handwriting_mask


# ---------------------------------------------------------------------------
# 3) 카메라 흉내
# ---------------------------------------------------------------------------
def simulate_camera(color_page: np.ndarray, seed: int = 0,
                    rotation_degrees: float | None = None) -> tuple[np.ndarray, float]:
    """스마트폰으로 찍은 것처럼 회전·원근·그림자·손가락·블러·노이즈·JPEG 를 입힌다.

    돌려주는 값: (사진처럼 만든 이미지, 실제로 넣은 회전 각도)
    """
    random_generator = random.Random(seed)

    margin = int(min(color_page.shape[:2]) * 0.10)
    desk_color = random_generator.randint(70, 120)
    photo = cv2.copyMakeBorder(color_page, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=(desk_color,) * 3)

    if rotation_degrees is None:
        rotation_degrees = random_generator.uniform(-15.0, 15.0)
    photo_center = (photo.shape[1] / 2.0, photo.shape[0] / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(photo_center, rotation_degrees, 1.0)
    photo = cv2.warpAffine(photo, rotation_matrix, (photo.shape[1], photo.shape[0]),
                           borderValue=(desk_color,) * 3)

    warp_strength = random_generator.uniform(0.0, 0.03)
    height_now, width_now = photo.shape[:2]
    source_points = np.float32([[0, 0], [width_now, 0], [width_now, height_now], [0, height_now]])
    destination_points = source_points + np.float32([
        [width_now * warp_strength, height_now * warp_strength],
        [-width_now * warp_strength, height_now * warp_strength * 0.5],
        [-width_now * warp_strength * 0.5, -height_now * warp_strength],
        [width_now * warp_strength * 0.5, -height_now * warp_strength * 0.5],
    ])
    photo = cv2.warpPerspective(photo, cv2.getPerspectiveTransform(source_points, destination_points),
                                (width_now, height_now), borderValue=(desk_color,) * 3)

    # 그림자(한쪽이 어두워지는 기울기)
    gradient_x = np.linspace(random_generator.uniform(0.55, 0.9), 1.0, photo.shape[1], dtype=np.float32)
    gradient_y = np.linspace(random_generator.uniform(0.65, 1.0), 1.0, photo.shape[0], dtype=np.float32)
    photo = np.clip(photo.astype(np.float32) * np.outer(gradient_y, gradient_x)[:, :, None],
                    0, 255).astype(np.uint8)

    # 손가락이 화면 가장자리에 들어오는 경우 (실제 샘플 사진에도 있었다)
    if random_generator.random() < 0.35:
        finger_height = random_generator.randint(photo.shape[0] // 6, photo.shape[0] // 3)
        finger_top = random_generator.randint(0, max(1, photo.shape[0] - finger_height))
        finger_width = random_generator.randint(30, 70)
        cv2.ellipse(photo, (0, finger_top + finger_height // 2), (finger_width, finger_height // 2),
                    0, 0, 360, (150, 170, 195), -1)
        photo = cv2.GaussianBlur(photo, (5, 5), 0)

    blur_size = random_generator.choice([1, 3, 3, 5])
    if blur_size > 1:
        photo = cv2.GaussianBlur(photo, (blur_size, blur_size), 0)
    noise = np.random.default_rng(seed).normal(0, 3.0, photo.shape)
    photo = np.clip(photo.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    encode_succeeded, encoded = cv2.imencode(".jpg", photo, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if encode_succeeded:
        photo = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return photo, rotation_degrees


# ---------------------------------------------------------------------------
# 4) 한 번에 만들기
# ---------------------------------------------------------------------------
def make_synthetic_page(seed: int = 0, difficulty: str = "보통",
                        rotation_degrees: float | None = None,
                        with_bleed_through: bool = True) -> SyntheticPage:
    """합성 시험지 한 장을 정답 마스크까지 갖춰 만든다."""
    (color_clean_page, gray_clean_page, printed_text_mask,
     figure_mask, color_print_mask) = make_clean_exam_page_with_masks(seed=seed)
    if with_bleed_through:
        gray_with_bleed = add_bleed_through(gray_clean_page, seed)
        bleed_amount = gray_clean_page.astype(np.int16) - gray_with_bleed.astype(np.int16)
        page_for_writing = np.clip(
            color_clean_page.astype(np.int16) - bleed_amount[:, :, None], 0, 255).astype(np.uint8)
    else:
        page_for_writing = color_clean_page
    written_page, handwriting_mask, student_graph_mask = add_student_handwriting(
        page_for_writing, seed=seed, difficulty=difficulty, return_graph_boxes=True)
    photo, applied_rotation = simulate_camera(written_page, seed=seed,
                                              rotation_degrees=rotation_degrees)
    return SyntheticPage(
        photo_image=photo,
        clean_page_image=gray_clean_page,
        written_page_image=written_page,
        handwriting_mask=handwriting_mask,
        printed_text_mask=printed_text_mask,
        figure_mask=figure_mask,
        color_print_mask=color_print_mask,
        student_graph_mask=student_graph_mask,
        applied_rotation_degrees=applied_rotation,
        notes={"difficulty": difficulty},
    )


def make_photo_pair(seed: int = 0, rotation_degrees: float | None = None):
    """예전 테스트가 쓰던 간단한 형태 (사진, 필기 마스크, 깨끗한 원본, 회전 각도)."""
    page = make_synthetic_page(seed=seed, rotation_degrees=rotation_degrees)
    return (page.photo_image, page.handwriting_mask, page.clean_page_image,
            page.applied_rotation_degrees)
