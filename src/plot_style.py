"""
그래프의 색과 글꼴을 프로젝트 전체에서 통일해 주는 모듈.

왜 따로 두나?
  - 노트북마다 색을 다르게 쓰면, 나중에 보고서에 그림을 모았을 때 따로 논다.
  - '정상은 파랑, 사기는 주황'을 한 곳에서 정해 두면 어느 그림에서든 같은 뜻이 된다.

색 선택 근거:
  파랑 #2a78d6 과 주황 #eb6834 두 색은 색각 이상(색맹) 검증을 통과한 조합이다.
  두 색의 차이(ΔE)는 색각 이상 시뮬레이션에서 24.7, 정상 시야에서 33.6 으로,
  안전 기준(각각 8 이상, 15 이상)을 크게 넘는다.
  다만 색만으로 구분하게 두지 않고, 범례와 직접 라벨을 항상 함께 붙인다.
"""

import matplotlib
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 색 정의
# ------------------------------------------------------------
COLOR_NORMAL = "#2a78d6"   # 정상 주소 (FLAG=0) — 파랑
COLOR_FRAUD = "#eb6834"    # 사기 주소 (FLAG=1) — 주황

# 클래스 값(0/1) -> 색, 이름
CLASS_COLORS = {0: COLOR_NORMAL, 1: COLOR_FRAUD}
CLASS_LABELS = {0: "정상 (FLAG=0)", 1: "사기 (FLAG=1)"}

# 글자·격자 색 (데이터보다 눈에 덜 띄어야 하므로 회색 계열)
COLOR_TEXT_PRIMARY = "#0b0b0b"    # 제목처럼 가장 중요한 글자
COLOR_TEXT_SECONDARY = "#52514e"  # 축 라벨, 설명
COLOR_GRID = "#dcdcd8"            # 격자선 — 아주 옅게

# 한글 글꼴 후보 목록.
# 운영체제마다 설치된 한글 글꼴이 다르므로 여러 개를 순서대로 시도한다.
#   윈도우 -> Malgun Gothic, 맥 -> AppleGothic, 리눅스 -> Nanum/WenQuanYi
KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",
    "AppleGothic",
    "Apple SD Gothic Neo",
    "NanumGothic",
    "Nanum Gothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "WenQuanYi Zen Hei",
    "Unifont",
]


def find_available_korean_font() -> str | None:
    """설치된 글꼴 중 한글을 표시할 수 있는 첫 번째 글꼴 이름을 돌려준다."""
    # matplotlib 이 인식하는 모든 글꼴 이름을 집합으로 모은다.
    installed_font_names = {font.name for font in matplotlib.font_manager.fontManager.ttflist}

    for candidate_name in KOREAN_FONT_CANDIDATES:
        if candidate_name in installed_font_names:
            return candidate_name
    return None


def apply_project_plot_style() -> str | None:
    """
    프로젝트 공통 그래프 스타일을 적용한다. 노트북 맨 위에서 한 번만 호출하면 된다.

    돌려주는 값: 실제로 적용된 한글 글꼴 이름 (없으면 None)
    """
    selected_font_name = find_available_korean_font()
    if selected_font_name is not None:
        plt.rcParams["font.family"] = selected_font_name

    # 마이너스 기호(−)가 네모로 깨지는 것을 막는다.
    # 일부 한글 글꼴에는 유니코드 마이너스 글리프가 없기 때문이다.
    plt.rcParams["axes.unicode_minus"] = False

    # ----- 격자와 축은 '데이터보다 덜 보이게' -----
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = COLOR_GRID
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["axes.axisbelow"] = True   # 격자를 데이터 뒤로 보낸다

    # 위쪽·오른쪽 테두리는 정보를 담지 않으므로 없앤다(잉크 절약).
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = COLOR_GRID

    # ----- 글자 색과 크기 -----
    plt.rcParams["text.color"] = COLOR_TEXT_PRIMARY
    plt.rcParams["axes.labelcolor"] = COLOR_TEXT_SECONDARY
    plt.rcParams["xtick.color"] = COLOR_TEXT_SECONDARY
    plt.rcParams["ytick.color"] = COLOR_TEXT_SECONDARY
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["font.size"] = 10

    # ----- 배경 -----
    plt.rcParams["figure.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    plt.rcParams["savefig.facecolor"] = "#fcfcfb"
    plt.rcParams["figure.dpi"] = 110

    # 범례 상자는 테두리 없이 (상자 선이 데이터와 경쟁하지 않도록)
    plt.rcParams["legend.frameon"] = False

    return selected_font_name
