"""화면 슬라이더(0~100) 네 개를 config.yaml 값으로 바꾼다.

강사는 임계값 숫자를 몰라도 된다. 슬라이더 가운데(50)가 config.yaml 의 기본값과 같다.
"""

from __future__ import annotations

import copy
from typing import Any

SLIDER_DEFAULTS = {
    "remove_strength": 50,     # 필기 제거 강도
    "color_sensitivity": 50,   # 색 필기 제거 민감도
    "graph_thickness": 33,     # 그래프 굵기 (0~3px → 33 이 1px)
    "text_darkness": 50,       # 글자 진하기
}


def clamp_ratio(value: Any) -> float:
    """0~100 슬라이더 값을 0.0~1.0 로. 이상한 값이 와도 죽지 않는다."""
    try:
        return max(0.0, min(1.0, float(value) / 100.0))
    except (TypeError, ValueError):
        return 0.5


def apply_sliders(config: dict[str, Any], sliders: dict[str, Any] | None) -> dict[str, Any]:
    """슬라이더 값을 반영한 설정 사본을 돌려준다. 원본 config 는 건드리지 않는다."""
    adjusted = copy.deepcopy(config)
    values = {**SLIDER_DEFAULTS, **(sliders or {})}

    # 필기 제거 강도: 올리면 "애매" 구간이 필기 쪽으로 넓어져 더 과감히 지운다
    strength = clamp_ratio(values["remove_strength"])
    adjusted["stroke_filter"]["remove_threshold"] = round(0.80 - 0.30 * strength, 3)   # 50 → 0.65
    adjusted["stroke_filter"]["keep_threshold"] = round(0.45 - 0.20 * strength, 3)     # 50 → 0.35

    # 색 필기 제거 민감도: 올리면 더 연한 색까지 색펜으로 본다
    sensitivity = clamp_ratio(values["color_sensitivity"])
    adjusted["color_filter"]["saturation_sensitivity"] = round(0.5 + sensitivity, 3)    # 50 → 1.0

    # 그래프 굵기: 0~3px
    thickness = clamp_ratio(values["graph_thickness"])
    adjusted["graph_enhance"]["dilate_px"] = int(round(thickness * 3))                  # 33 → 1

    # 글자 진하기: 감마가 낮을수록 진하다
    darkness = clamp_ratio(values["text_darkness"])
    adjusted["finalize"]["text_darken_gamma"] = round(1.0 - 0.4 * darkness, 3)         # 50 → 0.8

    return adjusted
