"""클로드 판정관 테스트 — 실제 API 는 부르지 않고 mock 으로 대체한다.

검증할 것:
  1. 키가 없으면 실패하지 않고, 애매한 획을 전부 "인쇄"로 두고 api_skipped 를 알린다
  2. 키가 틀렸을 때(인증 오류)도 500 이 아니라 규칙 결과 + 오류 종류를 돌려준다
  3. confidence 가 기준 미만이면 안전하게 "인쇄"로 처리한다
  4. 조각 묶기가 설정한 최대 크기를 넘지 않는다
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import engine.claude_judge as claude_judge
from engine.claude_judge import group_unsure_components, judge_unsure_components
from engine.settings import load_config
from engine.stroke_filter import (
    LABEL_FIGURE,
    LABEL_HANDWRITING,
    LABEL_PRINTED,
    LABEL_UNSURE,
    StrokeComponent,
)
from services.claude_client import CallStatistics, ClaudeCallError

PROMPT_PATH = Path(__file__).resolve().parent.parent / "services" / "prompts" / "judge.txt"


def make_component(component_index: int, left: int, top: int, label: str = LABEL_UNSURE) -> StrokeComponent:
    """테스트용 가짜 요소 하나를 만든다."""
    return StrokeComponent(
        component_index=component_index,
        bounding_box=(left, top, 20, 20),
        area_px=120,
        mean_darkness=90.0,
        dark_core_darkness=110.0,
        darkness_std=20.0,
        thickness_mean_px=3.0,
        thickness_cv=0.3,
        fill_ratio=0.35,
        local_darkness_reference=120.0,
        density_score=0.5,
        thickness_score=0.3,
        fill_score=0.4,
        alignment_score=0.5,
        handwriting_score=0.5,
        label=label,
    )


def test_no_key_keeps_everything_and_reports_skip() -> None:
    """키가 없으면 애매한 획을 남기고(= 인쇄 처리) api_skipped 를 True 로 알려야 한다."""
    config = load_config()
    components = [make_component(index, index * 30, 10) for index in range(1, 6)]
    image = np.full((400, 400, 3), 255, dtype=np.uint8)

    result = judge_unsure_components(
        corrected_color_image=image, components=components, config=config,
        api_key=None, model=config["claude"]["fast_model"],
        cache_directory=Path("/tmp/does_not_matter"), statistics=CallStatistics(),
        prompt_path=PROMPT_PATH,
    )

    assert result["api_skipped"] is True
    assert result["error_kind"] == "no_key"
    assert "키" in result["message"]
    assert all(component.label == LABEL_PRINTED for component in components), (
        "키가 없을 때는 안 지우는 쪽(인쇄)으로 둬야 한다."
    )


def test_authentication_error_falls_back_to_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """키가 틀려도 예외를 밖으로 던지지 않고 규칙 결과로 돌아가야 한다."""
    config = load_config()
    components = [make_component(index, index * 30, 10) for index in range(1, 4)]
    image = np.full((400, 400, 3), 255, dtype=np.uint8)

    def raise_authentication_error(**_kwargs):
        raise ClaudeCallError("auth", "Anthropic 키가 올바르지 않아요.")

    monkeypatch.setattr(claude_judge, "call_claude_with_tool", raise_authentication_error)

    result = judge_unsure_components(
        corrected_color_image=image, components=components, config=config,
        api_key="sk-ant-fake-key-for-test", model=config["claude"]["fast_model"],
        cache_directory=Path("/tmp/does_not_matter"), statistics=CallStatistics(),
        prompt_path=PROMPT_PATH,
    )

    assert result["error_kind"] == "auth"
    assert result["api_skipped"] is True
    assert all(component.label == LABEL_PRINTED for component in components)


def test_low_confidence_is_treated_as_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    """confidence 가 기준 미만이면 handwriting 이라고 해도 지우지 않아야 한다."""
    config = load_config()
    threshold = config["claude"]["confidence_threshold"]
    components = [make_component(1, 10, 10), make_component(2, 60, 10), make_component(3, 110, 10)]
    image = np.full((400, 400, 3), 255, dtype=np.uint8)

    def fake_call(**kwargs):
        return {"items": [
            {"box_id": 1, "label": LABEL_HANDWRITING, "confidence": threshold + 0.2},   # 확신 → 지움
            {"box_id": 2, "label": LABEL_HANDWRITING, "confidence": threshold - 0.2},   # 애매 → 살림
            {"box_id": 3, "label": LABEL_FIGURE, "confidence": 0.99},                   # 그래프
        ]}

    monkeypatch.setattr(claude_judge, "call_claude_with_tool", fake_call)

    result = judge_unsure_components(
        corrected_color_image=image, components=components, config=config,
        api_key="sk-ant-fake-key-for-test", model=config["claude"]["fast_model"],
        cache_directory=Path("/tmp/does_not_matter"), statistics=CallStatistics(),
        prompt_path=PROMPT_PATH,
    )

    labels_by_index = {component.component_index: component.label for component in components}
    assert labels_by_index[1] == LABEL_HANDWRITING
    assert labels_by_index[2] == LABEL_PRINTED, "확신이 낮으면 안 지우는 쪽이어야 한다."
    assert labels_by_index[3] == LABEL_FIGURE
    assert result["judged_count"] == 3
    assert result["api_skipped"] is False


def test_crop_groups_respect_maximum_size() -> None:
    """조각 묶기는 설정한 최대 한 변 길이를 넘지 않아야 한다."""
    maximum_side = 300
    padding = 12
    # 가로로 넓게 흩어진 요소들 — 한 조각에 다 들어갈 수 없다
    components = [make_component(index, index * 90, 40) for index in range(1, 12)]

    groups = group_unsure_components(components, (2000, 2000), maximum_side, padding)

    assert len(groups) > 1, "넓게 흩어진 요소가 한 조각에 다 들어가면 안 된다."
    for group in groups:
        _, _, width, height = group["crop_box"]
        assert width <= maximum_side + 2 * padding
        assert height <= maximum_side + 2 * padding
    covered = sum(len(group["components"]) for group in groups)
    assert covered == len(components), "모든 애매 요소가 빠짐없이 조각에 담겨야 한다."


@pytest.mark.live
def test_live_judge_against_real_api() -> None:
    """실제 Anthropic API 를 한 번 부른다. `pytest --live` 일 때만 실행된다.

    환경변수 ANTHROPIC_API_KEY 가 필요하고, 아주 적은 비용이 발생한다.
    키는 이 테스트가 끝나면 어디에도 남지 않는다.
    """
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY 환경변수가 없다.")

    import cv2
    from engine.clean import clean_image

    config = load_config()
    sample_path = Path(__file__).resolve().parent / "samples" / "01_기하_0071.jpg"
    if not sample_path.exists():
        pytest.skip("tests/samples 에 샘플 사진이 없다.")

    original_image = cv2.imread(str(sample_path))
    result = clean_image(original_image, config, api_key=api_key)

    print(f"\n판정관 호출 {result['api_call_count']}회, "
          f"판정한 획 {result['api_judged_count']}개, "
          f"비용 약 ${result['api_cost_usd']:.4f}")
    assert result["api_skipped"] is False
    assert result["api_call_count"] >= 1
    assert result["api_judged_count"] > 0
