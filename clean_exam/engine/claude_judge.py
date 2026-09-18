"""4) 클로드 판정관 — 규칙만으로 애매한 획만 골라 클로드 비전에게 물어본다.

핵심 설계:
  · **애매한 것만** 보낸다. 규칙이 확신하는 것은 안 보낸다(비용 때문).
  · 가까운 것끼리 묶어 한 조각으로 자르고, 각 요소를 얇은 번호 박스로 표시한다.
  · 조각 여러 개를 한 호출에 묶어 보낸다(config 의 max_crops_per_call).
  · 답은 도구 호출 스키마로 받는다(자유 문장 파싱 금지).
  · 키가 없거나 호출이 실패하면 **예외를 삼키고** 애매한 것을 전부 "인쇄"로 둔다.
    안 지우는 쪽이 항상 안전하기 때문이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from engine.stroke_filter import (
    LABEL_FIGURE,
    LABEL_HANDWRITING,
    LABEL_PRINTED,
    LABEL_UNSURE,
    StrokeComponent,
)
from services.claude_client import CallStatistics, ClaudeCallError, call_claude_with_tool

# 판정관에게 받을 JSON 의 모양. 도구 호출로 강제하므로 이 모양이 그대로 보장된다.
JUDGE_TOOL_NAME = "report_stroke_labels"
JUDGE_TOOL_DESCRIPTION = "각 번호 박스 안의 획이 필기인지, 인쇄 글자인지, 그래프·도형인지 보고한다."
JUDGE_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "번호 박스 하나당 판정 하나. 모든 박스에 대해 빠짐없이 answer 할 것.",
            "items": {
                "type": "object",
                "properties": {
                    "box_id": {"type": "integer", "description": "조각 이미지에 적힌 번호"},
                    "label": {
                        "type": "string",
                        "enum": [LABEL_HANDWRITING, LABEL_PRINTED, LABEL_FIGURE],
                        "description": "handwriting=학생 필기, printed_text=인쇄 글자, figure=인쇄된 그래프·도형",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0~1.0. 확신이 없으면 낮게 줄 것.",
                    },
                },
                "required": ["box_id", "label", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def group_unsure_components(unsure_components: list[StrokeComponent],
                            image_shape: tuple[int, int],
                            maximum_crop_side_px: int,
                            crop_padding_px: int) -> list[dict[str, Any]]:
    """애매한 요소들을 가까운 것끼리 묶어 "조각" 목록을 만든다.

    묶는 이유: 획 하나만 잘라 보내면 맥락이 없어 사람도 판정을 못 한다.
    주변 글자가 같이 보여야 "이건 인쇄 줄 위에 겹쳐 쓴 필기"라고 알 수 있다.

    묶는 방법: 위에서 아래로 훑으면서, 지금 조각의 사각형에 넣어도 maximum_crop_side_px 를
    넘지 않으면 같은 조각에 넣고, 넘으면 새 조각을 시작한다.

    돌려주는 값: [{"crop_box": (x, y, 너비, 높이), "components": [...]}, ...]
    """
    image_height, image_width = image_shape
    sorted_components = sorted(unsure_components, key=lambda c: (c.bounding_box[1], c.bounding_box[0]))

    crop_groups: list[dict[str, Any]] = []
    for component in sorted_components:
        left, top, width, height = component.bounding_box
        component_box = [left, top, left + width, top + height]

        placed_in_existing_group = False
        for group in crop_groups:
            merged_box = [
                min(group["box"][0], component_box[0]),
                min(group["box"][1], component_box[1]),
                max(group["box"][2], component_box[2]),
                max(group["box"][3], component_box[3]),
            ]
            merged_width = merged_box[2] - merged_box[0]
            merged_height = merged_box[3] - merged_box[1]
            if merged_width <= maximum_crop_side_px and merged_height <= maximum_crop_side_px:
                group["box"] = merged_box
                group["components"].append(component)
                placed_in_existing_group = True
                break

        if not placed_in_existing_group:
            crop_groups.append({"box": component_box, "components": [component]})

    # 여백을 주고 이미지 밖으로 나가지 않게 자른다
    result_groups: list[dict[str, Any]] = []
    for group in crop_groups:
        left = max(0, group["box"][0] - crop_padding_px)
        top = max(0, group["box"][1] - crop_padding_px)
        right = min(image_width, group["box"][2] + crop_padding_px)
        bottom = min(image_height, group["box"][3] + crop_padding_px)
        result_groups.append({
            "crop_box": (left, top, right - left, bottom - top),
            "components": group["components"],
        })
    return result_groups


def render_crop_with_numbered_boxes(corrected_color_image: np.ndarray,
                                    crop_group: dict[str, Any],
                                    box_id_by_component_index: dict[int, int]) -> bytes:
    """조각 하나를 잘라 내고 각 요소에 얇은 번호 박스를 그려 PNG 바이트로 돌려준다."""
    left, top, width, height = crop_group["crop_box"]
    crop_image = corrected_color_image[top:top + height, left:left + width].copy()

    for component in crop_group["components"]:
        component_left, component_top, component_width, component_height = component.bounding_box
        box_id = box_id_by_component_index[component.component_index]
        # 조각 안에서의 좌표로 옮긴다
        x1 = component_left - left
        y1 = component_top - top
        x2 = x1 + component_width
        y2 = y1 + component_height
        cv2.rectangle(crop_image, (x1, y1), (x2, y2), (0, 90, 255), 1)
        label_position = (x1, max(10, y1 - 3))
        cv2.putText(crop_image, str(box_id), label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 90, 255), 1, cv2.LINE_AA)

    encode_succeeded, encoded_buffer = cv2.imencode(".png", crop_image)
    if not encode_succeeded:
        return b""
    return encoded_buffer.tobytes()


def judge_unsure_components(
    corrected_color_image: np.ndarray,
    components: list[StrokeComponent],
    config: dict[str, Any],
    api_key: str | None,
    model: str,
    cache_directory: Path,
    statistics: CallStatistics,
    prompt_path: Path,
) -> dict[str, Any]:
    """애매한 요소들을 클로드에게 물어보고 최종 분류를 확정한다(components 를 제자리 수정).

    돌려주는 값: {"api_skipped": bool, "judged_count": int, "error_kind": str | None,
                  "message": str | None, "call_count": int}
    """
    unsure_components = [c for c in components if c.label == LABEL_UNSURE]
    claude_config = config["claude"]

    if not unsure_components:
        return {"api_skipped": False, "judged_count": 0, "error_kind": None,
                "message": None, "call_count": 0}

    if not api_key:
        # 키가 없으면 애매한 것은 전부 "인쇄"로 둔다 — 안 지우는 쪽이 안전하다.
        for component in unsure_components:
            component.label = LABEL_PRINTED
        return {
            "api_skipped": True,
            "judged_count": 0,
            "error_kind": "no_key",
            "message": "Anthropic 키를 넣으면 더 정확하게 지울 수 있어요. (지금은 애매한 획을 남겨 두었어요.)",
            "call_count": 0,
        }

    crop_groups = group_unsure_components(
        unsure_components,
        corrected_color_image.shape[:2],
        claude_config["crop_max_side_px"],
        claude_config["crop_padding_px"],
    )

    # 요소마다 1부터 시작하는 박스 번호를 붙인다
    box_id_by_component_index: dict[int, int] = {}
    component_by_box_id: dict[int, StrokeComponent] = {}
    next_box_id = 1
    for group in crop_groups:
        for component in group["components"]:
            box_id_by_component_index[component.component_index] = next_box_id
            component_by_box_id[next_box_id] = component
            next_box_id += 1

    with open(prompt_path, "r", encoding="utf-8") as prompt_file:
        system_prompt = prompt_file.read()

    crops_per_call = claude_config["max_crops_per_call"]
    maximum_calls = claude_config["max_judge_calls_per_page"]
    confidence_threshold = claude_config["confidence_threshold"]

    judged_count = 0
    call_count = 0
    first_error_kind: str | None = None
    first_error_message: str | None = None

    for batch_start in range(0, len(crop_groups), crops_per_call):
        if call_count >= maximum_calls:
            break
        batch_groups = crop_groups[batch_start:batch_start + crops_per_call]
        image_bytes_list = [
            render_crop_with_numbered_boxes(corrected_color_image, group, box_id_by_component_index)
            for group in batch_groups
        ]
        image_bytes_list = [image_bytes for image_bytes in image_bytes_list if image_bytes]
        if not image_bytes_list:
            continue

        batch_box_ids = sorted(
            box_id_by_component_index[component.component_index]
            for group in batch_groups for component in group["components"]
        )
        user_text = (
            f"조각 {len(image_bytes_list)}장을 보낸다. 이 조각들에 들어 있는 번호 박스는 "
            f"{batch_box_ids} 이다. 모든 번호에 대해 판정하라."
        )

        try:
            response_json = call_claude_with_tool(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                image_bytes_list=image_bytes_list,
                tool_name=JUDGE_TOOL_NAME,
                tool_description=JUDGE_TOOL_DESCRIPTION,
                tool_input_schema=JUDGE_TOOL_INPUT_SCHEMA,
                claude_config=claude_config,
                cache_directory=cache_directory,
                statistics=statistics,
            )
        except ClaudeCallError as error:
            # 키 문제·한도 초과로 전체 처리가 멈추면 안 된다. 원인만 기록하고 규칙 결과로 간다.
            if first_error_kind is None:
                first_error_kind = error.error_kind
                first_error_message = error.korean_message
            break

        call_count += 1
        for item in response_json.get("items", []):
            component = component_by_box_id.get(int(item.get("box_id", -1)))
            if component is None or component.label != LABEL_UNSURE:
                continue
            reported_label = item.get("label")
            reported_confidence = float(item.get("confidence", 0.0))
            if reported_label not in (LABEL_HANDWRITING, LABEL_PRINTED, LABEL_FIGURE):
                continue
            if reported_confidence < confidence_threshold:
                component.label = LABEL_PRINTED   # 확신이 낮으면 안 지운다
            else:
                component.label = reported_label
            judged_count += 1

    # 판정받지 못하고 남은 애매한 요소는 전부 "인쇄"로 둔다(안전 쪽)
    for component in components:
        if component.label == LABEL_UNSURE:
            component.label = LABEL_PRINTED

    return {
        "api_skipped": first_error_kind is not None or call_count == 0,
        "judged_count": judged_count,
        "error_kind": first_error_kind,
        "message": first_error_message,
        "call_count": call_count,
    }
