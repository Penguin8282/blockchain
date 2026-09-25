"""클로드 검수 — 결과 이미지 전체를 보내 "남은 필기 / 깨진 인쇄 / 끊긴 도형" 을 찾는다.

호출은 services/claude_client 를 통해서만 한다(키는 인자로만 흐른다).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from engine.preprocess import resize_to_max_side
from engine.settings import PROJECT_ROOT
from services.claude_client import CallStatistics, ClaudeCallError, append_usage_log, call_claude_with_tool

REVIEW_TOOL_NAME = "report_cleaning_issues"
REVIEW_TOOL_DESCRIPTION = "필기 지우기 결과 이미지에서 발견한 문제들을 보고한다."
REVIEW_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string",
                             "enum": ["remaining_handwriting", "damaged_print", "broken_figure"]},
                    "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                            "description": "[x, y, w, h] 0~1 비율"},
                    "note": {"type": "string", "description": "강사에게 보여줄 한국어 한 줄"},
                },
                "required": ["type", "box", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["issues"],
    "additionalProperties": False,
}

ISSUE_TYPE_KOREAN = {
    "remaining_handwriting": "남은 필기",
    "damaged_print": "깨진 인쇄",
    "broken_figure": "끊긴 도형",
}


def review_cleaned_image(cleaned_image: np.ndarray, api_key: str | None, model: str,
                         config: dict[str, Any], usage_log_path: Path | None) -> dict[str, Any]:
    """결과 이미지를 검수해 이슈 목록을 돌려준다. 키가 없거나 호출이 실패해도 예외를 내지 않는다."""
    claude_config = config["claude"]
    if not api_key:
        return {"issues": [], "api_skipped": True, "error_kind": "no_key",
                "message": "Anthropic 키를 넣으면 클로드가 결과를 검수해 줘요."}

    small = resize_to_max_side(cleaned_image, claude_config["page_max_side_px"])
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    ok, encoded = cv2.imencode(".png", small)
    if not ok:
        return {"issues": [], "api_skipped": True, "error_kind": "other", "message": "이미지를 보낼 형식으로 바꾸지 못했어요."}

    prompt_path = PROJECT_ROOT / "services" / "prompts" / "review.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    statistics = CallStatistics()
    try:
        response = call_claude_with_tool(
            api_key=api_key, model=model, system_prompt=system_prompt,
            user_text=f"이 결과 이미지를 검수하라. 문제는 최대 {claude_config['review_max_issues']}개까지만.",
            image_bytes_list=[encoded.tobytes()],
            tool_name=REVIEW_TOOL_NAME, tool_description=REVIEW_TOOL_DESCRIPTION,
            tool_input_schema=REVIEW_TOOL_INPUT_SCHEMA, claude_config=claude_config,
            cache_directory=PROJECT_ROOT / "data" / "api_cache", statistics=statistics,
        )
    except ClaudeCallError as error:
        return {"issues": [], "api_skipped": True, "error_kind": error.error_kind, "message": error.korean_message}

    if usage_log_path is not None:
        append_usage_log(usage_log_path, statistics, float(claude_config["usd_to_krw"]))

    issues = []
    for item in response.get("issues", [])[: claude_config["review_max_issues"]]:
        box = [float(v) for v in item.get("box", [])]
        if len(box) != 4:
            continue
        x, y, w, h = box
        x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
        w, h = max(0.0, min(1.0 - x, w)), max(0.0, min(1.0 - y, h))
        if w <= 0 or h <= 0:
            continue
        issue_type = item.get("type", "remaining_handwriting")
        issues.append({"type": issue_type, "type_ko": ISSUE_TYPE_KOREAN.get(issue_type, issue_type),
                       "box": [x, y, w, h], "note": str(item.get("note", ""))[:200]})
    return {"issues": issues, "api_skipped": False, "error_kind": None, "message": None,
            "call_count": statistics.call_count, "cache_hits": statistics.cache_hit_count,
            "cost_usd": round(statistics.total_cost_usd, 6)}
