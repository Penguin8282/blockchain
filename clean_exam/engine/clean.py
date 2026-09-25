"""필기 지우기 엔진의 지휘자 — 1~6 단계를 순서대로 실행한다.

CLI 로 바로 쓸 수 있다:
    python -m engine.clean sample.jpg --out sample_clean.png [--api-key sk-ant-...] [--debug]

--api-key 를 주지 않으면 판정관 없이 규칙만으로 동작한다(결과 JSON 에 api_skipped: true).
키는 인자나 환경변수로 받되, 어디에도 기록하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from engine import color_filter, finalize, graph_enhance, preprocess, stroke_filter
from engine.settings import PROJECT_ROOT, load_config
from engine.stroke_filter import LABEL_FIGURE, LABEL_HANDWRITING, LABEL_PRINTED
from services.claude_client import CallStatistics, append_usage_log


def clean_image(
    original_color_image: np.ndarray,
    config: dict[str, Any],
    api_key: str | None = None,
    use_judge: bool = True,
    model: str | None = None,
    debug: bool = False,
    label_overrides: list[dict[str, Any]] | None = None,
    manual_corners: list[list[float]] | None = None,
    usage_log_path: Path | None = None,
) -> dict[str, Any]:
    """사진 한 장에서 필기를 지우고 인쇄 글자·그래프를 진하게 만든다.

    인자:
        original_color_image: 읽어 들인 BGR 이미지
        config: load_config() 로 읽은 설정
        api_key: 강사 본인의 Anthropic 키. None 이면 판정관을 건너뛴다.
        use_judge: 강사가 화면에서 "판정관 사용"을 껐을 때 False
        model: 쓸 모델 ID. None 이면 config 의 fast_model.
        debug: True 면 중간 이미지를 결과에 담는다
        label_overrides: 판정을 강제할 영역들. [{"box": [x, y, w, h] (보정된 이미지 기준 0~1 비율),
            "label": "handwriting"|"printed_text"|"figure", "source": "judge"|"teacher"}, ...]
            · 판정관이 이미 정한 결과를 미리보기에서 **API 없이 재사용**할 때 (source=judge)
            · 강사가 화면에서 "이건 필기니까 지워 / 문제니까 살려" 로 고칠 때 (source=teacher)
            영역 안에 중심이 있는 획에 적용된다. teacher 는 어떤 규칙도 뒤집지 못하는 최종 결정이다.
        manual_corners: 강사가 직접 잡은 시험지 네 모서리 (원본 기준 0~1 비율). 자동 보정을 대신한다.

    돌려주는 값(dict):
        cleaned_image, compare_image, quality_warnings, rectify_method,
        label_counts, api_skipped, api_message, api_call_count, api_cost_usd,
        elapsed_seconds, elapsed_seconds_without_api, debug_images
    """
    overall_started_at = time.time()
    api_seconds = 0.0
    statistics = CallStatistics()

    # 1) 사진 보정
    preprocess_result = preprocess.preprocess_photo(original_color_image, config,
                                                    manual_corners=manual_corners)
    working_gray = preprocess_result.normalized_gray_image
    working_color = preprocess_result.corrected_color_image

    # 2) 색 기반 필기 제거
    working_gray, color_pen_mask, color_print_mask = color_filter.remove_color_pen(
        working_gray, working_color, config["color_filter"]
    )

    # 3) 연필/검정 필기 점수 매기기
    ink_mask = stroke_filter.binarize_ink(working_gray, config["stroke_filter"])
    labeled_image, components = stroke_filter.analyze_components(
        working_gray, ink_mask, config["stroke_filter"],
        exclude_from_reference_mask=color_print_mask,
    )
    # 색 단계에서 "컬러 인쇄"로 확정한 영역과 겹치는 획은 인쇄로 못 박는다.
    # 그레이스케일로 보면 색 글자가 중간 회색이라 그냥 두면 "흐리니까 연필"로 오인된다.
    _force_color_print_labels(labeled_image, components, color_print_mask)
    rule_label_counts = dict(Counter(component.label for component in components))

    # 판정관 결과 재사용(source=judge) — 애매한 것만 채운다. 규칙이 확신한 것은 그대로.
    image_height, image_width = working_gray.shape[:2]
    _apply_label_overrides(components, label_overrides or [], image_width, image_height,
                           only_source="judge")

    debug_images = dict(preprocess_result.debug_images) if debug else {}
    if debug:
        debug_images["04_규칙분류"] = stroke_filter.render_classification_debug_image(
            working_gray, labeled_image, components
        )

    # 4) 클로드 판정관 — 애매한 것만
    if use_judge:
        api_started_at = time.time()
        judge_result = _run_judge(working_color, components, config, api_key, model, statistics)
        api_seconds = time.time() - api_started_at
    else:
        for component in components:
            if component.label == stroke_filter.LABEL_UNSURE:
                component.label = LABEL_PRINTED
        judge_result = {"api_skipped": True, "judged_count": 0, "error_kind": "disabled",
                        "message": "판정관을 끈 상태로 처리했어요.", "call_count": 0}

    # 강사가 직접 정한 것(source=teacher) — 모든 규칙과 판정관보다 우선한다
    _apply_label_overrides(components, label_overrides or [], image_width, image_height,
                           only_source="teacher")

    # 5) 그래프·도형 보호 및 강화
    components = graph_enhance.mark_figure_components(
        labeled_image, components, ink_mask, config["graph_enhance"]
    )
    label_masks = graph_enhance.build_label_masks(labeled_image, components, working_gray.shape[:2])
    strengthened_figure_mask = graph_enhance.strengthen_figures(
        label_masks[LABEL_FIGURE], config["graph_enhance"]
    )
    # 그래프로 보호한 자리는 절대 지우지 않는다(겹치면 보호가 이긴다)
    handwriting_mask = label_masks[LABEL_HANDWRITING].copy()
    handwriting_mask[strengthened_figure_mask > 0] = 0
    # 강사가 "지워"라고 한 획은 어떤 보호도 무시하고 지운다
    for component in components:
        if component.is_manual_override and component.label == LABEL_HANDWRITING:
            left, top, width, height = component.bounding_box
            box_slice = (slice(top, top + height), slice(left, left + width))
            pixels = labeled_image[box_slice] == component.component_index
            handwriting_mask[box_slice][pixels] = 1
            strengthened_figure_mask[box_slice][pixels] = 0

    if debug:
        debug_images["05_최종분류"] = stroke_filter.render_classification_debug_image(
            working_gray, labeled_image, components
        )
        debug_images["06_지울곳"] = handwriting_mask * 255
        debug_images["07_그래프"] = strengthened_figure_mask * 255

    # "지키되 강조는 안 하는" 인쇄 부분(가는 선·회색으로 채운 면)의 자리를 모아 둔다.
    thin_line_mask = np.zeros(working_gray.shape[:2], dtype=np.uint8)
    for component in components:
        if not (component.is_protected_thin_line or component.is_shaded_area):
            continue
        left, top, width, height = component.bounding_box
        box_slice = (slice(top, top + height), slice(left, left + width))
        thin_line_mask[box_slice][labeled_image[box_slice] == component.component_index] = 1

    # 조명 정규화 단계가 찾은 회색 채움 면(색칠된 반원·표 칸)도 "지키되 강조 안 함"에 넣는다
    if preprocess_result.shaded_area_mask is not None:
        thin_line_mask[preprocess_result.shaded_area_mask > 0] = 1
        # 채움 면 위를 지나는 획 하나 때문에 면 전체가 새까맣게 칠해지지 않게 한다
        strengthened_figure_mask[preprocess_result.shaded_area_mask > 0] = 0

    # 6) 최종 마무리
    erased_image = finalize.erase_handwriting(
        working_gray, handwriting_mask, config["color_filter"]["inpaint_radius"]
    )
    cleaned_image = finalize.apply_final_contrast(
        erased_image, strengthened_figure_mask, config["finalize"],
        color_print_mask=color_print_mask, corrected_color_image=working_color,
        thin_line_mask=thin_line_mask,
    )
    # 처리를 위해 키웠던 사진은 원래 크기로 되돌린다(원본과 같은 해상도가 약속이다).
    working_scale = preprocess_result.working_scale
    if working_scale > 1.0:
        restored_size = (int(round(cleaned_image.shape[1] / working_scale)),
                         int(round(cleaned_image.shape[0] / working_scale)))
        cleaned_image = cv2.resize(cleaned_image, restored_size, interpolation=cv2.INTER_AREA)
        working_color = cv2.resize(working_color, restored_size, interpolation=cv2.INTER_AREA)
    compare_image = finalize.make_comparison_image(working_color, cleaned_image)

    # 판정관이 정한 것을 보정된 이미지 기준 0~1 비율 사각형으로 돌려준다 (미리보기 재사용용)
    component_by_index = {component.component_index: component for component in components}
    judge_overrides: list[dict[str, Any]] = []
    for judgement in judge_result.get("judgements", []):
        component = component_by_index.get(judgement["component_index"])
        if component is None:
            continue
        left, top, width, height = component.bounding_box
        judge_overrides.append({
            "box": [left / image_width, top / image_height, width / image_width, height / image_height],
            "label": judgement["label"],
            "source": "judge",
        })

    if usage_log_path is not None:
        append_usage_log(usage_log_path, statistics, float(config["claude"]["usd_to_krw"]))

    elapsed_seconds = time.time() - overall_started_at
    return {
        "cleaned_image": cleaned_image,
        "compare_image": compare_image,
        "quality_warnings": preprocess_result.quality_warnings,
        "rectify_method": preprocess_result.rectify_method,
        "working_scale": working_scale,
        "judge_overrides": judge_overrides,
        # 어두운 여백을 잘라냈다면 그 사각형(원본 좌표 기준). 결과를 원본 위에 겹칠 때 쓴다.
        "crop_box": tuple(int(round(value / working_scale)) for value in preprocess_result.crop_box)
        if preprocess_result.crop_box else None,
        "rule_label_counts": rule_label_counts,
        "label_counts": dict(Counter(component.label for component in components)),
        "color_pen_pixel_ratio": float(color_pen_mask.mean()),
        "api_skipped": judge_result["api_skipped"],
        "api_error_kind": judge_result["error_kind"],
        "api_message": judge_result["message"],
        "api_call_count": judge_result["call_count"],
        "api_judged_count": judge_result["judged_count"],
        "api_cache_hits": statistics.cache_hit_count,
        "api_cost_usd": round(statistics.total_cost_usd, 6),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_seconds_without_api": round(elapsed_seconds - api_seconds, 2),
        "debug_images": debug_images,
    }


def _apply_label_overrides(components: list[stroke_filter.StrokeComponent],
                           overrides: list[dict[str, Any]], image_width: int, image_height: int,
                           only_source: str) -> None:
    """영역 안에 중심이 있는 획의 판정을 강제한다(제자리 수정).

    source 가 "judge" 면 애매한 획만 채운다(판정관은 애매한 것만 봤으니까).
    source 가 "teacher" 면 무엇이든 덮어쓰고 수동 지정 표시를 남긴다.
    """
    selected = [override for override in overrides if override.get("source", "judge") == only_source]
    if not selected:
        return
    for component in components:
        left, top, width, height = component.bounding_box
        center_x = (left + width / 2.0) / image_width
        center_y = (top + height / 2.0) / image_height
        for override in selected:
            box_x, box_y, box_w, box_h = override["box"]
            if not (box_x <= center_x <= box_x + box_w and box_y <= center_y <= box_y + box_h):
                continue
            if only_source == "judge" and component.label != stroke_filter.LABEL_UNSURE:
                continue
            component.label = override["label"]
            if only_source == "teacher":
                component.is_manual_override = True
                component.is_protected_thin_line = False
                component.is_shaded_area = False
                component.is_ring_glued_blob = False
            break


def _force_color_print_labels(labeled_image: np.ndarray,
                              components: list[stroke_filter.StrokeComponent],
                              color_print_mask: np.ndarray,
                              minimum_overlap_ratio: float = 0.5) -> None:
    """컬러 인쇄로 확정된 영역과 충분히 겹치는 획을 "인쇄"로 확정한다(제자리 수정)."""
    if color_print_mask.sum() == 0:
        return
    for component in components:
        left, top, width, height = component.bounding_box
        box_slice = (slice(top, top + height), slice(left, left + width))
        component_pixels = labeled_image[box_slice] == component.component_index
        if not component_pixels.any():
            continue
        overlap_ratio = float((color_print_mask[box_slice][component_pixels] > 0).mean())
        if overlap_ratio >= minimum_overlap_ratio:
            component.label = LABEL_PRINTED


def _run_judge(working_color: np.ndarray, components: list[stroke_filter.StrokeComponent],
               config: dict[str, Any], api_key: str | None, model: str | None,
               statistics: CallStatistics) -> dict[str, Any]:
    """판정관을 부른다. anthropic 패키지가 없어도 전체 처리가 멈추지 않게 감싸 둔다."""
    from engine.claude_judge import judge_unsure_components

    return judge_unsure_components(
        corrected_color_image=working_color,
        components=components,
        config=config,
        api_key=api_key,
        model=model or config["claude"]["fast_model"],
        cache_directory=PROJECT_ROOT / "data" / "api_cache",
        statistics=statistics,
        prompt_path=PROJECT_ROOT / "services" / "prompts" / "judge.txt",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI 인자 정의."""
    parser = argparse.ArgumentParser(
        description="시험지 사진에서 학생 필기를 지우고 인쇄 글자·그래프만 남긴다."
    )
    parser.add_argument("input_path", help="처리할 시험지 사진 경로")
    parser.add_argument("--out", required=True, help="결과 PNG 를 저장할 경로")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic 키. 없으면 환경변수 ANTHROPIC_API_KEY 를 보고, "
                             "그것도 없으면 판정관 없이 규칙만으로 처리한다.")
    parser.add_argument("--no-judge", action="store_true", help="키가 있어도 판정관을 쓰지 않는다")
    parser.add_argument("--quality", action="store_true", help="고품질 모델을 쓴다(비용↑)")
    parser.add_argument("--debug", action="store_true", help="중간 이미지를 debug/ 에 저장한다")
    parser.add_argument("--config", default=None, help="다른 config.yaml 경로")
    return parser


def main() -> int:
    """CLI 진입점. 성공하면 0 을 돌려준다."""
    arguments = build_argument_parser().parse_args()
    config = load_config(arguments.config)

    original_color_image = cv2.imread(arguments.input_path)
    if original_color_image is None:
        print(f"[오류] 사진을 열 수 없어요: {arguments.input_path}")
        return 1

    api_key = arguments.api_key or os.environ.get("ANTHROPIC_API_KEY")
    model = config["claude"]["quality_model"] if arguments.quality else config["claude"]["fast_model"]

    result = clean_image(
        original_color_image, config,
        api_key=api_key, use_judge=not arguments.no_judge, model=model, debug=arguments.debug,
    )

    output_path = Path(arguments.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result["cleaned_image"])

    compare_path = output_path.parent / f"compare_{output_path.stem}.png"
    cv2.imwrite(str(compare_path), result["compare_image"])

    if arguments.debug:
        debug_directory = output_path.parent / "debug"
        debug_directory.mkdir(parents=True, exist_ok=True)
        for debug_name, debug_image in result["debug_images"].items():
            cv2.imwrite(str(debug_directory / f"{debug_name}.png"), debug_image)

    printable_result = {key: value for key, value in result.items()
                        if key not in ("cleaned_image", "compare_image", "debug_images")}
    print(json.dumps(printable_result, ensure_ascii=False, indent=2))
    print(f"\n결과:      {output_path}")
    print(f"비교 이미지: {compare_path}")
    if arguments.debug:
        print(f"디버그:     {output_path.parent / 'debug'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
