"""성능 보드 — 합성 시험지 시나리오별로 엔진의 정확도를 한 표로 보여준다.

    python -m train.benchmark              # 전체 시나리오
    python -m train.benchmark --seeds 8    # 시나리오당 8장

실제 사진이 드러낸 실패 유형(음영 도형, 저해상도, 흑백 저장, 화면 캡처)마다 따로 재서,
어떤 유형이 약한지 한눈에 보게 한다. 4단계에서 rule / model / hybrid 비교표의 바탕이 된다.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from engine.clean import clean_image
from engine.settings import load_config
from tests.synthetic_exam import SCENARIOS, make_synthetic_page


def measure_one(page, config) -> dict[str, float] | None:
    """합성 시험지 한 장의 종류별 제거율·손실률을 잰다. 크기가 바뀌면 None."""
    started_at = time.time()
    result = clean_image(page.written_page_image, config, api_key=None, debug=True)
    elapsed = time.time() - started_at
    cleaned = result["cleaned_image"]
    figure_mask = result["debug_images"]["07_그래프"] > 0

    # 엔진이 어두운 여백을 잘라냈으면 정답도 같은 사각형으로 잘라 정렬한다
    written_page = page.written_page_image
    truth = {"hand": page.handwriting_mask, "text": page.printed_text_mask, "fig": page.figure_mask,
             "cprint": page.color_print_mask, "graph": page.student_graph_mask}
    if result["crop_box"] is not None:
        left, top, width, height = result["crop_box"]
        written_page = written_page[top:top + height, left:left + width]
        truth = {name: mask[top:top + height, left:left + width] for name, mask in truth.items()}
    if cleaned.shape[:2] != written_page.shape[:2]:
        return None
    # 도형 마스크는 처리 좌표(확대·잘림 후) 기준이라 결과 크기에 맞춘다
    if figure_mask.shape[:2] != cleaned.shape[:2]:
        figure_mask = cv2.resize(figure_mask.astype(np.uint8), (cleaned.shape[1], cleaned.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0

    written_gray = cv2.cvtColor(written_page, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(written_page, cv2.COLOR_BGR2HSV)[:, :, 1]
    white = cleaned >= 250

    handwriting = (truth["hand"] > 0) & (written_gray < 215)
    pencil = handwriting & (saturation <= 60)
    color_pen = handwriting & (saturation > 60)
    printed = (truth["text"] > 0) & (truth["hand"] == 0) & (written_gray < 180)
    figure = (truth["fig"] > 0) & (truth["hand"] == 0) & (written_gray < 200)
    color_print = (truth["cprint"] > 0) & (truth["hand"] == 0) & (saturation > 60)
    student_graph = (truth["graph"] > 0) & (written_gray < 215)

    def ratio(mask, condition):
        return float(condition[mask].mean()) if mask.any() else float("nan")

    return {
        "연필제거": ratio(pencil, white),
        "색펜제거": ratio(color_pen, white),
        "인쇄손실": ratio(printed, white),
        "도형손실": ratio(figure, white),
        "컬러인쇄손실": ratio(color_print, white),
        "손그림강조": ratio(student_graph, figure_mask),
        "초": elapsed,
    }


def run_benchmark(seed_count: int, difficulty: str, scenarios: list[str]) -> dict[str, dict[str, float]]:
    """시나리오마다 seed_count 장씩 재서 평균을 낸다."""
    config = load_config()
    board: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        rows = []
        for seed in range(seed_count):
            page = make_synthetic_page(seed=seed, difficulty=difficulty, scenario=scenario)
            measurement = measure_one(page, config)
            if measurement is not None:
                rows.append(measurement)
        if not rows:
            continue
        board[scenario] = {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}
    return board


def print_board(board: dict[str, dict[str, float]]) -> None:
    """표로 출력한다. ↑ 는 높을수록 좋고 ↓ 는 낮을수록 좋다."""
    columns = ["연필제거", "색펜제거", "인쇄손실", "도형손실", "컬러인쇄손실", "손그림강조", "초"]
    arrows = {"연필제거": "↑", "색펜제거": "↑", "인쇄손실": "↓", "도형손실": "↓",
              "컬러인쇄손실": "↓", "손그림강조": "↓", "초": ""}
    header = f"{'시나리오':<10}" + "".join(f"{column + arrows[column]:>12}" for column in columns)
    print(header)
    print("-" * len(header))
    for scenario, values in board.items():
        cells = []
        for column in columns:
            value = values[column]
            cells.append(f"{value:12.2f}" if column == "초" else f"{value * 100:11.1f}%")
        print(f"{scenario:<10}" + "".join(cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--difficulty", default="보통")
    parser.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    arguments = parser.parse_args()
    print_board(run_benchmark(arguments.seeds, arguments.difficulty, arguments.scenarios))
