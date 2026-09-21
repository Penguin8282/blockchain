"""이미지 처리를 프로세스 풀에서 돌린다 (동시 사용 대비).

· 풀 크기 = CPU 코어 수 - 1 (config 로 바꿀 수 있다)
· 한 강사당 동시에 처리하는 장수는 config 의 per_teacher_concurrency (기본 2). 나머지는 줄을 선다.
· 풀로 넘어가는 것: 사진 바이트, 설정, 키(그 프로세스 메모리에서만 쓰고 버린다).
· 돌아오는 것: 결과 PNG 바이트 + 요약 dict (numpy 배열은 넘기지 않는다 — 크고 느리다)
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from engine.clean import clean_image


def decode_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    """업로드된 바이트를 BGR 이미지로. HEIC(아이폰)도 읽는다."""
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        import io
        from PIL import Image
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(io.BytesIO(image_bytes)) as opened:
            rgb = np.array(opened.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:   # 어떤 형식이든 못 읽으면 None — 호출 쪽이 한국어로 안내한다
        return None


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    return buffer.tobytes() if ok else b""


def run_clean_job(image_bytes: bytes, config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """풀 프로세스 안에서 실행되는 함수. 반드시 최상위(피클 가능)여야 한다."""
    image = decode_image_bytes(image_bytes)
    if image is None:
        return {"error": "사진 파일을 읽을 수 없어요. JPG·PNG·HEIC 인지 확인해 주세요."}

    result = clean_image(
        image, config,
        api_key=options.get("api_key"),
        use_judge=options.get("use_judge", True),
        model=options.get("model"),
        label_overrides=options.get("label_overrides"),
        manual_corners=options.get("manual_corners"),
        usage_log_path=Path(options["usage_log_path"]) if options.get("usage_log_path") else None,
    )
    cleaned = result.pop("cleaned_image")
    compare = result.pop("compare_image")
    result.pop("debug_images", None)
    return {
        "cleaned_png": encode_png(cleaned),
        "compare_png": encode_png(compare) if options.get("want_compare", True) else b"",
        "size": {"width": int(cleaned.shape[1]), "height": int(cleaned.shape[0])},
        "summary": result,
    }


class WorkerPool:
    """프로세스 풀 + 강사별 동시 처리 제한."""

    def __init__(self, worker_count: int, per_teacher_concurrency: int):
        if worker_count <= 0:
            worker_count = max(1, (os.cpu_count() or 2) - 1)
        self.executor = ProcessPoolExecutor(max_workers=worker_count)
        self.worker_count = worker_count
        self.per_teacher_concurrency = per_teacher_concurrency
        self._teacher_locks: dict[str, asyncio.Semaphore] = {}

    def _lock_for(self, teacher: str) -> asyncio.Semaphore:
        if teacher not in self._teacher_locks:
            self._teacher_locks[teacher] = asyncio.Semaphore(self.per_teacher_concurrency)
        return self._teacher_locks[teacher]

    async def run(self, teacher: str, image_bytes: bytes, config: dict[str, Any],
                  options: dict[str, Any]) -> dict[str, Any]:
        """강사별 제한 안에서 풀에 작업을 넘기고 결과를 기다린다."""
        async with self._lock_for(teacher):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, run_clean_job, image_bytes, config, options)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
