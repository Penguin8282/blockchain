"""깔끔 시험지 웹 서버 — 모든 라우트.

키 흐름: 브라우저 localStorage → 헤더 X-Anthropic-Key → get_identity → (요청 처리 중 메모리) → 풀 프로세스 → Anthropic.
디스크·로그·캐시·오류 메시지 어디에도 남지 않는다.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import jobs as job_store
from app.auth_key import RequestIdentity, get_identity
from app.params import SLIDER_DEFAULTS, apply_sliders
from app.review import review_cleaned_image
from app.workers import WorkerPool, decode_image_bytes
from engine.preprocess import resize_to_max_side
from engine.settings import PROJECT_ROOT, load_config
from services.claude_client import check_api_key
from services.key_safety import install_log_filter, mask_api_keys

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger("clean_exam")


def install_key_masking_everywhere() -> None:
    """루트뿐 아니라 uvicorn 로거들에도 키 마스킹 필터를 건다."""
    install_log_filter()
    from services.key_safety import ApiKeyMaskingFilter
    masking = ApiKeyMaskingFilter()
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "clean_exam"):
        target = logging.getLogger(name)
        target.addFilter(masking)
        for handler in target.handlers:
            handler.addFilter(masking)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 뜰 때 프로세스 풀을 만들고, 내려갈 때 정리한다."""
    install_key_masking_everywhere()
    config = load_config()
    app.state.pool = WorkerPool(config["server"]["worker_count"], config["server"]["per_teacher_concurrency"])
    logger.info("처리 프로세스 %d개로 시작", app.state.pool.worker_count)
    yield
    app.state.pool.shutdown()


app = FastAPI(title="깔끔 시험지", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    """예상 못 한 오류도 500 대신 한국어 메시지로. 키는 절대 섞이지 않게 마스킹한다."""
    logger.exception("처리 중 오류: %s", mask_api_keys(repr(error)))
    return JSONResponse(status_code=500, content={
        "detail": "처리 중 문제가 생겼어요. 사진을 다시 올려 보시고, 계속 그러면 관리자에게 알려 주세요.",
        "error": mask_api_keys(type(error).__name__),
    })


# ───────────────────────────── 화면·설정 ─────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
async def get_public_config() -> dict[str, Any]:
    """브라우저가 처음 접속할 때 필요한 것들. 비밀은 없다."""
    config = load_config()
    return {
        "teachers": config["teachers"],
        "slider_defaults": SLIDER_DEFAULTS,
        "models": {"fast": config["claude"]["fast_model"], "quality": config["claude"]["quality_model"]},
        "usd_to_krw": config["claude"]["usd_to_krw"],
        "max_upload_mb": config["server"]["max_upload_mb"],
    }


@app.get("/api/exams")
async def get_exam_names() -> dict[str, Any]:
    return {"exams": job_store.list_exam_names()}


@app.post("/api/key/check")
async def key_check(identity: RequestIdentity = Depends(get_identity)) -> dict[str, Any]:
    """키가 쓸 만한지 아주 싼 호출 1회로 확인한다. 확인 후 키는 저장하지 않는다."""
    if not identity.api_key:
        return {"ok": False, "status": "missing", "message": "키를 넣어 주세요. sk-ant- 로 시작해요."}
    return check_api_key(identity.api_key, identity.model, load_config()["claude"])


@app.get("/api/usage")
async def get_usage(identity: RequestIdentity = Depends(get_identity)) -> dict[str, Any]:
    return job_store.today_usage(identity.teacher)


@app.get("/api/jobs")
async def get_jobs(all: int = 0, identity: RequestIdentity = Depends(get_identity)) -> dict[str, Any]:
    """최근 작업. all=1 이면 다른 강사 작업까지."""
    return {"jobs": job_store.list_jobs(None if all else identity.teacher)}


# ───────────────────────────── 처리 ─────────────────────────────
def parse_json_form(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def usage_log_path_for(teacher: str) -> str:
    return str(job_store.DATA_ROOT / "usage" / f"{job_store.safe_name(teacher, '강사')}.jsonl")


def public_url(teacher: str, filename: str) -> str:
    return f"/files/{job_store.safe_name(teacher, '강사')}/{filename}"


def messages_for(summary: dict[str, Any]) -> list[str]:
    """엔진 요약을 강사에게 보여줄 한국어 문장들로."""
    messages = list(summary.get("quality_warnings") or [])
    if summary.get("rectify_method") in ("none", "deskew"):
        pass
    elif summary.get("rectify_method") == "cropped":
        messages.append("어두운 여백을 잘라내고 처리했어요.")
    if summary.get("api_skipped") and summary.get("api_message"):
        messages.append(summary["api_message"])
    return messages


async def process_and_store(request: Request, identity: RequestIdentity, image_bytes: bytes,
                            job: dict[str, Any], config: dict[str, Any], use_judge: bool,
                            want_compare: bool = True) -> dict[str, Any]:
    """풀에서 처리하고 결과 파일·작업 정보를 저장한다. 처리(/api/clean)와 재처리(/api/fix)가 함께 쓴다."""
    pool: WorkerPool = request.app.state.pool
    overrides = list(job.get("judge_overrides") or []) + list(job.get("teacher_overrides") or [])
    result = await pool.run(identity.teacher, image_bytes, config, {
        "api_key": identity.api_key if use_judge else None,
        "use_judge": use_judge,
        "model": identity.model,
        "label_overrides": overrides,
        "manual_corners": job.get("manual_corners"),
        "usage_log_path": usage_log_path_for(identity.teacher),
        "want_compare": want_compare,
    })
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    cleaned_file = job_store.cleaned_path(job["teacher"], job["job_id"])
    cleaned_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_file.write_bytes(result["cleaned_png"])
    if result["compare_png"]:
        job_store.cleaned_path(job["teacher"], job["job_id"], "_compare").write_bytes(result["compare_png"])

    summary = result["summary"]
    # 판정관이 새로 판정한 것이 있으면 기억해 둔다(다음 미리보기·수정에서 API 없이 재사용)
    if summary.get("judge_overrides"):
        job["judge_overrides"] = summary["judge_overrides"]
    job.update({
        "result_url": public_url(job["teacher"], cleaned_file.name),
        "compare_url": public_url(job["teacher"], cleaned_file.stem + "_compare.png"),
        "size": result["size"],
        "use_judge": use_judge,
        "quality_warnings": summary.get("quality_warnings", []),
        "api_skipped": summary.get("api_skipped"),
        "rectify_method": summary.get("rectify_method"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    job_store.save_job(job)
    return {
        "job_id": job["job_id"],
        "result_url": job["result_url"],
        "compare_url": job["compare_url"],
        "original_url": job["original_url"],
        "size": result["size"],
        "messages": messages_for(summary),
        "quality_warnings": summary.get("quality_warnings", []),
        "rectify_method": summary.get("rectify_method"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "api_skipped": summary.get("api_skipped"),
        "api_error_kind": summary.get("api_error_kind"),
        "api_message": summary.get("api_message"),
        "api_call_count": summary.get("api_call_count", 0),
        "api_judged_count": summary.get("api_judged_count", 0),
        "api_cost_usd": summary.get("api_cost_usd", 0.0),
        "label_counts": summary.get("label_counts", {}),
        "judged_boxes": len(job.get("judge_overrides") or []),
    }


@app.post("/api/clean")
async def clean(request: Request,
                file: UploadFile = File(...),
                exam_name: str = Form(default=""),
                sliders: str | None = Form(default=None),
                use_judge: int = Form(default=1),
                manual_corners: str | None = Form(default=None),
                identity: RequestIdentity = Depends(get_identity)) -> dict[str, Any]:
    """사진 한 장을 올려 처리한다. 키가 없어도 200 으로 답한다(api_skipped)."""
    config = load_config()
    image_bytes = await file.read()
    if len(image_bytes) > config["server"]["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"사진이 너무 커요. {config['server']['max_upload_mb']}MB 이하로 올려 주세요.")
    image = decode_image_bytes(image_bytes)
    if image is None:
        raise HTTPException(status_code=400, detail="사진 파일을 읽을 수 없어요. JPG·PNG·HEIC 인지 확인해 주세요.")

    # 큰 사진은 서버에서 먼저 줄인다. 원본은 시험지별 폴더에 그대로(줄인 것) 저장한다.
    image = resize_to_max_side(image, config["server"]["upload_max_side_px"])
    ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    stored_bytes = jpeg.tobytes()

    exam = job_store.safe_name(exam_name, job_store.UNCLASSIFIED_EXAM)
    raw_file = job_store.raw_path(exam, identity.teacher, file.filename or "photo.jpg")
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_bytes(stored_bytes)

    job_id = job_store.new_job_id()
    # 원본도 강사 폴더에서 바로 볼 수 있게 복사해 둔다(비교 슬라이더용)
    original_copy = job_store.cleaned_path(identity.teacher, job_id, "_original")
    original_copy.parent.mkdir(parents=True, exist_ok=True)
    original_copy.write_bytes(stored_bytes)

    job = {
        "job_id": job_id,
        "teacher": identity.teacher,
        "exam_name": exam,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "original_filename": file.filename,
        "raw_path": job_store.relative_data_path(raw_file),
        "original_url": public_url(identity.teacher, original_copy.name),
        "thumbnail_url": public_url(identity.teacher, original_copy.name),
        "sliders": {**SLIDER_DEFAULTS, **(parse_json_form(sliders, {}) or {})},
        "manual_corners": parse_json_form(manual_corners, None),
        "judge_overrides": [],
        "teacher_overrides": [],
    }
    adjusted = apply_sliders(config, job["sliders"])
    return await process_and_store(request, identity, stored_bytes, job, adjusted, bool(use_judge))


def load_job_or_404(job_id: str) -> dict[str, Any]:
    job = job_store.find_job_any_teacher(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="그 작업을 찾을 수 없어요. 사진을 다시 올려 주세요.")
    return job


def load_job_original(job: dict[str, Any]) -> bytes:
    original_file = job_store.cleaned_path(job["teacher"], job["job_id"], "_original")
    if not original_file.exists():
        raise HTTPException(status_code=404, detail="원본 사진 파일이 없어요. 사진을 다시 올려 주세요.")
    return original_file.read_bytes()


@app.post("/api/preview")
async def preview(request: Request,
                  job_id: str = Form(...),
                  sliders: str | None = Form(default=None),
                  use_judge: int = Form(default=1),
                  identity: RequestIdentity = Depends(get_identity)) -> dict[str, Any]:
    """슬라이더를 움직였을 때의 빠른 미리보기. **API 를 다시 부르지 않고** 저장된 판정 결과를 재사용한다."""
    config = load_config()
    job = load_job_or_404(job_id)
    job["sliders"] = {**SLIDER_DEFAULTS, **(parse_json_form(sliders, {}) or {})}
    adjusted = apply_sliders(config, job["sliders"])
    preview_side = config["server"]["preview_max_side_px"]
    adjusted["preprocess"]["work_max_side_px"] = preview_side
    adjusted["preprocess"]["work_min_side_px"] = min(adjusted["preprocess"]["work_min_side_px"], preview_side)

    pool: WorkerPool = request.app.state.pool
    overrides = list(job.get("judge_overrides") or []) if use_judge else []
    overrides += list(job.get("teacher_overrides") or [])
    result = await pool.run(identity.teacher, load_job_original(job), adjusted, {
        "api_key": None, "use_judge": False, "label_overrides": overrides,
        "manual_corners": job.get("manual_corners"), "want_compare": False,
    })
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    job_store.save_job(job)   # 슬라이더 값을 기억해 둔다
    import base64
    return {"preview_png_base64": base64.b64encode(result["cleaned_png"]).decode("ascii"),
            "size": result["size"], "elapsed_seconds": result["summary"].get("elapsed_seconds")}


@app.post("/api/render")
async def render_full(request: Request,
                      job_id: str = Form(...),
                      sliders: str | None = Form(default=None),
                      use_judge: int = Form(default=1),
                      identity: RequestIdentity = Depends(get_identity)) -> dict[str, Any]:
    """지금 슬라이더로 **원래 크기**로 다시 만들어 결과 파일을 갱신한다(내려받기 직전에 쓴다)."""
    config = load_config()
    job = load_job_or_404(job_id)
    job["sliders"] = {**SLIDER_DEFAULTS, **(parse_json_form(sliders, {}) or {})}
    adjusted = apply_sliders(config, job["sliders"])
    # 판정관은 다시 부르지 않는다 — 저장된 판정을 재사용한다. use_judge=0 이면 판정 없이.
    if not use_judge:
        job = {**job, "judge_overrides": []}
    identity_without_key = RequestIdentity(identity.teacher, None, identity.model_choice)
    return await process_and_store(request, identity_without_key, load_job_original(job), job, adjusted, use_judge=False)


@app.post("/api/review")
async def review(job_id: str = Form(...), identity: RequestIdentity = Depends(get_identity)) -> dict[str, Any]:
    """결과 이미지를 클로드에게 보내 남은 필기·깨진 인쇄·끊긴 도형을 찾는다."""
    job = load_job_or_404(job_id)
    cleaned_file = job_store.cleaned_path(job["teacher"], job["job_id"])
    if not cleaned_file.exists():
        raise HTTPException(status_code=404, detail="결과 파일이 없어요. 사진을 다시 올려 주세요.")
    cleaned = cv2.imread(str(cleaned_file))
    outcome = review_cleaned_image(cleaned, identity.api_key, identity.model, load_config(),
                                   Path(usage_log_path_for(identity.teacher)))
    job["last_review"] = {"at": datetime.now().isoformat(timespec="seconds"), "issues": outcome["issues"]}
    job_store.save_job(job)
    return outcome


@app.post("/api/fix")
async def fix(request: Request,
              job_id: str = Form(...),
              box: str = Form(...),
              decision: str = Form(...),
              identity: RequestIdentity = Depends(get_identity)) -> dict[str, Any]:
    """검수 박스 하나를 "지워"(erase) 또는 "살려"(keep) 로 고쳐 다시 만든다. 기록은 학습 라벨이 된다."""
    config = load_config()
    job = load_job_or_404(job_id)
    parsed_box = parse_json_form(box, None)
    if not (isinstance(parsed_box, list) and len(parsed_box) == 4):
        raise HTTPException(status_code=400, detail="영역 정보가 이상해요. 박스를 다시 눌러 주세요.")
    if decision not in ("erase", "keep"):
        raise HTTPException(status_code=400, detail="지워/살려 중 하나를 골라 주세요.")

    label = "handwriting" if decision == "erase" else "printed_text"
    override = {"box": [float(v) for v in parsed_box], "label": label, "source": "teacher"}
    job.setdefault("teacher_overrides", []).append(override)
    job_store.append_correction(job, {"영역": override["box"], "결정": "지워" if decision == "erase" else "살려", "라벨": label})

    adjusted = apply_sliders(config, job.get("sliders"))
    identity_without_key = RequestIdentity(identity.teacher, None, identity.model_choice)
    outcome = await process_and_store(request, identity_without_key, load_job_original(job), job, adjusted, use_judge=False)
    outcome["teacher_overrides"] = len(job["teacher_overrides"])
    return outcome


@app.post("/api/pdf")
async def make_pdf(job_ids: str = Form(...), identity: RequestIdentity = Depends(get_identity)) -> Response:
    """결과 여러 장을 PDF 한 파일로. (한 양식으로 통일하는 A4 배치는 2-B 단계에서 한다)"""
    from PIL import Image
    ids = parse_json_form(job_ids, [])
    pages = []
    for job_id in ids:
        job = job_store.find_job_any_teacher(str(job_id))
        if job is None:
            continue
        cleaned_file = job_store.cleaned_path(job["teacher"], job["job_id"])
        if cleaned_file.exists():
            pages.append(Image.open(cleaned_file).convert("L"))
    if not pages:
        raise HTTPException(status_code=404, detail="PDF 로 만들 결과가 없어요.")
    buffer = io.BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:], resolution=300)
    # HTTP 헤더는 latin-1 만 허용된다. 한글 파일명은 RFC 5987 대로 percent-encoding 해서 보낸다.
    filename = quote(f"{datetime.now().strftime('%Y%m%d')}_정리본.pdf")
    return Response(content=buffer.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"})


# ───────────────────────────── 파일 ─────────────────────────────
@app.get("/files/{teacher}/{filename}")
async def serve_file(teacher: str, filename: str) -> FileResponse:
    """결과·원본 파일. 내부용이라 접근 제한이 없다(다른 강사 결과도 볼 수 있어야 한다)."""
    safe_teacher = job_store.safe_name(teacher, "강사")
    safe_file = job_store.safe_name(filename, "")
    if not safe_file or not safe_file.endswith((".png", ".jpg")):
        raise HTTPException(status_code=404, detail="없는 파일이에요.")
    path = job_store.teacher_directory(safe_teacher) / "cleaned" / safe_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="없는 파일이에요.")
    return FileResponse(str(path))
