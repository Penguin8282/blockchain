"""작업(올린 사진 한 장)의 저장과 조회.

폴더 규칙 (CLAUDE.md):
  data/raw/{시험지이름}/{강사}_{날짜}_{원래파일명}   ← 올린 원본 전부 (학습 데이터)
  data/teachers/{강사}/cleaned/{작업ID}.png          ← 결과
  data/teachers/{강사}/cleaned/{작업ID}_compare.png  ← 원본/결과 비교
  data/teachers/{강사}/jobs/{작업ID}.json            ← 작업 정보 (슬라이더, 판정 결과, 강사 수정)
  data/labels/corrections/{작업ID}.jsonl             ← 강사가 "지워/살려" 로 고친 기록 (3단계 학습 라벨)
DB 는 쓰지 않는다. 전부 파일이다.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from engine.settings import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "data"
UNCLASSIFIED_EXAM = "미분류"


def safe_name(text: str, fallback: str) -> str:
    """폴더·파일 이름으로 써도 안전한 문자열로 만든다."""
    cleaned = re.sub(r'[/\\:*?"<>|\x00-\x1f]', "", (text or "").strip())
    cleaned = cleaned.strip(". ")
    return cleaned[:80] or fallback


def new_job_id() -> str:
    """시각 + 짧은 무작위 문자열. 예: 20260921_143012_a3f9"""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


def teacher_directory(teacher: str) -> Path:
    return DATA_ROOT / "teachers" / safe_name(teacher, "강사")


def job_path(teacher: str, job_id: str) -> Path:
    return teacher_directory(teacher) / "jobs" / f"{safe_name(job_id, 'job')}.json"


def cleaned_path(teacher: str, job_id: str, suffix: str = "") -> Path:
    return teacher_directory(teacher) / "cleaned" / f"{safe_name(job_id, 'job')}{suffix}.png"


def raw_path(exam_name: str, teacher: str, original_filename: str) -> Path:
    """올린 원본을 시험지별 폴더에 저장할 경로. 같은 시험지를 푼 여러 학생 사진이 여기 모인다."""
    exam_folder = safe_name(exam_name, UNCLASSIFIED_EXAM)
    stem = safe_name(Path(original_filename).stem, "photo")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DATA_ROOT / "raw" / exam_folder / f"{safe_name(teacher, '강사')}_{stamp}_{stem}.jpg"


def save_job(job: dict[str, Any]) -> None:
    path = job_path(job["teacher"], job["job_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")


def load_job(teacher: str, job_id: str) -> dict[str, Any] | None:
    path = job_path(teacher, job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_job_any_teacher(job_id: str) -> dict[str, Any] | None:
    """다른 강사의 작업도 열어볼 수 있어야 한다(내부용)."""
    teachers_root = DATA_ROOT / "teachers"
    if not teachers_root.exists():
        return None
    for job_file in teachers_root.glob(f"*/jobs/{safe_name(job_id, 'job')}.json"):
        return json.loads(job_file.read_text(encoding="utf-8"))
    return None


def list_jobs(teacher: str | None, limit: int = 50) -> list[dict[str, Any]]:
    """최근 작업 목록. teacher 가 None 이면 모든 강사의 작업."""
    teachers_root = DATA_ROOT / "teachers"
    if not teachers_root.exists():
        return []
    pattern = f"{safe_name(teacher, '강사')}/jobs/*.json" if teacher else "*/jobs/*.json"
    jobs = []
    for job_file in teachers_root.glob(pattern):
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        jobs.append({key: job.get(key) for key in (
            "job_id", "teacher", "exam_name", "created_at", "original_filename",
            "result_url", "compare_url", "thumbnail_url", "quality_warnings", "api_skipped")})
    jobs.sort(key=lambda job: job.get("created_at") or "", reverse=True)
    return jobs[:limit]


def list_exam_names() -> list[str]:
    raw_root = DATA_ROOT / "raw"
    names = sorted(p.name for p in raw_root.iterdir() if p.is_dir()) if raw_root.exists() else []
    if UNCLASSIFIED_EXAM not in names:
        names.insert(0, UNCLASSIFIED_EXAM)
    return names


def append_correction(job: dict[str, Any], correction: dict[str, Any]) -> None:
    """강사의 "지워/살려" 결정을 학습 라벨 원천으로 기록한다. 키는 절대 들어가지 않는다."""
    corrections_dir = DATA_ROOT / "labels" / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "시각": datetime.now().isoformat(timespec="seconds"),
        "작업ID": job["job_id"],
        "강사": job["teacher"],
        "시험지": job.get("exam_name"),
        "원본": job.get("raw_path"),
        **correction,
    }
    with open(corrections_dir / f"{safe_name(job['job_id'], 'job')}.jsonl", "a", encoding="utf-8") as opened:
        opened.write(json.dumps(record, ensure_ascii=False) + "\n")


def today_usage(teacher: str) -> dict[str, Any]:
    """오늘(현지 시각 기준) 이 강사의 API 호출 수와 비용을 합산한다."""
    usage_file = DATA_ROOT / "usage" / f"{safe_name(teacher, '강사')}.jsonl"
    total = {"calls": 0, "cost_usd": 0.0, "cost_krw": 0.0}
    if not usage_file.exists():
        return total
    today = datetime.now().date()
    for line in usage_file.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            stamp = datetime.fromisoformat(record["시각"]).astimezone().date()
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if stamp != today:
            continue
        total["calls"] += 1
        total["cost_usd"] += float(record.get("비용_USD", 0.0))
        total["cost_krw"] += float(record.get("비용_KRW", 0.0))
    total["cost_usd"] = round(total["cost_usd"], 4)
    total["cost_krw"] = round(total["cost_krw"])
    return total


def relative_data_path(file_path: Path) -> str:
    """기록용 경로: 데이터 폴더의 부모(프로젝트 루트) 기준 상대 경로. 밖에 있으면 절대 경로 그대로."""
    try:
        return str(file_path.relative_to(DATA_ROOT.parent))
    except ValueError:
        return str(file_path)
