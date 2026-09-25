"""API 키가 어디에도 새지 않는지 검증한다.

내부용이라도 동료 강사의 키가 로그에 남는 사고는 막아야 한다.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from services.claude_client import CallStatistics, UsageRecord, append_usage_log, build_cache_key
from services.key_safety import install_log_filter, mask_api_keys

FAKE_KEY = "sk-ant-api03-FAKEkey_for_test-1234567890abcdef"


def test_mask_api_keys_hides_the_key() -> None:
    """문자열 안의 키가 가려져야 한다."""
    text = f"요청 실패: x-api-key={FAKE_KEY} 가 거부됨"
    masked = mask_api_keys(text)
    assert FAKE_KEY not in masked
    assert "sk-ant-***" in masked


def test_logging_never_prints_a_key() -> None:
    """로그로 키를 흘려도 필터가 가려야 한다. (로그 전체를 grep 해 sk- 가 없어야 한다)"""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    install_log_filter()   # 핸들러를 붙인 뒤에 필터를 걸어야 이 핸들러에도 적용된다

    try:
        logging.getLogger("test").error("키가 틀렸습니다: %s", FAKE_KEY)
        logging.getLogger("test").info(f"헤더 X-Anthropic-Key: {FAKE_KEY}")
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)

    log_text = log_stream.getvalue()
    assert "sk-ant-api03" not in log_text, f"로그에 키가 남았다: {log_text}"
    assert log_text.count("sk-ant-***") >= 2


def test_usage_log_contains_no_key(tmp_path: Path) -> None:
    """사용량 기록 파일에는 시각·모델·토큰·비용만 있고 키는 없어야 한다."""
    statistics = CallStatistics()
    statistics.add(UsageRecord(model="claude-haiku-4-5", input_tokens=1200,
                               output_tokens=300, estimated_cost_usd=0.0027))

    usage_path = tmp_path / "강사1.jsonl"
    append_usage_log(usage_path, statistics, usd_to_krw=1400.0)

    file_text = usage_path.read_text(encoding="utf-8")
    assert "sk-" not in file_text
    record = json.loads(file_text.strip())
    assert set(record) == {"시각", "모델", "입력토큰", "출력토큰", "비용_USD", "비용_KRW"}


def test_cache_key_does_not_depend_on_the_teacher_key() -> None:
    """캐시 열쇠는 강사 키와 무관해야 한다 — 그래야 두 번째 강사는 호출 없이 재사용한다."""
    image_bytes = [b"\x89PNG-fake-image-bytes"]
    key_from_first_teacher = build_cache_key("claude-haiku-4-5", "판정 기준...", image_bytes, "질문")
    key_from_second_teacher = build_cache_key("claude-haiku-4-5", "판정 기준...", image_bytes, "질문")
    assert key_from_first_teacher == key_from_second_teacher

    # 반면 이미지가 다르면 열쇠도 달라야 한다
    other_key = build_cache_key("claude-haiku-4-5", "판정 기준...", [b"different"], "질문")
    assert other_key != key_from_first_teacher


def test_no_key_in_project_source_files() -> None:
    """프로젝트 소스 어디에도 진짜 키가 커밋되어 있으면 안 된다.

    "진짜 키처럼 보이는 것"의 기준: sk-ant- 뒤에 30자 이상이 붙은 문자열.
    테스트에 쓰는 가짜 키(FAKE 가 들어 있는 것)는 봐준다.
    """
    import re

    project_root = Path(__file__).resolve().parent.parent
    scanned_extensions = {".py", ".yaml", ".yml", ".txt", ".md", ".json", ".html", ".js"}
    real_looking_key_pattern = re.compile(r"sk-ant-[A-Za-z0-9_\-]{30,}")
    offending_files: list[str] = []

    for file_path in project_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix not in scanned_extensions:
            continue
        if "data" in file_path.parts or ".git" in file_path.parts:
            continue
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for match in real_looking_key_pattern.findall(content):
            if "FAKE" in match:
                continue
            offending_files.append(f"{file_path.relative_to(project_root)}: {match[:20]}...")

    assert not offending_files, f"소스에 진짜 키로 보이는 문자열이 있다: {offending_files}"
