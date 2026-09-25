"""2단계 웹 서버 테스트. 전부 mock — 키 없이, 인터넷 없이 돌아간다.

확인하는 것:
  · 키가 없어도, 가짜 키여도 /api/clean 은 200 + 규칙 결과 + 한국어 안내 (500 금지)
  · 서버 로그 어디에도 sk-ant- 키가 남지 않는다
  · 검수(mock) → 문제 박스 → 지워/살려 → 교정 기록 파일
  · PDF 내려받기
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from urllib.parse import quote

import cv2
import pytest
from fastapi.testclient import TestClient

from app import jobs as job_store
from tests.synthetic_exam import make_synthetic_page

FAKE_KEY = "sk-ant-api03-FAKEKEYFAKEKEYFAKEKEYFAKEKEYFAKEKEY-abcdefgh"
TEACHER_HEADERS = {"X-Teacher": quote("강사1"), "X-Model": "fast"}
KEYED_HEADERS = {**TEACHER_HEADERS, "X-Anthropic-Key": FAKE_KEY}


@pytest.fixture(scope="module")
def photo_bytes() -> bytes:
    """합성 시험지 사진 한 장을 JPG 바이트로."""
    page = make_synthetic_page(seed=11)
    ok, encoded = cv2.imencode(".jpg", page.photo_image)
    assert ok
    return encoded.tobytes()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """데이터 폴더를 임시 폴더로 돌린 서버 + 로그 캡처."""
    monkeypatch.setattr(job_store, "DATA_ROOT", tmp_path / "data")
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    previous_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    from app.main import app
    with TestClient(app) as test_client:
        test_client.log_stream = log_stream
        yield test_client
    root_logger.removeHandler(handler)
    root_logger.setLevel(previous_level)


def upload(client: TestClient, photo_bytes: bytes, headers: dict[str, str]) -> dict:
    """/api/clean 호출을 짧게."""
    response = client.post("/api/clean", headers=headers,
                           files={"file": ("시험지.jpg", photo_bytes, "image/jpeg")},
                           data={"exam_name": "테스트", "sliders": "{}", "use_judge": "1"})
    assert response.status_code == 200, response.text
    return response.json()


def test_index_and_config(client: TestClient) -> None:
    """첫 화면과 설정 정보가 나온다."""
    assert "깔끔 시험지" in client.get("/").text
    config = client.get("/api/config").json()
    assert "강사1" in config["teachers"]
    assert set(config["slider_defaults"]) == {"remove_strength", "color_sensitivity", "graph_thickness", "text_darkness"}


def test_clean_without_key_returns_rule_only_result(client: TestClient, photo_bytes: bytes) -> None:
    """키 없음 → 200, api_skipped, 한국어 안내, 결과 파일 존재."""
    result = upload(client, photo_bytes, TEACHER_HEADERS)
    assert result["api_skipped"] is True
    assert result["api_call_count"] == 0
    assert any("키를 넣으면" in message for message in result["messages"])
    assert client.get(result["result_url"]).status_code == 200
    assert client.get(result["original_url"]).status_code == 200


def test_clean_with_invalid_key_never_500(client: TestClient, photo_bytes: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    """가짜 키 → 규칙 결과 + '키가 올바르지 않아요'. 판정관 호출은 mock 으로 401 을 흉내 낸다."""
    from engine import claude_judge
    from services.claude_client import ClaudeCallError

    def fake_call(*args, **kwargs):
        # services/claude_client.py 가 401 을 받으면 이렇게 바꿔 던진다
        raise ClaudeCallError("auth", "Anthropic 키가 올바르지 않아요. ⚙ 설정에서 키를 다시 확인해 주세요.")
    monkeypatch.setattr(claude_judge, "call_claude_with_tool", fake_call)
    # 프로세스 풀 대신 현재 프로세스에서 돌려야 monkeypatch 가 먹는다
    from app import workers
    from app.main import app

    class InlinePool:
        async def run(self, teacher, image_bytes, config, options):
            return workers.run_clean_job(image_bytes, config, options)

        def shutdown(self):
            pass
    monkeypatch.setattr(app.state, "pool", InlinePool())

    result = upload(client, photo_bytes, KEYED_HEADERS)
    assert result["api_skipped"] is True
    assert result["api_error_kind"] == "auth"
    assert any("올바르지 않아요" in message for message in result["messages"])


def test_key_never_appears_in_logs_or_data(client: TestClient, photo_bytes: bytes, tmp_path: Path) -> None:
    """가짜 키로 여러 라우트를 두드린 뒤 로그·데이터 폴더를 grep 한다."""
    client.post("/api/key/check", headers=KEYED_HEADERS)
    result = upload(client, photo_bytes, KEYED_HEADERS)
    client.post("/api/review", headers=KEYED_HEADERS, data={"job_id": result["job_id"]})
    client.post("/api/clean", headers=KEYED_HEADERS, files={"file": ("x.jpg", b"not-an-image", "image/jpeg")},
                data={"exam_name": "", "sliders": "{}", "use_judge": "1"})
    log_text = client.log_stream.getvalue()
    assert FAKE_KEY not in log_text
    assert "sk-ant-api03" not in log_text
    for file_path in (tmp_path / "data").rglob("*"):
        if file_path.is_file() and file_path.suffix in (".json", ".jsonl", ".txt", ".log"):
            assert FAKE_KEY not in file_path.read_text(encoding="utf-8", errors="ignore"), file_path


def test_key_check_messages(client: TestClient) -> None:
    """키 없음/형식 틀림은 네트워크 없이 바로 한국어로 답한다."""
    missing = client.post("/api/key/check", headers=TEACHER_HEADERS).json()
    assert missing["ok"] is False and missing["status"] == "missing"
    wrong_shape = client.post("/api/key/check", headers={**TEACHER_HEADERS, "X-Anthropic-Key": "abc"}).json()
    assert wrong_shape["ok"] is False


def test_preview_and_render_reuse_without_api(client: TestClient, photo_bytes: bytes) -> None:
    """슬라이더 미리보기와 원래 크기 적용은 API 를 다시 부르지 않는다."""
    job = upload(client, photo_bytes, TEACHER_HEADERS)
    sliders = json.dumps({"remove_strength": 90, "text_darkness": 80})
    preview = client.post("/api/preview", headers=TEACHER_HEADERS,
                          data={"job_id": job["job_id"], "sliders": sliders, "use_judge": "0"}).json()
    assert preview["preview_png_base64"].startswith("iVBOR")   # PNG 매직
    assert preview.get("api_call_count", 0) == 0
    rendered = client.post("/api/render", headers=TEACHER_HEADERS,
                           data={"job_id": job["job_id"], "sliders": sliders, "use_judge": "0"}).json()
    assert rendered["result_url"] == job["result_url"]
    assert rendered["api_call_count"] == 0


def test_review_mock_then_fix_records_correction(client: TestClient, photo_bytes: bytes,
                                                 tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """검수(mock 2건) → 박스 → 지워/살려 → 결과 갱신 + corrections 기록."""
    from app import main as app_main

    def fake_review(cleaned_image, api_key, model, config, usage_log_path):
        if api_key is None:                 # 실제 함수와 같은 규칙: 키 없으면 건너뛴다
            return {"issues": [], "api_skipped": True, "error_kind": "no_key",
                    "message": "Anthropic 키를 넣으면 클로드가 결과를 검수해 줘요."}
        assert api_key == FAKE_KEY          # 키는 함수 인자로만 지나간다
        return {"issues": [
            {"type": "remaining_handwriting", "type_ko": "필기가 남음", "box": [0.1, 0.2, 0.15, 0.05], "note": "연필 계산"},
            {"type": "damaged_print", "type_ko": "인쇄가 손상됨", "box": [0.6, 0.5, 0.2, 0.05], "note": "분수 막대"},
        ], "api_skipped": False, "api_call_count": 1, "api_cost_usd": 0.001}
    monkeypatch.setattr(app_main, "review_cleaned_image", fake_review)

    job = upload(client, photo_bytes, TEACHER_HEADERS)
    review = client.post("/api/review", headers=KEYED_HEADERS, data={"job_id": job["job_id"]}).json()
    assert len(review["issues"]) == 2

    erased = client.post("/api/fix", headers=TEACHER_HEADERS,
                         data={"job_id": job["job_id"], "box": json.dumps(review["issues"][0]["box"]), "decision": "erase"})
    kept = client.post("/api/fix", headers=TEACHER_HEADERS,
                       data={"job_id": job["job_id"], "box": json.dumps(review["issues"][1]["box"]), "decision": "keep"})
    assert erased.status_code == 200 and kept.status_code == 200

    corrections_file = tmp_path / "data" / "labels" / "corrections" / f"{job['job_id']}.jsonl"
    records = [json.loads(line) for line in corrections_file.read_text(encoding="utf-8").splitlines()]
    assert [record["결정"] for record in records] == ["지워", "살려"]
    assert all("sk-ant" not in json.dumps(record) for record in records)

    no_key = client.post("/api/review", headers=TEACHER_HEADERS, data={"job_id": job["job_id"]}).json()
    assert no_key["api_skipped"] is True and "키를 넣으면" in no_key["message"]


def test_pdf_download(client: TestClient, photo_bytes: bytes) -> None:
    """정리본 여러 장을 PDF 한 파일로."""
    first = upload(client, photo_bytes, TEACHER_HEADERS)
    second = upload(client, photo_bytes, TEACHER_HEADERS)
    response = client.post("/api/pdf", headers=TEACHER_HEADERS,
                           data={"job_ids": json.dumps([first["job_id"], second["job_id"]])})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_usage_and_job_list(client: TestClient, photo_bytes: bytes) -> None:
    """사용량 배지와 최근 작업 목록. 다른 강사 작업 보기 토글."""
    upload(client, photo_bytes, TEACHER_HEADERS)
    upload(client, photo_bytes, {**TEACHER_HEADERS, "X-Teacher": quote("강사2")})
    usage = client.get("/api/usage", headers=TEACHER_HEADERS).json()
    assert usage["calls"] == 0 and usage["cost_krw"] == 0
    mine = client.get("/api/jobs?all=0", headers=TEACHER_HEADERS).json()["jobs"]
    everyone = client.get("/api/jobs?all=1", headers=TEACHER_HEADERS).json()["jobs"]
    assert {job["teacher"] for job in mine} == {"강사1"}
    assert {job["teacher"] for job in everyone} == {"강사1", "강사2"}


def test_unknown_teacher_rejected(client: TestClient) -> None:
    """config.yaml 목록에 없는 이름은 거절 (한국어 안내)."""
    response = client.get("/api/usage", headers={"X-Teacher": quote("없는사람")})
    assert response.status_code in (400, 403)
    assert "강사" in response.json()["detail"]
