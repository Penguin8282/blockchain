"""요청 헤더에서 강사 이름과 Anthropic 키를 꺼내는 FastAPI 의존성.

키에 대한 약속:
  · `X-Anthropic-Key` 헤더로 들어온 키는 이 요청을 처리하는 동안만 메모리에 있다.
  · 어디에도 저장하지 않고, 로그·오류 메시지에도 찍히지 않는다(services/key_safety 의 로그 필터).
  · 키가 없어도 요청은 실패하지 않는다 — API 가 필요한 단계만 건너뛴다.
강사 이름(`X-Teacher`)은 config.yaml 의 teachers 목록에 있는 것만 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote

from fastapi import Header, HTTPException

from engine.settings import load_config


@dataclass
class RequestIdentity:
    """이 요청을 보낸 강사와, 그 강사가 넣은 키(없으면 None)."""

    teacher: str
    api_key: str | None
    model_choice: str          # "fast" | "quality"

    @property
    def model(self) -> str:
        """선택에 맞는 모델 ID."""
        claude_config = load_config()["claude"]
        return claude_config["quality_model"] if self.model_choice == "quality" else claude_config["fast_model"]


def get_identity(
    x_teacher: str | None = Header(default=None, alias="X-Teacher"),
    x_anthropic_key: str | None = Header(default=None, alias="X-Anthropic-Key"),
    x_model: str | None = Header(default=None, alias="X-Model"),
) -> RequestIdentity:
    """헤더에서 강사 이름(필수)과 키(선택)를 꺼낸다."""
    # HTTP 헤더는 ASCII 만 허용된다(브라우저의 fetch 도 한글 헤더를 거부한다).
    # 그래서 화면은 encodeURIComponent 로 보내고 여기서 되돌린다. 안 인코딩된 값도 받아 준다.
    teacher_name = unquote(x_teacher) if x_teacher else ""
    teachers = load_config()["teachers"]
    if not teacher_name or teacher_name not in teachers:
        raise HTTPException(status_code=400, detail="화면 위에서 강사 이름을 먼저 골라 주세요.")

    api_key = (x_anthropic_key or "").strip() or None
    if api_key is not None and not api_key.startswith("sk-ant-"):
        # 모양이 틀린 키는 없는 것으로 친다(서버가 500 을 내면 안 된다)
        api_key = None

    model_choice = "quality" if (x_model or "").lower() == "quality" else "fast"
    return RequestIdentity(teacher=teacher_safe(teacher_name), api_key=api_key, model_choice=model_choice)


def teacher_safe(name: str) -> str:
    """강사 이름을 폴더 이름으로 써도 안전하게 만든다(경로 문자 제거)."""
    return "".join(ch for ch in name if ch not in '/\\:*?"<>|').strip() or "강사"
