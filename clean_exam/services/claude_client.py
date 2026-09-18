"""모든 Claude 호출이 반드시 지나가는 단 하나의 창구.

여기에 모아 둔 이유: 재시도·타임아웃·비용 기록·응답 캐시·키 마스킹을 한 곳에서만
관리하면, 다른 코드는 "무엇을 물어볼지"만 신경 쓰면 된다.

키에 대한 약속 (CLAUDE.md 의 규칙):
  · 키는 **함수 인자로만** 들어온다. 전역 변수·환경변수·설정 파일에서 읽지 않는다.
  · 키는 요청을 처리하는 동안만 메모리에 있고, 어디에도 저장하지 않는다.
  · 캐시 키에 강사 키를 넣지 않는다 → 같은 사진을 두 강사가 올리면 두 번째는 호출 0회.
  · 사용량 기록에는 시각·모델·토큰·비용만 남긴다.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.key_safety import install_log_filter, mask_api_keys

logger = logging.getLogger(__name__)
install_log_filter()


class ClaudeCallError(Exception):
    """Claude 호출이 실패했을 때, 강사에게 한국어로 보여줄 원인을 담아 던지는 예외."""

    def __init__(self, error_kind: str, korean_message: str):
        super().__init__(korean_message)
        self.error_kind = error_kind          # no_key / auth / rate_limit / credit / network / other
        self.korean_message = korean_message


@dataclass
class UsageRecord:
    """호출 한 번의 사용량. 키는 절대 담지 않는다."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    estimated_cost_usd: float = 0.0
    from_cache: bool = False


@dataclass
class CallStatistics:
    """한 장(또는 한 작업)을 처리하는 동안 쌓인 호출 통계."""

    call_count: int = 0
    cache_hit_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        """사용량 한 건을 합산한다."""
        self.records.append(record)
        if record.from_cache:
            self.cache_hit_count += 1
            return
        self.call_count += 1
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_cost_usd += record.estimated_cost_usd


def encode_image_to_base64(image_bytes: bytes) -> str:
    """이미지 바이트를 API 에 넣을 수 있는 base64 문자열로 바꾼다."""
    return base64.standard_b64encode(image_bytes).decode("ascii")


def build_cache_key(model: str, system_prompt: str, image_bytes_list: list[bytes],
                    user_text: str) -> str:
    """응답 캐시의 열쇠를 만든다 — 이미지 해시 + 프롬프트 해시 + 모델명.

    강사 키는 **일부러 넣지 않는다**. 같은 사진을 두 강사가 올리면 두 번째 사람은
    호출 없이 첫 번째 결과를 재사용해 비용이 0이 되게 하기 위해서다.
    """
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(system_prompt.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(user_text.encode("utf-8"))
    for image_bytes in image_bytes_list:
        hasher.update(b"\x00")
        hasher.update(hashlib.sha256(image_bytes).digest())
    return hasher.hexdigest()


def read_cached_response(cache_directory: Path, cache_key: str) -> dict[str, Any] | None:
    """캐시에 저장된 응답이 있으면 돌려준다(없으면 None)."""
    cache_file = cache_directory / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as opened_file:
            return json.load(opened_file)
    except (json.JSONDecodeError, OSError):
        return None


def write_cached_response(cache_directory: Path, cache_key: str, response_data: dict[str, Any]) -> None:
    """응답을 캐시에 저장한다. 저장에 실패해도 전체 처리를 멈추지 않는다."""
    try:
        cache_directory.mkdir(parents=True, exist_ok=True)
        with open(cache_directory / f"{cache_key}.json", "w", encoding="utf-8") as opened_file:
            json.dump(response_data, opened_file, ensure_ascii=False)
    except OSError as error:
        logger.warning("응답 캐시 저장 실패(무시하고 계속): %s", mask_api_keys(error))


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int,
                      pricing_table: dict[str, dict[str, float]]) -> float:
    """토큰 수와 단가표로 이번 호출의 대략적인 비용(달러)을 계산한다."""
    model_pricing = pricing_table.get(model)
    if model_pricing is None:
        return 0.0
    return (input_tokens * model_pricing["input"] + output_tokens * model_pricing["output"]) / 1_000_000.0


def call_claude_with_tool(
    api_key: str | None,
    model: str,
    system_prompt: str,
    user_text: str,
    image_bytes_list: list[bytes],
    tool_name: str,
    tool_description: str,
    tool_input_schema: dict[str, Any],
    claude_config: dict[str, Any],
    cache_directory: Path,
    statistics: CallStatistics,
) -> dict[str, Any]:
    """이미지 여러 장을 보내고, **도구 호출**로 검증된 JSON 을 받아 돌려준다.

    자유 문장을 파싱하지 않는 이유: 모델이 설명을 덧붙이거나 형식을 바꾸면 파싱이 깨진다.
    원하는 JSON 스키마를 가진 도구를 하나 정의하고 tool_choice 로 그 도구를 강제하면,
    응답의 tool_use.input 이 곧 스키마에 맞는 JSON 이다.

    인자:
        api_key: 강사 본인의 키. None 이면 ClaudeCallError("no_key") 를 던진다.
        statistics: 호출 통계를 누적할 상자(제자리 수정된다).
    """
    if not api_key:
        raise ClaudeCallError("no_key", "Anthropic 키를 넣으면 더 정확하게 지울 수 있어요.")

    # 1) 캐시 먼저 본다 (키와 무관하게 같은 사진 + 같은 질문이면 재사용)
    cache_key = build_cache_key(model, system_prompt, image_bytes_list, user_text)
    if claude_config.get("use_response_cache", True):
        cached_response = read_cached_response(cache_directory, cache_key)
        if cached_response is not None:
            statistics.add(UsageRecord(model=model, from_cache=True))
            return cached_response

    # 2) 실제 호출. anthropic 패키지는 여기서만 import 한다(키 없이 쓰는 강사는 설치 안 해도 되게).
    try:
        import anthropic
    except ImportError as error:
        raise ClaudeCallError("other", "anthropic 패키지가 설치되어 있지 않아요. pip install anthropic 을 해 주세요.") from error

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=float(claude_config["timeout_seconds"]),
        max_retries=int(claude_config["max_retries"]),
    )

    # 이미지 블록 + "몇 번 조각이 무엇인지" 설명 텍스트를 한 메시지에 담는다
    message_content: list[dict[str, Any]] = []
    for image_position, image_bytes in enumerate(image_bytes_list, start=1):
        message_content.append({
            "type": "text",
            "text": f"[조각 {image_position}]",
        })
        message_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encode_image_to_base64(image_bytes),
            },
        })
    message_content.append({"type": "text", "text": user_text})

    # 시스템 프롬프트는 길고 매번 같으므로 캐싱을 켜서 비용을 줄인다
    system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system_prompt}]
    if claude_config.get("use_prompt_cache", True):
        system_blocks[0]["cache_control"] = {"type": "ephemeral"}

    started_at = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_blocks,
            messages=[{"role": "user", "content": message_content}],
            tools=[{
                "name": tool_name,
                "description": tool_description,
                "input_schema": tool_input_schema,
            }],
            tool_choice={"type": "tool", "name": tool_name},
        )
    except anthropic.AuthenticationError as error:
        raise ClaudeCallError("auth", "Anthropic 키가 올바르지 않아요. ⚙ 설정에서 키를 다시 확인해 주세요.") from error
    except anthropic.PermissionDeniedError as error:
        raise ClaudeCallError("auth", "이 키로는 이 모델을 쓸 수 없어요. 키 권한을 확인해 주세요.") from error
    except anthropic.RateLimitError as error:
        raise ClaudeCallError("rate_limit", "Anthropic 호출이 너무 잦아요. 잠시 뒤 다시 시도해 주세요.") from error
    except anthropic.BadRequestError as error:
        # 크레딧 부족도 400 으로 온다. 메시지에 credit 이 들어 있으면 그렇게 안내한다.
        error_text = mask_api_keys(str(error)).lower()
        if "credit" in error_text or "balance" in error_text:
            raise ClaudeCallError("credit", "Anthropic 크레딧이 부족해요. Billing 에서 충전해 주세요.") from error
        raise ClaudeCallError("other", "Anthropic 요청이 거절됐어요(잘못된 요청).") from error
    except anthropic.APIConnectionError as error:
        raise ClaudeCallError("network", "Anthropic 에 연결하지 못했어요. 인터넷 연결을 확인해 주세요.") from error
    except anthropic.APIStatusError as error:
        raise ClaudeCallError("other", f"Anthropic 서버 오류({error.status_code})가 났어요.") from error

    elapsed_seconds = time.time() - started_at

    # 3) 도구 호출 결과를 꺼낸다
    tool_input: dict[str, Any] | None = None
    for content_block in response.content:
        if content_block.type == "tool_use" and content_block.name == tool_name:
            tool_input = dict(content_block.input)
            break
    if tool_input is None:
        raise ClaudeCallError("other", "Claude 가 예상한 형식으로 답하지 않았어요.")

    # 4) 사용량 기록 (키는 절대 기록하지 않는다)
    usage = response.usage
    cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    record = UsageRecord(
        model=model,
        input_tokens=int(usage.input_tokens),
        output_tokens=int(usage.output_tokens),
        cache_read_tokens=cache_read_tokens,
        estimated_cost_usd=estimate_cost_usd(
            model, int(usage.input_tokens), int(usage.output_tokens), claude_config["pricing"]
        ),
    )
    statistics.add(record)
    logger.info("Claude 호출 완료: 모델=%s 입력=%d 출력=%d 캐시읽기=%d %.1f초",
                model, record.input_tokens, record.output_tokens, cache_read_tokens, elapsed_seconds)

    if claude_config.get("use_response_cache", True):
        write_cached_response(cache_directory, cache_key, tool_input)
    return tool_input


def append_usage_log(usage_log_path: Path, statistics: CallStatistics, usd_to_krw: float) -> None:
    """사용량을 강사별 jsonl 파일에 덧붙인다. **키는 절대 쓰지 않는다.**"""
    if not statistics.records:
        return
    try:
        usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(usage_log_path, "a", encoding="utf-8") as opened_file:
            for record in statistics.records:
                if record.from_cache:
                    continue
                opened_file.write(json.dumps({
                    "시각": datetime.now(timezone.utc).isoformat(),
                    "모델": record.model,
                    "입력토큰": record.input_tokens,
                    "출력토큰": record.output_tokens,
                    "비용_USD": round(record.estimated_cost_usd, 6),
                    "비용_KRW": round(record.estimated_cost_usd * usd_to_krw, 2),
                }, ensure_ascii=False) + "\n")
    except OSError as error:
        logger.warning("사용량 기록 실패(무시하고 계속): %s", mask_api_keys(error))
