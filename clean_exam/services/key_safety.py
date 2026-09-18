"""API 키가 로그·오류 메시지에 절대 남지 않게 막는 장치.

내부용이라도 동료 강사의 키가 로그 파일에 남는 사고는 막아야 한다.
그래서 두 가지를 둔다:
  1) mask_api_keys(): 문자열 안의 `sk-ant-...` 를 `sk-ant-***` 로 바꾼다
  2) install_log_filter(): 파이썬 logging 전체에 그 마스킹을 자동 적용한다
"""

from __future__ import annotations

import logging
import re

# sk-ant- 로 시작해 영문·숫자·하이픈·밑줄이 이어지는 덩어리를 키로 본다
API_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")
MASKED_TEXT = "sk-ant-***"


def mask_api_keys(text: str) -> str:
    """문자열에 섞인 Anthropic API 키를 가린다."""
    return API_KEY_PATTERN.sub(MASKED_TEXT, str(text))


class ApiKeyMaskingFilter(logging.Filter):
    """모든 로그 기록에서 키를 가리는 필터."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_api_keys(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {key: mask_api_keys(value) if isinstance(value, str) else value
                               for key, value in record.args.items()}
            else:
                record.args = tuple(mask_api_keys(arg) if isinstance(arg, str) else arg
                                    for arg in record.args)
        return True


def install_log_filter() -> None:
    """루트 로거와 이미 붙어 있는 모든 핸들러에 마스킹 필터를 건다."""
    masking_filter = ApiKeyMaskingFilter()
    root_logger = logging.getLogger()
    root_logger.addFilter(masking_filter)
    for handler in root_logger.handlers:
        handler.addFilter(masking_filter)
