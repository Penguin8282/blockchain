"""pytest 공통 설정.

--live 플래그를 주면 실제 Anthropic API 를 부르는 테스트도 돈다.
기본값은 mock 이라 키 없이도 전체 테스트가 통과해야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# clean_exam/ 을 import 경로에 넣어 준다(패키지 설치 없이 바로 테스트하기 위해)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    """--live 옵션을 만든다."""
    parser.addoption("--live", action="store_true", default=False,
                     help="실제 Anthropic API 를 호출하는 테스트까지 실행한다(키 필요, 비용 발생)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """--live 가 없으면 live 표시된 테스트를 건너뛴다."""
    if config.getoption("--live"):
        return
    skip_marker = pytest.mark.skip(reason="실제 API 호출 테스트다. --live 를 줘야 실행된다.")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_marker)


def pytest_configure(config: pytest.Config) -> None:
    """마커 이름을 등록한다(경고 방지)."""
    config.addinivalue_line("markers", "live: 실제 Anthropic API 를 호출하는 테스트")
