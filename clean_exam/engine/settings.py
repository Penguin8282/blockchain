"""config.yaml 을 읽어 오는 곳.

이 프로젝트의 모든 조정 가능한 값은 config.yaml 한 파일에 있고,
코드는 이 모듈을 통해서만 그 값을 읽는다. (코드에 숫자를 박아 넣지 않는다.)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# 이 파일 기준으로 프로젝트 루트(clean_exam/)를 찾는다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_cached_config: dict[str, Any] | None = None
_cached_config_path: Path | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """config.yaml 을 읽어 dict 로 돌려준다.

    같은 파일을 여러 번 읽지 않도록 한 번 읽은 내용은 기억해 둔다.
    돌려주는 것은 복사본이라, 호출한 쪽에서 값을 바꿔도 원본은 안 바뀐다.

    인자:
        config_path: 읽을 설정 파일 경로. 없으면 clean_exam/config.yaml.
    """
    global _cached_config, _cached_config_path

    resolved_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if _cached_config is None or _cached_config_path != resolved_path:
        with open(resolved_path, "r", encoding="utf-8") as config_file:
            _cached_config = yaml.safe_load(config_file)
        _cached_config_path = resolved_path
    return copy.deepcopy(_cached_config)
