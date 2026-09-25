"""서버 실행 진입점. `python run.py` 하나로 뜬다.

같은 와이파이의 다른 컴퓨터에서는 http://이컴퓨터IP:8000 으로 접속한다.
(IP 확인: Windows `ipconfig`, Mac/Linux `ifconfig` 또는 `ip addr` — 192.168. 으로 시작하는 주소)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

from engine.settings import load_config


def main() -> None:
    server_config = load_config()["server"]
    print(f"깔끔 시험지 서버를 띄웁니다 → http://localhost:{server_config['port']}")
    print("같은 네트워크의 다른 컴퓨터에서는 http://이컴퓨터IP:%d" % server_config["port"])
    uvicorn.run("app.main:app", host=server_config["host"], port=int(server_config["port"]),
                log_level="info", access_log=True)


if __name__ == "__main__":
    main()
