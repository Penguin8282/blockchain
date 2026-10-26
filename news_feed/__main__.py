"""
명령줄에서 실행하는 입구.

사용 예:
  python -m news_feed run              # 한 번 수집 → 저장 → 새 뉴스 알림 → HTML 갱신
  python -m news_feed watch -i 300     # 5분(300초)마다 위 과정을 계속 반복 (Ctrl+C 로 종료)
  python -m news_feed list --min-level 2 --tag BTC   # 저장된 뉴스 중 BTC 주목 이상만 보기
  python -m news_feed html             # 저장된 뉴스로 HTML 만 다시 만들기
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from news_feed.classifier import classify
from news_feed.fetcher import fetch_all
from news_feed.notifier import format_console_line, notify
from news_feed.render import write_html
from news_feed.sources import DEFAULT_SOURCES
from news_feed.storage import NewsStore

try:  # .env 파일이 있으면 텔레그램 토큰 등을 읽어온다 (python-dotenv 가 없어도 동작)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = Path(os.getenv("NEWS_DB_PATH", PACKAGE_DIRECTORY / "output" / "news.db"))
DEFAULT_HTML_PATH = Path(os.getenv("NEWS_HTML_PATH", PACKAGE_DIRECTORY / "output" / "index.html"))
DEFAULT_ALERT_LEVEL = int(os.getenv("NEWS_ALERT_MIN_LEVEL", "2"))


def run_once(database_path: Path, html_path: Path, alert_level: int) -> int:
    """수집 1회. 새로 들어온 뉴스 개수를 돌려준다."""
    collected = [classify(news_item) for news_item in fetch_all(DEFAULT_SOURCES)]
    with NewsStore(database_path) as store:
        is_first_run = store.count() == 0
        new_items = store.add_new_items(collected)
        if is_first_run:
            # 처음 실행하면 지난 기사 수백 개가 전부 '새 뉴스'가 된다.
            # 그걸 모두 휴대폰으로 보내면 알림 폭탄이 되므로, 첫 회는 저장만 한다.
            print(f"첫 실행: 기존 뉴스 {len(new_items)}건을 저장했습니다. 다음 수집부터 알림을 보냅니다.")
        elif new_items:
            notify(new_items, min_importance=alert_level)
        write_html(store.recent(limit=300), html_path)
    logging.info("수집 %d건 중 새 뉴스 %d건", len(collected), len(new_items))
    return len(new_items)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="news_feed", description="블록체인 뉴스 수집·알림기")
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite 파일 경로")
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML_PATH, help="타임라인 HTML 경로")
    parser.add_argument("-v", "--verbose", action="store_true", help="수집 로그 자세히 보기")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="한 번 수집")
    watch_parser = commands.add_parser("watch", help="주기적으로 계속 수집")
    for sub_parser in (run_parser, watch_parser):
        sub_parser.add_argument("--alert-level", type=int, default=DEFAULT_ALERT_LEVEL, choices=[1, 2, 3],
                                help="메신저로 보낼 최소 중요도 (1=전부, 2=주목 이상, 3=중요만)")
    watch_parser.add_argument("-i", "--interval", type=int, default=300, help="수집 간격(초)")

    list_parser = commands.add_parser("list", help="저장된 뉴스 보기")
    list_parser.add_argument("-n", "--limit", type=int, default=30)
    list_parser.add_argument("--min-level", type=int, default=1, choices=[1, 2, 3])
    list_parser.add_argument("--tag", help="예: BTC, ETH, 규제, 보안, ETF")

    commands.add_parser("html", help="저장된 뉴스로 HTML 만 다시 만들기")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "run":
        run_once(args.db, args.html, args.alert_level)
        print(f"타임라인: {args.html}")
    elif args.command == "watch":
        print(f"{args.interval}초마다 수집합니다. 종료하려면 Ctrl+C. 타임라인: {args.html}")
        try:
            while True:
                try:
                    run_once(args.db, args.html, args.alert_level)
                except Exception:  # 한 번 실패해도 감시는 계속돼야 한다
                    logging.exception("수집 중 오류")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n종료합니다.")
    elif args.command == "list":
        with NewsStore(args.db) as store:
            for news_item in store.recent(limit=args.limit, min_importance=args.min_level, tag=args.tag):
                print(format_console_line(news_item))
    elif args.command == "html":
        with NewsStore(args.db) as store:
            print(f"타임라인: {write_html(store.recent(limit=300), args.html)}")


if __name__ == "__main__":
    main()
