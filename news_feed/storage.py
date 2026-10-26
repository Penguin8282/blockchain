"""
수집한 뉴스를 SQLite 데이터베이스에 저장하고, '처음 보는 뉴스'만 골라내는 단계.

SQLite 란?
  - 파일 하나(news.db)로 동작하는 작은 데이터베이스. 서버 설치가 필요 없다.
  - 엑셀 시트 하나를 '표(table)'라고 생각하면 된다. 여기서는 news 라는 표 하나만 쓴다.

중복 제거의 핵심:
  - url_key(주소 지문)에 UNIQUE(중복 금지) 제약을 건다.
  - 같은 주소의 기사를 또 넣으려 하면 DB 가 거절하므로, '새로 들어간 것 = 새 뉴스'가 된다.
  - 제목 지문(title_key)이 이미 있으면 다른 매체의 같은 기사로 보고 넣지 않는다.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from news_feed.fetcher import NewsItem

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS news (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    url_key      TEXT NOT NULL UNIQUE,
    title_key    TEXT NOT NULL,
    title        TEXT NOT NULL,
    link         TEXT NOT NULL,
    source       TEXT NOT NULL,
    language     TEXT NOT NULL,
    summary      TEXT NOT NULL DEFAULT '',
    tags         TEXT NOT NULL DEFAULT '[]',   -- 파이썬 리스트를 JSON 글자로 저장
    importance   INTEGER NOT NULL DEFAULT 1,
    published_at TEXT NOT NULL,                -- ISO 형식 UTC 시각 (예: 2026-10-26T04:00:00+00:00)
    collected_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_news_title_key ON news(title_key);
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news(published_at);
"""


class NewsStore:
    """뉴스 DB 를 다루는 창구. with 문으로 쓰면 끝날 때 자동으로 닫힌다."""

    def __init__(self, database_path: str | Path):
        database_path = Path(database_path)
        if str(database_path) != ":memory:":
            database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row  # 결과를 row["title"] 처럼 이름으로 꺼내기
        self.connection.executescript(SCHEMA_SQL)

    def __enter__(self) -> "NewsStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM news").fetchone()[0]

    def add_new_items(self, news_items: list[NewsItem]) -> list[NewsItem]:
        """뉴스 목록을 저장하고, 이번에 '처음 들어간' 뉴스만 돌려준다 (시간순)."""
        newly_added = []
        for news_item in sorted(news_items, key=lambda item: item.published_at):
            already_seen_title = self.connection.execute(
                "SELECT 1 FROM news WHERE title_key = ? LIMIT 1", (news_item.title_key,)
            ).fetchone()
            if already_seen_title:
                continue
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO news
                    (url_key, title_key, title, link, source, language, summary,
                     tags, importance, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    news_item.url_key,
                    news_item.title_key,
                    news_item.title,
                    news_item.link,
                    news_item.source,
                    news_item.language,
                    news_item.summary,
                    json.dumps(news_item.tags, ensure_ascii=False),
                    news_item.importance,
                    news_item.published_at.isoformat(),
                ),
            )
            # rowcount == 1 이면 실제로 들어갔다는 뜻, 0 이면 같은 주소가 이미 있어 무시된 것
            if cursor.rowcount == 1:
                newly_added.append(news_item)
        self.connection.commit()
        return newly_added

    def recent(
        self,
        limit: int = 200,
        min_importance: int = 1,
        tag: str | None = None,
    ) -> list[NewsItem]:
        """최근 뉴스를 최신순으로 꺼낸다. 중요도·태그로 거를 수 있다."""
        query = "SELECT * FROM news WHERE importance >= ?"
        parameters: list = [min_importance]
        if tag:
            # tags 열은 '["BTC", "규제"]' 같은 글자이므로, 따옴표까지 포함해 정확히 찾는다
            query += " AND tags LIKE ?"
            parameters.append(f'%"{tag}"%')
        query += " ORDER BY published_at DESC LIMIT ?"
        parameters.append(limit)
        rows = self.connection.execute(query, parameters).fetchall()
        return [_row_to_item(row) for row in rows]


def _row_to_item(row: sqlite3.Row) -> NewsItem:
    return NewsItem(
        title=row["title"],
        link=row["link"],
        source=row["source"],
        language=row["language"],
        published_at=datetime.fromisoformat(row["published_at"]),
        summary=row["summary"],
        tags=json.loads(row["tags"]),
        importance=row["importance"],
    )
