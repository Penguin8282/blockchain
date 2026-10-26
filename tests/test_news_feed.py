"""
news_feed 패키지 테스트. 실행: python -m pytest tests/test_news_feed.py

네트워크 없이도 돌아가도록, 진짜 RSS 대신 tests/fixtures/sample_rss.xml 을 쓰고
requests.get 을 가짜 함수로 바꿔치기(monkeypatch)한다.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from news_feed import fetcher
from news_feed.__main__ import run_once
from news_feed.classifier import classify, find_tags
from news_feed.fetcher import NewsItem, normalize_url, parse_feed, title_fingerprint
from news_feed.render import render_html
from news_feed.sources import FeedSource
from news_feed.storage import NewsStore

SAMPLE_RSS = (Path(__file__).parent / "fixtures" / "sample_rss.xml").read_bytes()
SAMPLE_SOURCE = FeedSource("샘플", "https://example.com/rss", "en")


def _items() -> list[NewsItem]:
    return [classify(item) for item in parse_feed(SAMPLE_RSS, SAMPLE_SOURCE)]


def test_parse_feed_cleans_html_and_skips_empty_titles():
    items = parse_feed(SAMPLE_RSS, SAMPLE_SOURCE)
    assert len(items) == 4  # 제목 없는 항목은 버려진다
    hack = items[0]
    assert hack.summary == "Attackers exploited a hot wallet & moved funds."
    assert hack.published_at == datetime(2026, 10, 26, 3, 10, tzinfo=timezone.utc)


def test_normalize_url_removes_tracking_and_fragment():
    assert normalize_url("https://Example.com/news/1/?utm_source=x&id=7#top") == "https://example.com/news/1?id=7"


def test_title_fingerprint_ignores_press_suffix_and_punctuation():
    assert title_fingerprint("비트코인, 1억 돌파 - 한국경제") == title_fingerprint("비트코인 1억 돌파!")


def test_tags_use_word_boundaries_for_english():
    assert find_tags("A new method for teaching") == []        # method 안의 eth 는 무시
    assert find_tags("ETH price rises") == ["ETH"]
    assert "BTC" in find_tags("비트코인이 상승했다")              # 한국어는 조사가 붙어도 잡힌다


def test_importance_levels():
    by_title = {item.title: item for item in _items()}
    assert by_title["BREAKING: Exchange hacked, $120M in ETH drained"].importance == 3
    assert by_title["SEC approves new spot Bitcoin ETF applications"].importance == 3
    assert by_title["A new method for teaching blockchain basics"].importance == 1
    upbit = by_title["업비트, 솔라나 기반 신규 토큰 상장"]
    assert {"SOL", "거래소"} <= set(upbit.tags)


def test_store_returns_only_new_items_and_dedupes_by_title():
    with NewsStore(":memory:") as store:
        assert len(store.add_new_items(_items())) == 4
        assert store.add_new_items(_items()) == []  # 같은 기사를 다시 넣으면 새 뉴스 0건

        same_story_other_site = NewsItem(
            title="SEC approves new spot Bitcoin ETF applications - Other Press",
            link="https://other.example.com/a/1",
            source="다른매체", language="en",
            published_at=datetime.now(timezone.utc),
        )
        assert store.add_new_items([same_story_other_site]) == []

        assert [item.importance for item in store.recent(min_importance=3)] == [3, 3]
        assert len(store.recent(tag="SOL")) == 1


def test_render_html_escapes_content():
    item = NewsItem(title="<script>alert(1)</script>", link="https://e.com/?a=1&b=2",
                    source="s", language="en", published_at=datetime.now(timezone.utc))
    page = render_html([item])
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def test_run_once_first_run_is_silent_then_alerts(tmp_path, monkeypatch, capsys):
    feed_contents = {"good": SAMPLE_RSS}

    def fake_get(url, **kwargs):
        if url.endswith("broken"):
            raise requests.ConnectionError("down")  # 한 출처가 죽어도 전체는 계속 돈다
        return _FakeResponse(feed_contents["good"])

    monkeypatch.setattr(fetcher.requests, "get", fake_get)
    monkeypatch.setattr("news_feed.__main__.DEFAULT_SOURCES",
                        [SAMPLE_SOURCE, FeedSource("고장", "https://x/broken", "en")])
    database_path, html_path = tmp_path / "news.db", tmp_path / "index.html"

    assert run_once(database_path, html_path, alert_level=2) == 4
    assert "첫 실행" in capsys.readouterr().out
    assert html_path.exists()

    feed_contents["good"] = SAMPLE_RSS.replace(
        b"<item>", b"<item><title>Bitcoin plunges after FOMC</title>"
        b"<link>https://example.com/news/fomc</link></item><item>", 1)
    assert run_once(database_path, html_path, alert_level=2) == 1
    assert "Bitcoin plunges after FOMC" in capsys.readouterr().out


@pytest.mark.parametrize("level", [1, 2, 3])
def test_render_handles_every_level(level):
    item = NewsItem(title="t", link="https://e.com", source="s", language="ko",
                    published_at=datetime.now(timezone.utc), importance=level)
    assert f"lv{level}" in render_html([item])
