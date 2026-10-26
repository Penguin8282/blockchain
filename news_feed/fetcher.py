"""
RSS 피드를 내려받아 '통일된 모양의 뉴스 한 건(NewsItem)'으로 바꾸는 단계.

매체마다 RSS 모양이 조금씩 다르다(날짜 형식, 요약에 HTML 태그 포함 여부 등).
그래서 여기서 한 번 '표준 규격'으로 맞춰 두면, 뒤 단계(분류·저장·알림)는
매체 차이를 신경 쓰지 않아도 된다. 공장의 '규격 검수대' 같은 역할이다.
"""

from __future__ import annotations

import calendar
import hashlib
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import requests

from news_feed.sources import FeedSource

logger = logging.getLogger(__name__)

# 일부 사이트는 '봇처럼 보이는' 요청을 막으므로 브라우저 비슷한 이름표를 단다.
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BlockchainNewsFeed/1.0)"}
REQUEST_TIMEOUT_SECONDS = 15
SUMMARY_MAX_CHARACTERS = 220

# 주소 뒤에 붙는 광고 추적용 꼬리표. 같은 기사인데 꼬리표만 다르면 중복으로 봐야 한다.
TRACKING_QUERY_PREFIXES = ("utm_", "fbclid", "gclid")


@dataclass
class NewsItem:
    """표준 규격으로 정리된 뉴스 한 건."""

    title: str
    link: str
    source: str
    language: str
    published_at: datetime          # 항상 UTC 기준 시각으로 저장한다
    summary: str = ""
    tags: list[str] = field(default_factory=list)  # classifier 가 채운다 (예: ["BTC", "규제"])
    importance: int = 1                             # 1=일반, 2=주목, 3=중요

    @property
    def url_key(self) -> str:
        """같은 기사인지 판별하는 '주소 지문'."""
        return normalize_url(self.link)

    @property
    def title_key(self) -> str:
        """같은 기사인지 판별하는 '제목 지문'.

        여러 매체가 같은 통신사 기사를 받아 쓰거나, 구글뉴스가 원문과 다른 주소로
        같은 기사를 내려줄 때를 잡기 위해 제목으로도 한 번 더 비교한다.
        """
        return title_fingerprint(self.title)


# ------------------------------------------------------------
# 1) 텍스트 정리 도우미
# ------------------------------------------------------------
_TAG_PATTERN = re.compile(r"<[^>]+>")
_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(raw_text: str) -> str:
    """HTML 태그를 지우고, &amp; 같은 기호를 원래 글자로 바꾸고, 공백을 하나로 줄인다."""
    without_tags = _TAG_PATTERN.sub(" ", raw_text or "")
    unescaped = html.unescape(without_tags)
    return _SPACE_PATTERN.sub(" ", unescaped).strip()


def shorten(text: str, max_characters: int = SUMMARY_MAX_CHARACTERS) -> str:
    """요약이 너무 길면 잘라서 말줄임표를 붙인다."""
    if len(text) <= max_characters:
        return text
    return text[: max_characters - 1].rstrip() + "…"


def normalize_url(url: str) -> str:
    """주소에서 추적용 꼬리표·#앵커·끝의 / 를 떼어 '같은 기사면 같은 문자열'이 되게 한다.

    예) https://a.com/news/1/?utm_source=x#top  ->  https://a.com/news/1
    """
    parts = urlsplit(url.strip())
    kept_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(TRACKING_QUERY_PREFIXES)
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(kept_query), ""))


def title_fingerprint(title: str) -> str:
    """제목을 소문자로 바꾸고 글자·숫자만 남긴 뒤 해시(sha1)한다.

    해시란? 긴 글을 짧은 '지문'으로 바꾸는 함수. 글이 같으면 지문도 항상 같다.
    구글뉴스 제목 끝의 " - 매체명" 꼬리도 떼어 비교한다.
    """
    without_press_suffix = re.sub(r"\s+-\s+[^-]{1,30}$", "", title)
    letters_only = re.sub(r"[\W_]+", "", without_press_suffix.lower())
    return hashlib.sha1(letters_only.encode("utf-8")).hexdigest()


def _entry_published_at(entry: dict) -> datetime:
    """RSS 항목의 발행 시각을 UTC datetime 으로 꺼낸다. 없으면 '지금'으로 둔다."""
    for key in ("published_parsed", "updated_parsed"):
        parsed_time = entry.get(key)
        if parsed_time:
            # feedparser 는 시각을 UTC 기준 time.struct_time 으로 준다.
            # calendar.timegm 은 그것을 '1970년부터 몇 초'로 바꿔준다.
            return datetime.fromtimestamp(calendar.timegm(parsed_time), tz=timezone.utc)
    return datetime.now(timezone.utc)


# ------------------------------------------------------------
# 2) 피드 하나 파싱 / 내려받기
# ------------------------------------------------------------
def parse_feed(feed_content: bytes | str, source: FeedSource) -> list[NewsItem]:
    """RSS(XML) 내용을 NewsItem 목록으로 바꾼다. (네트워크 없이 테스트할 수 있게 분리)"""
    parsed_feed = feedparser.parse(feed_content)
    news_items = []
    for entry in parsed_feed.entries:
        title = clean_text(entry.get("title", ""))
        link = entry.get("link", "").strip()
        if not title or not link:
            continue  # 제목이나 주소가 없는 항목은 쓸모가 없으므로 버린다
        news_items.append(
            NewsItem(
                title=title,
                link=link,
                source=source.name,
                language=source.language,
                published_at=_entry_published_at(entry),
                summary=shorten(clean_text(entry.get("summary", ""))),
            )
        )
    return news_items


def fetch_feed(source: FeedSource) -> list[NewsItem]:
    """출처 하나에서 뉴스를 받아온다. 실패하면 빈 목록을 돌려주고 경고만 남긴다."""
    try:
        response = requests.get(source.url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        logger.warning("[%s] 수집 실패: %s", source.name, error)
        return []
    return parse_feed(response.content, source)


def fetch_all(sources: list[FeedSource], max_workers: int = 8) -> list[NewsItem]:
    """여러 출처를 '동시에' 받아온다.

    출처가 10개이고 하나당 1초가 걸리면, 하나씩 받으면 10초지만
    스레드 여러 개로 동시에 받으면 1~2초면 끝난다. (식당에서 주문을 여러 직원이 나눠 받는 것)
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_per_source = executor.map(fetch_feed, sources)
    return [item for items in results_per_source for item in items]
