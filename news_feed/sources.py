"""
뉴스를 가져올 '출처(RSS 피드)' 목록.

RSS 란?
  - 뉴스 사이트가 "새 글 목록"을 기계가 읽기 좋은 XML 형식으로 공개해 둔 주소다.
  - 신문 가판대에 매일 새 신문을 꽂아 두는 것처럼, 사이트가 새 글을 이 주소에 계속 올린다.
  - 우리는 이 주소들을 주기적으로 둘러보며 "처음 보는 글"만 골라내면 된다.

주의:
  - 아래 주소들은 널리 알려진 공개 RSS 주소지만, 사이트 개편으로 바뀔 수 있다.
    (작성 시점에 이 개발 환경에서는 외부 접속이 막혀 있어 실제 응답을 확인하지 못했다.)
  - 한 출처가 실패해도 나머지는 계속 수집되도록 fetcher.py 에서 처리한다.
  - 코인니스(Coinness)는 공개 RSS 를 제공하지 않는 것으로 알고 있다(확실하지 않음).
    그래서 비슷한 역할을 하도록 여러 국내외 매체를 직접 모은다.
"""

from dataclasses import dataclass
from urllib.parse import quote


@dataclass(frozen=True)
class FeedSource:
    """RSS 출처 하나를 나타내는 작은 상자."""

    name: str      # 화면에 보여줄 매체 이름
    url: str       # RSS 주소
    language: str  # "ko" 또는 "en" — 화면에서 언어 필터에 쓴다


def google_news_search(query: str, language: str = "ko") -> str:
    """구글 뉴스 '검색어 RSS' 주소를 만든다.

    구글 뉴스는 검색 결과도 RSS 로 내려준다. 국내 매체 여러 곳을 한 번에 훑는 용도로 좋다.
    예) google_news_search("비트코인") -> 여러 한국 언론사의 비트코인 기사 목록
    """
    if language == "ko":
        region = "hl=ko&gl=KR&ceid=KR:ko"
    else:
        region = "hl=en-US&gl=US&ceid=US:en"
    return f"https://news.google.com/rss/search?q={quote(query)}&{region}"


# ------------------------------------------------------------
# 기본 출처 목록 — 필요하면 여기에 줄을 추가/삭제하면 된다.
# ------------------------------------------------------------
DEFAULT_SOURCES: list[FeedSource] = [
    # --- 해외 블록체인 전문 매체 ---
    FeedSource("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", "en"),
    FeedSource("Cointelegraph", "https://cointelegraph.com/rss", "en"),
    FeedSource("Decrypt", "https://decrypt.co/feed", "en"),
    FeedSource("The Block", "https://www.theblock.co/rss.xml", "en"),
    FeedSource("Bitcoin Magazine", "https://bitcoinmagazine.com/feed", "en"),
    FeedSource("CryptoSlate", "https://cryptoslate.com/feed/", "en"),
    # --- 국내 블록체인 전문 매체 ---
    FeedSource("블록미디어", "https://www.blockmedia.co.kr/feed", "ko"),
    FeedSource("토큰포스트", "https://www.tokenpost.kr/rss", "ko"),
    # --- 구글 뉴스 검색 (국내 일반 언론사의 코인 기사까지 함께 모으기) ---
    FeedSource("구글뉴스·비트코인", google_news_search("비트코인 when:1d"), "ko"),
    FeedSource("구글뉴스·가상자산", google_news_search("가상자산 when:1d"), "ko"),
]
