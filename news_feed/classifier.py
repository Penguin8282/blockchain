"""
뉴스 제목·요약을 보고 '태그'와 '중요도'를 붙이는 규칙 기반 분류기.

왜 머신러닝이 아니라 규칙(키워드)인가?
  - 학습시킬 '정답 라벨(이 뉴스는 중요/안 중요)'이 아직 없다.
  - 키워드 규칙은 왜 그렇게 분류됐는지 바로 설명할 수 있다(해석 가능성).
  - 나중에 사용자가 실제로 클릭한 기사 기록이 쌓이면, 그걸 라벨로 삼아
    로지스틱 회귀 같은 모델로 바꾸는 것이 자연스러운 다음 단계다.

중요도 점수 만드는 법 (아주 단순한 '가중합'):
  점수 = Σ w(제목 키워드) + ½ · Σ w(요약 키워드)
  - Σ : 모두 더한다는 기호
  - w : 가중치. "해킹"은 3점, "승인"은 2점처럼 키워드마다 정한 숫자
  - ½ : 요약은 제목보다 덜 중요하게 본다는 뜻
  점수가 4 이상이면 3단계(중요), 2 이상이면 2단계(주목), 그 밖은 1단계(일반).
"""

from __future__ import annotations

import re
from functools import lru_cache

from news_feed.fetcher import NewsItem

# ------------------------------------------------------------
# 1) 코인 태그 — 어떤 자산에 관한 뉴스인가?
# ------------------------------------------------------------
COIN_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "비트코인"],
    "ETH": ["ethereum", "ether", "eth", "이더리움"],
    "SOL": ["solana", "sol", "솔라나"],
    "XRP": ["xrp", "ripple", "리플"],
    "DOGE": ["dogecoin", "doge", "도지코인"],
    "스테이블코인": ["stablecoin", "usdt", "usdc", "tether", "스테이블코인", "테더"],
}

# ------------------------------------------------------------
# 2) 주제 태그 — 어떤 종류의 사건인가?
# ------------------------------------------------------------
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "규제": ["sec", "cftc", "regulation", "regulator", "lawsuit", "bill", "규제", "금융위",
             "금감원", "법안", "제재", "소송", "가상자산법"],
    "보안": ["hack", "hacked", "exploit", "drained", "stolen", "phishing", "vulnerability",
             "해킹", "탈취", "피싱", "취약점", "도난"],
    "ETF": ["etf", "spot etf", "현물 etf"],
    "거래소": ["binance", "coinbase", "upbit", "bithumb", "listing", "delist", "바이낸스",
              "코인베이스", "업비트", "빗썸", "상장", "상장폐지"],
    "거시경제": ["fed", "fomc", "interest rate", "cpi", "inflation", "연준", "금리", "물가",
               "환율"],
    "기관": ["blackrock", "fidelity", "microstrategy", "institutional", "블랙록", "기관"],
    "DeFi": ["defi", "dex", "uniswap", "aave", "staking", "디파이", "스테이킹"],
}

# ------------------------------------------------------------
# 3) 중요도 가중치 — 시장을 크게 움직일 만한 단어일수록 높게
# ------------------------------------------------------------
IMPORTANCE_WEIGHTS: dict[str, int] = {
    # 긴급성
    "breaking": 3, "속보": 3, "긴급": 3, "단독": 2,
    # 사고
    "hack": 3, "hacked": 3, "exploit": 3, "drained": 3, "해킹": 3, "탈취": 3,
    # 승인·거부 같은 결정
    # (영어는 단어 경계로 찾으므로 approves 처럼 변형된 형태도 따로 적어 둔다)
    "approve": 2, "approves": 2, "approved": 2, "approval": 2,
    "reject": 2, "rejects": 2, "rejected": 2,
    "승인": 2, "거부": 2, "기각": 2,
    # 상장/상장폐지 (국내 투자자에게 특히 중요)
    "상장폐지": 3, "delist": 3, "상장": 1, "listing": 1,
    # 급격한 가격 변동
    "surge": 1, "plunge": 2, "crash": 2, "all-time high": 2, "ath": 1,
    "급등": 1, "급락": 2, "폭락": 2, "사상 최고": 2, "신고가": 2,
    # 규제·거시 이벤트
    "sec": 1, "fomc": 1, "etf": 1, "금리": 1,
}

IMPORTANT_THRESHOLD = 4   # 이 점수 이상이면 3단계(중요)
NOTABLE_THRESHOLD = 2     # 이 점수 이상이면 2단계(주목)


@lru_cache(maxsize=None)
def _keyword_pattern(keyword: str) -> re.Pattern:
    """키워드 하나를 찾는 정규식을 만든다.

    영어는 '단어 경계'를 지켜야 한다. 그렇지 않으면 "eth" 가 "method" 안에서도 잡힌다.
    한국어는 조사가 붙으므로("비트코인이", "비트코인은") 경계 없이 부분 일치로 찾는다.
    """
    if keyword.isascii():
        return re.compile(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", re.IGNORECASE)
    return re.compile(re.escape(keyword))


def _contains(text: str, keyword: str) -> bool:
    return _keyword_pattern(keyword).search(text) is not None


def find_tags(text: str) -> list[str]:
    """글에 등장하는 코인·주제 태그를 순서대로 모은다."""
    tags = []
    for tag_name, keywords in {**COIN_KEYWORDS, **TOPIC_KEYWORDS}.items():
        if any(_contains(text, keyword) for keyword in keywords):
            tags.append(tag_name)
    return tags


def importance_score(text: str) -> int:
    """중요도 점수 = 등장한 키워드 가중치의 합."""
    return sum(weight for keyword, weight in IMPORTANCE_WEIGHTS.items() if _contains(text, keyword))


def importance_level(score: float) -> int:
    """점수를 1~3 단계로 바꾼다."""
    if score >= IMPORTANT_THRESHOLD:
        return 3
    if score >= NOTABLE_THRESHOLD:
        return 2
    return 1


def classify(news_item: NewsItem) -> NewsItem:
    """뉴스 한 건에 태그와 중요도를 채워 넣는다. 제목은 요약보다 더 중요하게 본다."""
    title_text = news_item.title
    full_text = f"{news_item.title} {news_item.summary}"
    news_item.tags = find_tags(full_text)
    # 요약에 나온 키워드는 절반만 센다: 제목이 기자가 가장 강조하고 싶은 내용이기 때문
    score = importance_score(title_text) + importance_score(news_item.summary) / 2
    news_item.importance = importance_level(score)
    return news_item
