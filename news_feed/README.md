# ⚡ 블록체인 라이브 뉴스 수집기

코인니스 라이브(Coinness Live)처럼 **국내외 블록체인 뉴스를 한곳에 모아 시간순으로 보여주고,
중요한 뉴스는 바로 알려주는** 작은 프로그램입니다.

```
[RSS 10곳] → 수집(fetcher) → 분류(classifier) → 저장·중복제거(storage) → 알림(notifier)
                                                                     └→ 타임라인 HTML(render)
```

| 단계 | 파일 | 하는 일 |
|---|---|---|
| 출처 | `sources.py` | CoinDesk, Cointelegraph, Decrypt, The Block, 블록미디어, 토큰포스트, 구글뉴스 검색 등 |
| 수집 | `fetcher.py` | RSS 를 동시에 내려받아 제목·링크·시각·요약을 같은 규격으로 정리 |
| 분류 | `classifier.py` | 키워드로 태그(BTC, ETH, 규제, 보안, ETF…)와 중요도(🔴중요/🟠주목/⚪일반) 부여 |
| 저장 | `storage.py` | SQLite 에 저장. 같은 주소·같은 제목 기사는 한 번만 저장 → "새 뉴스"만 골라냄 |
| 알림 | `notifier.py` | 터미널 출력 + (선택) 텔레그램·디스코드로 푸시 |
| 화면 | `render.py` | 날짜별 타임라인 웹페이지. 태그·중요도·국내/해외 필터, 검색, 5분 자동 새로고침 |

## 빠른 시작

```bash
pip install -r news_feed/requirements.txt

python -m news_feed run            # 한 번 수집 (첫 실행은 저장만 하고 알림은 생략)
python -m news_feed watch -i 300   # 5분마다 계속 수집 → 새 뉴스가 오면 바로 출력/알림
```

그다음 브라우저에서 `news_feed/output/index.html` 을 열면 타임라인이 보입니다.
`watch` 를 켜 두면 페이지가 5분마다 새로고침되어 '라이브'처럼 동작합니다.

```bash
python -m news_feed list --min-level 2           # 저장된 뉴스 중 주목 이상만
python -m news_feed list --tag 보안 -n 10        # 해킹·보안 뉴스 최근 10건
python -m news_feed -v run                       # 출처별 수집 실패 로그까지 보기
```

터미널 출력 예시:

```
🔴 10/26 12:10 [CoinDesk] BREAKING: Exchange hacked, $120M in ETH drained #ETH #보안
🔴 10/26 11:00 [CoinDesk] SEC approves new spot Bitcoin ETF applications #BTC #규제 #ETF
⚪ 10/25 23:30 [토큰포스트] 업비트, 솔라나 기반 신규 토큰 상장 #SOL #거래소
```

## 휴대폰으로 알림 받기 (텔레그램)

1. 텔레그램에서 `@BotFather` 에게 `/newbot` → 봇 토큰을 받습니다.
2. 만든 봇에게 아무 메시지나 보낸 뒤
   `https://api.telegram.org/bot<토큰>/getUpdates` 를 열어 `chat.id` 숫자를 찾습니다.
3. 프로젝트 루트의 `.env` 에 적습니다 (`.env` 는 깃에 올라가지 않습니다).

```
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=987654321
NEWS_ALERT_MIN_LEVEL=2      # 1=전부, 2=주목 이상(기본), 3=중요만
```

디스코드는 채널 설정 → 연동 → 웹후크 주소를 `DISCORD_WEBHOOK_URL` 에 넣으면 됩니다.

## 중요도는 어떻게 매기나?

```
점수 = Σ w(제목에 나온 키워드) + ½ · Σ w(요약에 나온 키워드)
```

- `w` : 키워드 가중치. 예) `해킹`·`hack`·`속보`·`상장폐지` = 3, `승인`·`급락`·`plunge` = 2, `SEC`·`ETF`·`금리` = 1
- `½` : 요약은 제목보다 덜 중요하게 반영
- 점수 ≥ 4 → 🔴 중요, ≥ 2 → 🟠 주목, 나머지 ⚪ 일반

예) "SEC approves new spot Bitcoin ETF" → approves(2) + SEC(1) + ETF(1) = 4 → 🔴

키워드와 가중치는 `classifier.py` 맨 위 사전(dict)만 고치면 바로 바뀝니다.
영어 키워드는 **단어 경계**로 찾아서 `method` 안의 `eth` 같은 오탐을 막고,
한국어는 조사(`비트코인이`, `비트코인은`)가 붙으므로 부분 일치로 찾습니다.

## 출처 추가·삭제

`sources.py` 의 `DEFAULT_SOURCES` 에 한 줄 추가하면 됩니다.

```python
FeedSource("매체 이름", "https://매체주소/rss", "ko"),
FeedSource("구글뉴스·이더리움", google_news_search("이더리움 when:1d"), "ko"),
```

## 알아두면 좋은 한계

- **RSS 주소는 바뀔 수 있습니다.** 작성 환경에서는 외부 사이트 접속이 막혀 있어 각 주소의 실제
  응답을 확인하지 못했습니다. `-v` 옵션으로 실행해 `수집 실패` 가 뜨는 출처는 주소를 고쳐 주세요.
  한 출처가 실패해도 나머지는 정상 수집됩니다.
- 코인니스 자체 뉴스는 공개 RSS 가 없는 것으로 알고 있어(확실하지 않음) 포함하지 않았습니다.
  사이트를 긁어오는(스크래핑) 방식은 이용약관 문제가 있을 수 있어 넣지 않았습니다.
- RSS 는 매체가 글을 올린 뒤 몇 분 늦게 갱신되기도 해서, 코인니스처럼 '초 단위' 속보는 아닙니다.
- 중요도는 키워드 규칙이라 문맥을 모릅니다 (예: "해킹 피해 없었다"도 🔴).
  다음 단계로, 실제로 클릭/저장한 기사를 정답 라벨로 모아 로지스틱 회귀나 LLM 분류로 바꿔 볼 수 있습니다.

## 테스트

```bash
python -m pytest tests/test_news_feed.py
```

네트워크 없이 `tests/fixtures/sample_rss.xml` 로 파싱·태그·중요도·중복제거·첫 실행 알림 생략·HTML 이스케이프를 검사합니다.
