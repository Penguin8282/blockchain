"""
새 뉴스를 사람에게 '알려주는' 단계.

알림 채널 3가지:
  1) 터미널 출력  — 항상 동작
  2) 텔레그램 봇  — TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 환경변수가 있을 때만
  3) 디스코드     — DISCORD_WEBHOOK_URL 환경변수가 있을 때만

휴대폰 푸시 알림처럼 쓰고 싶으면 텔레그램이 가장 간단하다. (README 참고)
"""

from __future__ import annotations

import html
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from news_feed.fetcher import NewsItem

logger = logging.getLogger(__name__)

KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")
IMPORTANCE_ICONS = {3: "🔴", 2: "🟠", 1: "⚪"}
IMPORTANCE_LABELS = {3: "중요", 2: "주목", 1: "일반"}


def to_korea_time(moment: datetime) -> datetime:
    return moment.astimezone(KOREA_TIMEZONE)


def format_console_line(news_item: NewsItem) -> str:
    """터미널에 찍을 한 줄. 예) 🔴 10/26 13:05 [CoinDesk] 제목 #BTC #보안"""
    timestamp = to_korea_time(news_item.published_at).strftime("%m/%d %H:%M")
    tag_text = " ".join(f"#{tag}" for tag in news_item.tags)
    icon = IMPORTANCE_ICONS[news_item.importance]
    return f"{icon} {timestamp} [{news_item.source}] {news_item.title} {tag_text}".rstrip()


def format_telegram_message(news_item: NewsItem) -> str:
    """텔레그램용 메시지 (HTML 형식: <b>굵게</b>, <a href>링크</a>)."""
    icon = IMPORTANCE_ICONS[news_item.importance]
    timestamp = to_korea_time(news_item.published_at).strftime("%H:%M")
    lines = [
        f"{icon} <b>{html.escape(news_item.title)}</b>",
        f"{timestamp} · {html.escape(news_item.source)}",
    ]
    if news_item.summary:
        lines.append(html.escape(news_item.summary))
    if news_item.tags:
        lines.append(" ".join(f"#{html.escape(tag)}" for tag in news_item.tags))
    lines.append(f'<a href="{html.escape(news_item.link, quote=True)}">원문 보기</a>')
    return "\n".join(lines)


def send_telegram(news_item: NewsItem, bot_token: str, chat_id: str) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": format_telegram_message(news_item),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=payload, timeout=10).raise_for_status()
        return True
    except requests.RequestException as error:
        # 토큰이 들어 있는 URL 을 그대로 로그에 남기지 않도록 오류 종류만 기록한다
        logger.warning("텔레그램 전송 실패: %s", type(error).__name__)
        return False


def send_discord(news_item: NewsItem, webhook_url: str) -> bool:
    icon = IMPORTANCE_ICONS[news_item.importance]
    tag_text = " ".join(f"#{tag}" for tag in news_item.tags)
    content = f"{icon} **{news_item.title}**\n{news_item.source} {tag_text}\n<{news_item.link}>"
    try:
        requests.post(webhook_url, json={"content": content[:2000]}, timeout=10).raise_for_status()
        return True
    except requests.RequestException as error:
        logger.warning("디스코드 전송 실패: %s", type(error).__name__)
        return False


def notify(news_items: list[NewsItem], min_importance: int = 1) -> None:
    """새 뉴스를 모든 설정된 채널로 보낸다.

    터미널에는 전부 찍고, 메신저(텔레그램·디스코드)에는 min_importance 이상만 보낸다.
    모든 뉴스를 휴대폰으로 받으면 하루 수백 개라 알림 피로가 생기기 때문이다.
    """
    for news_item in news_items:
        print(format_console_line(news_item), flush=True)

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    for news_item in news_items:
        if news_item.importance < min_importance:
            continue
        if telegram_token and telegram_chat_id:
            send_telegram(news_item, telegram_token, telegram_chat_id)
        if discord_webhook_url:
            send_discord(news_item, discord_webhook_url)
