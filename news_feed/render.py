"""
모은 뉴스를 코인니스 라이브처럼 '시간순 타임라인' 웹페이지(HTML 파일 하나)로 만드는 단계.

  - 날짜별로 묶고, 각 뉴스 왼쪽에 시각을 붙인다.
  - 중요도에 따라 왼쪽 점 색이 달라진다 (빨강=중요, 주황=주목, 회색=일반).
  - 위쪽 버튼으로 태그(BTC, 규제…)·중요도·언어별로 거를 수 있다. (브라우저 안 JavaScript)
  - 5분마다 페이지가 스스로 새로고침된다 → watch 모드와 함께 쓰면 '라이브'처럼 보인다.

브라우저에서 파일을 그냥 열면 된다: news_feed/output/index.html
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from news_feed.classifier import COIN_KEYWORDS, TOPIC_KEYWORDS
from news_feed.fetcher import NewsItem
from news_feed.notifier import IMPORTANCE_LABELS, to_korea_time

WEEKDAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]

PAGE_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>블록체인 라이브 뉴스</title>
<style>
  :root {{
    --bg: #f6f7f9; --card: #ffffff; --text: #16181d; --muted: #6b7280; --line: #e5e7eb;
    --accent: #2563eb; --important: #e11d48; --notable: #f59e0b; --normal: #9ca3af;
    --chip: #eef2ff; --chip-text: #3730a3;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f1115; --card: #171a21; --text: #e5e7eb; --muted: #9ca3af; --line: #262a33;
      --accent: #60a5fa; --chip: #1e2235; --chip-text: #c7d2fe;
    }}
  }}
  * {{ box-sizing: border-box; }}
  [hidden] {{ display: none !important; }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
         font-family: -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 1; background: var(--card);
           border-bottom: 1px solid var(--line); padding: 12px 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .updated {{ color: var(--muted); font-size: 12px; }}
  .filters {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .filters button {{ border: 1px solid var(--line); background: var(--bg); color: var(--text);
                    border-radius: 999px; padding: 4px 10px; font-size: 12px; cursor: pointer; }}
  .filters button.on {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  input[type=search] {{ width: 100%; margin-top: 10px; padding: 8px 10px; font-size: 14px;
                       border: 1px solid var(--line); border-radius: 8px;
                       background: var(--bg); color: var(--text); }}
  main {{ max-width: 760px; margin: 0 auto; padding: 8px 16px 40px; }}
  .day {{ font-weight: 700; font-size: 14px; color: var(--muted); margin: 20px 0 8px; }}
  .item {{ display: grid; grid-template-columns: 46px 14px 1fr; gap: 8px;
          padding: 12px 0; border-bottom: 1px solid var(--line); }}
  .time {{ color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; margin-top: 4px; background: var(--normal); }}
  .lv3 .dot {{ background: var(--important); }}
  .lv2 .dot {{ background: var(--notable); }}
  .lv3 .title {{ color: var(--important); }}
  .title {{ font-weight: 600; font-size: 15px; line-height: 1.45; color: var(--text);
           text-decoration: none; overflow-wrap: anywhere; }}
  .title:hover {{ text-decoration: underline; }}
  .summary {{ color: var(--muted); font-size: 13px; line-height: 1.5; margin: 4px 0 0; }}
  .meta {{ margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
          font-size: 12px; color: var(--muted); }}
  .chip {{ background: var(--chip); color: var(--chip-text); border-radius: 4px; padding: 1px 6px; }}
  .empty {{ color: var(--muted); text-align: center; padding: 40px 0; }}
</style>
</head>
<body>
<header>
  <h1>⚡ 블록체인 라이브 뉴스</h1>
  <div class="updated">마지막 갱신 {updated_at} (KST) · {item_count}건 · 5분마다 자동 새로고침</div>
  <div class="filters" id="filters">
    <button data-filter="all" class="on">전체</button>
    <button data-filter="lv3">🔴 중요</button>
    <button data-filter="lv2">🟠 주목 이상</button>
    <button data-filter="ko">국내</button>
    <button data-filter="en">해외</button>
    {tag_buttons}
  </div>
  <input type="search" id="search" placeholder="제목·요약 검색">
</header>
<main id="timeline">
{timeline}
<p class="empty" id="empty" hidden>조건에 맞는 뉴스가 없습니다.</p>
</main>
<script>
  const items = [...document.querySelectorAll(".item")];
  const days = [...document.querySelectorAll(".day")];
  let activeFilter = "all";
  function matches(item, filter) {{
    if (filter === "all") return true;
    if (filter === "lv3") return item.dataset.level === "3";
    if (filter === "lv2") return Number(item.dataset.level) >= 2;
    if (filter === "ko" || filter === "en") return item.dataset.lang === filter;
    return item.dataset.tags.split("|").includes(filter);
  }}
  function apply() {{
    const keyword = document.getElementById("search").value.trim().toLowerCase();
    let shown = 0;
    for (const item of items) {{
      const ok = matches(item, activeFilter) &&
                 (!keyword || item.textContent.toLowerCase().includes(keyword));
      item.hidden = !ok;
      if (ok) shown++;
    }}
    // 보이는 뉴스가 하나도 없는 날짜 제목은 숨긴다
    for (const day of days) {{
      let next = day.nextElementSibling, any = false;
      while (next && next.classList.contains("item")) {{ if (!next.hidden) any = true; next = next.nextElementSibling; }}
      day.hidden = !any;
    }}
    document.getElementById("empty").hidden = shown > 0;
  }}
  document.getElementById("filters").addEventListener("click", (event) => {{
    const button = event.target.closest("button");
    if (!button) return;
    activeFilter = button.dataset.filter;
    document.querySelectorAll("#filters button").forEach(b => b.classList.toggle("on", b === button));
    apply();
  }});
  document.getElementById("search").addEventListener("input", apply);
</script>
</body>
</html>
"""


def _render_item(news_item: NewsItem) -> str:
    escape = html.escape
    time_text = to_korea_time(news_item.published_at).strftime("%H:%M")
    chips = "".join(f'<span class="chip">{escape(tag)}</span>' for tag in news_item.tags)
    summary_html = f'<p class="summary">{escape(news_item.summary)}</p>' if news_item.summary else ""
    level_label = IMPORTANCE_LABELS[news_item.importance]
    return (
        f'<article class="item lv{news_item.importance}" data-level="{news_item.importance}" '
        f'data-lang="{escape(news_item.language)}" data-tags="{escape("|".join(news_item.tags))}">'
        f'<div class="time">{time_text}</div><div class="dot" title="{level_label}"></div>'
        f'<div><a class="title" href="{escape(news_item.link, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{escape(news_item.title)}</a>{summary_html}'
        f'<div class="meta"><span>{escape(news_item.source)}</span>{chips}</div></div>'
        f"</article>"
    )


def render_html(news_items: list[NewsItem], now: datetime | None = None) -> str:
    """뉴스 목록(최신순)으로 HTML 문서 전체를 만든다."""
    now = now or datetime.now(timezone.utc)
    sorted_items = sorted(news_items, key=lambda item: item.published_at, reverse=True)

    timeline_parts = []
    for day, items_of_day in groupby(sorted_items, key=lambda item: to_korea_time(item.published_at).date()):
        day_label = f"{day.year}년 {day.month}월 {day.day}일 ({WEEKDAY_NAMES[day.weekday()]})"
        timeline_parts.append(f'<h2 class="day">{day_label}</h2>')
        timeline_parts.extend(_render_item(news_item) for news_item in items_of_day)

    tag_buttons = "\n    ".join(
        f'<button data-filter="{html.escape(tag)}">{html.escape(tag)}</button>'
        for tag in [*COIN_KEYWORDS, *TOPIC_KEYWORDS]
    )
    return PAGE_TEMPLATE.format(
        updated_at=to_korea_time(now).strftime("%Y-%m-%d %H:%M"),
        item_count=len(sorted_items),
        tag_buttons=tag_buttons,
        timeline="\n".join(timeline_parts),
    )


def write_html(news_items: list[NewsItem], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(news_items), encoding="utf-8")
    return output_path
