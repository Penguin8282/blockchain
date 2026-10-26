#!/usr/bin/env bash
# ============================================================
# GitHub Actions 에서 뉴스 DB(news.db)와 타임라인(index.html)을
# 'news-data' 브랜치에 보관하고 꺼내 오는 스크립트.
#
# 왜 필요한가?
#   Actions 는 매번 '새 빈 컴퓨터'에서 실행된다. 그냥 두면 지난번에 모은 뉴스를
#   기억하지 못해서 매번 "전부 새 뉴스"가 되어 버린다.
#   그래서 DB 파일을 별도 브랜치에 저장해 두고, 다음 실행 때 다시 꺼내 쓴다.
#   (사물함에 넣어 두고 다음 날 다시 꺼내는 것과 같다)
#
# 사용법:
#   news_state.sh restore <폴더>   # news-data 브랜치 내용을 <폴더>에 꺼내기 (없으면 빈 폴더)
#   news_state.sh save <폴더>      # <폴더> 내용을 news-data 브랜치에 저장하기
# ============================================================
set -euo pipefail

STATE_BRANCH="${NEWS_STATE_BRANCH:-news-data}"
command_name="$1"
state_directory="$2"

case "$command_name" in
  restore)
    # git worktree: 같은 저장소의 다른 브랜치를 옆 폴더에 함께 꺼내 두는 기능
    if git ls-remote --exit-code --heads origin "$STATE_BRANCH" >/dev/null; then
      git fetch --depth=1 origin "$STATE_BRANCH"
      git worktree add --detach "$state_directory" FETCH_HEAD
      echo "기존 뉴스 데이터를 불러왔습니다."
    else
      git worktree add --detach "$state_directory"
      git -C "$state_directory" rm -rfq --ignore-unmatch .
      echo "처음 실행입니다. 빈 상태로 시작합니다."
    fi
    ;;
  save)
    cd "$state_directory"
    git config user.name "github-actions[bot]"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    # 매번 '이력 없는 커밋 하나'로 덮어써서 브랜치 용량이 계속 불어나지 않게 한다.
    git checkout -q --orphan "save-$$"
    git add -A
    git commit -qm "뉴스 데이터 갱신 $(date -u +%Y-%m-%dT%H:%MZ)"
    git push -q --force origin "HEAD:refs/heads/$STATE_BRANCH"
    echo "뉴스 데이터를 $STATE_BRANCH 브랜치에 저장했습니다."
    ;;
  *)
    echo "사용법: $0 {restore|save} <폴더>" >&2
    exit 2
    ;;
esac
