# 깔끔 시험지

학생이 푼 수학 시험지 사진에서 **필기만 지우고 인쇄된 문제 글자와 그림은 원래 양식 그대로**
남기는 도구입니다. 학원 내부에서만 씁니다.

현재 **1단계(규칙 기반 필기 지우기 엔진 + 클로드 판정관)** 까지 되어 있습니다.
2단계(웹사이트) 이후는 아직 없습니다.

---

## 설치 → 실행

```bash
pip install -r requirements.txt
python -m engine.clean tests/samples/01_기하_0071.jpg --out 결과.png
```

끝나면 `결과.png` 와 `compare_결과.png`(원본/결과 나란히 비교) 두 장이 생깁니다.

### 더 정확하게 하려면 (Anthropic 키)

```bash
python -m engine.clean 사진.jpg --out 결과.png --api-key sk-ant-...
```

키가 없어도 **그냥 됩니다.** 규칙만으로 처리하고, 결과 JSON 에 `api_skipped: true` 와
한국어 안내가 담깁니다. 키를 넣으면 규칙이 애매해한 획만 클로드에게 물어봐 정확도가 올라갑니다.

| 옵션 | 뜻 |
|---|---|
| `--api-key sk-ant-...` | 판정관을 쓴다. 안 주면 환경변수 `ANTHROPIC_API_KEY` 를 본다 |
| `--no-judge` | 키가 있어도 판정관을 쓰지 않는다 |
| `--quality` | 고품질 모델을 쓴다(비용↑) |
| `--debug` | 중간 이미지를 `debug/` 에 저장한다 (어디서 잘못됐는지 볼 때) |
| `--config 경로` | 다른 `config.yaml` 로 실험한다 |

---

## 키는 어디를 지나가고 어디에 남지 않나

이 프로젝트는 **서버에 운영자 키를 두지 않습니다(BYOK).** 강사가 각자 자기 키를 넣고
비용도 각자 부담합니다.

- 키가 지나가는 곳: CLI 인자(또는 환경변수) → `engine.clean.clean_image(api_key=...)` →
  `engine/claude_judge.py` → `services/claude_client.py` → Anthropic 서버.
  **전부 함수 인자**이고, 전역 변수나 설정 파일에 담기지 않습니다.
- 키가 **절대 남지 않는 곳**: 디스크, 로그, 응답 캐시(`data/api_cache/`),
  사용량 기록(`data/usage/`), 오류 메시지, 예외 스택.
  `sk-ant-` 로 시작하는 문자열은 로그 필터가 자동으로 `sk-ant-***` 로 가립니다
  (`services/key_safety.py`, 테스트 `tests/test_key_safety.py`).
- 응답 캐시의 열쇠는 **이미지 해시 + 프롬프트 해시 + 모델명**뿐입니다. 키는 안 들어갑니다.
  그래서 같은 사진을 다른 강사가 올리면 두 번째 사람은 호출 0회로 재사용합니다.

---

## 조정하기

모든 값은 `config.yaml` 한 파일에 있고, 항목마다 "이 값을 올리면 어떻게 되는지"가 주석으로
붙어 있습니다. 자주 만지게 되는 것:

| 값 | 언제 만지나 |
|---|---|
| `stroke_filter.remove_threshold` / `keep_threshold` | 필기가 덜 지워지거나 인쇄가 지워질 때. 두 값 사이가 "애매" 구간이고, **이 구간만 판정관에게 가므로 넓히면 정확도↑·비용↑** |
| `color_filter.saturation_threshold` | 색펜이 덜 지워지거나(내린다), 컬러 인쇄가 지워질 때(올린다) |
| `graph_enhance.dilate_px` | 그래프를 더 굵게/얇게 |
| `finalize.bleed_through_cutoff` | 뒷장 비침이 남을 때(내린다), 흐린 인쇄가 날아갈 때(올린다) |
| `claude.max_judge_calls_per_page` | 한 장당 비용 상한 |

---

## 테스트

```bash
pytest -q          # 전부 mock. 키 없이 돌아간다
pytest -q --live   # 실제 API 를 한 번 부른다 (ANTHROPIC_API_KEY 필요, 소액 과금)
```

실제 시험지 사진이 없어도 `tests/synthetic_exam.py` 가 합성 시험지를 만들어
보정 정확도·필기 제거율·처리 시간을 자동으로 잽니다.

---

## 폴더

```
engine/     이미지 처리 (웹과 독립, CLI 로도 실행 가능)
services/   Claude 호출 단일 창구 + 프롬프트 텍스트(코드에 하드코딩 안 함)
tests/      샘플 사진 + 합성 시험지 생성기 + 테스트
config.yaml 모든 조정 값
data/       올린 사진·결과·라벨·캐시·사용량 (깃에 올리지 않음)
```
