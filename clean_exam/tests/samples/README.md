# 테스트용 시험지 사진을 두는 곳

여기에 실제 시험지 사진(.jpg/.png/.heic)을 넣으면 눈으로 결과를 확인할 수 있습니다.

```bash
python -m engine.clean tests/samples/내사진.jpg --out /tmp/결과.png --debug
```

**사진 파일은 깃에 올라가지 않습니다.** 이 저장소가 공개(public)라서, 학생 필기와 이름이
담긴 사진을 올리면 그대로 인터넷에 공개되기 때문입니다(`.gitignore` 에서 막아 두었습니다).

사진이 하나도 없어도 `pytest -q` 는 전부 통과합니다.
`tests/synthetic_exam.py` 가 정답을 아는 합성 시험지를 만들어 쓰기 때문입니다.
