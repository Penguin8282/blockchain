"""
프로젝트 전체가 공유하는 '설정' 모음.

왜 이런 파일을 따로 두는가?
  - 경로("data/raw/...")나 난수 시드(42) 같은 값을 여러 파일에 흩어 쓰면,
    나중에 하나만 바꿔도 다른 곳이 안 맞아 버그가 난다.
  - 그래서 "값은 한 곳에서만 정의하고, 나머지는 가져다 쓴다"는 원칙을 따른다.
  - 비밀값(API 키)은 코드에 직접 쓰지 않고 .env 파일에서 읽는다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ------------------------------------------------------------
# 1) 프로젝트 최상위 폴더 찾기
# ------------------------------------------------------------
# __file__ 은 지금 이 파일(src/config.py)의 경로다.
# .resolve() 로 절대경로를 만들고, .parent 를 두 번 올라가면 프로젝트 루트가 된다.
#   src/config.py -> (parent) src -> (parent) 프로젝트 루트
PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# 2) .env 파일 읽어오기
# ------------------------------------------------------------
# load_dotenv 는 .env 파일의 "KEY=VALUE" 들을 환경변수로 올려준다.
# .env 가 없어도 오류가 나지 않고, 그 경우 아래 기본값이 쓰인다.
ENV_FILE_PATH = PROJECT_ROOT_DIRECTORY / ".env"
load_dotenv(dotenv_path=ENV_FILE_PATH)


def _read_setting(setting_name: str, default_value: str) -> str:
    """환경변수 하나를 읽되, 없으면 기본값을 돌려주는 작은 도우미 함수."""
    # os.getenv(이름, 기본값) 은 환경변수가 없을 때 기본값을 반환한다.
    return os.getenv(setting_name, default_value)


# ------------------------------------------------------------
# 3) 폴더 경로 설정
# ------------------------------------------------------------
# 원본 데이터 폴더: 외부에서 받은 파일을 '절대 수정하지 않고' 그대로 두는 곳
RAW_DATA_DIRECTORY = PROJECT_ROOT_DIRECTORY / _read_setting("RAW_DATA_DIRECTORY", "data/raw")

# 가공 데이터 폴더: 결측 처리·특징 변환 등을 거친 결과를 저장하는 곳
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT_DIRECTORY / _read_setting(
    "PROCESSED_DATA_DIRECTORY", "data/processed"
)

# 보고서 폴더: 표(.md)와 그림(.png)을 저장하는 곳
REPORTS_DIRECTORY = PROJECT_ROOT_DIRECTORY / _read_setting("REPORTS_DIRECTORY", "reports")

# 그림 전용 하위 폴더
FIGURES_DIRECTORY = REPORTS_DIRECTORY / "figures"

# ------------------------------------------------------------
# 4) 분석 대상 원본 파일 경로
# ------------------------------------------------------------
RAW_DATASET_FILENAME = _read_setting("RAW_DATASET_FILENAME", "transaction_dataset.csv")
RAW_DATASET_PATH = RAW_DATA_DIRECTORY / RAW_DATASET_FILENAME

# ------------------------------------------------------------
# 5) 실험 재현성 설정
# ------------------------------------------------------------
# 난수 시드: 데이터를 나누거나 모델이 무작위 선택을 할 때 쓰는 '출발 숫자'.
# 같은 시드를 쓰면 몇 번을 돌려도 똑같은 결과가 나온다 → 결과 재현이 가능해진다.
RANDOM_SEED = int(_read_setting("RANDOM_SEED", "42"))

# 교차검증 fold 개수: 데이터를 5조각으로 나눠 5번 학습/평가한다는 뜻
CROSS_VALIDATION_FOLDS = int(_read_setting("CROSS_VALIDATION_FOLDS", "5"))

# ------------------------------------------------------------
# 6) 목표 변수(정답 열) 이름
# ------------------------------------------------------------
# 이 데이터셋에서 FLAG 열이 1이면 사기 주소, 0이면 정상 주소다.
TARGET_COLUMN_NAME = "FLAG"

# ------------------------------------------------------------
# 7) 외부 API 키 (현재 파이프라인에서는 사용하지 않음)
# ------------------------------------------------------------
# 값이 비어 있어도 문제없다. 나중에 온체인 API 를 붙일 때 사용할 자리.
ETHERSCAN_API_KEY = _read_setting("ETHERSCAN_API_KEY", "")
COVALENT_API_KEY = _read_setting("COVALENT_API_KEY", "")


def ensure_directories_exist() -> None:
    """
    분석에 필요한 폴더들이 없으면 만들어 준다.

    parents=True  : 중간 폴더까지 한꺼번에 생성
    exist_ok=True : 이미 있어도 오류를 내지 않음
    """
    for directory_path in (
        RAW_DATA_DIRECTORY,
        PROCESSED_DATA_DIRECTORY,
        REPORTS_DIRECTORY,
        FIGURES_DIRECTORY,
    ):
        directory_path.mkdir(parents=True, exist_ok=True)
