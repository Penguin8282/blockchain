"""
단계 1 확인용 스크립트.

이 스크립트가 하는 일:
  1) 필요한 패키지가 설치되어 있는지 확인
  2) 필요한 폴더가 있는지 확인(없으면 생성)
  3) .env 파일이 있는지 확인
  4) 분석 대상 원본 CSV 가 있는지 확인하고, 있으면 크기/열 개수를 알려준다

실행:  python -m src.check_environment
"""

import importlib
import sys

from src import config

# 확인할 패키지 목록: (파이썬에서 import 할 때 쓰는 이름, pip 에 설치할 때 쓰는 이름)
# 두 이름이 다른 경우가 있어서 쌍으로 관리한다. 예) import sklearn <-> pip install scikit-learn
REQUIRED_PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("dotenv", "python-dotenv"),
]


def check_packages() -> list[str]:
    """설치된 패키지를 확인하고, 빠진 패키지의 'pip 이름' 목록을 돌려준다."""
    missing_package_names: list[str] = []

    print("[1/4] 패키지 설치 확인")
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            # importlib.import_module 은 문자열 이름으로 패키지를 불러온다.
            module = importlib.import_module(import_name)
            # 어떤 패키지는 __version__ 속성이 없을 수 있으므로 getattr 로 안전하게 읽는다.
            version_text = getattr(module, "__version__", "(버전 정보 없음)")
            print(f"    OK   {pip_name:<16} {version_text}")
        except ImportError:
            print(f"    없음 {pip_name:<16} <- 설치 필요")
            missing_package_names.append(pip_name)

    return missing_package_names


def check_directories() -> None:
    """분석에 필요한 폴더를 확인하고 없으면 만든다."""
    print("\n[2/4] 폴더 확인 (없으면 자동 생성)")
    config.ensure_directories_exist()
    for directory_path in (
        config.RAW_DATA_DIRECTORY,
        config.PROCESSED_DATA_DIRECTORY,
        config.REPORTS_DIRECTORY,
        config.FIGURES_DIRECTORY,
    ):
        # relative_to 로 프로젝트 루트 기준의 짧은 경로만 보여준다(출력이 깔끔해짐).
        short_path = directory_path.relative_to(config.PROJECT_ROOT_DIRECTORY)
        print(f"    OK   {short_path}")


def check_env_file() -> None:
    """.env 파일이 있는지 확인한다. 없으면 기본값으로 동작한다고 안내한다."""
    print("\n[3/4] .env 설정 파일 확인")
    if config.ENV_FILE_PATH.exists():
        print("    OK   .env 파일을 찾았습니다.")
    else:
        print("    안내 .env 파일이 없습니다. 지금은 코드의 기본값으로 동작합니다.")
        print("         비밀값을 쓰려면:  cp .env.example .env")

    print(f"    설정 RANDOM_SEED            = {config.RANDOM_SEED}")
    print(f"    설정 CROSS_VALIDATION_FOLDS = {config.CROSS_VALIDATION_FOLDS}")


def check_raw_dataset() -> bool:
    """원본 CSV 파일이 있는지 확인하고, 있으면 간단한 요약을 출력한다."""
    print("\n[4/4] 원본 데이터 확인")
    dataset_path = config.RAW_DATASET_PATH
    short_path = dataset_path.relative_to(config.PROJECT_ROOT_DIRECTORY)

    if not dataset_path.exists():
        print(f"    없음 {short_path}")
        print("         Kaggle 'Ethereum Fraud Detection Dataset' 의 CSV 파일을")
        print("         위 경로에 넣어 주세요. (파일명이 다르면 .env 의")
        print("         RAW_DATASET_FILENAME 값을 바꾸면 됩니다.)")
        return False

    # 파일 크기를 메가바이트 단위로 환산해서 보여준다. (1 MB = 1024 * 1024 바이트)
    file_size_in_megabytes = dataset_path.stat().st_size / (1024 * 1024)
    print(f"    OK   {short_path}  ({file_size_in_megabytes:.2f} MB)")

    try:
        import pandas as pd

        # nrows=5 : 확인용이므로 앞의 5줄만 읽어 빠르게 끝낸다.
        sample_dataframe = pd.read_csv(dataset_path, nrows=5)
        print(f"    OK   열 개수 = {sample_dataframe.shape[1]}개")
        has_target = config.TARGET_COLUMN_NAME in sample_dataframe.columns
        print(f"    OK   목표 열 '{config.TARGET_COLUMN_NAME}' 존재 여부 = {has_target}")
    except ImportError:
        print("    안내 pandas 가 아직 설치되지 않아 내용 확인은 건너뜁니다.")

    return True


def main() -> int:
    """전체 점검을 순서대로 실행하고, 종료 코드(0=정상)를 돌려준다."""
    print("=" * 60)
    print("이더리움 사기 주소 분류 프로젝트 — 환경 점검")
    print("=" * 60)

    missing_package_names = check_packages()
    check_directories()
    check_env_file()
    dataset_exists = check_raw_dataset()

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)

    if missing_package_names:
        print("- 빠진 패키지가 있습니다. 아래 명령으로 설치하세요:")
        print(f"    pip install {' '.join(missing_package_names)}")
    else:
        print("- 패키지: 모두 설치됨")

    if dataset_exists:
        print("- 데이터: 준비됨 -> 단계 2(EDA)로 넘어갈 수 있습니다.")
    else:
        print("- 데이터: 없음 -> CSV 파일을 data/raw/ 에 넣어 주세요.")

    # 종료 코드: 문제가 하나라도 있으면 1, 모두 정상이면 0
    return 0 if (not missing_package_names and dataset_exists) else 1


if __name__ == "__main__":
    # sys.exit 에 종료 코드를 넘기면, 터미널이나 CI 가 성공/실패를 판단할 수 있다.
    sys.exit(main())
