"""
Kaggle 에서 원본 데이터셋을 내려받아 data/raw/ 에 저장하는 스크립트.

핵심 규칙: **캐시**
  - 이미 data/raw/transaction_dataset.csv 가 있으면 다시 내려받지 않는다.
  - 즉, 몇 번을 실행해도 네트워크 호출은 처음 한 번뿐이다.
  - 강제로 다시 받고 싶으면 --force 옵션을 준다.

실행:
    python -m src.download_dataset            # 없으면 받고, 있으면 그냥 통과
    python -m src.download_dataset --force    # 무조건 다시 받기

참고: kagglehub 는 인증 정보가 필요할 수 있다.
  - Kaggle 사이트 > Settings > API > "Create New Token" 으로 kaggle.json 을 받아
    ~/.kaggle/kaggle.json 에 두거나,
  - .env 에 KAGGLE_USERNAME / KAGGLE_KEY 를 넣어 두면 된다.
"""

import argparse
import shutil
import sys
from pathlib import Path

from src import config

# Kaggle 데이터셋 식별자: "계정이름/데이터셋이름" 형식
KAGGLE_DATASET_IDENTIFIER = "vagifa/ethereum-frauddetection-dataset"


def find_csv_file_in_directory(directory_path: Path) -> Path | None:
    """
    폴더 안에서 CSV 파일을 하나 찾아 돌려준다.

    kagglehub 는 압축을 푼 '폴더 경로'를 돌려주는데, 그 안의 파일 이름이
    데이터셋마다 다를 수 있다. 그래서 이름을 추측하지 않고 실제로 찾아본다.
    rglob("*.csv") 는 하위 폴더까지 뒤져 CSV 를 모두 찾는다는 뜻이다.
    """
    csv_file_paths = sorted(directory_path.rglob("*.csv"))
    if not csv_file_paths:
        return None

    # 파일이 여러 개면 가장 큰 파일을 '본 데이터'로 본다.
    # (보조 파일보다 본 데이터가 큰 것이 일반적이기 때문)
    largest_csv_path = max(csv_file_paths, key=lambda path: path.stat().st_size)
    return largest_csv_path


def download_dataset(force_redownload: bool = False) -> Path:
    """
    데이터셋을 내려받아 data/raw/ 로 복사하고, 최종 파일 경로를 돌려준다.

    force_redownload=False 이면, 파일이 이미 있을 때 네트워크를 쓰지 않고 바로 반환한다.
    """
    config.ensure_directories_exist()
    destination_path = config.RAW_DATASET_PATH

    # ----- 캐시 확인: 이미 받아 둔 파일이 있으면 재호출하지 않는다 -----
    if destination_path.exists() and not force_redownload:
        file_size_in_megabytes = destination_path.stat().st_size / (1024 * 1024)
        print(f"[캐시 사용] 이미 존재합니다: {destination_path} ({file_size_in_megabytes:.2f} MB)")
        print("           다시 받으려면 --force 옵션을 주세요.")
        return destination_path

    # ----- 실제 내려받기 -----
    try:
        import kagglehub
    except ImportError:
        print("[오류] kagglehub 가 설치되어 있지 않습니다.")
        print("       pip install kagglehub")
        raise

    print(f"[다운로드] Kaggle 에서 '{KAGGLE_DATASET_IDENTIFIER}' 를 받는 중...")
    # kagglehub 는 파일을 자기 캐시 폴더에 풀고, 그 '폴더 경로'를 문자열로 돌려준다.
    downloaded_directory_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_IDENTIFIER))
    print(f"[다운로드] 완료: {downloaded_directory_path}")

    source_csv_path = find_csv_file_in_directory(downloaded_directory_path)
    if source_csv_path is None:
        raise FileNotFoundError(
            f"내려받은 폴더에서 CSV 파일을 찾지 못했습니다: {downloaded_directory_path}"
        )

    # 프로젝트 안(data/raw/)으로 복사해 둔다.
    # 이렇게 해야 kagglehub 캐시가 지워져도 프로젝트가 계속 동작한다.
    shutil.copy2(source_csv_path, destination_path)
    file_size_in_megabytes = destination_path.stat().st_size / (1024 * 1024)
    print(f"[저장] {destination_path} ({file_size_in_megabytes:.2f} MB)")

    return destination_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle 이더리움 사기 탐지 데이터셋 내려받기")
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 파일이 있어도 무시하고 다시 내려받는다",
    )
    arguments = parser.parse_args()

    try:
        download_dataset(force_redownload=arguments.force)
    except Exception as error:  # 네트워크 차단·인증 실패 등 모든 경우를 안내로 바꿔 준다
        print(f"\n[실패] {type(error).__name__}: {error}")
        print("\n대안: 브라우저에서 아래 주소로 직접 내려받아")
        print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET_IDENTIFIER}")
        print(f"  CSV 파일을 {config.RAW_DATASET_PATH} 에 두면 됩니다.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
