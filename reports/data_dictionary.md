# 데이터 사전 (Data Dictionary)

`data/raw/transaction_dataset.csv` 의 모든 열을 한국어 한 줄로 정리한 문서입니다.

- 전체 크기: **9,841행 × 51열**
- 이 문서는 `python -m src.build_data_dictionary` 로 다시 생성됩니다.
  (설명 문장은 사람이 쓰고, 결측·고유값 숫자는 매번 코드가 다시 계산합니다.)

### 확신 수준 표기

| 표기 | 뜻 |
|---|---|
| 확실 | 열 이름과 실제 값 분포가 일치해 해석에 이견이 없음 |
| 추정 | 이름상 의미는 분명하나 세부 계산 방식(0값 포함 여부 등)까지는 확인 못 함 |
| **확인 필요** | 이름만으로 의미를 알 수 없음. **추측하지 않고 남겨 둠** |

---

## 식별자 (3개)

| 열 이름 | 한국어 설명 | 단위 | 결측 | 고유값 | 확신 |
|---|---|---|---|---|---|
| `Unnamed: 0` | CSV 를 저장할 때 딸려 들어온 행 번호. 분석에 쓰지 않음 | - | 0 | 9,841 | 확실 |
| `Index` | 출처 불명의 번호. 9841행인데 고유값이 4729개뿐이라 단순 행 번호가 아님 | - | 0 | 4,729 | **확인 필요** |
| `Address` | 이더리움 지갑 주소(0x로 시작하는 문자열). 행을 구분하는 이름표 | - | 0 | 9,816 | 확실 |

## 목표 변수 (1개)

| 열 이름 | 한국어 설명 | 단위 | 결측 | 고유값 | 확신 |
|---|---|---|---|---|---|
| `FLAG` | 목표 변수. 1 = 사기(피싱·스캠) 주소, 0 = 정상 주소 | 0/1 | 0 | 2 | 확실 |

## 거래 빈도 (6개)

| 열 이름 | 한국어 설명 | 단위 | 결측 | 고유값 | 확신 |
|---|---|---|---|---|---|
| `Sent tnx` | 이 지갑이 이더를 '보낸' 거래의 총 횟수 | 건 | 0 | 641 | 확실 |
| `Received Tnx` | 이 지갑이 이더를 '받은' 거래의 총 횟수 | 건 | 0 | 727 | 확실 |
| `Number of Created Contracts` | 이 지갑이 새로 만든 스마트 컨트랙트의 개수 | 개 | 0 | 20 | 확실 |
| `Unique Received From Addresses` | 이 지갑에 이더를 보내 준 '서로 다른' 주소의 수 | 개 | 0 | 256 | 확실 |
| `Unique Sent To Addresses` | 이 지갑이 이더를 보낸 '서로 다른' 주소의 수 | 개 | 0 | 258 | 확실 |
| `total transactions (including tnx to create contract` | 보낸 거래 + 받은 거래 + 컨트랙트 생성 거래를 모두 합한 총 거래 수 | 건 | 0 | 897 | 추정 |

## 금액 (13개)

| 열 이름 | 한국어 설명 | 단위 | 결측 | 고유값 | 확신 |
|---|---|---|---|---|---|
| `min value received` | 받은 거래들 중 가장 작은 금액 | Ether | 0 | 4,589 | 추정 |
| `max value received` | 받은 거래들 중 가장 큰 금액 | Ether | 0 | 6,302 | 추정 |
| `avg val received` | 받은 거래 1건당 평균 금액 | Ether | 0 | 6,767 | 추정 |
| `min val sent` | 보낸 거래들 중 가장 작은 금액 | Ether | 0 | 4,719 | 추정 |
| `max val sent` | 보낸 거래들 중 가장 큰 금액 | Ether | 0 | 6,647 | 추정 |
| `avg val sent` | 보낸 거래 1건당 평균 금액 | Ether | 0 | 5,854 | 추정 |
| `min value sent to contract` | 스마트 컨트랙트로 보낸 거래 중 가장 작은 금액 | Ether | 0 | 3 | 추정 |
| `max val sent to contract` | 스마트 컨트랙트로 보낸 거래 중 가장 큰 금액 | Ether | 0 | 4 | 추정 |
| `avg value sent to contract` | 스마트 컨트랙트로 보낸 거래 1건당 평균 금액 | Ether | 0 | 4 | 추정 |
| `total Ether sent` | 이 지갑이 보낸 이더의 총합 | Ether | 0 | 5,868 | 확실 |
| `total ether received` | 이 지갑이 받은 이더의 총합 | Ether | 0 | 6,728 | 확실 |
| `total ether sent contracts` | 스마트 컨트랙트로 보낸 이더의 총합 | Ether | 0 | 4 | 확실 |
| `total ether balance` | 잔액. '받은 총액 - 보낸 총액'으로 계산된 것으로 보임(음수도 존재) | Ether | 0 | 5,717 | 추정 |

## 시간 간격 (3개)

| 열 이름 | 한국어 설명 | 단위 | 결측 | 고유값 | 확신 |
|---|---|---|---|---|---|
| `Avg min between sent tnx` | 보낸 거래와 그 다음 보낸 거래 사이의 평균 시간 간격 | 분 | 0 | 5,013 | 확실 |
| `Avg min between received tnx` | 받은 거래와 그 다음 받은 거래 사이의 평균 시간 간격 | 분 | 0 | 6,223 | 확실 |
| `Time Diff between first and last (Mins)` | 첫 거래부터 마지막 거래까지의 총 기간. 사실상 '지갑의 활동 수명' | 분 | 0 | 7,810 | 확실 |

## 토큰 활동 (25개)

| 열 이름 | 한국어 설명 | 단위 | 결측 | 고유값 | 확신 |
|---|---|---|---|---|---|
| `Total ERC20 tnxs` | ERC20 토큰 거래의 총 횟수(보낸 것 + 받은 것) | 건 | 829 (8.4%) | 300 | 추정 |
| `ERC20 total Ether received` | 받은 ERC20 토큰 수량의 총합. 이름에 'Ether'가 있지만 실제로는 토큰 수량으로 보임 | 토큰 수량 | 829 (8.4%) | 3,460 | **확인 필요** |
| `ERC20 total ether sent` | 보낸 ERC20 토큰 수량의 총합. 위와 같은 이유로 단위가 불명확 | 토큰 수량 | 829 (8.4%) | 1,415 | **확인 필요** |
| `ERC20 total Ether sent contract` | 컨트랙트로 보낸 ERC20 토큰 수량의 총합 | 토큰 수량 | 829 (8.4%) | 29 | **확인 필요** |
| `ERC20 uniq sent addr` | ERC20 토큰을 보낸 '서로 다른' 상대 주소의 수 | 개 | 829 (8.4%) | 107 | 추정 |
| `ERC20 uniq rec addr` | ERC20 토큰을 받은 '서로 다른' 상대 주소의 수 | 개 | 829 (8.4%) | 147 | 추정 |
| `ERC20 uniq sent addr.1` | 원본 CSV 에 같은 이름의 열이 두 개 있어 pandas 가 '.1'을 붙인 열. 앞의 'ERC20 uniq sent addr' 와 값이 83%만 일치하고 분포도 전혀 달라 별개의 의미로 추정됨 | 개 | 829 (8.4%) | 4 | **확인 필요** |
| `ERC20 uniq rec contract addr` | ERC20 토큰을 받은 '서로 다른' 컨트랙트 주소의 수 | 개 | 829 (8.4%) | 123 | 추정 |
| `ERC20 avg time between sent tnx` | ERC20 토큰을 보낸 거래 사이의 평균 간격. 전 행이 0 | 분 | 829 (8.4%) | 1 | **확인 필요** |
| `ERC20 avg time between rec tnx` | ERC20 토큰을 받은 거래 사이의 평균 간격. 전 행이 0 | 분 | 829 (8.4%) | 1 | **확인 필요** |
| `ERC20 avg time between rec 2 tnx` | 이름의 'rec 2'가 무엇을 뜻하는지 알 수 없음. 전 행이 0 | 분 | 829 (8.4%) | 1 | **확인 필요** |
| `ERC20 avg time between contract tnx` | ERC20 컨트랙트 거래 사이의 평균 간격. 전 행이 0 | 분 | 829 (8.4%) | 1 | **확인 필요** |
| `ERC20 min val rec` | 받은 ERC20 토큰 거래 중 가장 작은 수량 | 토큰 수량 | 829 (8.4%) | 1,276 | 추정 |
| `ERC20 max val rec` | 받은 ERC20 토큰 거래 중 가장 큰 수량 | 토큰 수량 | 829 (8.4%) | 2,647 | 추정 |
| `ERC20 avg val rec` | 받은 ERC20 토큰 거래 1건당 평균 수량 | 토큰 수량 | 829 (8.4%) | 3,380 | 추정 |
| `ERC20 min val sent` | 보낸 ERC20 토큰 거래 중 가장 작은 수량 | 토큰 수량 | 829 (8.4%) | 476 | 추정 |
| `ERC20 max val sent` | 보낸 ERC20 토큰 거래 중 가장 큰 수량 | 토큰 수량 | 829 (8.4%) | 1,130 | 추정 |
| `ERC20 avg val sent` | 보낸 ERC20 토큰 거래 1건당 평균 수량 | 토큰 수량 | 829 (8.4%) | 1,309 | 추정 |
| `ERC20 min val sent contract` | 컨트랙트로 보낸 토큰 최소 수량. 전 행이 0 | 토큰 수량 | 829 (8.4%) | 1 | **확인 필요** |
| `ERC20 max val sent contract` | 컨트랙트로 보낸 토큰 최대 수량. 전 행이 0 | 토큰 수량 | 829 (8.4%) | 1 | **확인 필요** |
| `ERC20 avg val sent contract` | 컨트랙트로 보낸 토큰 평균 수량. 전 행이 0 | 토큰 수량 | 829 (8.4%) | 1 | **확인 필요** |
| `ERC20 uniq sent token name` | 보낸 적 있는 '서로 다른' 토큰 종류의 수 | 개 | 829 (8.4%) | 70 | 확실 |
| `ERC20 uniq rec token name` | 받은 적 있는 '서로 다른' 토큰 종류의 수 | 개 | 829 (8.4%) | 121 | 확실 |
| `ERC20 most sent token type` | 가장 많이 보낸 토큰의 이름(문자열) | 문자열 | 2,697 (27.4%) | 304 | 확실 |
| `ERC20_most_rec_token_type` | 가장 많이 받은 토큰의 이름(문자열) | 문자열 | 871 (8.9%) | 466 | 확실 |

---

## 확인 필요 열 정리 (12개)

아래 열들은 **의미를 추측하지 않았습니다.** 원 데이터 제작자의 설명이나
Etherscan API 문서를 확인한 뒤 채워 넣어야 합니다.

- `Index` — 출처 불명의 번호. 9841행인데 고유값이 4729개뿐이라 단순 행 번호가 아님
- `ERC20 total Ether received` — 받은 ERC20 토큰 수량의 총합. 이름에 'Ether'가 있지만 실제로는 토큰 수량으로 보임
- `ERC20 total ether sent` — 보낸 ERC20 토큰 수량의 총합. 위와 같은 이유로 단위가 불명확
- `ERC20 total Ether sent contract` — 컨트랙트로 보낸 ERC20 토큰 수량의 총합
- `ERC20 uniq sent addr.1` — 원본 CSV 에 같은 이름의 열이 두 개 있어 pandas 가 '.1'을 붙인 열. 앞의 'ERC20 uniq sent addr' 와 값이 83%만 일치하고 분포도 전혀 달라 별개의 의미로 추정됨
- `ERC20 avg time between sent tnx` — ERC20 토큰을 보낸 거래 사이의 평균 간격. 전 행이 0
- `ERC20 avg time between rec tnx` — ERC20 토큰을 받은 거래 사이의 평균 간격. 전 행이 0
- `ERC20 avg time between rec 2 tnx` — 이름의 'rec 2'가 무엇을 뜻하는지 알 수 없음. 전 행이 0
- `ERC20 avg time between contract tnx` — ERC20 컨트랙트 거래 사이의 평균 간격. 전 행이 0
- `ERC20 min val sent contract` — 컨트랙트로 보낸 토큰 최소 수량. 전 행이 0
- `ERC20 max val sent contract` — 컨트랙트로 보낸 토큰 최대 수량. 전 행이 0
- `ERC20 avg val sent contract` — 컨트랙트로 보낸 토큰 평균 수량. 전 행이 0
