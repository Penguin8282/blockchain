# 특징 4묶음 분류표

전체 특징 **47개**를 4묶음으로 분류했습니다.

| 묶음 | 개수 | 상수 열 | 확인 필요 |
|---|---|---|---|
| 거래 빈도 | 6 | 0 | 0 |
| 금액 | 13 | 0 | 0 |
| 시간 간격 | 3 | 0 | 0 |
| 토큰 활동 | 25 | 7 | 2 |

---

| 묶음      | 열 이름                                              | 측정 축      | 자료형   |   결측 |   고유값 | 상수열   | 확인필요   |
|:----------|:-----------------------------------------------------|:-------------|:---------|-------:|---------:|:---------|:-----------|
| 거래 빈도 | Sent tnx                                             | 거래 빈도    | int64    |      0 |      641 |          |            |
| 거래 빈도 | Received Tnx                                         | 거래 빈도    | int64    |      0 |      727 |          |            |
| 거래 빈도 | Number of Created Contracts                          | 거래 빈도    | int64    |      0 |       20 |          |            |
| 거래 빈도 | Unique Received From Addresses                       | 거래 빈도    | int64    |      0 |      256 |          |            |
| 거래 빈도 | Unique Sent To Addresses                             | 거래 빈도    | int64    |      0 |      258 |          |            |
| 거래 빈도 | total transactions (including tnx to create contract | 거래 빈도    | int64    |      0 |      897 |          |            |
| 금액      | min value received                                   | 금액         | float64  |      0 |     4589 |          |            |
| 금액      | max value received                                   | 금액         | float64  |      0 |     6302 |          |            |
| 금액      | avg val received                                     | 금액         | float64  |      0 |     6767 |          |            |
| 금액      | min val sent                                         | 금액         | float64  |      0 |     4719 |          |            |
| 금액      | max val sent                                         | 금액         | float64  |      0 |     6647 |          |            |
| 금액      | avg val sent                                         | 금액         | float64  |      0 |     5854 |          |            |
| 금액      | min value sent to contract                           | 금액         | float64  |      0 |        3 |          |            |
| 금액      | max val sent to contract                             | 금액         | float64  |      0 |        4 |          |            |
| 금액      | avg value sent to contract                           | 금액         | float64  |      0 |        4 |          |            |
| 금액      | total Ether sent                                     | 금액         | float64  |      0 |     5868 |          |            |
| 금액      | total ether received                                 | 금액         | float64  |      0 |     6728 |          |            |
| 금액      | total ether sent contracts                           | 금액         | float64  |      0 |        4 |          |            |
| 금액      | total ether balance                                  | 금액         | float64  |      0 |     5717 |          |            |
| 시간 간격 | Avg min between sent tnx                             | 시간 간격    | float64  |      0 |     5013 |          |            |
| 시간 간격 | Avg min between received tnx                         | 시간 간격    | float64  |      0 |     6223 |          |            |
| 시간 간격 | Time Diff between first and last (Mins)              | 시간 간격    | float64  |      0 |     7810 |          |            |
| 토큰 활동 | Total ERC20 tnxs                                     | 빈도         | float64  |    829 |      300 |          |            |
| 토큰 활동 | ERC20 total Ether received                           | 금액         | float64  |    829 |     3460 |          |            |
| 토큰 활동 | ERC20 total ether sent                               | 금액         | float64  |    829 |     1415 |          |            |
| 토큰 활동 | ERC20 total Ether sent contract                      | 금액         | float64  |    829 |       29 |          |            |
| 토큰 활동 | ERC20 uniq sent addr                                 | 빈도         | float64  |    829 |      107 |          |            |
| 토큰 활동 | ERC20 uniq rec addr                                  | 빈도         | float64  |    829 |      147 |          |            |
| 토큰 활동 | ERC20 uniq sent addr.1                               | 빈도         | float64  |    829 |        4 |          | 예         |
| 토큰 활동 | ERC20 uniq rec contract addr                         | 빈도         | float64  |    829 |      123 |          |            |
| 토큰 활동 | ERC20 avg time between sent tnx                      | 시간         | float64  |    829 |        1 | 예       |            |
| 토큰 활동 | ERC20 avg time between rec tnx                       | 시간         | float64  |    829 |        1 | 예       |            |
| 토큰 활동 | ERC20 avg time between rec 2 tnx                     | 시간         | float64  |    829 |        1 | 예       | 예         |
| 토큰 활동 | ERC20 avg time between contract tnx                  | 시간         | float64  |    829 |        1 | 예       |            |
| 토큰 활동 | ERC20 min val rec                                    | 금액         | float64  |    829 |     1276 |          |            |
| 토큰 활동 | ERC20 max val rec                                    | 금액         | float64  |    829 |     2647 |          |            |
| 토큰 활동 | ERC20 avg val rec                                    | 금액         | float64  |    829 |     3380 |          |            |
| 토큰 활동 | ERC20 min val sent                                   | 금액         | float64  |    829 |      476 |          |            |
| 토큰 활동 | ERC20 max val sent                                   | 금액         | float64  |    829 |     1130 |          |            |
| 토큰 활동 | ERC20 avg val sent                                   | 금액         | float64  |    829 |     1309 |          |            |
| 토큰 활동 | ERC20 min val sent contract                          | 금액         | float64  |    829 |        1 | 예       |            |
| 토큰 활동 | ERC20 max val sent contract                          | 금액         | float64  |    829 |        1 | 예       |            |
| 토큰 활동 | ERC20 avg val sent contract                          | 금액         | float64  |    829 |        1 | 예       |            |
| 토큰 활동 | ERC20 uniq sent token name                           | 빈도         | float64  |    829 |       70 |          |            |
| 토큰 활동 | ERC20 uniq rec token name                            | 빈도         | float64  |    829 |      121 |          |            |
| 토큰 활동 | ERC20 most sent token type                           | 범주(문자열) | str      |   2697 |      304 |          |            |
| 토큰 활동 | ERC20_most_rec_token_type                            | 범주(문자열) | str      |    871 |      466 |          |            |