# IV(정보가치) 순위표

특징별 **정보가치(IV)** 를 계산해 순위를 매긴 표입니다.
계산식과 기호 설명은 `notebooks/02_woe_iv.ipynb` 또는 `src/woe_iv.py` 를 보세요.

## IV 해석 기준

| IV 범위 | 해석 |
|---|---|
| < 0.02 | 거의 쓸모없음 |
| 0.02 ~ 0.1 | 약함 |
| 0.1 ~ 0.3 | 중간 |
| 0.3 ~ 0.5 | 강함 |
| > 0.5 | 매우 강함 — **데이터 누수 의심** |

---

## IV 상위 15개 — full 트랙 (누수 포함)

|   순위 | 특징                                                 |     IV | 해석                  |   구간 수 |
|-------:|:-----------------------------------------------------|-------:|:----------------------|----------:|
|      1 | ERC20_most_rec_token_type                            | 9.6459 | 매우 강함 (누수 의심) |        10 |
|      2 | ERC20 most sent token type                           | 9.6285 | 매우 강함 (누수 의심) |         8 |
|      3 | ERC20 이력 없음(표시)                                | 3.4797 | 매우 강함 (누수 의심) |         2 |
|      4 | total ether received                                 | 2.9779 | 매우 강함 (누수 의심) |        11 |
|      5 | Time Diff between first and last (Mins)              | 2.8842 | 매우 강함 (누수 의심) |        11 |
|      6 | avg val received                                     | 2.6622 | 매우 강함 (누수 의심) |        11 |
|      7 | total transactions (including tnx to create contract | 2.5465 | 매우 강함 (누수 의심) |        10 |
|      8 | max value received                                   | 2.0936 | 매우 강함 (누수 의심) |        11 |
|      9 | Received Tnx                                         | 1.7667 | 매우 강함 (누수 의심) |         9 |
|     10 | Unique Received From Addresses                       | 1.717  | 매우 강함 (누수 의심) |         7 |
|     11 | min value received                                   | 1.608  | 매우 강함 (누수 의심) |        11 |
|     12 | total ether balance                                  | 1.5148 | 매우 강함 (누수 의심) |        11 |
|     13 | ERC20 min val rec                                    | 1.397  | 매우 강함 (누수 의심) |        10 |
|     14 | total Ether sent                                     | 1.2873 | 매우 강함 (누수 의심) |        11 |
|     15 | Avg min between sent tnx                             | 1.2581 | 매우 강함 (누수 의심) |        11 |

## IV 상위 15개 — no_leak 트랙 (누수 최소 차단)

|   순위 | 특징                                                 |     IV | 해석                  |   구간 수 |
|-------:|:-----------------------------------------------------|-------:|:----------------------|----------:|
|      1 | ERC20_most_rec_token_type                            | 9.6459 | 매우 강함 (누수 의심) |        10 |
|      2 | ERC20 most sent token type                           | 9.6285 | 매우 강함 (누수 의심) |         8 |
|      3 | total ether received                                 | 2.9779 | 매우 강함 (누수 의심) |        11 |
|      4 | Time Diff between first and last (Mins)              | 2.8842 | 매우 강함 (누수 의심) |        11 |
|      5 | avg val received                                     | 2.6622 | 매우 강함 (누수 의심) |        11 |
|      6 | total transactions (including tnx to create contract | 2.5465 | 매우 강함 (누수 의심) |        10 |
|      7 | max value received                                   | 2.0936 | 매우 강함 (누수 의심) |        11 |
|      8 | Received Tnx                                         | 1.7667 | 매우 강함 (누수 의심) |         9 |
|      9 | Unique Received From Addresses                       | 1.717  | 매우 강함 (누수 의심) |         7 |
|     10 | min value received                                   | 1.608  | 매우 강함 (누수 의심) |        11 |
|     11 | total ether balance                                  | 1.5148 | 매우 강함 (누수 의심) |        11 |
|     12 | ERC20 min val rec                                    | 1.397  | 매우 강함 (누수 의심) |        10 |
|     13 | total Ether sent                                     | 1.2873 | 매우 강함 (누수 의심) |        11 |
|     14 | Avg min between sent tnx                             | 1.2581 | 매우 강함 (누수 의심) |        11 |
|     15 | Avg min between received tnx                         | 1.1248 | 매우 강함 (누수 의심) |        11 |

---

## 오염 경로 — 정답이 완벽하게 갈리는 구간

어떤 구간에 30건 이상이 들어 있는데 사기 비율이 정확히 0% 또는 100% 라면,
그것은 예측이 아니라 **정답이 새는 통로**입니다.

| 특징                                                 | 구간                  |   건수 |   사기 비율(%) |   IV 기여 |
|:-----------------------------------------------------|:----------------------|-------:|---------------:|----------:|
| ERC20 most sent token type                           |                       |   1191 |            100 |     4.927 |
| ERC20 most sent token type                           | 0                     |   4388 |              0 |     4.494 |
| ERC20_most_rec_token_type                            | 0                     |   4388 |              0 |     4.493 |
| ERC20_most_rec_token_type                            | Blockwell say NOTSAFU |    779 |            100 |     3.07  |
| total transactions (including tnx to create contract | 0                     |    529 |            100 |     1.992 |
| total ether received                                 | (100.695, 101.0]      |   1475 |              0 |     1.299 |
| min value received                                   | (23.67, 101.0]        |   1197 |              0 |     1.022 |
| avg val received                                     | (50.5, 101.0]         |    944 |              0 |     0.776 |
| total Ether sent                                     | (100.998, 100.999]    |    465 |              0 |     0.339 |
| ERC20 min val sent                                   | (0.01, 0.62]          |     76 |              0 |     0.037 |

## 비정상적으로 반복되는 값

진짜 거래라면 금액이 소수점까지 제각각이어야 합니다.
수백~수천 개 주소가 정확히 같은 금액을 가졌다면 인위적으로 만들어진 데이터일 수 있습니다.

| 특징                 |   반복된 값 |   반복 횟수 |   사기 비율(%) |
|:---------------------|------------:|------------:|---------------:|
| total ether received |     101     |        1476 |            0   |
| total ether received |       0     |         707 |           92.9 |
| total ether received |    2001     |         342 |            0   |
| total Ether sent     |       0     |        2067 |           40.2 |
| total Ether sent     |     100.999 |         454 |            0   |
| total Ether sent     |     100.998 |         384 |            0   |
| avg val received     |     101     |         765 |            0   |
| avg val received     |       0     |         709 |           92.7 |
| avg val received     |      50.5   |         639 |            0   |
| max value received   |     101     |         766 |            0   |
| max value received   |       0     |         707 |           92.9 |
| max value received   |       1     |         176 |           52.3 |

---

## 전체 IV 순위 (no_leak 트랙)

|   순위 | 특징                                                 |     IV | 해석                  |   구간 수 |
|-------:|:-----------------------------------------------------|-------:|:----------------------|----------:|
|      1 | ERC20_most_rec_token_type                            | 9.6459 | 매우 강함 (누수 의심) |        10 |
|      2 | ERC20 most sent token type                           | 9.6285 | 매우 강함 (누수 의심) |         8 |
|      3 | total ether received                                 | 2.9779 | 매우 강함 (누수 의심) |        11 |
|      4 | Time Diff between first and last (Mins)              | 2.8842 | 매우 강함 (누수 의심) |        11 |
|      5 | avg val received                                     | 2.6622 | 매우 강함 (누수 의심) |        11 |
|      6 | total transactions (including tnx to create contract | 2.5465 | 매우 강함 (누수 의심) |        10 |
|      7 | max value received                                   | 2.0936 | 매우 강함 (누수 의심) |        11 |
|      8 | Received Tnx                                         | 1.7667 | 매우 강함 (누수 의심) |         9 |
|      9 | Unique Received From Addresses                       | 1.717  | 매우 강함 (누수 의심) |         7 |
|     10 | min value received                                   | 1.608  | 매우 강함 (누수 의심) |        11 |
|     11 | total ether balance                                  | 1.5148 | 매우 강함 (누수 의심) |        11 |
|     12 | ERC20 min val rec                                    | 1.397  | 매우 강함 (누수 의심) |        10 |
|     13 | total Ether sent                                     | 1.2873 | 매우 강함 (누수 의심) |        11 |
|     14 | Avg min between sent tnx                             | 1.2581 | 매우 강함 (누수 의심) |        11 |
|     15 | Avg min between received tnx                         | 1.1248 | 매우 강함 (누수 의심) |        11 |
|     16 | ERC20 avg val rec                                    | 1.0073 | 매우 강함 (누수 의심) |        11 |
|     17 | ERC20 max val rec                                    | 0.9202 | 매우 강함 (누수 의심) |        11 |
|     18 | max val sent                                         | 0.8332 | 매우 강함 (누수 의심) |        11 |
|     19 | Sent tnx                                             | 0.7761 | 매우 강함 (누수 의심) |         8 |
|     20 | ERC20 total Ether received                           | 0.7344 | 매우 강함 (누수 의심) |        11 |
|     21 | avg val sent                                         | 0.6526 | 매우 강함 (누수 의심) |        11 |
|     22 | Total ERC20 tnxs                                     | 0.6225 | 매우 강함 (누수 의심) |         8 |
|     23 | ERC20 uniq rec addr                                  | 0.5776 | 매우 강함 (누수 의심) |         7 |
|     24 | ERC20 uniq rec contract addr                         | 0.5655 | 매우 강함 (누수 의심) |         7 |
|     25 | ERC20 uniq rec token name                            | 0.5641 | 매우 강함 (누수 의심) |         7 |
|     26 | Unique Sent To Addresses                             | 0.428  | 강함                  |         6 |
|     27 | ERC20 uniq sent token name                           | 0.1972 | 중간                  |         8 |
|     28 | ERC20 avg val sent                                   | 0.1679 | 중간                  |        11 |
|     29 | min val sent                                         | 0.1678 | 중간                  |        11 |
|     30 | ERC20 max val sent                                   | 0.1522 | 중간                  |        11 |
|     31 | ERC20 total ether sent                               | 0.1497 | 중간                  |        11 |
|     32 | ERC20 uniq sent addr                                 | 0.1462 | 중간                  |         8 |
|     33 | ERC20 min val sent                                   | 0.1095 | 중간                  |        11 |
|     34 | Number of Created Contracts                          | 0.0454 | 약함                  |         2 |
|     35 | ERC20 total Ether sent contract                      | 0.0022 | 거의 쓸모없음         |        11 |
|     36 | ERC20 uniq sent addr.1                               | 0.0005 | 거의 쓸모없음         |         2 |
|     37 | avg value sent to contract                           | 0      | 거의 쓸모없음         |         4 |
|     38 | total ether sent contracts                           | 0      | 거의 쓸모없음         |         4 |
|     39 | max val sent to contract                             | 0      | 거의 쓸모없음         |         4 |
|     40 | min value sent to contract                           | 0      | 거의 쓸모없음         |         3 |
