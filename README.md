# Quoridor Search Agents

**인공지능입문 프로젝트 | BFS + Greedy 기반 탐색 알고리즘 비교 및 쿼리도 AI 리그전**

9×9 쿼리도에서 BFS 최단거리 기반 평가 함수를 공통으로 사용하고, Minimax·Alpha-Beta·탐색 깊이에 따른 **실제 경기력과 탐색 효율**을 비교합니다. 팀원과 합의한 *평가방법 및 평가함수.pdf*를 반영한 업데이트 버전입니다.

> **주의:** 이것은 독립 실행형 Python 게임 엔진 + AI 실험 코드입니다. 팀원 웹게임 서버에 직접 합친 버전이 아닙니다. 서버 연동은 `server_adapter_example.py`를 참고하되 실제 API/좌표 규칙 확인이 필요합니다.

## 비교 대상

| 번호 | 실행 모드 | 구성 | 리그전 |
|:--:|---|---|:--:|
| ① | `greedy` | BFS + Greedy (기준선) | 기본 제외 |
| ② | `minimax2` | Greedy 후보 정렬 + Minimax 깊이 2 | ✅ |
| ③ | `minimax3` | Greedy 후보 정렬 + Minimax 깊이 3 | ✅ |
| ④ | `minimax2_ab` | Minimax 깊이 2 + Alpha-Beta | ✅ |
| ⑤ | `minimax3_ab` | Minimax 깊이 3 + Alpha-Beta | ✅ |
| ⑥ | `mcts` | **선택적 실험용** heuristic-guided MCTS/UCT | 옵션 |

- 깊이는 **ply**, 즉 한 플레이어의 행동 한 번을 의미합니다.
- 모든 Minimax 변형은 **동일한 평가 함수 / 합법적 행동 규칙 / 후보 범위 / top-k**를 사용합니다.
- 상위 `k`개 선택은 Greedy 기반 **beam 근사**이고, Alpha-Beta 가지치기와는 다릅니다.
- ⑥은 수업 외 알고리즘의 **임시 후보**입니다. 신경망 학습이나 AlphaZero가 아닙니다.

## 빠른 시작

Python 3.10 이상, 별도 패키지 설치 불필요. 아래 명령은 **레포 루트**에서 실행하세요. Windows에서 `python`이 안 되면 `py`를 사용해도 됩니다.

```bash
python -m unittest discover -s tests -v
```

### 1) 예비 실험: 0.2초에 깊이 3까지 도달하는지 확인

```bash
python pilot_budget.py --cases 20 --budgets 0.2 0.5 1.0 --output results/pilot
```

`results/pilot/depth_summary.csv`에 알고리즘별 **완료 깊이·깊이 3 완료율·평균 시간**이 저장됩니다. `budget_choice.json`은 **②~⑤에 공통으로 적용할 시간 예산**을 제안합니다.

- 두 깊이 3 모델 **모두** 표본 상태 중 75% 이상에서 깊이 3을 완료하는 **가장 짧은 후보 예산**을 선택합니다.
- 해당 예산이 없으면 `recommended_budget_s: null`로 저장됩니다. **그대로 리그전에 자동 적용하지 않습니다.** `--budgets 0.5 1.0 2.0`으로 다시 시험하거나 모두에게 공통으로 `--root-k 5 --reply-k 3`처럼 분기 폭을 줄여 다시 시험하세요.
- 실험에서 예산을 0.5초로 바꾸면 허용 한계도 **0.55초**로 함께 바꿉니다(초기 팀원 안은 0.2/0.25초). 실제로 더 깊게 봐야만 두 깊이의 차이를 검증할 수 있기 때문입니다.
- 이 단계는 실험 설정 결정용 **pilot**이며 리그전 승률을 뜻하지 않습니다.

- **0.2초를 유지하고 싶다면** 아래처럼 후보 수를 공통으로 줄여 예비 실험을 해볼 수 있습니다. 다만 원래보다 많은 행동을 배제하므로 근사 오차가 커집니다.

```bash
python pilot_budget.py --cases 20 --budgets 0.2 0.5 --root-k 4 --reply-k 2 --output results/pilot_small_beam
python league.py --pilot-config results/pilot_small_beam/budget_choice.json --root-k 4 --reply-k 2 --pairs 10 --max-plies 200 --output results/league_small_beam
```

위의 두 번째 명령은 pilot에서 충분한 깊이 3 완료율을 확인한 경우에만 실행 가능합니다.

### 2) 리그전: ②~⑤ 전체 맞대결 / Score로 순위

먼저 소규모 확인:

```bash
python league.py --pairs 1 --max-plies 20 --output results/league_smoke
```

팀원 평가 방식 기준의 본 실험 (시간제한 0.2초 / 허용 한계 0.25초):

```bash
python league.py --pairs 10 --max-plies 200 --time-limit 0.2 --hard-limit 0.25 --output results/league
```

**예비 실험에서 결정한 시간 예산을 반영**하려면 대신 다음처럼 실행:

```bash
python league.py --pilot-config results/pilot/budget_choice.json --pairs 10 --max-plies 200 --output results/league_pilot
```

`--pilot-config`를 사용하면 파일의 시간 예산/허용 한계를 읽고, 분기 설정(`--root-k`, `--reply-k`, `--margin`)이 pilot과 같은지 검사합니다. pilot에서 분기 설정을 변경했다면 league에도 동일하게 지정해야 합니다.

⑥ MCTS 참가를 결정하면 `--include-mcts`를 추가합니다. 기본은 **②~⑤ 네 AI의 6개 대진**이고, MCTS 포함 시 **5개 AI의 10개 대진**입니다. 대진별로 시작 판을 고정하고 선후공을 바꾸어 2판씩 진행하므로 `--pairs 10`은 각각 **120판 / 200판**입니다. 단순한 최대 ply 종료는 프로젝트 실험용 무승부 규칙으로 취급합니다.

**결과 파일:** `results/league/standings.csv`, `head_to_head.csv`, `games.csv`, `moves.csv`, `summary.md`, `config.json`.

### 3) 별도 실험: 시간제한 **없이** 동일 깊이 Minimax vs Alpha-Beta

```bash
python benchmark_ab.py --cases 12 --depths 2 3 --output results/fixed_depth/ab_verification.csv
```

이 실험은 리그전과 **완전히 별개**입니다. 깊이 2끼리(② vs ④), 깊이 3끼리(③ vs ⑤) 동일한 판에서 같은 수와 Minimax 평가값을 내는지 검증하며, **방문 노드·BFS 호출 수·실행 시간·가지치기 횟수**를 기록합니다. Alpha-Beta는 같은 탐색 트리에서 최종 값/선택을 유지하지만 **실행 시간이 반드시 더 짧은 것은 아닙니다**(후보 생성/정렬 비용 때문).

### 4) 기본 모델과 대결하거나 직접 플레이

```bash
# ① Greedy vs ② Minimax: 기본 모델과의 상대 성능
python experiment.py --a greedy --b minimax2 --pairs 10 --time-limit 0.2 --hard-limit 0.25 --output results/baseline

# 터미널에서 직접 플레이
python play_cli.py --ai minimax3_ab --time-limit 0.2
```

## 팀원이 정한 평가 함수

### AI가 탐색 중 사용하는 함수: `eval(s)`

```text
승리:       +(10000 + 남은 탐색 깊이 k)
패배:       -(10000 + 남은 탐색 깊이 k)
그 외:      (상대 BFS 거리 - 내 BFS 거리)
            + 0.5 * (내 남은 벽 - 상대 남은 벽)
            + tau

tau = +0.5 (내 차례) 또는 -0.5 (상대 차례)
```

BFS는 **벽만 장애물**로 보고 계산하며, 상대 말의 점유/점프를 최단거리 계산에 반영하지 않습니다. `wall_weight=0.5`는 PDF의 기본값이며 전체 참가 AI에 일관되게 적용합니다.

### 리그전 종료 후 평가 점수: `Score(A)`

```text
WR(A)       = (승리 수 + 0.5 * 무승부 수) / 전체 경기 수
margin_j    = 승리: +min(1, 진 사람의 남은 BFS 거리 / 8)
              패배: -min(1, 내 남은 BFS 거리 / 8)
              상대 시간 초과: +1 / 내 시간 초과: -1 / 최대 ply 무승부: 0
S_margin(A) = (1 + 평균 margin) / 2
Score(A)    = 0.7 * WR(A) + 0.3 * S_margin(A)
```

**시간제한 규칙:** 판단의 탐색 예산은 기본 **0.2초**. 실제 판단 시간이 **0.25초를 초과하면 그 수는 두지 않고 시간 초과 패배**로 처리합니다. 시간제한 탐색은 **Iterative Deepening**으로 깊이 1→2→3을 탐색하고, **마지막으로 완전히 완료한 깊이의 행동만 채택**합니다. 아무 깊이도 완료되지 않으면 안전한 합법 말 이동을 사용합니다. 이는 Python 내부 협력적 시간 확인이며 OS 수준의 선점형 실행 중단을 보장하지는 않습니다.

최종 순위는 `Score`로 결정합니다. `standings.csv`에는 팀원이 제안한 **0.6/0.4와 0.8/0.2 가중치 민감도 평가 순위**, 평균 탐색 깊이, 실제 한 수 판단 시간, 탐색 노드 수, 초과율 및 **선후공 쌍 묶음 단위의 설명적 Bootstrap 95% 승률 구간**도 저장됩니다. 표본이 적으면 신뢰구간을 과도하게 해석하지 마세요.

## 주요 코드

| 파일 | 용도 |
|---|---|
| `quoridor_engine.py` | 쿼리도 규칙, BFS 최단거리, 벽 설치 합법성, 상태 직렬화 |
| `quoridor_ai.py` | Greedy / Minimax 2·3 / Alpha-Beta / 선택적 MCTS, 공통 평가 함수, 시간제한 |
| `pilot_budget.py` | 0.2·0.5·1초 실제 완료 깊이 예비 실험 및 공통 예산 제안 |
| `league.py` | ②~⑤ 리그전, 선택적 ⑥, Score 순위표 / 신뢰구간 / CSV |
| `experiment.py` | 두 AI의 선후공 교대 대전 + 종료 마진/시간 초과 규칙 |
| `benchmark_ab.py` | **무제한 시간·고정 깊이** Minimax vs Alpha-Beta 정확성/효율 비교 |
| `play_cli.py` | 터미널에서 AI와 게임 |
| `server_adapter_example.py` | 기존 웹서버 연동 시 참고할 JSON 예시 (미검증) |
| `tests/` | 규칙, 평가 점수, 타임아웃, 알고리즘 단위 테스트 |

**주의할 해석:** 깊이 3이 실제로 완료되더라도 깊이 2와 **같은 수를 고르는 것 자체는 정상적인 결과**입니다. 또한 리그전에는 beam 근사가 포함되므로 전 게임 트리를 탐색하는 '완전한 Minimax'라고 주장하면 안 됩니다. MCTS의 `mcts_iterations`는 Minimax 깊이·노드 수와 동일한 의미가 아닙니다.
