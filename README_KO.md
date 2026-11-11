# 쿼리도 AI 입문 프로젝트 — 실행 가능한 최종 코드

**Greedy → Minimax(깊이 2) → Minimax + 알파–베타(깊이 2) → Minimax + 알파–베타(깊이 3)** 비교 실험을 위한 Python 프로젝트입니다.

**중요:** 이 패키지는 제공된 설명을 바탕으로 **독립적으로 작성·테스트한 9×9 게임 엔진과 탐색 AI**입니다. 웹사이트의 기존 `Game_hub/quoridor/server/ai.py`, `game.py`를 수정하거나 서버에 배포한 것이 아닙니다. 기존 코드의 실제 인터페이스를 가져오지 못했기 때문에 **웹게임 연동은 별도 검증**이 필요합니다. 연결할 때는 `server_adapter_example.py`를 참고하세요.

## 파일 설명

| 파일 | 내용 |
|---|---|
| `quoridor_engine.py` | 9×9 게임 규칙, 말 이동과 점프, 벽 합법성 검사, BFS, 불변 상태 |
| `quoridor_ai.py` | Greedy, Minimax, α–β, 깊이 제한, 후보 정렬/상위 k, 탐색 통계, 선택적 시간제한 |
| `experiment.py` | 자동 AI 대전: 동일한 시작 상태를 사용하고 선후공을 바꿔 대국, CSV 저장 |
| `benchmark_ab.py` | 동일한 판에서 일반 Minimax와 α–β의 **수/점수 동일성**을 검증하고 계산량 비교 |
| `play_cli.py` | 터미널에서 직접 AI와 대결할 수 있는 게임 |
| `server_adapter_example.py` | JSON 상태 입력과 행동 결과 출력의 서버 연결 경계 예시 |
| `tests/test_project.py` | 규칙·탐색 검증 단위 테스트 12개 |
| `example_results/` | 작은 실행 확인용 CSV 예시(결과 해석에 주의) |

**외부 라이브러리가 필요 없습니다.** Python 3.10+를 권장합니다.

## 1. 실행

이 폴더에서 터미널(PowerShell/명령 프롬프트)을 열고:

```bash
python -m unittest discover -s tests -v
python play_cli.py --ai minimax2_ab --human-side 0
```

Windows에서 `python` 명령이 안 되면 `py`를 사용하세요.

게임 직접 플레이 시 좌표는 0부터 시작하며 `p 1 4`(말 이동), `w 3 4 h`(가로 벽), `w 2 5 v`(세로 벽)처럼 입력합니다. `quit`을 입력하면 종료합니다.

## 2. 가장 중요한 실험: Minimax vs Minimax + α–β

먼저 **같은 판에서 같은 수를 선택하는지** 확인하세요. 아래 명령은 기본 12개의 판 상태를 만들고 일치성을 검사한 뒤 CSV로 저장합니다.

```bash
python benchmark_ab.py --cases 12 --output results/ab_verification.csv
```

그다음 대전을 진행합니다.

```bash
python experiment.py --a minimax2 --b minimax2_ab --pairs 10 --max-plies 120 --output results/ab
```

각 시작 판마다 두 AI의 플레이어 색(선후공)을 바꿔 **2판씩** 진행합니다. `--pairs 10`이면 **총 20판**입니다. `--max-plies 120`에 도달하면 **실험상의 미종료(draw)** 로 처리합니다. 쿼리도의 공식 무승부 규칙이라고 주장하지 마세요.

다른 비교:

```bash
python experiment.py --a greedy --b minimax2 --pairs 10 --output results/greedy_vs_minimax
python experiment.py --a minimax2_ab --b minimax3_ab --reply-k 4 --pairs 4 --output results/depth
```

> 깊이 3은 계산량이 커질 수 있습니다. `--reply-k 4`는 **중간 노드에서 상대 후보를 상위 4개로 제한**하므로, 엄밀한 전체 트리 Minimax가 아니라 beam 근사입니다. 깊이 2는 최종 응수 노드에서 이 후보 제한을 적용하지 않습니다. 따라서 `C vs D` 대결은 깊이뿐 아니라 후보 제한 효과도 섞일 수 있습니다. 깊이 효과를 순수하게 비교하려면 `--reply-k`를 제거하고 **동일한 `--root-k`, `--margin`, 평가 함수**로 비교하세요(느릴 수 있음).

출력은 `results/.../games.csv`, `moves.csv`, `summary.txt` 세 파일입니다. Excel로 CSV를 열어 바로 표·그래프를 만들 수 있어요.

## 3. 알고리즘 정의

평가 함수 (검색 중 root player `me` 관점):

```
score = 10 * (opponent_BFS_distance - my_BFS_distance)
        + 0.5 * (my_remaining_walls - opponent_remaining_walls)
```

승리 시 +10000, 패배 시 −10000입니다. BFS 거리는 **설치된 벽만 고려**하며 상대 말 점유/점프는 제외합니다. 거리의 엣지 비용은 모두 1이므로 BFS를 사용합니다.

| 모드 | 깊이(ply) | 상대 응수 고려 | 알파–베타 |
|---|---:|---|---|
| `greedy` | 1 | X | X |
| `minimax2` | 2 | O | X |
| `minimax2_ab` | 2 | O | O |
| `minimax3_ab` | 3 | O | O |

`--root-k 8`에서는 첫 행동들을 즉시 평가해서 상위 8개만 Minimax로 확인합니다. 이것은 **후보 제한(beam approximation)**이지 알파–베타 가지치기가 아닙니다. `--root-k`와 `--margin`을 같게 둬야 `minimax2`와 `minimax2_ab`를 공정하게 비교할 수 있습니다.

**속도 개선 포인트:** 말 이동을 먼저 검사하고 벽 후보는 *lazy generator*로 생성합니다. 탐색 중 `beta <= alpha`가 되면 뒤쪽 벽 후보는 **벽 합법성 BFS도 수행하지 않으므로** 시간 절감이 큽니다. 반대로 모든 벽을 미리 BFS로 검사하여 리스트로 만드는 구조라면 가지치기를 해도 리스트 구성 비용이 남습니다.

### 측정 항목

- `nodes`: 트리 탐색으로 들어간 자식 노드 수(루트 제외).
- `leaves`: 깊이 0 또는 종료 노드 수.
- `evaluations`: 점수 함수 호출 수. 루트 상위 k개 선별용 평가도 포함.
- `bfs_calls`: **캐시 미스**로 실제 수행한 BFS 탐색 횟수(거리 계산·벽 경로검사 포함).
- `wall_checks`: 경로 검사까지 들어간 벽 설치 시도 횟수.
- `cutoffs`: 알파–베타로 중단한 지점 수.
- `decision_ms`: 실제 전체 판단 시간, 후보 생성 및 BFS 포함.
- `search_depth`: 해당 수에서 **완료된** 탐색 깊이.
- `timed_out`: 시간제한에 걸려 덜 깊은 판단/합법적인 말 이동으로 대응했는지.

## 4. 선택적 시간제한 실험 (0.2초)

```bash
python experiment.py --a minimax2_ab --b minimax3_ab --pairs 3 --reply-k 4 --time-limit 0.2 --iterative --output results/timeout
```

시간제한이 있으면 반복 심화 탐색을 사용해 **완료한 깊이의 수만** 선택합니다. 시간이 너무 짧아 루트 후보 생성조차 완료하지 못하면 합법적인 말 이동으로 폴백합니다. Python 루프에서 시간 확인을 수행하는 **소프트 제한**이므로 OS 수준의 절대적인 0.2초 보장은 아닙니다. 정식 실험에서는 `timed_out`과 `search_depth`를 반드시 같이 보세요.

## 5. 웹서버에 연동하기 위한 체크리스트

`server_adapter_example.py`를 실행하면 다음처럼 **입력 상태 → AI 행동**이 나옵니다.

```bash
python server_adapter_example.py
# or
python server_adapter_example.py saved_state.json minimax2_ab
```

필요한 입력 데이터 모양은 다음과 같습니다.

```json
{
  "pawns": [[0, 4], [8, 4]],
  "horizontal_walls": [],
  "vertical_walls": [],
  "remaining": [10, 10],
  "turn": 0
}
```

`Action.as_dict()`의 결과는 예를 들어 말 이동 `{"type":"move", "row":1, "col":4}` 또는 벽 설치 `{"type":"wall", "row":3, "col":4, "orientation":"h"}`입니다.

기존 사이트 `game.py`와 연결할 때 확인할 요소:

1. **좌표**: `(row, col)`가 0부터 시작하고 선공 말은 `(0,4)`인지, UI 방향은 일치하는지.
2. **벽 정보**: 기준점은 8×8 anchor로 H는 아래쪽 두 행을 막는 벽, V는 옆쪽 두 열을 막는 벽인지. 원래 서버는 다른 좌표를 사용할 수 있습니다.
3. **남은 벽, 현재 플레이어**, 서버가 기대하는 **행동 JSON 타입·필드명**, 타임아웃 처리.
4. 기존 서버의 합법성 검사와 이 엔진의 결과가 일치하는지 **샘플 대국별로 테스트**.
5. 서버는 업데이트 전 기존 `server/ai.py`를 백업하고, 연결 코드에서만 `choose_project_ai`를 호출하도록 만드는 편이 안전합니다.

**기존 웹사이트에서 바로 작동한다고 보증하는 드롭인 교체 파일은 아닙니다.** 원본 `ai.py`, `game.py`, 서버 라우팅 로직을 복사해서 대조하면 어댑터를 완성할 수 있습니다.

## 6. 보고서에 쓰기 좋은 비교

- **알고리즘 적절성:** 2인 완전 관측·결정적·턴제 게임에서 상대의 적대적인 행동을 가정하는 Minimax; BFS는 평가와 합법성 검사.
- **실험 1:** Greedy와 Minimax2의 승률 비교 = 상대 응수 고려 여부.
- **실험 2:** Minimax2와 Minimax2+AB의 선택 수/평가값 일치 검증, BFS 호출·시간 비교 = 가지치기 효율.
- **실험 3 (확장):** 깊이 2와 3 비교 = 더 멀리 보되 비용 증가; 후보 제한의 영향 명시.
- **한계:** 경로 기반 heuristic은 점프/교착·벽의 전략적 위치를 정확히 평가하지 못함, `top_k`는 최적 행동을 누락할 가능성, 0.2초 soft deadline.

### 구현/실험 기록

구현 후 `python -m unittest discover -s tests -v`에서 총 12개 테스트를 통과했으며, A/B 대전을 1쌍(40 ply 제한 × 2판)으로 연동 실행하여 CSV 저장도 확인했습니다. **이 작은 샘플의 승률은 성능 결론으로 해석하지 마세요.**
