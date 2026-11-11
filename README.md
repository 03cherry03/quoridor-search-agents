# Quoridor Search Agents

**인공지능입문 프로젝트 | Greedy, Minimax, Alpha-Beta Pruning 비교**

9×9 보드게임 **쿼리도(Quoridor)** 에서 여러 탐색 알고리즘으로 다음 행동을 선택하는 AI를 구현하고, **승률·의사결정 시간·탐색량**을 비교하는 프로젝트입니다.

## 주요 기능

- **BFS**: 목표 행까지의 최단거리 계산 및 벽 설치 합법성 검사
- **Greedy**: 다음 상태를 즉시 평가하여 행동 선택
- **Minimax (Depth 2)**: 상대방의 최선의 응수를 고려
- **Minimax + Alpha-Beta (Depth 2/3)**: 불필요한 탐색을 줄이고 깊이 확장
- **자동 대전**: 선후공을 바꿔 대결하고 결과를 CSV로 저장

## 실행 방법

Python **3.10 이상**이 권장되며, 별도의 외부 패키지 설치는 필요하지 않습니다.

```bash
# 규칙 및 알고리즘 테스트
python -m unittest discover -s tests -v

# AI와 직접 대전 (터미널)
python play_cli.py --ai minimax2_ab

# Minimax vs Alpha-Beta: 동일한 선택/평가값 및 탐색량 비교
python benchmark_ab.py --cases 12 --output results/ab_verification.csv

# AI 간 자동 대전 (선후공 교대, 10쌍 = 총 20판)
python experiment.py --a minimax2 --b minimax2_ab --pairs 10 --output results/minimax_vs_ab
```

## 알고리즘 비교

| AI 모드 | 탐색 깊이 (ply) | 상대 응수 고려 | Alpha-Beta |
|---|---:|:---:|:---:|
| `greedy` | 1 | X | X |
| `minimax2` | 2 | O | X |
| `minimax2_ab` | 2 | O | O |
| `minimax3_ab` | 3 | O | O |

평가 함수는 **상대 BFS 거리 − 내 BFS 거리**에 남은 벽 개수 차이를 반영합니다. 동일한 탐색 조건에서 Alpha-Beta는 Minimax의 평가값을 유지하면서 탐색을 줄이는 것이 목표입니다.

## 프로젝트 파일

- `quoridor_engine.py`: 게임 규칙, 합법적 이동·벽 설치, BFS
- `quoridor_ai.py`: 탐색 AI 및 평가 함수
- `experiment.py`: 자동 대전과 CSV 기록
- `benchmark_ab.py`: Minimax와 Alpha-Beta 정확성·효율 검증
- `play_cli.py`: 터미널 게임
- `tests/`: 단위 테스트

> **참고:** 이 저장소는 비교 실험을 위한 **독립 실행형 구현**입니다. 팀원의 기존 웹게임 서버와의 연동은 별도 작업이며, 예시 어댑터는 `server_adapter_example.py`를 참고하세요. 후보를 상위 `k`개로 제한하는 방식은 Alpha-Beta와 별개의 근사 기법입니다.
