# congress_map — 미국 의회를 모델의 경제·사회 이념 공간에 배치하기

레포의 경제/사회 페르소나 벡터로 1947–2023년(80–118대) 미국 상·하원 의원 21,288명-의회를
Qwen2.5-7B-Instruct의 2차원 이념 공간에 투영하고, 의회별 애니메이션 지도를 만드는 전체 과정.

- 지도 (인터랙티브): https://claude.ai/artifact/9oKUnXGzM6EcDiQGSDWoGv
- 결과물 사본: `results/` (지도 HTML, GIF, 의회별 통계, 의원별 좌표, 벡터)

## 실행

GPU 1장(24GB 이상; L40S·A40에서 확인), OpenAI 키 불필요.

```bash
pip install -r requirements.txt
cd congress_map
bash run_all.sh          # 또는 SLURM: sbatch run_all.sbatch (파티션/환경 줄은 클러스터에 맞게 수정)
```

| 단계 | 파일 | 하는 일 | 출력 | GPU |
|---|---|---|---|---|
| 1 | `01_extract_vectors.py` | `data_generation/trait_data_extract`의 지시문 5쌍 × 질문 20개 × 5회로 좌/우(진보/전통) 답변 생성 → 답변 토큰 평균 은닉 상태 차이 → 17층에서 Löwdin 직교화 | `out/vectors.pt` | ~15분 |
| 2 | `02_get_voteview.py` | Voteview 의원 파일(DW-NOMINATE) 다운로드 | `data/HSall_members.csv` | – |
| 3 | `03_project_legislators.py` | 의원별 페르소나(이름·원·주·연도, 정당 없음) + 평가 문항 10개의 마지막 토큰 17층 상태를 벡터에 투영, 정당 회상 확인 | `out/temporal_member_scores.csv` | ~25분 |
| 4 | `04_analyze.py` | 의회별 정당 평균·분리도(Cohen's d)·NOMINATE dim1 상관 | `out/temporal_congress_stats.csv` | – |
| 5 | `05_build_map.py` | 애니메이션 지도 (데이터 내장 단일 HTML) | `out/congress_map.html` | – |
| 6 | `06_make_gif.py` | GIF | `out/congress_map.gif` | – |

`out/vectors.pt`가 있으면 1단계를, `out/temporal_member_scores.csv`가 있으면 3단계를 건너뜀
(`results/vectors.pt`를 `out/`에 복사하면 1단계 생략 가능).

## 방법 메모

- **좌표**: 원점 = 페르소나 없는 모델 자신의 위치. x = 경제(오른쪽 = 우파), y = 사회(위 = 전통). 단위는 은닉 상태 투영값(모델 단위).
- **현대 기준 축**: 시대 보정 없음. 1947년 의원도 오늘날 모델이 가진 경제·사회 축 위의 절대 위치로 찍힘.
- **노트북 01과 차이**: GPT judge 필터 대신 길이·거부 필터만 사용(벡터는 거의 동일, 17층 cos(경제, 사회) ≈ 0.45).
  샘플링은 `top_k=0, repetition_penalty=1.0` 명시(Qwen 기본값 top_k=20, 1.05 상속 방지).
- **투영은 생성 없이** forward 한 번(17층에서 중단). 정당 회상 확인만 8토큰 greedy 생성.
- **주의**: 초기 의회일수록 모델이 의원을 잘 모름(정당 회상 1947년 62% → 2015년 97%, dim1 상관 0.12 → 0.84).
  1990년 이전 수치는 정당 평균의 방향 정도로만 읽을 것. 지도의 "Only members whose party the model recalls" 토글로 확인 가능.
