# uncertainty_map: 모델은 누구를 대변할 때 망설이는가

Junsol Kim의 `congress_map`(의원 1947–2023을 Qwen2.5-7B-Instruct의 경제·사회 공간에 투영)을 그대로 두고, 그 위에
**생성 엔트로피(망설임)**를 얹어 다섯 가지를 본다. `congress_map`의 파일은 하나도 바꾸지 않았고, 그의 결과물
(`congress_map/results/vectors.pt`, `temporal_member_scores.csv`)과 페르소나 문장을 그대로 읽는다.

| 아이디어 | 질문 |
|---|---|
| A | 모델은 어느 주(州)·시대의 의원을 망설이며 연기하나? 의원의 실제 이념을 통제해도 남는가? |
| B | 캐리커처: 모델이 의원을 표결 기록보다 더 전통적으로 놓을 때 더 망설이나? |
| C | 같은 사람, 다른 조건: 연도만 바꾸면? 당을 바꾼 의원의 전과 후는? |
| D | 이름 없는 페르소나("1960년 앨라배마 하원의원")로 본 모델의 정치 지리 |
| E | 모델은 현재에서 과거를 읽는가: 모델 지도 속 양극화와 실제 표결(DW-NOMINATE) 양극화 비교 |
| T1 | 모델 기본값의 **양쪽**에 있는 페르소나: 망설임이 기본값을 중심으로 대칭(자기와의 거리)인가, 한쪽(오른쪽·과거)으로만 오르는가? |

탐색 단계 결과다(p값 보정 없음, 8비트 모델, 질문당 1회 샘플). 확정은 H200 원본 정밀도 재실행 후.
진행 기록: `../EXPLORATION_LEDGER.md` (E14 이후).

## 실행

GPU 1장(8비트 기준 여유 메모리 12GB 이상; RTX 5060 Ti 16GB에서 확인), 외부 API 불필요.

```bash
pip install -r requirements.txt
cd kim/uncertainty_map
bash run_all.sh                    # 기본 8비트. 큰 GPU에서는 QUANT=bf16 OUT=out_bf16 bash run_all.sh
```

| 단계 | 파일 | 하는 일 | 출력 | GPU(RTX 5060 Ti, int8) |
|---|---|---|---|---|
| 1 | `01_generate_members.py` | 격년 의회(80–118대)마다 주별 의원 1명 + 기존 탐색 생성 1,064명(같은 설정) 재사용. Kim 페르소나 + 보류 문항 10개 생성, 엔트로피 기록 | `out/member_generations.csv` | ~30분 |
| 2 | `02_counterfactual.py` | (a) 아는 의원 60명을 연도만 1955/1985/2015/본인 연도로 바꿔 투영·생성 (b) 당적 변경 의원 30명의 변경 직전·직후 | `out/counterfactual_*.csv` | ~10분 |
| 3 | `03_anonymous_personas.py` | 이름 없는 페르소나 50주 × 상·하원 × 6개 의회 투영·생성, 같은 의회 실제 의원 전원 투영(같은 정밀도) | `out/anonymous_*.csv`, `out/real_member_projections.csv` | ~25분 |
| 4 | `04_analyze.py` | A–E 분석(OLS, CR1 군집 표준오차를 행렬로 다시 계산해 검증) | `out/summary.json`, `out/*.csv`, `out/fig_BED.png` | – |
| 5 | `05_build_map.py` | 망설임 지도(데이터 내장 단일 HTML) | `out/uncertainty_map.html` | – |
| 6 | `06_make_gif.py` | GIF | `out/uncertainty_map.gif` | – |
| 7 | `07_left_of_default.py` | T1: 입장 서술 페르소나 25칸 × 문구 4개, 의원이 아닌 공적 인물 좌 18·우 18, 페르소나 없음. 투영(조작 점검) + 10문항 × 2회 생성 | `out/t1_*.csv` | ~12분 |
| 9 | `09_persona_offset.py` | 사회 축 원점 분해: 기본 문구, 일반 어시스턴트, 그냥 사람, 비정치 실명 인물 30명(중립/Kim 끝맺음), 이름만, 1955년. 투영 + 10문항 × 2회 | `out/offset_*.csv` | ~10분 |
| 10 | `10_analyze_offset.py` | 조건별 이동량, 중립 인물을 원점으로 T1·의원 재분석 | `out/OFFSET_RESULTS.md`, `out/fig_offset.png` | – |
| 11 | `11_dynamics.py` | R1 경력 동학(Kim 좌표 패널 + Nokken-Poole, 의원·의회 고정효과), R2 시계열(수준과 차분) | `out/DYNAMICS_RESULTS.md`, `out/fig_dynamics.png` | – |
| 12 | `12_get_rollcalls.py` | Voteview 의회별 표결·안건 파일 다운로드(100/106/112/118대, SHA-256 목록) | `../data/voteview/` | – |
| 13 | `13_vote_prediction.py` | R4 페르소나로 실제 표결 예측(강제선택 다음 토큰 확률), 기준선: 정당 다수, DW-NOMINATE | `out/vote_predictions.csv` | ~8분 |
| 14 | `14_default_swap.py` | 분석 1: 시스템 기본값 4종(Qwen, 진보 어시스턴트, 보수 어시스턴트, 중립 역사가) + T1 페르소나 61명 | `out/swap_*.csv` | ~20분 |
| 15 | `15_topics.py` | 분석 2: 질문별·주제별 비대칭(Kim 10문항, WVS 6주제 steering) | `out/TOPICS_RESULTS.md`, `out/fig_topics.png` | – |
| 16 | `16_analyze_votes_swap.py` | R4와 분석 1 분석 | `out/VOTES_SWAP_RESULTS.md`, `out/fig_votes_swap.png` | – |
| 17 | `17_representation.py` | 대표성 측정: Achen 반응성·근접성·분산비(정당×시대), 개별 대 집합 대표, 정당별 동적 대표(Kim 원본 좌표) | `out/REPRESENTATION_RESULTS.md`, `out/fig_representation.png` | – |
| 18 | `18_time_direction.py` | 시간 방향 추출(비정치 실명 인물 30명 × 1900~2060, 15명 추출/15명 검증), 의원 패널 300명 투영(시간 성분 제거 y_perp) | `out/time_*.csv`, `out/time_direction.npy` | ~15분 |
| 19 | `19_rep_and_time.py` | (a) NOMINATE와 독립적인 표결 PCA로 공화당 반응성 점검 (b) 시간 방향 분석 | `out/REP_TIME_RESULTS.md`, `out/fig_rep_time.png` | – |
| 20 | `20_corrected_map.py` | A: 전 의원·의회 재투영, 보정 1(같은 해 평범한 사람 기준), 보정 2(시간 방향 제거). 의회별 저장·재개 | `out/corrected/` | ~55분 |
| 21 | `21_niche.py` | A 검증(실제 이념 대비), B 틈새 생태학(위치·폭·겹침·축별 분리·남부 민주당), C 교차 압력 틈새와 엔트로피 | `out/NICHE_RESULTS.md`, `out/fig_niche.png` | – |
| 22 | `22_build_corrected_map.py` | 보정 지도 HTML(모델이 본 역사 / 보정 / 시간 제거 전환) | `out/corrected_map.html` | – |
| 23 | `23_output_stance.py` | #2·#3: 판정자 검증 세트(지시문 답변 100개), 평범한 사람 문항별 투영, 판정자(Qwen2.5-3B, 눈가림)로 기존 답변 약 2.6만 개 채점 | `out/stance_*.csv`, `out/ordinary_by_question.csv` | ~15분 |
| 24 | `24_output_econ_magnitude.py` | #2 출력에서의 왜곡, #3 경제 축 원인, #5 왜곡 크기(DW 단위 환산) | `out/OUTPUT_ECON_MAGNITUDE.md`, `out/fig_output_econ_magnitude.png` | – |
| 25 | `25_mechanism_robustness.py` | 메커니즘·견고성(프로토콜 R3): 시간 방향 steering, 연도 표현, WVS 48문항, 기준 집단 8개, 위약 판정 차원 | `out/r3/` | ~1.5시간 |
| 26 | `26_analyze_mech_robust.py` | R3 판정표(PASS/FAIL)와 분석 | `out/r3/MECH_ROBUST_RESULTS.md`, `out/r3/fig_mech_robust.png` | – |
| 8 | `08_analyze_t1.py` | T1: 원점 기준 꺾은선 회귀(왼쪽·오른쪽 기울기), 거리 대 선형 모형 비교 | `out/T1_RESULTS.md`, `out/fig_T1.png` | – |

GPU 단계는 끝나면 `out/stepN.done`을 남기고, `run_all.sh`는 그 파일이 있으면 건너뛴다. 중간에 멈추면 배치 단위로 이어서 돈다.

## 방법 메모

- **원점은 Qwen 기본 시스템 문구다.** 시스템 턴이 없으면 Qwen2.5 템플릿이 "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."를 넣는다. 페르소나는 이 문구를 대체한다. 9단계: 정치와 무관한 실명 인물만으로도 y가 약 +1.65 이동한다(2023년 민주당 평균보다 전통 쪽).

- **T1이 필요한 이유**: Kim 지도의 의원-의회 21,288행 중 사회 축에서 모델보다 진보적인 행은 0개, 경제 축에서 왼쪽인 행은 91개(최대 0.26)다. 의원만으로는 기본값의 한쪽만 볼 수 있다. 입장 서술 페르소나의 위치는 가정하지 않고 Kim의 투영으로 잰다.

- **측정은 저장소 노트북 03과 Kim의 congress_map을 따른다.** 페르소나 문장, 주 이름, 이름 정리 함수는 Kim의
  `03_project_legislators.py`에서 AST로 읽는다(복사하지 않음). 투영은 17층 마지막 토큰, 원점 = 페르소나 없는 모델,
  x = 경제(오른쪽 = 우파), y = 사회(위 = 전통). 엔트로피는 상위 40개 토큰 재정규화, 비트, 첫 EOS까지 포함, 처음 64토큰 평균.
- **정당은 프롬프트에 넣지 않는다**(Kim과 같음). 모델이 아는 의원인지(Kim의 `known`)는 하위 집단으로 보고한다.
- **샘플링**: `do_sample, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0` 명시. 시드는 라벨의 SHA-256 앞 8자리(배치 단위).
- **Kim 파이프라인과 다른 점**
  - **8비트(LLM.int8) 모델.** 이 PC(16GB)에서 7B 생성을 하려고 썼다. 원본 정밀도와 비교(`../2_runs/F2/f2_int8.json`):
    의원 200명 투영 r = 0.992(경제), 0.990(사회), 질문 상태 코사인 0.998. 다만 경제 좌표 평균이 0.19 정도 밀린다
    (순위는 유지). 그래서 A·B·E는 Kim의 원본 정밀도 좌표를 쓰고, C·D처럼 새로 투영하는 값은 같은 실행의 8비트 값끼리만 비교한다.
  - **생성은 100단어 제한 접미어를 붙인다**(노트북 03과 같음). 투영 프롬프트에는 붙이지 않는다(Kim과 같음).
  - **엔트로피는 처음 64토큰.** 노트북의 주 지표(512토큰 평균)보다 8배 싸다. 노트북이 함께 저장하는 `entropy_first64`와 같은 정의.
- **당적 변경 의원은 `bioguide_id`로 묶는다.** Voteview는 당을 바꾸면 새 ICPSR 번호를 주는 경우가 있다.
- **주의**: 초기 의회일수록 모델이 의원을 잘 모른다(Kim README). E의 DW-NOMINATE 변화량은 척도 구조상 크기를 직접
  비교하기 어렵다. 방향과 상대 크기만 보고, 쓰기 전에 Voteview 1차 자료로 확인한다.
