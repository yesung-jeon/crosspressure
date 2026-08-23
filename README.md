# latentbias — 2-Axis Ideological Persona Vectors

LLM에서 **경제축(Economic Left ↔ Right)** 과 **사회축(Social Progressive ↔ Conservative)**
페르소나 벡터를 추출·검증하고, 대칭 직교화(Löwdin) 2D steering으로 두 축의
상호작용을 분석하는 파이프라인.

Model: `Qwen/Qwen2.5-7B-Instruct` (주 모델). Llama-3.1-8B는 선택 사항 — 쓰려면 HF 승인 필요.

## 파이프라인 (순서대로 실행)

```
notebooks/01_extract_vectors.ipynb   벡터 추출  → persona_vectors/{model}/*.pt
notebooks/02_validate_vectors.ipynb  검증 + common layer 자동 산출
notebooks/03_main_analysis.ipynb     2D steering 분석 → output/{model}/...
```

의존 관계:

```
data_economic.json  ─┐
data_social.json     ├─► 01 ─► persona_vectors/*.pt ─► 02 ─► common_layer_*.pt ─► 03
system_prompts.json ─┘                                 │
                                                        └─► steering_validation.csv
```

**공통 레이어는 손으로 고르는 게 아니라 02가 자동 계산한다** (각 축 layer별 최고점을
순위합으로 합산 → 최소 레이어). 03은 그 결과를 로드만 함.

## 실행 전 반드시 채워야 할 것 (지금은 템플릿)


- [ ] `data_generation/trait_data_extract/data_economic.json` — instructions(pos/neg 5쌍) + eval_prompt + questions(≥40)
- [ ] `data_generation/trait_data_extract/data_social.json` — 동일 스키마
- [ ] `output/{model}/{economic,social}/system_prompts.json` — suppress→promote 순서 리스트 (모델·축별)
- [ ] `.env` — `cp .env.example .env` 후 `OPENAI_API_KEY`, (Llama 쓰면) `HF_TOKEN`

## 환경

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch/transformers 버전은 네 환경에 맞게 핀 고정
cp .env.example .env              # 키 입력
```

- GPU 필요 (재현 환경: NVIDIA H200). 01·03은 모델 generation, 02는 steering.
- Judge: `gpt-4.1-mini-2025-04-14`
- Llama는 HuggingFace 승인 필요(gated).

## 실행 매트릭스

`MODEL_KEY`(llama/qwen)와 `AXIS`(economic/social)는 각 노트북 상단에서 손으로 바꿈.
`VECTOR_TYPE="response_avg"` 고정.

| 노트북 | MODEL_KEY | AXIS | 비고 |
|--------|-----------|------|------|
| 01, 02 | qwen | economic → social | AXIS 바꿀 때 **커널 재시작** |
| 01, 02 | qwen | economic → social | 커널 재시작 |
| 02 Cell 10~11 | (모델별) | — | 두 축 완료 후 1회, common layer 산출 |
| 03 | (모델별 1회) | — | common layer 로드 → 2D 분석 |


