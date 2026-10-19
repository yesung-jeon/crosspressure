# Replication package: LLM simulations of U.S. legislators and cross-pressured members

Authors: [to be filled by the authors]

This folder reproduces the numbers behind the paper's main claim:

> LLM simulations of U.S. legislators recover the aligned structure of Congress but miss its cross-pressured members.

The claim rests on four sets of results in three open models (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, OLMo-2-1124-7B-Instruct, all run in FP8):

| Act | Claim | Results reproduced here |
|---|---|---|
| 1 | The simulated Congress recovers the large structure | Map coordinates follow roll-call scales (A3, A4); the party gap is larger on the economic axis than on the social axis (A6) |
| 2 | It misses cross-pressured legislators | Recall by roll-call quadrant (A9); placement of 60 legislators chosen by their votes, 15 per quadrant (A12) |
| 3 | Stated cross-pressured profiles are followed less faithfully than aligned ones | Profile consistency (H3), quadrant arrival (H4) and choice entropy (H1) for stated profiles (A'1, A'3, A'4) |
| 4 | Crossed packages are harder in two of three models, with or without a persona | Reversed minus party-line bundle entropy by party (B1, B2) and without any persona (two-way item bootstrap); economic share of decided conflicts (B5, descriptive) |

The paper's steering results (constructed cross-pressure, act 5) are not reproduced here. The steering code that selects the layer of the model directions is included, because acts 1 to 4 use that layer.

## Folder layout

The package runs at repository level. Keep these three folders side by side, as in the repository:

```
<repository>/
  replication/        this folder
  congress_map/       original direction-extraction code and the shipped map files it reads (results/)
  data_generation/    direction-extraction prompts (trait_data_extract/, trait_data_eval/)
```

Both paths read files from `congress_map/` and `data_generation/`; `run_analysis.sh` and `run_all.sh` stop at once with a message if either folder is missing. Copying `replication/` alone is not enough.

## Two ways to reproduce

**Path B: analysis from intermediates (CPU, about 30 minutes).** The package ships the model outputs that the analyses read (directions, legislator coordinates, profile runs, bundled-choice runs). Path B reruns every analysis step on them and checks each number against the sealed value.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r replication/requirements-analysis.txt
bash replication/run_analysis.sh                            # ends with compare_to_sealed.py --path B, then build_paper.sh
```

Path B requires every number to equal the sealed value after rounding to three decimals.

**Path A: full rerun from raw data (GPU).** Path A downloads Voteview, rebuilds the roll-call scales, extracts the model directions with the original direction-extraction code (`congress_map/01_extract_vectors.py` at the repository root, run unchanged), selects the layer by steering and judging, projects legislators, and reruns the profile and bundled-choice experiments.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r replication/requirements-gpu.txt
huggingface-cli login                                       # Llama-3.1 needs approved access; Llama is the layer-selection judge for every model
python replication/data/fetch_data.py                       # Voteview, about 700 MB; then add the PAP file (see Data)
SMOKE=1 bash replication/run_all.sh qwen                    # smoke test of the GPU steps
bash replication/run_all.sh                                 # all three models; or: bash replication/run_all.sh qwen
```

FP8 batched inference is not bit-reproducible: the same prompt can give slightly different logits when the batch around it changes. Path A therefore checks agreement at the level of the verdicts, with tolerances fixed before any reproduction run (`compare_to_sealed.py`, docstring):

- for rows with an interval, the estimate must fall within a quarter of the sealed half-width, with the same verdict;
- descriptive shares, correlations and gaps must fall within 0.05;
- placement counts must fall within 2 of 15;
- in every model, aligned recall must exceed cross recall and the economic party gap must exceed the social party gap.

Because each script skips a stage whose output already exists (so an interrupted run resumes), `run_all.sh` first moves the shipped outputs of the selected models to `_shipped_intermediates/` (`clear_intermediates.py`); every GPU stage then runs from scratch. Run Path A in a separate copy of the repository, or restore the shipped files afterwards with `python replication/clear_intermediates.py qwen llama olmo --restore`. When only one model is run, the joint selection steps read the other models' shipped coordinates. On a 16 GB GPU, the full run takes many hours per model.

## Results of our reproduction

<!-- BEGIN GENERATED: fill_readme.py -->
**Path B** (qwen, llama, olmo): 63 of 63 rows pass; patterns 3 of 3; overall PASS.

| Act | Row | Sealed | Reproduced | Status |
|---|---|---|---|---|
| 1 | qwen A3 r(x, DW-NOMINATE dim1) | 0.713 | 0.713 | PASS |
| 1 | qwen A4 r(x, economic scale) | 0.769 | 0.769 | PASS |
| 1 | qwen A4 r(y, abortion scale) | 0.612 | 0.612 | PASS |
| 1 | qwen A6 party gap x | 1.253 | 1.253 | PASS |
| 1 | qwen A6 party gap y | 0.474 | 0.474 | PASS |
| 2 | qwen A9 recall EL-SP | 0.725 | 0.725 | PASS |
| 2 | qwen A9 recall ER-ST | 0.700 | 0.700 | PASS |
| 2 | qwen A9 recall EL-ST | 0.214 | 0.214 | PASS |
| 2 | qwen A9 recall ER-SP | 0.142 | 0.142 | PASS |
| 2 | qwen A12 placed EL-SP (/15) | 13 | 13 | PASS |
| 2 | qwen A12 placed ER-ST (/15) | 13 | 13 | PASS |
| 2 | qwen A12 placed EL-ST (/15) | 6 | 6 | PASS |
| 2 | qwen A12 placed ER-SP (/15) | 2 | 2 | PASS |
| 3 | qwen A'3 H3 consistency | -0.087 | -0.087 [-0.095, -0.078] | PASS |
| 3 | qwen A'4 H4 quadrant hit | -0.127 | -0.127 [-0.192, -0.065] | PASS |
| 3 | qwen A'1 H1 choice entropy | 0.006 | 0.006 [-0.002, 0.014] | PASS |
| 4 | qwen B1 reversed - party-like entropy, Democrat | 0.182 | 0.182 [0.172, 0.192] | PASS |
| 4 | qwen B2 reversed - party-like entropy, Republican | 0.148 | 0.148 [0.141, 0.156] | PASS |
| 4 | qwen B5 economic share, Democrat (descriptive) | 0.840 | 0.840 | PASS |
| 4 | qwen B5 economic share, Republican (descriptive) | 0.898 | 0.898 | PASS |
| 4 | qwen no-persona reversed - party-like (two-way item bootstrap) | 0.172 | 0.172 [0.037, 0.309] | PASS |
| 1 | llama A3 r(x, DW-NOMINATE dim1) | 0.785 | 0.785 | PASS |
| 1 | llama A4 r(x, economic scale) | 0.843 | 0.843 | PASS |
| 1 | llama A4 r(y, abortion scale) | 0.685 | 0.685 | PASS |
| 1 | llama A6 party gap x | 1.395 | 1.395 | PASS |
| 1 | llama A6 party gap y | 0.994 | 0.994 | PASS |
| 2 | llama A9 recall EL-SP | 0.851 | 0.851 | PASS |
| 2 | llama A9 recall ER-ST | 0.825 | 0.825 | PASS |
| 2 | llama A9 recall EL-ST | 0.193 | 0.193 | PASS |
| 2 | llama A9 recall ER-SP | 0.150 | 0.150 | PASS |
| 2 | llama A12 placed EL-SP (/15) | 15 | 15 | PASS |
| 2 | llama A12 placed ER-ST (/15) | 14 | 14 | PASS |
| 2 | llama A12 placed EL-ST (/15) | 5 | 5 | PASS |
| 2 | llama A12 placed ER-SP (/15) | 3 | 3 | PASS |
| 3 | llama A'3 H3 consistency | -0.127 | -0.127 [-0.140, -0.114] | PASS |
| 3 | llama A'4 H4 quadrant hit | -0.319 | -0.319 [-0.373, -0.265] | PASS |
| 3 | llama A'1 H1 choice entropy | 0.061 | 0.061 [0.050, 0.072] | PASS |
| 4 | llama B1 reversed - party-like entropy, Democrat | -0.136 | -0.136 [-0.165, -0.106] | PASS |
| 4 | llama B2 reversed - party-like entropy, Republican | -0.153 | -0.153 [-0.190, -0.116] | PASS |
| 4 | llama B5 economic share, Democrat (descriptive) | 0.364 | 0.364 | PASS |
| 4 | llama B5 economic share, Republican (descriptive) | 0.925 | 0.925 | PASS |
| 4 | llama no-persona reversed - party-like (two-way item bootstrap) | -0.051 | -0.051 [-0.128, 0.025] | PASS |
| 1 | olmo A3 r(x, DW-NOMINATE dim1) | 0.700 | 0.700 | PASS |
| 1 | olmo A4 r(x, economic scale) | 0.790 | 0.790 | PASS |
| 1 | olmo A4 r(y, abortion scale) | 0.686 | 0.686 | PASS |
| 1 | olmo A6 party gap x | 1.290 | 1.290 | PASS |
| 1 | olmo A6 party gap y | 0.970 | 0.970 | PASS |
| 2 | olmo A9 recall EL-SP | 0.788 | 0.788 | PASS |
| 2 | olmo A9 recall ER-ST | 0.786 | 0.786 | PASS |
| 2 | olmo A9 recall EL-ST | 0.139 | 0.139 | PASS |
| 2 | olmo A9 recall ER-SP | 0.114 | 0.114 | PASS |
| 2 | olmo A12 placed EL-SP (/15) | 13 | 13 | PASS |
| 2 | olmo A12 placed ER-ST (/15) | 13 | 13 | PASS |
| 2 | olmo A12 placed EL-ST (/15) | 2 | 2 | PASS |
| 2 | olmo A12 placed ER-SP (/15) | 3 | 3 | PASS |
| 3 | olmo A'3 H3 consistency | -0.175 | -0.175 [-0.188, -0.163] | PASS |
| 3 | olmo A'4 H4 quadrant hit | -0.473 | -0.473 [-0.519, -0.423] | PASS |
| 3 | olmo A'1 H1 choice entropy | 0.068 | 0.068 [0.056, 0.081] | PASS |
| 4 | olmo B1 reversed - party-like entropy, Democrat | 0.256 | 0.256 [0.236, 0.276] | PASS |
| 4 | olmo B2 reversed - party-like entropy, Republican | 0.246 | 0.246 [0.225, 0.266] | PASS |
| 4 | olmo B5 economic share, Democrat (descriptive) | 0.684 | 0.684 | PASS |
| 4 | olmo B5 economic share, Republican (descriptive) | 0.713 | 0.713 | PASS |
| 4 | olmo no-persona reversed - party-like (two-way item bootstrap) | 0.482 | 0.482 [0.350, 0.607] | PASS |

**Path A** (qwen): 18 of 21 rows pass; patterns 1 of 1; overall FAIL.

| Act | Row | Sealed | Reproduced | Status |
|---|---|---|---|---|
| 1 | qwen A3 r(x, DW-NOMINATE dim1) | 0.713 | 0.665 | PASS |
| 1 | qwen A4 r(x, economic scale) | 0.769 | 0.747 | PASS |
| 1 | qwen A4 r(y, abortion scale) | 0.612 | 0.631 | PASS |
| 1 | qwen A6 party gap x | 1.253 | 1.167 | FAIL |
| 1 | qwen A6 party gap y | 0.474 | 0.701 | FAIL |
| 2 | qwen A9 recall EL-SP | 0.725 | 0.732 | PASS |
| 2 | qwen A9 recall ER-ST | 0.700 | 0.714 | PASS |
| 2 | qwen A9 recall EL-ST | 0.214 | 0.186 | PASS |
| 2 | qwen A9 recall ER-SP | 0.142 | 0.129 | PASS |
| 2 | qwen A12 placed EL-SP (/15) | 13 | 14 | PASS |
| 2 | qwen A12 placed ER-ST (/15) | 13 | 13 | PASS |
| 2 | qwen A12 placed EL-ST (/15) | 6 | 4 | PASS |
| 2 | qwen A12 placed ER-SP (/15) | 2 | 1 | PASS |
| 3 | qwen A'3 H3 consistency | -0.087 | -0.087 [-0.096, -0.078] | PASS |
| 3 | qwen A'4 H4 quadrant hit | -0.127 | -0.204 [-0.265, -0.142] | FAIL |
| 3 | qwen A'1 H1 choice entropy | 0.006 | 0.005 [-0.003, 0.013] | PASS |
| 4 | qwen B1 reversed - party-like entropy, Democrat | 0.182 | 0.182 [0.172, 0.192] | PASS |
| 4 | qwen B2 reversed - party-like entropy, Republican | 0.148 | 0.148 [0.141, 0.156] | PASS |
| 4 | qwen B5 economic share, Democrat (descriptive) | 0.840 | 0.840 | PASS |
| 4 | qwen B5 economic share, Republican (descriptive) | 0.898 | 0.898 | PASS |
| 4 | qwen no-persona reversed - party-like (two-way item bootstrap) | 0.172 | 0.172 [0.037, 0.309] | PASS |

In this Path A rerun the layer rule selected layer 19 for Qwen, while the sealed results use layer 20. Rerunning with layer 20 forced, 20 of 21 rows pass: qwen A6 party gap x; qwen A6 party gap y then match, while qwen A'4 H4 quadrant hit stays outside the tolerance with the same preregistered verdict as the sealed result (qwen A'4 H4 quadrant hit). Reruns on the same machine reproduce the rerun's own values exactly.

<!-- END GENERATED -->

## Data

- **Voteview** member, roll-call and vote files (Lewis et al., voteview.com). `data/fetch_data.py` downloads them and compares their sha256 with the copies used for the paper (downloaded 4 and 5 October 2026). Voteview updates its files as Congress votes. A mismatch caused only by new 119th-Congress rows does not change the inputs, because the analyses use Congresses 80 to 118.
- **Policy Agendas Project** roll-call topics ("Roll Call Votes", file `rcv_voteview_v3_4.csv`, 1947–2024). This file is not redistributed here. Download it from the Comparative Agendas Project's U.S. datasets page (https://www.comparativeagendas.net/project/us/datasets; direct link https://minio.la.utexas.edu/compagendas/datasetfiles/rcv_voteview_v3_4.csv), place it in `data/agendas/`, and check its sha256 with `fetch_data.py` (expected value in the script). The project asks users to cite the dataset (Voteview, Lewis et al.) together with the Policy Agendas Project codebook (Jones et al., 2025).
- **Items and prompts**: `STUDY_B_polarization/issue_data/` (profile items), `STUDY_J_social_topics/items_j.json` (bundled-choice items, with sha256), and `data_generation/trait_data_extract/` at the repository root (direction-extraction prompts).
- **Wikipedia-based codes for A12**: `STUDY_D_quadrant/out/s4b/` and `out/s4c/` hold `coding_codes.csv`, with short quotations from English Wikipedia articles (CC BY-SA 4.0; article titles and revision ids in `wiki_ledger.csv`). The codes were written blind to the map positions by the same AI assistant that ran the analyses, not by human coders. They cannot be regenerated by a script, so they are shipped as fixed data. In Path A, `check_selection.py` stops the run if the rerun selection of legislators differs from the coded one.

## Layout

The folder keeps the layout of the research workspace, so every script runs unchanged; `COPY_LEDGER.json` lists each file with its sha256, identical to the file that produced the sealed results.

| Folder | Contents |
|---|---|
| `STUDY_B_polarization/` | issue-specific roll-call scales (`00_data.py`) |
| `STUDY_D_quadrant/` | directions, layer rule, legislator map, quadrant recall, 60-legislator check, stated profiles |
| `STUDY_L_battery/` | bundled-choice experiment and its analysis |
| `STUDY_F_...` to `STUDY_K_...` | shared helper modules imported by Study L (prompts, personas, items, steering hooks) |
| `uncertainty_map/common.py` | shared measurement helpers imported by the roll-call scale step |
| `post/no_persona/` | no-persona effect with the two-way item bootstrap |
| `post/members_by_era/`, `post/benchmark/` | descriptive follow-ups computed after sealing (Section 7, Tables B2, B6, B7), each with its own `SEAL.json` |
| `sealed/` | byte copies of the sealed result files that `compare_to_sealed.py` checks against |
| `paper/` | the manuscript: `*.template.md` sources (abstract, introduction, `sections/`), `fill_intro.py` (fills every number from the results), `make_appendix_b.py`, `figures/`, `assemble.py` (`manuscript.md`), `build_pdf.py` (`pdf/main.pdf`, `pdf/appendix.pdf`), `check_generated.py`, and `CLAIMS_TO_CODE.md`, which maps each result sentence to its script and output file |
| `build_paper.sh` | runs the post/ follow-ups and every paper step in order, checks the post/ outputs against their seals, and reports whether the rebuilt files are byte-identical to the shipped ones; called at the end of `run_analysis.sh` |

## Known limits

- Path A matches verdicts, not digits (FP8; see above).
- The layer rule can select an adjacent layer in a rerun, which moves the map coordinates; the Path A results above report our rerun and the rerun with the sealed layer forced.
- The A12 text codes come from one AI coder and are fixed data.
- The paper's results come from one Windows machine with an RTX 5060 Ti (16 GB). Other GPUs and library versions can move FP8 logits further.

## License

Code: MIT (`LICENSE`). Data, items and codes written for this project: CC BY 4.0 (`LICENSE-DATA`). Wikipedia quotations: CC BY-SA 4.0. Voteview and Policy Agendas Project data remain under their providers' terms.
