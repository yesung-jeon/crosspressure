# release_core: sealed core results (steps 2-5)

Scope: `../manuscript/RESEARCH_DESIGN.md` v2, steps 2-5 (step 1, the validity baseline, is excluded). Exploratory; one model
(Qwen2.5-7B-Instruct int8 generator, Qwen2.5-3B-Instruct judge).

| file | contents |
|---|---|
| `reproduce_core.py` | one linear script: copy frozen intermediates -> re-run the 7 analysis scripts unchanged -> extract 36 cited values -> compare with the original outputs -> seal (`--seal`) or verify against the seal |
| `RELEASE.json` | sealed values, SHA-256 of 106 intermediate files, 23 external inputs and 8 code files, environment |
| `CLAIMS.md` | every cited number with its key, generated from RELEASE.json (the only place prose may take numbers from) |
| `rerun_*/` | each clean re-run with logs and the comparison files |

Checks done (2026-10-04): sealing run 36/36 values match the original outputs; a second clean run 36/36 match the seal, no input changed.

Run: `python reproduce_core.py` (verify, about 1 minute on CPU; exit code 1 on any mismatch). Tier 2 (regenerating the intermediates
on a GPU with `../uncertainty_map/` steps 02, 09, 18, 20, 23, 25) is not part of this check: sampled generations reproduce in
distribution only, so tier-2 results are compared with these sealed values with tolerances.
