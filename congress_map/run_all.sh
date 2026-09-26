#!/bin/bash
# Full pipeline: repo data -> vectors -> legislator projections -> stats -> map (HTML) + GIF.
# Run from this folder on a machine with one GPU (>= 24 GB; tested on NVIDIA L40S / A40).
# Steps 1 and 3 need the GPU (~15 min and ~25 min on an L40S); steps 2, 4-6 are CPU and take a few minutes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out data logs
[ -f out/vectors.pt ] || python 01_extract_vectors.py --repo_root .. --out out
python 02_get_voteview.py data
[ -f out/temporal_member_scores.csv ] || python 03_project_legislators.py --repo_root .. --vectors out/vectors.pt --voteview data/HSall_members.csv --out out
python 04_analyze.py out
python 05_build_map.py out
python 06_make_gif.py out
echo "done: out/congress_map.html  out/congress_map.gif  out/temporal_congress_stats.csv"
