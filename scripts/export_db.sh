#!/usr/bin/env bash
# Package the finished vector database + source data for the "Vector Database & Original Data
# Files" attachment (Kaggle Dataset / Hugging Face dataset repo / shared drive).
#
#   bash scripts/export_db.sh            -> dist/safeharbor-chroma-db.zip
#
# Contents: data/chroma (persistent Chroma DB + manifest.json), data/raw (MedQuAD snapshot,
# eval questions, synthetic notes before/after redaction), results/ (all benchmark tables).
# The PHI vault is deliberately EXCLUDED: shipping it would undo the de-identification.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/safeharbor-chroma-db.zip
zip -r -q dist/safeharbor-chroma-db.zip data/chroma data/raw results config.yaml -x "*.sqlite"
ls -lh dist/safeharbor-chroma-db.zip
cat <<'EOF'

Upload options:
  Kaggle:        kaggle datasets init -p dist   # edit dataset-metadata.json, then
                 kaggle datasets create -p dist --public
  Hugging Face:  huggingface-cli upload <user>/safeharbor-chroma-db dist/safeharbor-chroma-db.zip --repo-type dataset
EOF
