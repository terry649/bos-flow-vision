# Reference ground truth

Small files that let committed predictions be re-scored without regenerating images:

- `v1_test/test/_annotations.coco.json` and `v1_test/metadata.jsonl`: labels and flow
  physics for the 100-image v1 test split.
- `sequences/blast_*/truth.json`: time, true shock radius, energy, and ambient density
  for each frame of the 20 blast-wave sequences.

They are copies of generator outputs (seeded; see `configs/`), used by
`notebooks/reproduce.ipynb`.
