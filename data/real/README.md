# data/real

Experimental images live here locally and are never committed by default.

This project uses synthetic data only unless the author confirms in writing that a
specific image is cleared for public release. Some source data was produced under
AFRL funding and may carry distribution restrictions.

To clear an image, add its file name and its SHA-256 hash to `CLEARED.md` together
with the date and the written clearance reference. The pre-commit hook in
`.githooks/pre-commit` rejects any file under `data/real/` that is not listed there
with a matching hash.
