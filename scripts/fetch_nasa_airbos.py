"""Download the NASA AirBOS 4 reference images listed in data/public/nasa_airbos/SOURCES.md
and verify each file against its recorded SHA-256.

    uv run python scripts/fetch_nasa_airbos.py

The images are NASA content (see SOURCES.md for the quoted usage guidelines); they are
not committed to the repository and are used for qualitative illustration only.
"""

import hashlib
import re
import sys
import urllib.request
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data" / "public" / "nasa_airbos"


def entries():
    text = (DIR / "SOURCES.md").read_text()
    for block in text.split("\n### ")[1:]:
        name = re.match(r"`([^`]+)`", block).group(1)
        url = re.search(r"^- File: (\S+)", block, re.MULTILINE).group(1)
        sha = re.search(r"^- SHA-256: `([0-9a-f]{64})`", block, re.MULTILINE).group(1)
        yield name, url, sha


def main() -> int:
    bad = 0
    for name, url, sha in entries():
        path = DIR / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = digest == sha
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
