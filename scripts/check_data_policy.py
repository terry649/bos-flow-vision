"""Enforce the data-release policy on staged files (pre-commit) or the whole tree (CI).

Every file under data/real/ other than README.md and CLEARED.md must be listed in
data/real/CLEARED.md as `sha256  filename  date  clearance-reference`. Staged or
tracked changes must not contain a ROBOFLOW_API_KEY value.

    python scripts/check_data_policy.py --staged     # used by .githooks/pre-commit
    python scripts/check_data_policy.py --all        # used by CI
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ALLOWED = {"data/real/README.md", "data/real/CLEARED.md"}
KEY_RE = re.compile(r"ROBOFLOW_API_KEY\s*[:=]\s*['\"]?[A-Za-z0-9]{12,}")


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def cleared_hashes(text: str) -> set[tuple[str, str]]:
    out = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            out.add((parts[0], parts[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.staged:
        files = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").decode().split()
        read = lambda f: git("show", f":{f}")
        cleared_text = (git("show", ":data/real/CLEARED.md").decode()
                        if "data/real/CLEARED.md" in git("ls-files", "--cached").decode()
                        else "")
    else:
        files = git("ls-files").decode().split()
        read = lambda f: Path(f).read_bytes()
        cleared_text = Path("data/real/CLEARED.md").read_text()
    cleared = cleared_hashes(cleared_text)

    problems = []
    for f in files:
        if f.startswith("data/real/") and f not in ALLOWED:
            h = hashlib.sha256(read(f)).hexdigest()
            if (h, Path(f).name) not in cleared:
                problems.append(f"{f}: not in data/real/CLEARED.md with a matching sha256")
        if f.endswith((".py", ".yaml", ".yml", ".toml", ".json", ".md", ".txt", ".ipynb",
                       ".sh", ".cfg", ".ini")) or Path(f).name.startswith(".env"):
            try:
                text = read(f).decode("utf-8", errors="ignore")
            except (OSError, subprocess.CalledProcessError):
                continue
            if KEY_RE.search(text):
                problems.append(f"{f}: appears to contain a ROBOFLOW_API_KEY value")

    for p in problems:
        print(f"data policy: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
