"""Launch the drama episode script writer Streamlit app (primary entrypoint)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    app = root / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    proc = subprocess.run(cmd, check=False)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
