"""
Compatibility wrapper for the older r&d.docx generation command.

The canonical generator is write_resultandd.py. This wrapper regenerates
resultandd.docx from corrected outputs and then copies it to r&d.docx so older
notes or commands do not produce the stale hard-coded narrative.
"""

from pathlib import Path
from shutil import copyfile

import write_resultandd


ROOT = Path(__file__).parent


def main() -> None:
    write_resultandd.main()
    copyfile(ROOT / "resultandd.docx", ROOT / "r&d.docx")
    print(f"Saved -> {ROOT / 'r&d.docx'}")


if __name__ == "__main__":
    main()
