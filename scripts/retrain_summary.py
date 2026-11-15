import json
import sys
from pathlib import Path

from housing_mlops.retraining import render_outcome


def main(argv: list[str]) -> int:
    payload = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    sys.stdout.write(render_outcome(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
