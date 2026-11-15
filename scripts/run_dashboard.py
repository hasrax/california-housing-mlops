import sys
from pathlib import Path

from streamlit.web import cli as stcli

import housing_mlops.dashboard.app as dashboard_app


def main() -> int:
    app_path = Path(dashboard_app.__file__).resolve()
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
