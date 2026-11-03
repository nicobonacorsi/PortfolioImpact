"""Execute the UI state regression tests without provider calls."""
from pathlib import Path
import shutil
import subprocess


def test_browser_portfolio_state_regressions():
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the UI regression tests"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, str(root / "tests" / "ui_portfolio_state.cjs")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": 28' in result.stdout
