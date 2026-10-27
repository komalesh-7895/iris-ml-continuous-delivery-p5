import json
import subprocess
import sys
from pathlib import Path


def test_quality_gate_passes():
    metrics_path = Path("metrics.json")

    original_metrics = (
        metrics_path.read_text(encoding="utf-8")
        if metrics_path.exists()
        else None
    )

    try:
        metrics_path.write_text(
            json.dumps({"accuracy": 0.90}),
            encoding="utf-8"
        )

        result = subprocess.run(
            [sys.executable, "quality_gate.py"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "QUALITY GATE PASSED" in result.stdout

    finally:
        if original_metrics is not None:
            metrics_path.write_text(
                original_metrics,
                encoding="utf-8"
            )
        else:
            metrics_path.unlink(missing_ok=True)
