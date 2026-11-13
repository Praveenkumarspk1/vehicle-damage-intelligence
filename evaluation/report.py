from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_report(results: dict[str, Any], output_path: Any = None, title: str = "Vehicle Damage Detection Evaluation Report") -> dict[str, Any]:
    lines: list[str] = [f"# {title}", ""]
    for key, value in results.items():
        lines.append(f"## {key}")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    lines.append(f"### {sub_key}")
                    for detail_key, detail_value in sub_value.items():
                        lines.append(f"- {detail_key}: {detail_value}")
                else:
                    lines.append(f"- {sub_key}: {sub_value}")
        else:
            lines.append(f"{value}")
        lines.append("")
    report = "\n".join(lines)
    target = Path(str(output_path)) if output_path is not None else PROJECT_ROOT / "outputs" / "evaluation_report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    return {"report_path": str(target), "content": report}
