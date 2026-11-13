"""Unified Vehicle Damage Intelligence Platform — Senior-Level Single Entrypoint.

Supports multi-mode execution:
1. UI Mode:      python run.py --mode ui       (Interactive Streamlit Web Portal)
2. API Mode:     python run.py --mode api      (Production Flask REST API)
3. Test Mode:    python run.py --mode test     (Full Pytest regression & integration suite)
4. Eval Mode:    python run.py --mode eval     (COCO mAP50-95 & confusion matrix validation)
5. Train Mode:   python run.py --mode train --target [gbr|yolo|unet|vgg19]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent


def run_ui(args: argparse.Namespace) -> int:
    """Launch the interactive Streamlit Web Portal."""
    print("=" * 60)
    print("[*] Launching Vehicle Damage Intelligence — Streamlit Web Portal")
    print("    IRDAI Indian Bodyshop Schedule | 3D Mesh | Surveyor Console")
    print(f"    URL: http://localhost:{args.port}")
    print("=" * 60)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.port",
        str(args.port),
        "--server.headless",
        "false",
    ]
    return subprocess.call(cmd)


def run_api(args: argparse.Namespace) -> int:
    """Launch the Production Flask REST API."""
    print("=" * 60)
    print("[*] Launching Vehicle Damage Intelligence — Flask REST API")
    print("    Endpoints: /health, /predict, /compare, /video, /pricing, /report")
    print(f"    Listening on: http://{args.host}:{args.port}")
    print("=" * 60)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from api.flask_api import app
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def run_tests(args: argparse.Namespace) -> int:
    """Run full pytest regression & integration test suite."""
    print("=" * 60)
    print("[*] Running Pytest Regression & Integration Test Suite")
    print("=" * 60)
    cmd = [sys.executable, "-m", "pytest", "-v"]
    if args.filter:
        cmd.extend(["-k", args.filter])
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def run_eval(args: argparse.Namespace) -> int:
    """Run model evaluation."""
    print("=" * 60)
    print("[*] Evaluating Vehicle Damage Model (mAP50-95 & Metrics)")
    print("=" * 60)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate_model.py"),
        "--model",
        args.model or "models/best.pt",
    ]
    if args.data:
        cmd.extend(["--data", args.data])
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def run_train(args: argparse.Namespace) -> int:
    """Run model training scripts."""
    target = (args.target or "gbr").lower()
    print("=" * 60)
    print(f"[*] Running Training Pipeline for: {target.upper()}")
    print("=" * 60)
    script_map = {
        "gbr": "training/train_gbr.py",
        "yolo": "training/train_yolo.py",
        "unet": "training/train_unet.py",
        "vgg19": "training/train_vgg19.py",
    }
    script = script_map.get(target)
    if not script:
        print(f"[!] Unknown target: {target}. Available: {list(script_map.keys())}")
        return 1
    return subprocess.call([sys.executable, str(PROJECT_ROOT / script)], cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Vehicle Damage Intelligence Engine — Master Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "api", "test", "eval", "train"],
        default="ui",
        help="Execution mode (ui, api, test, eval, train). Default: ui",
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="API Host")
    parser.add_argument("--port", type=int, default=8501, help="Port (Default: 8501 for UI, 8000 for API)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")


    parser.add_argument("--filter", "-k", type=str, default=None, help="Pytest filter expression")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default=None, help="Dataset yaml for evaluation")
    parser.add_argument("--target", choices=["gbr", "yolo", "unet", "vgg19"], default="gbr", help="Training target")

    args = parser.parse_args()


    if args.mode == "api" and args.port == 8501:
        args.port = 8000

    if args.mode == "ui":
        return run_ui(args)
    elif args.mode == "api":
        return run_api(args)
    elif args.mode == "test":
        return run_tests(args)
    elif args.mode == "eval":
        return run_eval(args)
    elif args.mode == "train":
        return run_train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
