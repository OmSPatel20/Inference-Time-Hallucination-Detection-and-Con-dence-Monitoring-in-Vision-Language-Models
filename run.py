"""
run.py — Single entry point for the entire project.
Replaces all .bat and .sh files. Works on Windows, Mac, and Linux.

Usage:
    python run.py setup          # Check environment + install deps
    python run.py download       # Download COCO, POPE, cache model
    python run.py test           # Quick sanity check (~20 min)
    python run.py full           # Full experiment (~4-6 hours)
    python run.py pope           # POPE evaluation only
    python run.py chair          # CHAIR evaluation only
    python run.py drift          # Confidence drift only
    python run.py plots          # Generate all figures
    python run.py dashboard      # Launch Streamlit dashboard
    python run.py status         # Show what has been completed
    python run.py clean          # Delete all outputs (fresh start)

    python run.py full --resume  # Skip already-completed steps
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
import importlib
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.resolve()
SCRIPTS_DIR = PROJECT_DIR / "scripts"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CONFIG_PATH = PROJECT_DIR / "configs" / "experiment_config.yaml"

# Track completed steps
PROGRESS_FILE = OUTPUT_DIR / ".progress.json"


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "✓", "WARN": "⚠", "ERROR": "✗", "RUN": "▶", "SKIP": "⏭"}
    symbol = prefix.get(level, "•")
    print(f"  [{timestamp}] {symbol} {msg}")


def run_script(script_name, args=None, description=None):
    """Run a script from the scripts/ directory."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        log(f"Script not found: {script_path}", "ERROR")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    desc = description or script_name
    log(f"{desc}", "RUN")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            # Stream output directly to console
            stdout=None,
            stderr=None,
        )
        if result.returncode != 0:
            log(f"FAILED: {desc} (exit code {result.returncode})", "ERROR")
            return False
        return True
    except KeyboardInterrupt:
        log("Interrupted by user", "WARN")
        return False
    except Exception as e:
        log(f"Error running {desc}: {e}", "ERROR")
        return False


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(step_name):
    progress = load_progress()
    progress[step_name] = {
        "completed": True,
        "timestamp": datetime.now().isoformat(),
    }
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def is_completed(step_name):
    progress = load_progress()
    return progress.get(step_name, {}).get("completed", False)


def time_since(start):
    elapsed = time.time() - start
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    elif elapsed < 3600:
        return f"{elapsed/60:.1f}min"
    else:
        return f"{elapsed/3600:.1f}h"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_setup():
    """Check environment and verify all dependencies."""
    print("\n" + "=" * 60)
    print("  ENVIRONMENT CHECK")
    print("=" * 60)

    errors = []

    # Python version
    v = sys.version_info
    log(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.minor < 10:
        log("Python 3.10+ recommended", "WARN")

    # CUDA / PyTorch
    try:
        import torch
        log(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            log(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            if vram < 8:
                log(f"Only {vram:.1f}GB VRAM — may struggle with INT4", "WARN")
        else:
            log("No CUDA GPU detected — this project requires a GPU", "ERROR")
            errors.append("No GPU")
    except ImportError:
        log("PyTorch not installed", "ERROR")
        errors.append("No PyTorch")

    # Key packages
    packages = [
        "transformers", "accelerate", "bitsandbytes",
        "PIL", "scipy", "sklearn", "matplotlib", "seaborn",
        "pandas", "yaml", "tqdm", "psutil",
    ]
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            # Some package names differ from import names
            alt_names = {"PIL": "Pillow", "yaml": "pyyaml", "sklearn": "scikit-learn"}
            missing.append(alt_names.get(pkg, pkg))

    if missing:
        log(f"Missing packages: {', '.join(missing)}", "ERROR")
        log(f"Run: pip install {' '.join(missing)}", "INFO")
        errors.append("Missing packages")
    else:
        log("All required packages installed")

    # pycocotools (often fails on Windows)
    try:
        import pycocotools
        log("pycocotools installed")
    except ImportError:
        log("pycocotools missing — needed for CHAIR evaluation", "WARN")
        log("Run: pip install pycocotools", "INFO")

    # Streamlit (optional)
    try:
        import streamlit
        log(f"Streamlit {streamlit.__version__} (dashboard available)")
    except ImportError:
        log("Streamlit not installed — dashboard won't work", "WARN")
        log("Run: pip install streamlit plotly", "INFO")

    # Disk space
    free_gb = shutil.disk_usage(str(PROJECT_DIR)).free / 1024**3
    log(f"Free disk space: {free_gb:.1f} GB")
    if free_gb < 25:
        log("Need ~25GB free (6GB COCO + 14GB model cache + outputs)", "WARN")

    # Config
    if CONFIG_PATH.exists():
        log(f"Config found: {CONFIG_PATH}")
    else:
        log(f"Config missing: {CONFIG_PATH}", "ERROR")
        errors.append("No config")

    print()
    if errors:
        log(f"Issues found: {', '.join(errors)}", "ERROR")
        print("  Fix the above issues before running experiments.\n")
        return False
    else:
        log("Environment OK — ready to run")
        print()
        return True


def cmd_download(resume=False):
    """Download all data and cache the model."""
    print("\n" + "=" * 60)
    print("  DOWNLOADING DATA & MODEL")
    print("=" * 60)

    if resume and is_completed("download"):
        log("Already completed — skipping (use --no-resume to force)", "SKIP")
        return True

    t0 = time.time()
    ok = run_script("01_download_data.py", description="Download COCO + POPE + cache model")
    if ok:
        save_progress("download")
        log(f"Download complete ({time_since(t0)})")
    return ok


def cmd_test():
    """Quick sanity check — verifies everything works."""
    print("\n" + "=" * 60)
    print("  QUICK TEST (~20 minutes)")
    print("=" * 60)

    steps = [
        ("01_download_data.py", [], "Download data (if needed)"),
        ("02_basic_inference.py", [], "Basic inference test"),
        ("04_entropy_detector.py", [], "Entropy detector test"),
        ("07_pope_evaluation.py",
         ["--split", "random", "--max_samples", "200", "--no_contrastive"],
         "POPE quick test (200 samples, entropy only)"),
    ]

    t0 = time.time()
    for script, args, desc in steps:
        print()
        if not run_script(script, args, desc):
            log("Quick test FAILED — fix errors above before running full experiments", "ERROR")
            return False

    print()
    log(f"Quick test PASSED ({time_since(t0)})")
    print()
    print("  Everything works. Next: python run.py full")
    print()
    return True


def cmd_pope(resume=False):
    """Run POPE evaluation across all splits."""
    print("\n" + "=" * 60)
    print("  POPE EVALUATION")
    print("=" * 60)

    if resume and is_completed("pope"):
        log("Already completed — skipping", "SKIP")
        return True

    t0 = time.time()
    ok = run_script("10_experiment_runner.py", description="POPE: all splits × INT4")
    if ok:
        save_progress("pope")
        log(f"POPE complete ({time_since(t0)})")
    return ok


def cmd_chair(resume=False):
    """Run CHAIR evaluation."""
    print("\n" + "=" * 60)
    print("  CHAIR EVALUATION")
    print("=" * 60)

    if resume and is_completed("chair"):
        log("Already completed — skipping", "SKIP")
        return True

    t0 = time.time()
    ok = run_script("08_chair_evaluation.py", description="CHAIR: 200 captions")
    if ok:
        save_progress("chair")
        log(f"CHAIR complete ({time_since(t0)})")
    return ok


def cmd_drift(resume=False):
    """Run confidence drift monitoring."""
    print("\n" + "=" * 60)
    print("  CONFIDENCE DRIFT MONITORING")
    print("=" * 60)

    if resume and is_completed("drift"):
        log("Already completed — skipping", "SKIP")
        return True

    t0 = time.time()
    ok = run_script("09_confidence_monitor.py", description="Drift: 1000 consecutive inferences")
    if ok:
        save_progress("drift")
        log(f"Drift monitoring complete ({time_since(t0)})")
    return ok


def cmd_plots():
    """Generate all analysis figures."""
    print("\n" + "=" * 60)
    print("  GENERATING PLOTS")
    print("=" * 60)

    t0 = time.time()
    ok = run_script("11_analysis_plots.py", description="Generate all figures")
    if ok:
        save_progress("plots")
        log(f"Plots generated ({time_since(t0)})")
        plots_dir = OUTPUT_DIR / "plots"
        if plots_dir.exists():
            print(f"\n  Saved to: {plots_dir}")
            for f in sorted(plots_dir.iterdir()):
                print(f"    {f.name}")
        print()
    return ok


def cmd_full(resume=False):
    """Run the complete experiment pipeline."""
    total_start = time.time()

    print("\n" + "=" * 60)
    print("  FULL EXPERIMENT PIPELINE")
    print("  Estimated time: 4-6 hours on RTX 4060")
    print("  Close Chrome, Discord, and other GPU apps!")
    print("=" * 60)

    steps = [
        ("download", cmd_download),
        ("verify",   lambda r: run_script("02_basic_inference.py",
                                          description="Verify inference")),
        ("benchmark", lambda r: run_script("03_batched_pipeline.py",
                                           description="Throughput benchmark")),
        ("pope",     cmd_pope),
        ("chair",    cmd_chair),
        ("drift",    cmd_drift),
        ("plots",    cmd_plots),
    ]

    results = {}
    for step_name, step_fn in steps:
        print()
        step_start = time.time()

        if resume and step_name not in ("verify", "plots") and is_completed(step_name):
            log(f"Step '{step_name}' already done — skipping", "SKIP")
            results[step_name] = "skipped"
            continue

        try:
            ok = step_fn(resume)
        except Exception as e:
            log(f"Step '{step_name}' crashed: {e}", "ERROR")
            ok = False

        elapsed = time_since(step_start)
        if ok:
            results[step_name] = f"done ({elapsed})"
        else:
            results[step_name] = f"FAILED ({elapsed})"
            log(f"Step '{step_name}' failed — continuing with remaining steps", "WARN")

    # Final summary
    total_elapsed = time_since(total_start)
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    for step, status in results.items():
        symbol = "✓" if "done" in status or "skip" in status else "✗"
        print(f"  {symbol} {step:12s}  {status}")
    print(f"\n  Total time: {total_elapsed}")
    print(f"  Results:    {OUTPUT_DIR}")
    print(f"  Plots:      {OUTPUT_DIR / 'plots'}")
    print()
    print("  Next: python run.py dashboard")
    print()


def cmd_dashboard():
    """Launch the Streamlit results dashboard."""
    dashboard_path = PROJECT_DIR / "dashboard.py"
    if not dashboard_path.exists():
        log("dashboard.py not found", "ERROR")
        return False

    try:
        import streamlit
    except ImportError:
        log("Streamlit not installed", "ERROR")
        log("Run: pip install streamlit plotly", "INFO")
        return False

    # Check if any results exist
    if not (OUTPUT_DIR / "pope_results").exists():
        log("No results found — run experiments first: python run.py full", "WARN")

    print("\n" + "=" * 60)
    print("  LAUNCHING DASHBOARD")
    print("  Opening browser at http://localhost:8501")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "true",
    ], cwd=str(PROJECT_DIR))


def cmd_status():
    """Show what experiments have been completed."""
    print("\n" + "=" * 60)
    print("  PROJECT STATUS")
    print("=" * 60)

    progress = load_progress()
    all_steps = ["download", "pope", "chair", "drift", "plots"]

    for step in all_steps:
        info = progress.get(step, {})
        if info.get("completed"):
            ts = info.get("timestamp", "?")
            print(f"  ✓ {step:12s}  completed at {ts}")
        else:
            print(f"  ○ {step:12s}  not started")

    # Check for result files
    print()
    result_dirs = {
        "POPE results": OUTPUT_DIR / "pope_results",
        "CHAIR results": OUTPUT_DIR / "chair_results",
        "Drift data": OUTPUT_DIR / "drift_monitor",
        "Plots": OUTPUT_DIR / "plots",
    }
    for name, path in result_dirs.items():
        if path.exists():
            count = len(list(path.iterdir()))
            print(f"  📁 {name}: {count} files in {path}")
        else:
            print(f"  📁 {name}: (not yet created)")
    print()


def cmd_clean():
    """Delete all output files for a fresh start."""
    print("\n  This will delete ALL experiment results in outputs/")
    confirm = input("  Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("  Cancelled.")
        return

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print("  Deleted outputs/")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Clean complete — ready for fresh run")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VLM Hallucination Detection — Project Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  setup       Check environment and dependencies
  download    Download COCO, POPE, cache LLaVA model
  test        Quick sanity check (~20 min)
  full        Run complete pipeline (~4-6 hours)
  pope        POPE evaluation only
  chair       CHAIR evaluation only
  drift       Confidence drift monitoring only
  plots       Generate analysis figures
  dashboard   Launch Streamlit results viewer
  status      Show what's been completed
  clean       Delete all outputs

Examples:
  python run.py setup
  python run.py test
  python run.py full --resume
  python run.py dashboard
        """,
    )
    parser.add_argument("command", nargs="?", default="status",
                        choices=["setup", "download", "test", "full",
                                 "pope", "chair", "drift", "plots",
                                 "dashboard", "status", "clean"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed steps")

    args = parser.parse_args()

    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║  VLM Hallucination Detection — Project Runner    ║")
    print("  ║  RTX 4060 (8GB) · INT4 · LLaVA-1.5-7B           ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    commands = {
        "setup": lambda: cmd_setup(),
        "download": lambda: cmd_download(args.resume),
        "test": lambda: cmd_test(),
        "full": lambda: cmd_full(args.resume),
        "pope": lambda: cmd_pope(args.resume),
        "chair": lambda: cmd_chair(args.resume),
        "drift": lambda: cmd_drift(args.resume),
        "plots": lambda: cmd_plots(),
        "dashboard": lambda: cmd_dashboard(),
        "status": lambda: cmd_status(),
        "clean": lambda: cmd_clean(),
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
