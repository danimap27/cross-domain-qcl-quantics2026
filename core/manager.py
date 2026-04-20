"""
manager.py — Experiment Control Hub for Hercules HPC.

Reads phases and labels from config.yaml. The monitor screen [M] shows
real-time progress with a bar that refreshes every 2 seconds until the
user presses any key.

Usage:
    python manager.py
    python manager.py --config config.yaml
"""

import glob
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(name: str):
    w = 70
    print("=" * w)
    print(f"{'EXPERIMENT HUB — ' + name.upper():^{w}}")
    print("=" * w)


def run_cmd(cmd: str, capture: bool = False):
    try:
        if capture:
            r = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return r.stdout.strip()
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {cmd}\n{e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"[STDERR] {e.stderr.strip()}")
        return None


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    for enc in ["utf-8-sig", "utf-16", "latin1"]:
        try:
            with open(path, encoding=enc) as f:
                return sum(1 for l in f if l.strip())
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 0


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _progress_bar(done: int, total: int, width: int = 40) -> str:
    """Return an ASCII progress bar string."""
    if total == 0:
        return f"[{'?' * width}] ?/?  ?%"
    pct = done / total
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total}  {pct*100:.1f}%"


def _scan_progress(cfg: dict) -> tuple[int, dict, Optional[object]]:
    """
    Scan results directory.

    Returns (completed, mean_drops_by_ansatz, dataframe_or_None).
    """
    results_dir = cfg.get("output_dir", "./results")
    pattern = os.path.join(results_dir, "*", "runs.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        return 0, {}, None

    if HAS_PANDAS:
        try:
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=["run_id"])
            completed = len(df)
            drops = {}
            if "forgetting_drop" in df.columns and "ansatz" in df.columns:
                drops = (
                    df.groupby("ansatz")["forgetting_drop"]
                    .mean()
                    .to_dict()
                )
            return completed, drops, df
        except Exception:
            pass

    return len(csv_files), {}, None


def _kbhit_nonblock() -> bool:
    """Return True if a key has been pressed (Unix only, non-blocking)."""
    if not HAS_TERMIOS:
        return False
    import select
    return select.select([sys.stdin], [], [], 0)[0] != []


def show_monitor(cfg: dict):
    """
    Live progress monitor. Refreshes every 2 seconds.
    Press any key (or Enter on non-Unix) to exit.
    """
    expected = cfg.get("expected_runs", 0)
    results_dir = cfg.get("output_dir", "./results")
    ansatz_labels = cfg.get("labels", {}).get("ansatze", {})

    # Switch terminal to raw mode for non-blocking key detection (Unix only)
    old_settings = None
    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            clear_screen()
            print_header(cfg.get("experiment_name", "experiment"))
            print()
            print(f"  Results dir : {results_dir}")
            print(f"  Expected    : {expected} runs")
            print()

            completed, drops, df = _scan_progress(cfg)
            bar = _progress_bar(completed, expected)
            print(f"  Progress  {bar}")
            print()

            if drops:
                print("  Mean forgetting drop by ansatz:")
                for ansatz, val in sorted(drops.items()):
                    label = ansatz_labels.get(ansatz, ansatz)
                    print(f"    {label:<25}: {val*100:>6.2f}%")
                print()

            # Per-phase progress
            print("  Phase breakdown:")
            for phase in cfg.get("phases", []):
                phase_done = 0
                phase_total = count_lines(phase.get("file", ""))
                if df is not None and HAS_PANDAS:
                    try:
                        phase_filters = phase.get("filters", {})
                        mask = pd.Series([True] * len(df), index=df.index)
                        if "noise" in phase_filters:
                            mask &= df["noise_model"] == phase_filters["noise"]
                        if "source" in phase_filters:
                            mask &= df["source"] == phase_filters["source"]
                        if "ansatz" in phase_filters:
                            mask &= df["ansatz"] == phase_filters["ansatz"]
                        phase_done = int(mask.sum())
                    except Exception:
                        phase_done = 0
                pbar = _progress_bar(phase_done, phase_total, width=20)
                print(f"    [{phase['id']}] {phase['description']:<45} {pbar}")
            print()

            # SLURM queue
            squeue = run_cmd(
                "squeue -u $USER --format='%.10i %.9P %.30j %.8T %.10M' 2>/dev/null",
                capture=True,
            )
            if squeue:
                lines = squeue.splitlines()
                active = len(lines) - 1  # subtract header
                print(f"  Active SLURM jobs: {max(active, 0)}")
                for line in lines[:6]:   # show up to 5 jobs
                    print(f"    {line}")
            else:
                print("  SLURM queue: not available")

            print()
            print("  " + "─" * 60)
            print("  [Press any key to return to the main menu]")

            # Check for keypress
            if HAS_TERMIOS:
                if _kbhit_nonblock():
                    sys.stdin.read(1)   # consume the key
                    break
            else:
                # Fallback: wait 2s then loop — user must Ctrl+C or use menu
                time.sleep(2)
                continue

            time.sleep(2)

    finally:
        if old_settings is not None and HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    print()


# ---------------------------------------------------------------------------
# Phase submission helpers
# ---------------------------------------------------------------------------

def refresh_commands(config_path: str, cfg: dict):
    print(f"\n[INFO] Regenerating command files from {config_path}...")
    ok = run_cmd(f"python runner.py --config {config_path} --export-commands")
    if ok:
        print("[OK] Command files regenerated.")
        for phase in cfg.get("phases", []):
            n = count_lines(phase.get("file", ""))
            print(f"  - [{phase['id']}] {phase['description']}: {n} tasks")
    else:
        print("[FAIL] Check runner.py output.")
    input("\nEnter to return...")


def check_completed(cfg: dict, phase: Optional[dict] = None, view_only: bool = False) -> Optional[str]:
    """
    Comprueba runs completados/pendientes.
    - view_only=True : solo muestra estado (opción [C] del menú)
    - view_only=False: si hay completados, pregunta qué hacer (2 opciones)
      Devuelve "skip_all", "overwrite_all", o None (cancelar)
    """
    results_dir = cfg.get("output_dir", "./results")
    import glob as _glob
    import shutil
    completed_ids = {
        Path(p).parent.name
        for p in _glob.glob(os.path.join(results_dir, "*", "runs.csv"))
    }

    phases = [phase] if phase else cfg.get("phases", [])
    all_run_ids: list[str] = []
    for ph in phases:
        cmd_file = ph.get("file", "")
        if not os.path.exists(cmd_file):
            continue
        with open(cmd_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "--run-id" and i + 1 < len(parts):
                        all_run_ids.append(parts[i + 1])
                        break

    if not all_run_ids:
        print("\n  No hay archivos de comandos. Ejecuta [R] primero.")
        input("\nEnter para volver...")
        return None

    done = [r for r in all_run_ids if r in completed_ids]
    pending = [r for r in all_run_ids if r not in completed_ids]

    print(f"\n  Total: {len(all_run_ids)}  |  Completados: {len(done)}  |  Pendientes: {len(pending)}")

    if view_only:
        if done:
            print(f"\n  Completados ({len(done)}):")
            for r in done[:20]:
                print(f"    [x] {r}")
            if len(done) > 20:
                print(f"    ... y {len(done) - 20} más")
        if pending:
            print(f"\n  Pendientes ({len(pending)}):")
            for r in pending[:20]:
                print(f"    [ ] {r}")
            if len(pending) > 20:
                print(f"    ... y {len(pending) - 20} más")
        input("\nEnter para volver...")
        return None

    # Sin completados: ejecutar directo
    if not done:
        return "skip_all"

    # Con completados: 2 opciones claras
    print(f"\n  Hay {len(done)} run(s) ya completados.")
    print(f"  [1] Mantener resultados y ejecutar solo los {len(pending)} pendientes")
    print(f"  [2] Borrar todo y reejecutar los {len(all_run_ids)} runs desde cero")
    print(f"  [C] Cancelar")
    while True:
        choice = input("  Opción: ").strip().upper()
        if choice == "1":
            return "skip_all"
        if choice == "2":
            print(f"\n  Borrando resultados de {len(done)} run(s)...")
            for run_id in done:
                run_dir = Path(results_dir) / run_id
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    print(f"    Eliminado: {run_dir}")
            print(f"  Listo. {len(done)} carpeta(s) eliminadas.")
            return "overwrite_all"
        if choice == "C":
            return None
        print("  Introduce 1, 2 o C.")


def submit_phase(
    phase: dict,
    dependency_id: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[str]:
    n = count_lines(phase.get("file", ""))
    if n == 0:
        print(f"\n[WARN] {phase['file']} is empty or missing. Run [R] first.")
        return None

    dep = f"--dependency=afterok:{dependency_id}" if dependency_id else ""
    job_name = f"QCL_{phase['id']}_{phase['name']}"
    export_vars = f"CMD_FILE={phase['file']}"
    if overwrite:
        export_vars += ",EXTRA_ARGS=--overwrite"
    cmd = (
        f"sbatch --parsable --job-name='{job_name}' "
        f"--array=1-{n}%30 {dep} "
        f"--export={export_vars} "
        f"core/slurm_generic.sh"
    )
    print(f"\n[SUBMIT] {phase['description']} ({n} tasks)...")
    job_id = run_cmd(cmd, capture=True)
    if job_id:
        print(f"[OK] Job ID: {job_id}")
    return job_id


def launch_full_pipeline(cfg: dict, overwrite: bool = False):
    phases = cfg.get("phases", [])
    print(f"\n[PIPELINE] Submitting {len(phases)} phases with sequential dependencies...")
    prev_id = None
    ids = []
    for phase in phases:
        job_id = submit_phase(phase, dependency_id=prev_id, overwrite=overwrite)
        ids.append(str(job_id) if job_id else "?")
        if job_id:
            prev_id = job_id
    print(f"\n[OK] Chain: {' -> '.join(ids)}")
    input("\nEnter to return...")


def generate_tables(config_path: str):
    print("\n[TABLES] Generating LaTeX tables...")
    run_cmd(f"python core/generate_tables.py --config {config_path}")
    input("\nEnter to return...")


def generate_plots():
    print("\n[PLOTS] Generating paper figures...")
    run_cmd("python plots/plot_forgetting_curves.py --out paper/figure2_ansatz_decay.png")
    run_cmd("python plots/plot_convergence.py --out paper/figure3_convergence.png")
    input("\nEnter to return...")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment HUB")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment_name = cfg.get("experiment_name", "experiment")
    phases = cfg.get("phases", [])

    os.makedirs("logs", exist_ok=True)
    os.makedirs(cfg.get("output_dir", "./results"), exist_ok=True)

    while True:
        clear_screen()
        print_header(experiment_name)
        print()
        print("  [R]  Refresh command files from config.yaml")
        print()
        for phase in phases:
            n = count_lines(phase.get("file", ""))
            print(f"  [{phase['id']}]  {phase['description']}  ({n} tasks)")
        print()
        print("  [F]  Launch FULL PIPELINE (all phases, sequential deps)")
        print("  [M]  Monitor progress  (live, refreshes every 2s)")
        print("  [C]  Check completed / pending runs")
        print("  [T]  Generate LaTeX tables")
        print("  [P]  Generate paper figures (Figure 2 & 3)")
        print("  [X]  Exit")
        print("-" * 70)

        choice = input("Option: ").strip().upper()

        if choice == "R":
            refresh_commands(args.config, cfg)
        elif choice == "F":
            mode = check_completed(cfg)
            if mode is not None:
                launch_full_pipeline(cfg, overwrite=(mode == "overwrite_all"))
        elif choice == "M":
            show_monitor(cfg)
        elif choice == "C":
            check_completed(cfg, view_only=True)
        elif choice == "T":
            generate_tables(args.config)
        elif choice == "P":
            generate_plots()
        elif choice == "X":
            print("\nExiting.\n")
            break
        elif choice in {p["id"] for p in phases}:
            phase = next(p for p in phases if p["id"] == choice)
            mode = check_completed(cfg, phase=phase)
            if mode is not None:
                submit_phase(phase, overwrite=(mode == "overwrite_all"))
                input("\nEnter to return...")
        else:
            print("Invalid option.")
            time.sleep(1)


if __name__ == "__main__":
    main()
