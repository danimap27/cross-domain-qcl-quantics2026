"""
generate_tables.py — Generic LaTeX table generator.

Reads all results/<run_id>/runs.csv files and generates tables
defined in config.yaml. Labels, metrics, and table structure
are fully driven by the configuration file.

Usage:
    python generate_tables.py
    python generate_tables.py --config config.yaml --results-dir ./results --out-dir ./paper/tables
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config and data loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_results(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "*", "runs.csv")
    dfs = []
    for csv_path in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  [WARN] {csv_path}: {e}")
    if not dfs:
        print(f"[ERROR] No runs.csv files found in {results_dir}")
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values("timestamp", na_position="last").drop_duplicates(
        subset=["run_id"], keep="last"
    )
    print(f"[OK] Loaded {len(all_df)} unique runs from {len(dfs)} folders.")
    return all_df


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

def fmt_ms(series: pd.Series, as_pct: bool = True) -> str:
    """Format mean ± std for a pandas Series."""
    if series.empty or series.isna().all():
        return "---"
    mean = series.mean()
    std = series.std(ddof=1) if len(series) > 1 else 0.0
    if pd.isna(mean) or pd.isna(std):
        return "---"
    if as_pct:
        return f"${mean*100:.1f} \\pm {std*100:.1f}$"
    return f"${mean:.2f} \\pm {std:.2f}$"


def fmt_time(series: pd.Series) -> str:
    """Format mean training time in seconds."""
    if series.empty or series.isna().all():
        return "---"
    mean = series.mean()
    return f"${mean:.0f}$"


def write_tex(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] {path}")


# ---------------------------------------------------------------------------
# Table 1 — Topology comparison (ideal vs noisy)
# ---------------------------------------------------------------------------

def make_topology_table(df: pd.DataFrame, cfg: dict, out_dir: str):
    """
    Topology comparison: SEL vs Basic Entangler vs TTN.
    Columns: ansatz, noise, Acc_A_init, Acc_B, Acc_A_final, Delta_A, Time_A, Time_B.
    """
    labels = cfg.get("labels", {})
    ansatz_labels = labels.get("ansatze", {})
    noise_labels = labels.get("noise_models", {})
    ansatz_order = [a["name"] for a in cfg.get("ansatze", [])]
    noise_order = [n["name"] for n in cfg.get("noise_models", [])]

    sub = df[df["source"] == "scratch"]
    if sub.empty:
        print("  [INFO] No topology data (source=scratch); skipping.")
        return

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Topology comparison across three ans\"{a}tze under ideal simulation "
        r"and IBM Heron~r2 noise. Values represent mean\,$\pm$\,std over "
        + str(len(cfg.get("seeds", [])))
        + r" seeds. $\Delta_A$ denotes the forgetting drop on Task~A after Task~B training.}",
        r"\label{tab:topology}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Ans\"{a}tz} & \textbf{Noise} & "
        r"$\text{Acc}_A^{\text{init}}$\,(\%) & "
        r"$\text{Acc}_B$\,(\%) & "
        r"$\text{Acc}_A^{\text{final}}$\,(\%) & "
        r"$\Delta_A$\,(\%) & "
        r"Time$_A$ (s) \\",
        r"\midrule",
    ]

    for ansatz in ansatz_order:
        a_label = ansatz_labels.get(ansatz, ansatz)
        first_noise = True
        rows_added = 0
        for noise in noise_order:
            n_label = noise_labels.get(noise, noise)
            row_df = sub[(sub["ansatz"] == ansatz) & (sub["noise_model"] == noise)]
            if row_df.empty:
                continue
            row = [
                a_label if first_noise else "",
                n_label,
                fmt_ms(row_df["acc_a_initial"]),
                fmt_ms(row_df["acc_b_final"]),
                fmt_ms(row_df["acc_a_final"]),
                fmt_ms(row_df["forgetting_drop"]),
                fmt_time(row_df["train_time_a_s"]),
            ]
            first_noise = False
            lines.append(" & ".join(row) + r" \\")
            rows_added += 1
        if rows_added > 0:
            lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    write_tex(os.path.join(out_dir, "tab_topology.tex"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 2 — Cross-domain pre-training comparison
# ---------------------------------------------------------------------------

def make_crossdomain_table(df: pd.DataFrame, cfg: dict, out_dir: str):
    """
    Cross-domain comparison: scratch vs synthetic vs MobileNetV2.
    Uses only TTN ansatz.
    """
    labels = cfg.get("labels", {})
    source_labels = labels.get("sources", {})
    noise_labels = labels.get("noise_models", {})
    source_order = [s["name"] for s in cfg.get("sources", [])]
    noise_order = [n["name"] for n in cfg.get("noise_models", [])]

    sub = df[df["ansatz"] == "ttn"]
    if sub.empty:
        print("  [INFO] No TTN data for cross-domain table; skipping.")
        return

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Cross-domain pre-training comparison using the TTN ans\"{a}tz. "
        r"Scratch denotes random initialization. Values are mean\,$\pm$\,std over "
        + str(len(cfg.get("seeds", [])))
        + r" seeds.}",
        r"\label{tab:crossdomain}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Initialization} & \textbf{Noise} & "
        r"$\text{Acc}_\text{src}$\,(\%) & "
        r"$\text{Acc}_A^{\text{init}}$\,(\%) & "
        r"$\text{Acc}_B$\,(\%) & "
        r"$\text{Acc}_A^{\text{final}}$\,(\%) & "
        r"$\Delta_A$\,(\%) \\",
        r"\midrule",
    ]

    for source in source_order:
        s_label = source_labels.get(source, source)
        first_noise = True
        rows_added = 0
        for noise in noise_order:
            n_label = noise_labels.get(noise, noise)
            row_df = sub[(sub["source"] == source) & (sub["noise_model"] == noise)]
            if row_df.empty:
                continue
            row = [
                s_label if first_noise else "",
                n_label,
                fmt_ms(row_df["acc_source"]),
                fmt_ms(row_df["acc_a_initial"]),
                fmt_ms(row_df["acc_b_final"]),
                fmt_ms(row_df["acc_a_final"]),
                fmt_ms(row_df["forgetting_drop"]),
            ]
            first_noise = False
            lines.append(" & ".join(row) + r" \\")
            rows_added += 1
        if rows_added > 0:
            lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    write_tex(os.path.join(out_dir, "tab_crossdomain.tex"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 3 — Dataset summary
# ---------------------------------------------------------------------------

def make_dataset_table(cfg: dict, out_dir: str):
    """Static dataset description table (does not read results)."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Dataset summary. All image datasets are reduced to "
        + str(cfg.get("n_qubits", 4))
        + r" features via PCA before encoding. Samples correspond to the binary subset used.}",
        r"\label{tab:datasets}",
        r"\begin{tabular}{lcccl}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Train} & \textbf{Test} & \textbf{Classes} & \textbf{Role} \\",
        r"\midrule",
        r"Synthetic Gaussian & 1{,}000 & --- & 2 & Pre-training source \\",
        r"Fashion-MNIST~(T-shirt/Trouser) & 12{,}000 & 2{,}000 & 2 & Task~A \\",
        r"MNIST~(digit~2 / digit~3) & 11{,}879 & 1{,}990 & 2 & Task~B \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    write_tex(os.path.join(out_dir, "tab_datasets.tex"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Master include file
# ---------------------------------------------------------------------------

def make_master(out_dir: str, files: list):
    lines = [
        "% AUTO-GENERATED — include all tables",
        r"% \input{paper/tables/tables.tex}",
        "",
    ]
    for f in files:
        lines.append(r"\input{" + f.replace("\\", "/") + "}")
    write_tex(os.path.join(out_dir, "tables.tex"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out-dir", default="./paper/tables")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = args.results_dir or cfg.get("output_dir", "./results")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LaTeX Table Generator")
    print(f"  Results : {results_dir}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*60}\n")

    df = load_results(results_dir)
    generated = []

    print("[1/3] Dataset table...")
    make_dataset_table(cfg, out_dir)
    generated.append(os.path.join(out_dir, "tab_datasets.tex"))

    if not df.empty:
        print("[2/3] Topology comparison table...")
        make_topology_table(df, cfg, out_dir)
        generated.append(os.path.join(out_dir, "tab_topology.tex"))

        print("[3/3] Cross-domain pre-training table...")
        make_crossdomain_table(df, cfg, out_dir)
        generated.append(os.path.join(out_dir, "tab_crossdomain.tex"))
    else:
        print("[2/3] No results yet — topology and cross-domain tables skipped.")
        print("[3/3] Skipped.")

    make_master(out_dir, [os.path.relpath(f, start=os.path.dirname(out_dir)) for f in generated])
    print(f"\n[SUCCESS] {len(generated)} table(s) in {out_dir}/")


if __name__ == "__main__":
    main()
