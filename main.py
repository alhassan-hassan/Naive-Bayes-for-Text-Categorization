"""
Runs:
- Learning curves for m=0 and m=1 on each dataset
- Smoothing sweep for required m values on each dataset
Optionally: extra credit preprocessing experiments.

To Run:
  python3 main.py --data_dir pldata --out_dir out --seed 0
"""

from __future__ import annotations
import argparse
import os
import numpy as np

from data import load_labelled_sentences, dataset_paths
from cv import stratified_kfold_indices
from experiments import learning_curves, summarize_curve, smoothing_values, smoothing_sweep, summarize_sweep
from plots import ensure_dir, plot_learning_curve, plot_smoothing_sweep


# Extra credit is optional, so import it only if the file exists.
try:
    from extra_credit import run_extra_credit
except Exception:
    run_extra_credit = None


def run_dataset(name: str, path: str, out_dir: str, seed: int) -> None:
    """Run the required experiments for one dataset and write plots + .npz outputs."""
    # Load tokenized examples from disk.
    examples = load_labelled_sentences(path)

    # Build stratified folds once and reuse them across experiments.
    labels = [e.label for e in examples]
    folds = stratified_kfold_indices(labels, k=10, seed=seed)

    # Compute learning curves for m=0 and m=1.
    lc = learning_curves(examples, folds, m_values=(0.0, 1.0), seed=seed)

    # Extract the fractions used (same keys for both m values).
    fractions = np.array(sorted(next(iter(lc.values())).keys()), dtype=float)

    # Summarize per-fold accuracies into mean/std arrays for plotting.
    curve_by_m = {}
    for m in (0.0, 1.0):
        mean, std = summarize_curve(lc[m])
        curve_by_m[m] = (mean, std)

    # Save the learning curve plot.
    plot_learning_curve(
        curve_by_m=curve_by_m,
        fractions=fractions,
        title=f"{name}: learning curves (m=0 vs m=1)",
        outpath=os.path.join(out_dir, f"{name}_learning_curves.png"),
    )

    # Run the full smoothing sweep over the required m values.
    m_list = smoothing_values()
    sweep = smoothing_sweep(examples, folds, m_list=m_list, seed=seed)

    # Summarize sweep results for plotting.
    ms, mean, std = summarize_sweep(sweep)

    # Save the sweep plot.
    plot_smoothing_sweep(
        ms=ms,
        mean=mean,
        std=std,
        title=f"{name}: smoothing sweep",
        outpath=os.path.join(out_dir, f"{name}_smoothing_sweep.png"),
    )

    # Save numeric results so the report can be written from the exact values.
    np.savez(
        os.path.join(out_dir, f"{name}_learning_curves.npz"),
        fractions=fractions,
        m0_mean=curve_by_m[0.0][0],
        m0_std=curve_by_m[0.0][1],
        m1_mean=curve_by_m[1.0][0],
        m1_std=curve_by_m[1.0][1],
    )

    np.savez(
        os.path.join(out_dir, f"{name}_smoothing_sweep.npz"),
        ms=ms,
        mean=mean,
        std=std,
    )


def main() -> None:
    """Parse args, run experiments, and optionally run extra credit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="pldata", help="Directory containing dataset .txt files (default: pldata)")
    ap.add_argument("--out_dir", default="out", help="Output directory for plots/results (default: out)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    ap.add_argument("--extra_credit", action="store_true", help="Run extra credit preprocessing experiments")
    args = ap.parse_args()

    # Make sure the output directory exists.
    ensure_dir(args.out_dir)

    # Run the required experiments for each dataset.
    for name, path in dataset_paths(args.data_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        print(f"[INFO] Running experiments for {name} ({path})")
        run_dataset(name, path, args.out_dir, args.seed)

    # Run extra credit experiments if requested.
    if args.extra_credit:
        if run_extra_credit is None:
            print("[WARN] extra_credit module not available")
            return

        # Keep extra credit outputs in a separate folder.
        ec_dir = os.path.join(args.out_dir, "extra_credit")
        ensure_dir(ec_dir)

        for name, path in dataset_paths(args.data_dir):
            print(f"[INFO] Extra credit experiments for {name}")
            table = run_extra_credit(path, ec_dir, seed=args.seed)

            # Write a TSV summary for easy copy/paste into the report.
            tsv_path = os.path.join(ec_dir, f"{name}_extra_credit.tsv")
            with open(tsv_path, "w", encoding="utf-8") as f:
                f.write("variant\tbest_m\tbest_mean_acc\tstd\n")
                for variant, best_m, best_mean, best_std in table:
                    f.write(f"{variant}\t{best_m}\t{best_mean:.4f}\t{best_std:.4f}\n")

            print(f"[INFO] Wrote {tsv_path}")

    print("[DONE] All requested plots/results written to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
