from pathlib import Path
import argparse

import pandas as pd
import torch

from pathduet.data import load_fold, load_resources
from pathduet.runtime import set_seeds
from pathduet.training import train_one_fold

ROOT = Path(__file__).resolve().parent
COHORTS = ("COAD", "KIRP", "THCA", "UCS")
SEEDS = (1028, 2024, 3407)


def run(args):
    if len(args.cohorts) != len(set(args.cohorts)) or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Cohorts and seeds must not contain duplicates.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    mask, memory = load_resources(args.data / "resources")
    rows = []
    for cohort in args.cohorts:
        for seed in args.seeds:
            set_seeds(seed)
            for fold in range(1, 6):
                split = args.data / "stage" / cohort / "splits" / f"seed_{seed}" / f"fold_{fold}.csv"
                data = load_fold(split, mask.index)
                output = args.output / cohort / f"seed_{seed}" / f"fold_{fold}"
                print(f"{cohort} seed={seed} fold={fold}", flush=True)
                metrics = train_one_fold(data, mask, memory, device, output)
                rows.append({"cohort": cohort, "seed": seed, "fold": fold, **metrics})
                pd.DataFrame(rows).to_csv(args.output / "fold_metrics.csv", index=False)
                print(f"AUC={metrics['auc']:.6f} AUPR={metrics['aupr']:.6f}", flush=True)
    folds = pd.DataFrame(rows)
    seeds = folds.groupby(["cohort", "seed"])[["auc", "aupr"]].mean()
    seeds.to_csv(args.output / "seed_metrics.csv")
    summary = seeds.groupby("cohort").agg(["mean", "std"])
    summary.to_csv(args.output / "summary.csv")
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(description="PathDUET cancer-stage classification")
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=list(COHORTS))
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--data", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
