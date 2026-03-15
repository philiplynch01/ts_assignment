import csv
import json
import os
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import torch
from aeon.datasets.tsc_datasets import univariate
from sklearn.metrics import (accuracy_score, f1_score)

from experiments.init_models import LRRawModel, RandomForestModel, DTWKNNModel
from experiments.utils import RunResult, load_dataset, dataset_meta, log
from init_models import HydraModel, MrSQMModel

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
np.errstate(divide="ignore", over="ignore", invalid="ignore")

RESULTS_DIR = Path("results")
PREDICTIONS_DIR = RESULTS_DIR / "optimised_hydra_predictions"
SUMMARY_CSV = RESULTS_DIR / "optimised_hydra_summary.csv"

SEEDS = [42]            # extend to e.g. [42, 0, 7] for multi-seed runs
MODELS = [
    # "hydra",
    "mps_hydra",
    "cpu_optimised_hydra",
    # "mrsqm"
    # "lr_classifier",
    # "rf_classifier",
    # "dtw_1nn"
          ]


# Set to a list of dataset names to restrict the run, e.g. ["Chinatown", "ECG200"]
DATASET_SUBSET = None

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

def run_model(model_name: str,
              seed: int,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_test: np.ndarray,
              y_test: np.ndarray,
              input_dim: int) -> np.ndarray | None:
    """Instantiate and run a model. Returns (predictions, fit_time, predict_time)."""
    # TODO: Make this dynamic rather than increasing the if/else statements
    if model_name == "hydra":
        model = HydraModel(input_dim=input_dim, seed=seed, use_latency_optimisation=False)
    elif model_name == "mps_hydra":
        model = HydraModel(input_dim=input_dim, seed=seed, device=torch.device("mps"))
    elif model_name == "cpu_optimised_hydra":
        model = HydraModel(input_dim=input_dim, seed=seed, device=torch.device("cpu"))
    elif model_name == "mrsqm":
        model = MrSQMModel(random_state=seed)
    elif model_name == "lr_classifier":
        model = LRRawModel()
    elif model_name == "rf_classifier":
        model = RandomForestModel()
    elif model_name == "dtw_1nn":
        model = DTWKNNModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    predictions = model(x_train, y_train, x_test, y_test)
    return predictions

def evaluate(dataset: str, model_name: str, seed: int) -> RunResult:
    result = RunResult(dataset=dataset, model=model_name, seed=seed)

    try:
        x_train, y_train, x_test, y_test, _ = load_dataset(dataset)
        print(f"Loaded dataset '{dataset}': ")
        print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"  x_test: {x_test.shape}, y_test: {y_test.shape}")
    except Exception as e:
        result.status = "error"
        result.error_msg = f"Data loading failed: {e}"
        log.error(f"[{dataset}] Data load error: {e}")
        return result

    meta = dataset_meta(x_train, x_test, y_train)
    result.n_train = meta["n_train"]
    result.n_test = meta["n_test"]
    result.series_length = meta["series_length"]
    result.n_classes = meta["n_classes"]

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1e6

    t_total_start = time.perf_counter()
    try:
        predictions = run_model(
            model_name, seed, x_train, y_train, x_test, y_test,
            input_dim=meta["series_length"]
        )
    except Exception as e:
        result.status = "error"
        result.error_msg = traceback.format_exc()
        result.total_time = time.perf_counter() - t_total_start
        log.error(f"[{dataset}][{model_name}][seed={seed}] Runtime error: {e}")
        return result

    result.total_time = time.perf_counter() - t_total_start
    result.peak_memory_mb = process.memory_info().rss / 1e6 - mem_before

    result.accuracy = accuracy_score(y_test, predictions)
    result.f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)
    result.f1_weighted = f1_score(y_test, predictions, average="weighted", zero_division=0)

    log.info(
        f"[{dataset}][{model_name}][seed={seed}] "
        f"acc={result.accuracy:.4f} f1={result.f1_macro:.4f} "
        f"time={result.total_time:.2f}s mem={result.peak_memory_mb:+.1f}MB"
    )

    # Save predictions for statistical tests
    pred_path = PREDICTIONS_DIR / f"{dataset}__{model_name}__seed{seed}.json"
    pred_path.write_text(json.dumps(predictions.tolist()))

    return result

def run_pipeline(datasets: list[str], models: list[str], seeds: list[int]):
    all_results: list[RunResult] = []

    total = len(datasets) * len(models) * len(seeds)
    done = 0

    for dataset in datasets:
        for model_name in models:
            for seed in seeds:
                done += 1
                log.info(f"--- [{done}/{total}] {dataset} | {model_name} | seed={seed} ---")
                result = evaluate(dataset, model_name, seed)
                all_results.append(result)
                _append_csv(SUMMARY_CSV, result)

    return all_results


def _append_csv(path: Path, result: RunResult):
    row = asdict(result)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def print_summary(results: list[RunResult]):
    import pandas as pd

    df = pd.DataFrame([asdict(r) for r in results if r.status == "ok"])
    if df.empty:
        log.warning("No successful results to summarise.")
        return

    log.info("\n=== ACCURACY SUMMARY (mean per dataset) ===")
    pivot = df.pivot_table(index="dataset", columns="model", values="accuracy")
    log.info("\n" + pivot.to_string())

    # Rank each model per dataset by accuracy (lower rank = better)
    ranked = pivot.rank(axis=1, ascending=False)
    mean_rank = ranked.mean()
    log.info(f"\nMean rank across datasets:\n{mean_rank.to_string()}")

    # Worst ~10% of datasets by mean accuracy across models
    pivot["mean_acc"] = pivot.mean(axis=1)
    n_worst = max(1, int(len(pivot) * 0.10))
    worst = pivot.nsmallest(n_worst, "mean_acc")
    log.info(f"\n=== WORST {n_worst} DATASETS (candidates for deep-dive) ===")
    log.info("\n" + worst.to_string())

    errors = [r for r in results if r.status == "error"]
    if errors:
        log.warning(f"\n{len(errors)} runs failed:")
        for r in errors:
            log.warning(f"  {r.dataset} | {r.model} | seed={r.seed}: {r.error_msg[:120]}")


def main():
    datasets = DATASET_SUBSET if DATASET_SUBSET else list(univariate)
    log.info(f"Running pipeline on {len(datasets)} datasets, "
             f"{len(MODELS)} models, {len(SEEDS)} seeds "
             f"({len(datasets) * len(MODELS) * len(SEEDS)} total runs)")

    results = run_pipeline(datasets, MODELS, SEEDS)
    print_summary(results)
    log.info(f"\nResults saved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()