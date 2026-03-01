import csv
import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import psutil
import os
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report)
from sklearn.preprocessing import LabelEncoder

from init_models import HydraModel, MrSQMModel

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
np.errstate(divide="ignore", over="ignore", invalid="ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOG_FILE = RESULTS_DIR / "pipeline.log"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

SEEDS = [42]            # extend to e.g. [42, 0, 7] for multi-seed runs
MODELS = [
    "hydra",
    "mrsqm"
          ]

# Set to a list of dataset names to restrict the run, e.g. ["Chinatown", "ECG200"]
DATASET_SUBSET = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

class MrSQMFilter(logging.Filter):
    _noise = {
        "Filter subsequences",
        "Random sampling",
        "Symbolic Parameters",
        "Sampling window size",
        "Found ",
        "Fit training data",
        "Search for subsequences",
        "Transform time series",
        "Select ",
        "Compute ",
    }
    def filter(self, record):
        return not any(record.getMessage().startswith(p) for p in self._noise)

# Apply to root logger so it catches the pyx calls
logging.getLogger().addFilter(MrSQMFilter())
# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    dataset: str
    model: str
    seed: int
    # Dataset metadata
    n_train: int = 0
    n_test: int = 0
    series_length: int = 0
    n_classes: int = 0
    # Timing (seconds)
    fit_time: float = 0.0
    predict_time: float = 0.0
    total_time: float = 0.0
    # Memory (MB)
    peak_memory_mb: float = 0.0
    # Metrics
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    # Status
    status: str = "ok"          # ok | error
    error_msg: str = ""

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(name: str):
    """Load UCR dataset using canonical train/test splits."""
    x_train, y_train = load_classification(name, split="train")
    x_test, y_test = load_classification(name, split="test")

    # aeon returns shape (N, n_channels, T) — squeeze to (N, T) for univariate
    if x_train.ndim == 3 and x_train.shape[1] == 1:
        x_train = x_train[:, 0, :]
        x_test = x_test[:, 0, :]

    # Encode string labels to integers
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    return x_train, y_train, x_test, y_test, le


def dataset_meta(x_train, x_test, y_train) -> dict:
    return {
        "n_train": len(x_train),
        "n_test": len(x_test),
        "series_length": x_train.shape[-1],
        "n_classes": len(np.unique(y_train)),
    }

# ---------------------------------------------------------------------------
# Single model run
# ---------------------------------------------------------------------------

def run_model(model_name: str,
              seed: int,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_test: np.ndarray,
              y_test: np.ndarray,
              input_dim: int) -> tuple[np.ndarray | None, float, float]:
    """Instantiate and run a model. Returns (predictions, fit_time, predict_time)."""

    if model_name == "hydra":
        model = HydraModel(input_dim=input_dim, seed=seed)
    elif model_name == "mrsqm":
        model = MrSQMModel(random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # --- fit ---
    t0 = time.perf_counter()
    if model_name == "hydra":
        # HydraModel.__call__ fits and predicts in one pass;
        # split timing by doing a dry transform-only pass isn't straightforward,
        # so we time the full call and attribute it to fit+predict together.
        predictions = model(x_train, y_train, x_test, y_test)
        fit_time = time.perf_counter() - t0
        predict_time = 0.0          # inseparable in current HydraModel design
    else:
        # MrSQMModel fits transformer inside __call__ too, same approach
        predictions = model(x_train, y_train, x_test, y_test)
        fit_time = time.perf_counter() - t0
        predict_time = 0.0

    return predictions, fit_time, predict_time

# ---------------------------------------------------------------------------
# Evaluate one dataset x model x seed
# ---------------------------------------------------------------------------

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
        predictions, fit_time, predict_time = run_model(
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
    result.fit_time = fit_time
    result.predict_time = predict_time
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

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Summary & worst performers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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