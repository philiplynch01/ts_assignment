import numpy as np
import pandas as pd


def saliency_to_importance(saliency: np.ndarray, use_absolute: bool = True):
    '''Convert raw saliency to importance values
    '''
    saliency = np.asarray(saliency, dtype=np.float32)
    return np.abs(saliency) if use_absolute else saliency


def select_contiguous_window(importance: np.ndarray, fraction: float, mode: str = "top", rng: np.random.Generator | None = None):
    '''Select a contiguous mask region using top, bottom, or random importance
    '''
    importance = np.asarray(importance, dtype=np.float32)
    T = len(importance)

    # Window length from masking fraction
    window = max(1, int(round(T * fraction)))
    mask = np.zeros(T, dtype=bool)

    if window >= T:
        mask[:] = True
        return mask

    # NOTE - Random contiguous window
    if mode == "random":
        rng = np.random.default_rng() if rng is None else rng
        start = int(rng.integers(0, T - window + 1))
    
    # NOTE - Score all windows by summed importance
    else:
        scores = np.convolve(importance, np.ones(window, dtype=np.float32), mode="valid")
        if mode == "top":
            start = int(np.argmax(scores))
        elif mode == "bottom":
            start = int(np.argmin(scores))
    mask[start:start + window] = True
    return mask


def apply_mask(x: np.ndarray, mask: np.ndarray):
    '''Apply mean masking to the selected region
    '''
    x = np.asarray(x, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    x[mask] = float(np.mean(x))
    return x


def _predicted_class_score(decision_output: np.ndarray, pred_label: int):
    ''' Return the decision score for the predicted class
    '''
    decision_output = np.asarray(decision_output)

    if decision_output.ndim == 0:
        return float(decision_output)

    if decision_output.ndim == 1:
        # Binary case: single decision margin
        margin = float(decision_output[0])
        return margin if pred_label == 1 else -margin

    # Multiclass case
    return float(decision_output[0, pred_label])


def explain_single_cached(model, x, pred_before, use_absolute=True):
    ''' Compute importance and baseline score once per sample
    '''
    x = np.asarray(x, dtype=np.float32)
    x_batch = x[None, :]

    score_before = _predicted_class_score(model.decision_function(x_batch), pred_before)
    saliency = np.asarray(model.explain(x), dtype=np.float32)
    importance = saliency_to_importance(saliency, use_absolute=use_absolute)
    return {"importance": importance, "score_before": score_before}


def mask_from_cached(model, x, importance, pred_before, score_before, fraction, mode="top", rng=None):
    ''' Apply masking from cached importance and evaluate the prediction change
    '''
    # Generate mask
    mask = select_contiguous_window(importance=importance, fraction=fraction, mode=mode, rng=rng)
    x_masked = apply_mask(x, mask)

    x_masked_batch = x_masked[None, :]
    pred_after = int(model.predict(x_masked_batch)[0])
    # Get predicted score
    score_after = _predicted_class_score(model.decision_function(x_masked_batch), pred_before)
    return {
        "mask": mask,
        "x_masked": x_masked,
        "pred_after": pred_after,
        "score_after": score_after,
        "score_drop": score_before - score_after,
        "flipped": int(pred_before != pred_after),
    }


def evaluate_masking_dataset(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fractions=(0.05, 0.10, 0.20),
    use_absolute: bool = True,
    only_correct: bool = True,
    random_repeats: int = 5,
    seed: int = 42,
    max_samples: int | None = None,
    ):
    '''Evaluate top, random, and bottom masking across a dataset
    '''
    rng = np.random.default_rng(seed)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    # Baseline predictions and accuracy
    base_preds = model.predict(X_test)
    base_acc = float(np.mean(base_preds == y_test))

    # Testing restricting eval to correct predictions
    candidate_indices = np.arange(len(X_test))
    if only_correct:
        candidate_indices = candidate_indices[base_preds == y_test]

    if max_samples is not None:
        candidate_indices = candidate_indices[:max_samples]

    rows = []
    for count, i in enumerate(candidate_indices, 1):
        x = X_test[i]
        pred_before = int(base_preds[i])

        # NOTE - Compute saliency once and reuse it for all masking settings
        cached = explain_single_cached(model=model, x=x, pred_before=pred_before, use_absolute=use_absolute)
        importance = cached["importance"]
        score_before = cached["score_before"]

        if count % 10 == 0:
            print(f"  processed {count}/{len(candidate_indices)} samples")

        for frac in fractions:
            # Top & bottom masking using cache to avoid recomputing
            for mode in ("top", "bottom"):
                out = mask_from_cached(model=model, x=x, importance=importance, pred_before=pred_before, score_before=score_before, fraction=frac, mode=mode, rng=rng)
                rows.append({
                    "sample_idx": i,
                    "fraction": frac,
                    "mode": mode,
                    "true_label": int(y_test[i]),
                    "base_pred": pred_before,
                    "masked_pred": out["pred_after"],
                    "score_before": score_before,
                    "score_after": out["score_after"],
                    "score_drop": out["score_drop"],
                    "flipped": out["flipped"],
                })

            # Random masking averaged across repeated runs
            random_drops = []
            random_flips = []
            random_preds = []
            for _ in range(random_repeats):
                out = mask_from_cached(model=model, x=x, importance=importance, pred_before=pred_before, score_before=score_before, fraction=frac, mode="random", rng=rng)
                random_drops.append(out["score_drop"])
                random_flips.append(out["flipped"])
                random_preds.append(out["pred_after"])

            rows.append({
                "sample_idx": i,
                "fraction": frac,
                "mode": "random",
                "true_label": int(y_test[i]),
                "base_pred": pred_before,
                "masked_pred": int(round(np.mean(random_preds))),
                "score_before": score_before,
                "score_after": np.nan,
                "score_drop": float(np.mean(random_drops)),
                "flipped": float(np.mean(random_flips)),
            })

    sample_df = pd.DataFrame(rows)

    if sample_df.empty:
        summary_df = pd.DataFrame(columns=["fraction", "mode", "n_samples", "mean_score_drop", "flip_rate", "base_accuracy",])
        return sample_df, summary_df

    # AGG dataset level summary metrics
    summary_df = (
        sample_df.groupby(["fraction", "mode"], as_index=False)
        .agg(
            n_samples=("sample_idx", "count"),
            mean_score_drop=("score_drop", "mean"),
            flip_rate=("flipped", "mean"),
        )
    )
    summary_df["base_accuracy"] = base_acc

    return sample_df, summary_df