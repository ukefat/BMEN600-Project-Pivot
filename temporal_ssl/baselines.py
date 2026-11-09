"""Observation-only baselines and transition-conditioned reconstruction metrics."""
import numpy as np


def fill_masked(patches, mask, method="linear"):
    """Interpolate in sample time between visible samples; one-sided constant at edges."""
    if method not in {"linear", "nearest"}:
        raise ValueError(method)
    n, p, c = patches.shape
    x = patches.reshape(n*p, c)
    hidden = np.repeat(mask, p)
    visible = np.flatnonzero(~hidden)
    if not len(visible):
        raise ValueError("At least one visible patch is required")
    times = np.arange(n*p)
    if method == "linear":
        result = np.column_stack([np.interp(times, visible, x[visible, j]) for j in range(c)])
    else:
        right = np.searchsorted(visible, times).clip(0, len(visible)-1)
        left = (right-1).clip(0)
        nearest = np.where(abs(visible[left]-times) <= abs(visible[right]-times), left, right)
        result = x[visible[nearest]]
    return result.reshape(n, p, c)


def causal_baselines(patches, horizon, context):
    """Predict each future patch using only the same preceding window as the model."""
    persistence, linear = [], []
    p = patches.shape[1]
    for endpoint in range(len(patches)-horizon):
        history = patches[max(0, endpoint-context+1):endpoint+1].reshape(-1, patches.shape[-1])
        persistence.append(np.repeat(history[-1:], p, axis=0))
        # Short-context linear trend, fit independently per channel; no target values used.
        t = np.arange(len(history), dtype=float)
        design = np.column_stack([np.ones(len(t)), t-t[-1]])
        coef = np.linalg.lstsq(design, history, rcond=None)[0]
        future_t = (horizon-1)*p + np.arange(1, p+1)
        linear.append(np.column_stack([np.ones(p), future_t]) @ coef)
    return {"constant": np.array(persistence), "linear_trend": np.array(linear)}


def near_transitions(latents, length, radius):
    near = np.zeros(length, dtype=bool)
    if latents is not None and "transition" in latents:
        for index in np.flatnonzero(latents["transition"][:, 0]):
            near[max(0, index-radius):min(length, index+radius+1)] = True
    return near


def error_summary(error, selected, near):
    """Inputs are per-sample errors/masks. Counts refer to time samples, not replicates."""
    result = {}
    for name, subset in {"all": selected, "near_transition": selected & near,
                         "stable": selected & ~near}.items():
        result[name] = float(error[subset].mean()) if subset.any() else None
        result[f"{name}_samples"] = int(subset.sum())
    return result
