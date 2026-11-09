"""Train-only feature scaling, validation-selected probes, trajectory-level reporting."""
import numpy as np
import torch


def standardize(train, others):
    mean, std = train.mean(0), np.maximum(train.std(0), 1e-5)
    return [(x-mean)/std for x in [train, *others]]


def design_features(features, nonlinear, seed):
    values = standardize(features[0], features[1:])
    if nonlinear:
        rng = np.random.default_rng(seed)
        w = rng.normal(size=(values[0].shape[1], 64))/np.sqrt(values[0].shape[1])
        bias = rng.normal(size=64)
        values = [np.concatenate([x, np.tanh(x @ w+bias)], -1) for x in values]
    return [np.column_stack([np.ones(len(x)), x]) for x in values]


def fit_ridge(x, y, alpha):
    penalty = np.eye(x.shape[1])*alpha
    penalty[0, 0] = 0
    return np.linalg.lstsq(x.T @ x/len(x)+penalty, x.T @ y/len(x), rcond=None)[0]


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -35, 35)))


def fit_logistic(x, y, alpha):
    """Binary logistic regression with damped Newton updates and L2 penalty."""
    coef = np.zeros((x.shape[1], 1))
    penalty = np.eye(x.shape[1])*alpha
    penalty[0, 0] = 0
    for _ in range(40):
        p = sigmoid(x @ coef)
        gradient = x.T @ (p-y)/len(x)+penalty @ coef
        weight = (p*(1-p))[:, 0]
        hessian = x.T @ (weight[:, None]*x)/len(x)+penalty+1e-7*np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        # Bound changes in logits for nearly separable labels.
        step /= max(1, float(np.max(np.abs(x @ step)))/5)
        coef -= step
        if np.linalg.norm(step) < 1e-6:
            break
    return coef


def regression_metrics(y, prediction):
    mse = np.mean((y-prediction)**2, axis=0)
    variance = np.mean((y-y.mean(0))**2, axis=0)
    valid = variance > 1e-10
    return {"mse": float(mse.mean()),
            "r2": float(np.mean(1-mse[valid]/variance[valid])) if valid.any() else None}


def classification_metrics(y, prediction):
    truth, predicted = y[:, 0].astype(int), (prediction[:, 0] >= 0.5).astype(int)
    recalls = [float(np.mean(predicted[truth == k] == k)) for k in (0, 1) if (truth == k).any()]
    return {"accuracy": float(np.mean(predicted == truth)), "balanced_accuracy": float(np.mean(recalls))}


@torch.no_grad()
def extract(model, dataset, batch_size, device):
    model.eval()
    local, sequence = [], []
    for start in range(0, len(dataset), batch_size):
        x = torch.stack([dataset[i]["x"] for i in range(start, min(start+batch_size, len(dataset)))]).to(device)
        rep = model.representations(x)
        local.append(rep["local"].cpu().numpy())
        sequence.append(rep["sequence"].cpu().numpy())
    return {"local": np.concatenate(local), "sequence": np.concatenate(sequence)}


def evaluate_probes(model, splits, cfg, device):
    names = ["train", "validation", "test"]
    representations = {k: extract(model, splits[k], cfg.get("batch_size", 8), device) for k in names}
    return evaluate_representations(representations, splits, cfg, model.patch_size)


def observation_representations(dataset, patch_size, context, statistics=False):
    """Observation-only controls. Statistics use past windows locally and the full sequence globally."""
    x = np.stack([r["x"].numpy() for r in dataset])
    patches = x.reshape(len(x), x.shape[1]//patch_size, -1)
    if not statistics:
        return {"local": patches, "sequence": patches.mean(1)}

    def summary(window):
        centered = window-window.mean(1, keepdims=True)
        features = [window.mean(1), window.std(1)]
        variance = np.mean(centered**2, axis=1)+1e-6
        for lag in (1, 2, 4, 8):
            # A missing lag is zero, never filled from future samples.
            covariance = (centered[:, lag:]*centered[:, :-lag]).mean(1) if window.shape[1] > lag else np.zeros_like(variance)
            features.append(covariance/variance)
        return np.concatenate(features, -1)

    local = []
    for j in range(patches.shape[1]):
        window = x[:, max(0, j-context+1)*patch_size:(j+1)*patch_size]
        local.append(np.concatenate([patches[:, j], summary(window)], -1))
    return {"local": np.stack(local, 1), "sequence": summary(x)}


def evaluate_representations(representations, splits, cfg, patch_size):
    """Evaluate any fixed feature source with identical train/validation/test probe rules."""
    names = ["train", "validation", "test"]
    first = splits["train"].records[0]
    tasks = []
    for key in ("u", "phase", "q", "state"):
        if key in (first.get("latents") or {}):
            tasks.append(("local", key, key == "q"))
    for key in (first.get("parameters") or {}):
        # Constant parameters cannot have a meaningful R2 and are not useful probe targets.
        if np.std([r["parameters"][key] for r in splits["train"].records]) > 1e-8:
            tasks.append(("sequence", key, False))
    results = {}
    for level, key, classification in tasks:
        features, labels, sizes = [], [], []
        for name in names:
            rep = representations[name][level]
            if level == "local":
                # Complete patch is available; target state is the LAST sample of that patch.
                y = np.stack([r["latents"][key][patch_size-1::patch_size]
                              for r in splits[name].records])
                sizes.append(rep.shape[1])
                rep, y = rep.reshape(-1, rep.shape[-1]), y.reshape(-1, y.shape[-1])
            else:
                y = np.array([[r["parameters"][key]] for r in splits[name].records])
                sizes.append(1)
            features.append(rep.astype(np.float64))
            labels.append(y.astype(np.float64))
        for nonlinear in ([False, True] if cfg.get("nonlinear", False) else [False]):
            xtrain, xval, xtest = design_features(features, nonlinear, cfg.get("seed", 0))
            best = None
            # Regression target scaling uses train data only, preserving comparable penalties.
            mean = labels[0].mean(0) if not classification else 0
            std = np.maximum(labels[0].std(0), 1e-6) if not classification else 1
            ytrain = (labels[0]-mean)/std
            for alpha in cfg.get("alphas", [0.001, 0.1, 10.0]):
                if alpha <= 0:
                    raise ValueError("Probe regularization must be positive")
                coef = (fit_logistic if classification else fit_ridge)(xtrain, ytrain, alpha)
                val = sigmoid(xval @ coef) if classification else (xval @ coef)*std+mean
                score = (-(classification_metrics(labels[1], val)["balanced_accuracy"]) if classification
                         else float(np.mean(((val-labels[1])/std)**2)))
                if best is None or score < best[0]:
                    best = (score, alpha, coef)
            pred = sigmoid(xtest @ best[2]) if classification else (xtest @ best[2])*std+mean
            metric = classification_metrics if classification else regression_metrics
            per_trajectory = [metric(labels[2][i:i+sizes[2]], pred[i:i+sizes[2]])
                              for i in range(0, len(pred), sizes[2])] if level == "local" else []
            aggregate = {}
            for name in metric(labels[2], pred):
                values = [r[name] for r in per_trajectory if r[name] is not None]
                aggregate[name] = {"mean": float(np.mean(values)),
                                   "sem": float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None,
                                   "n_trajectories": len(values)} if values else None
            results[f"{level}/{key}/{'nonlinear' if nonlinear else 'linear'}"] = {
                "alpha": best[1], "validation_selection_score": best[0],
                "test_pooled": metric(labels[2], pred), "test_trajectory_aggregate": aggregate,
                "test_per_trajectory": per_trajectory,
                "test_trajectories": len(splits["test"]),
                "test_predictions": pred.tolist(), "test_targets": labels[2].tolist()}
    return results
