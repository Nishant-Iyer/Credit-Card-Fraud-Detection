"""
Train the pipeline, then export everything the in-browser demo needs as plain JSON:

  demo/fraud-model.json   scaler, autoencoder weights, XGBoost + LightGBM trees, meta-model
  demo/fraud-data.json    sample test transactions, cost table over thresholds, headline metrics

Before writing, the exported model is re-scored with a small NumPy implementation (the same
logic the browser runs) and compared with the trained pipeline. The job fails if they disagree.

Run:  python -m scripts.export_demo
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import AppConfig  # noqa: E402
from src.pipeline.train_pipeline import run_train_pipeline  # noqa: E402

CONFIG = "config/config.yaml"
OUT_DIR = "demo"
F32 = np.float32


# ----------------------------------------------------------------------------- export
def export_xgb(clf, feat_names):
    booster = clf.get_booster()
    names = booster.feature_names or feat_names
    index = {n: i for i, n in enumerate(names)}
    trees = []
    for raw in booster.get_dump(dump_format="json"):
        node = json.loads(raw)
        flat = {"f": [], "t": [], "l": [], "r": [], "m": [], "v": []}

        def walk(n):
            i = len(flat["f"])
            for k in flat:
                flat[k].append(0)
            if "leaf" in n:
                flat["f"][i] = -1
                flat["v"][i] = float(n["leaf"])
                return i
            split = n["split"]
            flat["f"][i] = index[split] if split in index else int(str(split).lstrip("f"))
            flat["t"][i] = float(n["split_condition"])
            kids = {c["nodeid"]: c for c in n["children"]}
            left = walk(kids[n["yes"]])
            right = walk(kids[n["no"]])
            flat["l"][i], flat["r"][i] = left, right
            flat["m"][i] = 1 if n.get("missing", n["yes"]) == n["yes"] else 0
            return i

        walk(node)
        trees.append(flat)
    return trees


def export_lgb(clf):
    dump = clf.booster_.dump_model()
    trees = []
    for info in dump["tree_info"]:
        flat = {"f": [], "t": [], "l": [], "r": [], "m": [], "v": []}

        def walk(n):
            i = len(flat["f"])
            for k in flat:
                flat[k].append(0)
            if "leaf_value" in n:
                flat["f"][i] = -1
                flat["v"][i] = float(n["leaf_value"])
                return i
            assert n["decision_type"] == "<=", n["decision_type"]
            flat["f"][i] = int(n["split_feature"])
            flat["t"][i] = float(n["threshold"])
            left = walk(n["left_child"])
            right = walk(n["right_child"])
            flat["l"][i], flat["r"][i] = left, right
            flat["m"][i] = 1 if n.get("default_left", True) else 0
            return i

        walk(info["tree_structure"])
        trees.append(flat)
    return trees


# ------------------------------------------------------------------ reference scorer
def tree_sum(trees, X, strict):
    # XGBoost compares in float32 and goes left on x < split; LightGBM compares doubles, x <= split.
    dtype = F32 if strict else np.float64
    out = np.zeros(len(X), dtype=np.float64)
    Xf = X.astype(dtype)
    for t in trees:
        f = np.array(t["f"])
        thr = np.array(t["t"], dtype=dtype)
        left = np.array(t["l"])
        right = np.array(t["r"])
        val = np.array(t["v"])
        node = np.zeros(len(X), dtype=np.int64)
        for _ in range(64):
            fi = f[node]
            active = fi >= 0
            if not active.any():
                break
            x = Xf[np.arange(len(X)), np.where(active, fi, 0)]
            go_left = (x < thr[node]) if strict else (x <= thr[node])
            nxt = np.where(go_left, left[node], right[node])
            node = np.where(active, nxt, node)
        out += val[node]
    return out


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def reference_score(model, X_raw):
    """X_raw: V1..V28, Amount (in model['input'] order). Returns dict of intermediate scores."""
    X = X_raw.astype(np.float64).copy()
    a = model["input"].index("Amount")
    X[:, a] = (X[:, a] - model["scaler"]["center"]) / model["scaler"]["scale"]
    h = X
    for layer in model["autoencoder"]:
        W = np.array(layer["w"])
        b = np.array(layer["b"])
        h = h @ W.T + b
        if layer["relu"]:
            h = np.maximum(h, 0)
    err = ((X - h) ** 2).mean(axis=1)
    Xa = np.column_stack([X, err])
    p_x = sigmoid(tree_sum(model["xgb"]["trees"], Xa, strict=True) + model["xgb"]["offset"])
    p_l = sigmoid(tree_sum(model["lgb"]["trees"], Xa, strict=False) + model["lgb"]["offset"])
    meta = model["meta"]
    p = sigmoid(meta["coef"][0] * p_x + meta["coef"][1] * p_l + meta["intercept"])
    return {"recon": err, "p_xgb": p_x, "p_lgb": p_l, "p": p}


# ---------------------------------------------------------------------------- main
def main():
    config = AppConfig.load_from_yaml(CONFIG)
    run_train_pipeline(CONFIG, use_mlflow=False)
    pipe = joblib.load("artifacts/full_pipeline.joblib")
    with open("artifacts/metadata.json") as fh:
        meta_info = json.load(fh)

    test = pd.read_csv(config.data.test_file)
    y = test[config.data.target_col].values
    X_in = test.drop(columns=[config.data.target_col])

    # Features exactly as the classifier sees them
    steps = pipe.named_steps
    Z = X_in
    for name in ("col_dropper", "time_encoder", "scaler", "autoencoder"):
        Z = steps[name].transform(Z)
    feat_names = list(Z.columns)
    input_cols = feat_names[:-1]
    assert feat_names[-1] == "reconstruction_error", feat_names

    stack = steps["classifier"].stacking_clf
    xgb_clf, lgb_clf = stack.estimators_
    lr = stack.final_estimator_

    scaler = steps["scaler"].scaler
    a_idx = steps["scaler"].scale_cols.index("Amount")

    ae = steps["autoencoder"].model
    sd = {k: v.detach().cpu().numpy().astype(float) for k, v in ae.state_dict().items()}
    ae_layers = [
        {"w": sd["encoder.0.weight"].tolist(), "b": sd["encoder.0.bias"].tolist(), "relu": True},
        {"w": sd["encoder.2.weight"].tolist(), "b": sd["encoder.2.bias"].tolist(), "relu": True},
        {"w": sd["decoder.0.weight"].tolist(), "b": sd["decoder.0.bias"].tolist(), "relu": True},
        {"w": sd["decoder.2.weight"].tolist(), "b": sd["decoder.2.bias"].tolist(), "relu": False},
    ]

    xgb_trees = export_xgb(xgb_clf, feat_names)
    lgb_trees = export_lgb(lgb_clf)

    # Offsets (base score / init score) calibrated against each library's own margin output
    Za = Z.values
    xgb_margin = xgb_clf.predict(Z, output_margin=True)
    lgb_raw = lgb_clf.predict(Z, raw_score=True)
    xgb_off = float(np.median(xgb_margin - tree_sum(xgb_trees, Za, strict=True)))
    lgb_off = float(np.median(lgb_raw - tree_sum(lgb_trees, Za, strict=False)))

    model = {
        "input": input_cols,
        "features": feat_names,
        "scaler": {"center": float(scaler.center_[a_idx]), "scale": float(scaler.scale_[a_idx])},
        "autoencoder": ae_layers,
        "xgb": {"offset": xgb_off, "trees": xgb_trees},
        "lgb": {"offset": lgb_off, "trees": lgb_trees},
        "meta": {"coef": lr.coef_[0].astype(float).tolist(), "intercept": float(lr.intercept_[0])},
        "threshold": float(meta_info["optimal_threshold"]),
    }

    # ---- parity check against the real pipeline on the whole test set
    X_raw = X_in[input_cols].values
    ref = reference_score(model, X_raw)
    true_p = pipe.predict_proba(X_in)[:, 1]
    diff = np.abs(ref["p"] - true_p)
    flips = int(((ref["p"] >= model["threshold"]) != (true_p >= model["threshold"])).sum())
    print(f"parity: max |dp| = {diff.max():.2e}, 99.9th pct = {np.quantile(diff, 0.999):.2e}, "
          f"decision flips = {flips}")
    if np.quantile(diff, 0.999) > 1e-4 or diff.max() > 2e-2 or flips > 2:
        raise SystemExit("Exported model does not match the trained pipeline")

    # ---- cost table over thresholds (test set), independent of the per-review cost
    amounts = X_in["Amount"].values
    grid = np.unique(np.concatenate([
        np.round(np.geomspace(1e-4, 0.999, 260), 6),
        [round(model["threshold"], 6)],
    ]))
    rows = []
    for t in grid:
        flag = true_p >= t
        rows.append([
            float(t), int(flag.sum()), int((flag & (y == 1)).sum()),
            round(float(amounts[(~flag) & (y == 1)].sum()), 2),
        ])

    # ---- sample transactions: every test fraud plus a random set of legitimate ones
    rng = np.random.default_rng(7)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = rng.choice(np.where(y == 0)[0], size=300, replace=False)
    pick = np.concatenate([fraud_idx, legit_idx])
    sample = [
        {"x": [round(float(v), 5) for v in X_raw[i]], "y": int(y[i]), "p": round(float(true_p[i]), 6)}
        for i in pick
    ]

    m = meta_info["metrics"]
    b = meta_info["business_metrics"]
    data = {
        "input": input_cols,
        "threshold": model["threshold"],
        "review_cost": config.business_costs.manual_review_cost,
        "fraud_cost_factor": config.business_costs.fraud_cost_factor,
        "test": {
            "n": int(len(y)), "frauds": int(y.sum()),
            "fraud_amount": round(float(amounts[y == 1].sum()), 2),
            "auprc": m["auprc"], "roc_auc": m["roc_auc"], "precision": m["precision"],
            "recall": m["recall"], "confusion": m["confusion_matrix"],
            "cost": b["min_business_cost"], "do_nothing": b["do_nothing_cost"],
        },
        "sizes": meta_info.get("sizes"),
        "cost_table": {"cols": ["threshold", "flagged", "caught", "missed_amount"], "rows": rows},
        "sample": sample,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/fraud-model.json", "w") as fh:
        json.dump(model, fh, separators=(",", ":"))
    with open(f"{OUT_DIR}/fraud-data.json", "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    with open(f"{OUT_DIR}/parity.txt", "w") as fh:
        fh.write(f"max_abs_diff={diff.max():.3e}\nq999={np.quantile(diff, 0.999):.3e}\nflips={flips}\n")
    print(json.dumps(data["test"], indent=2))


if __name__ == "__main__":
    main()
