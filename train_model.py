"""
train_model.py — Train the entity matcher on the labeled feature table.

Uses XGBoost with an entity-grouped 80/20 split.
Auto-tunes classification threshold on the held-out validation set.

Usage:
    python train_model.py \\
        --features output/train_features.tsv \\
        --model-out output/matcher_model.joblib \\
        --val-out output/val_features.tsv
"""
import argparse
import gc
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier

from features import FEATURE_COLS


def split_by_entity(df, test_size=0.2, random_state=42):
    """80/20 split grouped by source1_entity_id — no leakage across splits."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["source1_entity_id"]
    train_idx, val_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    overlap  = set(train_df["source1_entity_id"]) & set(val_df["source1_entity_id"])
    assert not overlap, f"leakage: {len(overlap)} S1 entities in both splits"
    return train_df, val_df


def prepare_xy(df):
    X = df[FEATURE_COLS].astype(np.float32)
    y = df["label"].astype(int)
    return X, y


def train(df):
    X, y = prepare_xy(df)
    pos = int(y.sum())
    neg = len(y) - pos
    # Weight positives by imbalance ratio so the model doesn't ignore matches
    sample_weight = np.where(y == 1, neg / max(pos, 1), 1.0).astype(np.float32)
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def tune_threshold(model, val_df, beta=0.5):
    """
    Sweep thresholds 0.01–0.99 and return the one that maximises F-beta
    on the validation set.  beta=0.5 → F0.5 (precision-weighted).
    """
    X_val, y_val = prepare_xy(val_df)
    proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_thresh, best_f = 0.5, -1.0
    rows = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f  = fbeta_score(y_val, preds, beta=beta, zero_division=0)
        p  = precision_score(y_val, preds, zero_division=0)
        r  = recall_score(y_val, preds, zero_division=0)
        rows.append((t, f, p, r))
        if f > best_f:
            best_f, best_thresh = f, t

    # Print top 5 by F0.5 for inspection
    rows.sort(key=lambda x: -x[1])
    print("[train_model.py] Top thresholds by F0.5:")
    for t, f, p, r in rows[:5]:
        print(f"  thresh={t:.2f}  F0.5={f:.4f}  P={p:.4f}  R={r:.4f}")

    return float(best_thresh), float(best_f)


def main():
    # SageMaker supplies these paths for input channels and training outputs.
    # The fallbacks preserve the existing local project layout.
    output_dir = os.environ.get("SM_OUTPUT_DIR", "output")
    model_dir = os.environ.get("SM_MODEL_DIR", output_dir)
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", output_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=os.path.join(train_dir, "train_features.tsv"))
    parser.add_argument("--model-out", default=os.path.join(model_dir, "matcher_model.joblib"))
    parser.add_argument("--val-out", default=os.path.join(output_dir, "val_features.tsv"))
    parser.add_argument(
        "--threshold-out",
        default=os.path.join(output_dir, "best_threshold.txt"),
    )
    parser.add_argument("--test-size",     type=float, default=0.2)
    parser.add_argument("--random-state",  type=int,   default=42)
    args = parser.parse_args()

    print(f"[train_model.py] loading {args.features} ...")
    df = pd.read_csv(args.features, sep="\t", encoding="utf-8")
    required_cols = {"source1_entity_id", "candidate_entity_id", "label", *FEATURE_COLS}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(
            f"{args.features} is missing required columns: {missing_cols}. "
            "Regenerate it with features.py using the current feature definitions."
        )
    print(f"[train_model.py] {len(df):,} rows, {int(df['label'].sum()):,} positives")

    train_df, val_df = split_by_entity(df, test_size=args.test_size,
                                        random_state=args.random_state)
    print(f"[train_model.py] train: {len(train_df):,} rows "
          f"({int(train_df['label'].sum()):,} pos)")
    print(f"[train_model.py] val:   {len(val_df):,} rows "
          f"({int(val_df['label'].sum()):,} pos)")
    del df
    gc.collect()

    print("[train_model.py] training XGBoost ...")
    model = train(train_df)
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.val_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.threshold_out)), exist_ok=True)
    joblib.dump(model, args.model_out)
    val_df.to_csv(args.val_out, sep="\t", index=False)
    print(f"[train_model.py] model saved → {args.model_out}")
    print(f"[train_model.py] val set saved → {args.val_out}")

    best_thresh, best_f = tune_threshold(model, val_df, beta=0.5)
    print(f"[train_model.py] best val F0.5={best_f:.4f} at threshold={best_thresh:.2f}")
    print(f"[train_model.py] pass --threshold {best_thresh:.2f} to score_and_submit.py")

    # Save threshold to file for automatic pickup
    with open(args.threshold_out, "w", encoding="utf-8") as fh:
        fh.write(str(best_thresh))


if __name__ == "__main__":
    main()
