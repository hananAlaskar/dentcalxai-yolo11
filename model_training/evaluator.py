
"""Validation helper that iterates over all runs in the `RUNS_DIR`
and saves a CSV summary with mAP metrics.
"""
import os
import yaml
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from config import RUNS_DIR, DATA_YAML, DEVICE

def evaluate_all_runs(
    runs_dir: Path = RUNS_DIR,
    output_csv: str = "validation_summary.csv",
    imgsz: int = 800,
    batch_size: int = 16,
    device: str = DEVICE,
) -> pd.DataFrame:
    """Evaluate every trained run inside *runs_dir* and save results.

    Returns
    -------
    pandas.DataFrame
    """
    runs = [d for d in os.listdir(runs_dir) if (runs_dir / d).is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found in {runs_dir}")

    records = []
    for run in runs:
        run_dir   = runs_dir / run
        model_pt  = run_dir / "weights" / "best.pt"
        args_yaml = run_dir / "args.yaml"
        if not model_pt.exists():
            print(f"[evaluator] Skipping {run}: no best.pt")
            continue

        print(f"[evaluator] Evaluating {run} …")
        model   = YOLO(str(model_pt))
        results = model.val(
            data=str(DATA_YAML),
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            save_json=False
        )
        precision, recall, mAP50, mAP50_95 = results.mean_results()

        # Metadata ----------
        epochs = None
        base_model = None
        if args_yaml.exists():
            with open(args_yaml, "r") as f:
                args = yaml.safe_load(f) or {}
            epochs     = args.get("epochs")
            base_model = os.path.basename(args.get("model", "")).replace(".pt", "")

        records.append({
            "run":        run,
            "precision":  precision,
            "recall":     recall,
            "mAP50":      mAP50,
            "mAP50_95":   mAP50_95,
            "epochs":     epochs,
            "base_model": base_model,
        })

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    print(f"[evaluator] Saved → {output_csv}")
    return df
