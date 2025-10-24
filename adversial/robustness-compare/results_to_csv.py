#!/usr/bin/env python3
import json
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

def flatten_results(results: dict):
    rows = []
    for model_name, payload in results.items():
        corruptions = payload.get("corruptions", {})
        summary = payload.get("summary", {})
        for cname, sev_list in corruptions.items():
            # sev_list is a list of per-severity errors (1 - top1)
            for idx, err in enumerate(sev_list, start=1):
                rows.append({
                    "model": model_name,
                    "corruption": cname,
                    "severity": idx,
                    "top1_error": err,
                    "top1_accuracy": 1.0 - err,
                })
        yield model_name, summary, rows


def write_flat_csv(rows, out_path: Path):
    fieldnames = ["model", "corruption", "severity", "top1_error", "top1_accuracy"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_csv(summaries: list[dict], out_path: Path):
    fieldnames = ["model", "clean_top1", "clean_top5", "avg_corruption_top1_error", "avg_corruption_top1_accuracy"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            w.writerow(s)


def main():
    results_path = HERE / "results.json"
    if not results_path.exists():
        raise SystemExit(f"results not found: {results_path}")
    data = json.loads(results_path.read_text())

    flat_rows = []
    summary_rows = []
    for model_name, payload in data.items():
        # flatten corruption rows
        corruptions = payload.get("corruptions", {})
        for cname, sev_list in corruptions.items():
            for idx, err in enumerate(sev_list, start=1):
                flat_rows.append({
                    "model": model_name,
                    "corruption": cname,
                    "severity": idx,
                    "top1_error": err,
                    "top1_accuracy": 1.0 - err,
                })
        # collect summary row
        s = payload.get("summary", {})
        summary_rows.append({
            "model": model_name,
            "clean_top1": s.get("clean_top1"),
            "clean_top5": s.get("clean_top5"),
            "avg_corruption_top1_error": s.get("avg_corruption_top1_error"),
            "avg_corruption_top1_accuracy": s.get("avg_corruption_top1_accuracy"),
        })

    write_flat_csv(flat_rows, HERE / "results_flat.csv")
    write_summary_csv(summary_rows, HERE / "summary.csv")
    print("wrote", HERE / "results_flat.csv")
    print("wrote", HERE / "summary.csv")


if __name__ == "__main__":
    main()
