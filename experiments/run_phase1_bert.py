import csv
import os
from typing import List, Dict, Any

from training.train import run_phase1_bert


def compute_peak_live_activation_memory_mb(activations: dict) -> float:
    """
    Sweep-line computation of peak live activation memory.
    """
    if not activations:
        return 0.0

    events = []
    for act in activations.values():
        start = act["first_use_index"]
        end = act["last_use_index"]
        size = act["size_bytes"]

        events.append((start, size))
        events.append((end + 1, -size))

    events.sort()

    current_bytes = 0
    peak_bytes = 0
    for _, delta in events:
        current_bytes += delta
        peak_bytes = max(peak_bytes, current_bytes)

    return peak_bytes / (1024 ** 2)


def summarize_trace(batch_size: int, seq_len: int, trace: Dict[str, Any]) -> Dict[str, Any]:
    total_activation_bytes = sum(
        act["size_bytes"] for act in trace["activations"].values()
    )

    peak_live_activation_mb = compute_peak_live_activation_memory_mb(trace["activations"])

    avg_forward_ms = (
        sum(op["duration_ms"] for op in trace["forward_ops"]) / len(trace["forward_ops"])
        if trace["forward_ops"] else 0.0
    )
    avg_backward_ms = (
        sum(op["duration_ms"] for op in trace["backward_ops"]) / len(trace["backward_ops"])
        if trace["backward_ops"] else 0.0
    )

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "loss": trace["loss"],
        "iteration_time_ms": trace["iteration_time_ms"],
        "backward_only_time_ms": trace["backward_only_time_ms"],
        "peak_memory_allocated_mb": trace["peak_memory_allocated_mb"],
        "peak_memory_reserved_mb": trace["peak_memory_reserved_mb"],
        "num_forward_ops": len(trace["forward_ops"]),
        "num_backward_ops": len(trace["backward_ops"]),
        "num_activations": len(trace["activations"]),
        "total_activation_mb": total_activation_bytes / (1024 ** 2),
        "peak_live_activation_mb": peak_live_activation_mb,
        "parameter_memory_mb": trace["parameter_memory_mb"],
        "gradient_memory_mb": trace["gradient_memory_mb"],
        "avg_forward_op_ms": avg_forward_ms,
        "avg_backward_op_ms": avg_backward_ms,
    }


def save_summary_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    batch_sizes = [1, 2, 4, 6, 8, 12, 16]
    seq_len = 128
    summary_rows = []

    for bs in batch_sizes:
        print("=" * 80)
        print(f"Running BERT Phase 1 profiling for batch size = {bs}, seq_len = {seq_len}")

        try:
            trace = run_phase1_bert(
                batch_size=bs,
                seq_len=seq_len,
                save_dir="results/logs",
                save_trace=True,
            )
            summary = summarize_trace(bs, seq_len, trace)
            summary_rows.append(summary)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch size {bs}, seq_len {seq_len}. Skipping.")
                continue
            raise

    if not summary_rows:
        raise RuntimeError("No BERT runs completed successfully.")

    summary_path = "results/tables/bert_phase1_summary.csv"
    save_summary_csv(summary_rows, summary_path)

    print("=" * 80)
    print(f"Saved summary table to: {summary_path}")


if __name__ == "__main__":
    main()