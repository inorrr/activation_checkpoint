import os
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet import load_resnet152
from models.bert import load_bert_classifier
from profiler.graph_profiler import GraphProfiler


def get_parameter_memory_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.element_size() * p.nelement()
    return total_bytes / (1024 ** 2)


def get_gradient_memory_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        if p.grad is not None:
            total_bytes += p.grad.element_size() * p.grad.nelement()
    return total_bytes / (1024 ** 2)


def make_dummy_resnet_batch(batch_size: int, num_classes: int, device: torch.device):
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def make_dummy_bert_batch(
    batch_size: int,
    seq_len: int,
    num_labels: int,
    device: torch.device,
    vocab_size: int = 30522,
):
    """
    Create synthetic BERT inputs without using a tokenizer.

    input_ids: random token IDs
    attention_mask: all ones
    token_type_ids: all zeros
    labels: random class labels
    """
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones(
        (batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    token_type_ids = torch.zeros(
        (batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    labels = torch.randint(
        low=0,
        high=num_labels,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    batch_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    return batch_inputs, labels


def run_phase1_resnet(
    batch_size: int = 4,
    num_classes: int = 1000,
    save_dir: str = "results/logs",
    save_trace: bool = True,
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_resnet152(num_classes=num_classes).to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    profiler = GraphProfiler(model, device)
    profiler.attach()

    x, y = make_dummy_resnet_batch(batch_size, num_classes, device)
    trace = profiler.profile_one_iteration(
        batch_inputs=x,
        batch_targets=y,
        optimizer=optimizer,
        criterion=criterion,
    )

    parameter_memory_mb = get_parameter_memory_mb(model)
    gradient_memory_mb = get_gradient_memory_mb(model)

    trace["parameter_memory_mb"] = parameter_memory_mb
    trace["gradient_memory_mb"] = gradient_memory_mb
    trace["model_name"] = "resnet152"
    trace["batch_size"] = batch_size

    profiler.detach()

    if save_trace:
        out_path = os.path.join(save_dir, f"resnet152_bs{batch_size}_phase1_trace.json")
        profiler.save_trace(trace, out_path)
        print(f"Saved trace to: {out_path}")

    print(f"Loss: {trace['loss']:.4f}")
    print(f"Iteration time (ms): {trace['iteration_time_ms']:.2f}")
    print(f"Peak allocated memory (MB): {trace['peak_memory_allocated_mb']:.2f}")
    print(f"Peak reserved memory (MB): {trace['peak_memory_reserved_mb']:.2f}")
    print(f"Forward ops recorded: {len(trace['forward_ops'])}")
    print(f"Backward ops recorded: {len(trace['backward_ops'])}")
    print(f"Tracked activations: {len(trace['activations'])}")

    return trace


def run_phase1_bert(
    batch_size: int = 2,
    seq_len: int = 128,
    num_labels: int = 2,
    save_dir: str = "results/logs",
    save_trace: bool = True,
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_bert_classifier(num_labels=num_labels).to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    profiler = GraphProfiler(model, device)
    profiler.attach()

    batch_inputs, labels = make_dummy_bert_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        num_labels=num_labels,
        device=device,
    )

    trace = profiler.profile_one_iteration(
        batch_inputs=batch_inputs,
        batch_targets=labels,
        optimizer=optimizer,
        criterion=criterion,
    )

    parameter_memory_mb = get_parameter_memory_mb(model)
    gradient_memory_mb = get_gradient_memory_mb(model)

    trace["parameter_memory_mb"] = parameter_memory_mb
    trace["gradient_memory_mb"] = gradient_memory_mb
    trace["model_name"] = "bert_base_sequence_classification"
    trace["batch_size"] = batch_size
    trace["seq_len"] = seq_len

    profiler.detach()

    if save_trace:
        out_path = os.path.join(
            save_dir,
            f"bert_bs{batch_size}_seq{seq_len}_phase1_trace.json",
        )
        profiler.save_trace(trace, out_path)
        print(f"Saved trace to: {out_path}")

    print(f"Loss: {trace['loss']:.4f}")
    print(f"Iteration time (ms): {trace['iteration_time_ms']:.2f}")
    print(f"Peak allocated memory (MB): {trace['peak_memory_allocated_mb']:.2f}")
    print(f"Peak reserved memory (MB): {trace['peak_memory_reserved_mb']:.2f}")
    print(f"Forward ops recorded: {len(trace['forward_ops'])}")
    print(f"Backward ops recorded: {len(trace['backward_ops'])}")
    print(f"Tracked activations: {len(trace['activations'])}")

    return trace


if __name__ == "__main__":
    run_phase1_resnet(batch_size=4)