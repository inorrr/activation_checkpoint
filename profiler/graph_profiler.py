import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from profiler.memory_tracker import MemoryTracker
from profiler.op_profiler import OpRecord


def tensor_nbytes(x: Any) -> int:
    """
    Recursively estimate memory in bytes for tensors in nested structures.
    """
    if torch.is_tensor(x):
        return x.element_size() * x.nelement()
    if isinstance(x, (list, tuple)):
        return sum(tensor_nbytes(v) for v in x)
    if isinstance(x, dict):
        return sum(tensor_nbytes(v) for v in x.values())
    return 0


def categorize_tensor(name: str, op_type: str) -> str:
    """
    Coarse categorization for Phase 1.
    """
    lname = name.lower()
    if "weight" in lname or "bias" in lname:
        return "parameter"
    if op_type == "backward":
        return "gradient"
    if op_type == "optimizer":
        return "optimizer_state"
    return "activation"


class GraphProfiler:
    """
    Phase 1 profiler:
    - profiles module-level forward pass
    - profiles module-level backward pass
    - records optimizer step timing
    - tracks activation lifetimes (first/last use)
    - records memory snapshots
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.memory_tracker = MemoryTracker(device)

        self.forward_records: List[OpRecord] = []
        self.backward_records: List[OpRecord] = []
        self.optimizer_records: List[OpRecord] = []

        self.activation_info: Dict[str, Dict[str, Any]] = {}
        self.activation_counter = 0
        self.execution_index = 0

        self.handles = []
        self.forward_start_times: Dict[str, float] = {}
        self.backward_start_times: Dict[str, float] = {}

        self.module_execution_order: List[Tuple[int, str, str]] = []

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _now_ms(self) -> float:
        return time.perf_counter() * 1000.0

    def _register_activation(self, module_name: str, output: Any, module_type: str) -> None:
        if not torch.is_tensor(output):
            return

        act_name = f"{module_name}:act_{self.activation_counter}"
        self.activation_counter += 1

        self.activation_info[act_name] = {
            "module_name": module_name,
            "module_type": module_type,
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
            "size_bytes": tensor_nbytes(output),
            "first_use_index": self.execution_index,
            "last_use_index": self.execution_index,
        }

    def _update_activation_last_use(self, module_name: str) -> None:
        for act_name, info in self.activation_info.items():
            if info["module_name"] == module_name:
                info["last_use_index"] = self.execution_index

    def _forward_pre_hook(self, module_name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            self._sync()
            self.forward_start_times[module_name] = self._now_ms()
        return hook

    def _forward_hook(self, module_name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            self._sync()
            end_time = self._now_ms()
            start_time = self.forward_start_times.get(module_name, end_time)
            duration = end_time - start_time

            self.execution_index += 1
            self.module_execution_order.append(
                (self.execution_index, module_name, "forward")
            )

            mem = self.memory_tracker.snapshot()
            input_bytes = tensor_nbytes(inputs)
            output_bytes = tensor_nbytes(output)

            record = OpRecord(
                name=module_name,
                op_type="forward",
                module_type=module.__class__.__name__,
                start_time_ms=start_time,
                end_time_ms=end_time,
                duration_ms=duration,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                input_category="activation",
                output_category="activation",
                memory_allocated_mb=mem.allocated_mb,
                memory_reserved_mb=mem.reserved_mb,
                extra={
                    "output_shape": tuple(output.shape) if torch.is_tensor(output) else None,
                },
            )
            self.forward_records.append(record)

            self._register_activation(module_name, output, module.__class__.__name__)
            self._update_activation_last_use(module_name)

            # Register backward event hook on tensor outputs
            if torch.is_tensor(output) and output.requires_grad:
                backward_start_time = self._now_ms()

                def grad_hook(grad: torch.Tensor) -> torch.Tensor:
                    self._sync()
                    backward_end_time = self._now_ms()

                    self.execution_index += 1
                    self.module_execution_order.append(
                        (self.execution_index, module_name, "backward")
                    )

                    mem = self.memory_tracker.snapshot()

                    backward_record = OpRecord(
                        name=module_name,
                        op_type="backward",
                        module_type=module.__class__.__name__,
                        start_time_ms=backward_start_time,
                        end_time_ms=backward_end_time,
                        duration_ms=backward_end_time - backward_start_time,
                        input_bytes=tensor_nbytes(grad),
                        output_bytes=tensor_nbytes(grad),
                        input_category="gradient",
                        output_category="gradient",
                        memory_allocated_mb=mem.allocated_mb,
                        memory_reserved_mb=mem.reserved_mb,
                        extra={
                            "grad_shape": tuple(grad.shape)
                        },
                    )
                    self.backward_records.append(backward_record)
                    self._update_activation_last_use(module_name)
                    return grad

                output.register_hook(grad_hook)

        return hook

    def attach(self) -> None:
        """
        Register hooks on leaf modules only to avoid too much duplication.
        Use forward hooks only. Backward events are captured via tensor gradient hooks
        registered on forward outputs.
        """
        for module_name, module in self.model.named_modules():
            if module_name == "":
                continue
            if len(list(module.children())) > 0:
                continue

            self.handles.append(
                module.register_forward_pre_hook(self._forward_pre_hook(module_name))
            )
            self.handles.append(
                module.register_forward_hook(self._forward_hook(module_name))
            )

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def profile_one_iteration(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Dict[str, Any]:
        self.forward_records.clear()
        self.backward_records.clear()
        self.optimizer_records.clear()
        self.activation_info.clear()
        self.module_execution_order.clear()
        self.execution_index = 0
        self.activation_counter = 0

        self.memory_tracker.reset_peak_stats()
        optimizer.zero_grad(set_to_none=True)

        iteration_start = self._now_ms()

        # forward
        outputs = self.model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        # backward
        self._sync()
        backward_start = self._now_ms()
        loss.backward()
        self._sync()
        backward_end = self._now_ms()

        # optimizer step
        self._sync()
        opt_start = self._now_ms()
        optimizer.step()
        self._sync()
        opt_end = self._now_ms()

        iteration_end = self._now_ms()

        mem = self.memory_tracker.snapshot()

        self.optimizer_records.append(
            OpRecord(
                name="optimizer_step",
                op_type="optimizer",
                module_type=optimizer.__class__.__name__,
                start_time_ms=opt_start,
                end_time_ms=opt_end,
                duration_ms=opt_end - opt_start,
                input_bytes=0,
                output_bytes=0,
                input_category="gradient",
                output_category="optimizer_state",
                memory_allocated_mb=mem.allocated_mb,
                memory_reserved_mb=mem.reserved_mb,
                extra=None,
            )
        )

        return {
            "loss": float(loss.item()),
            "iteration_time_ms": iteration_end - iteration_start,
            "backward_only_time_ms": backward_end - backward_start,
            "peak_memory_allocated_mb": mem.max_allocated_mb,
            "peak_memory_reserved_mb": mem.max_reserved_mb,
            "forward_ops": [r.to_dict() for r in self.forward_records],
            "backward_ops": [r.to_dict() for r in self.backward_records],
            "optimizer_ops": [r.to_dict() for r in self.optimizer_records],
            "activations": self.activation_info,
            "execution_order": self.module_execution_order,
        }

    def save_trace(self, trace: Dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)