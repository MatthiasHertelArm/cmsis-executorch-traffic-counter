# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""The methods create_ai_layer.py exports into the ExecuTorch program.

create_ai_layer.py calls get_methods() and turns every MethodSpec into one
method of the .pte: it exports the module, quantizes it with the samples as
calibration data (the first sample is also the example input), delegates it
to the Ethos-U and compiles it with Vela. The model of this repository is
the traffic counter's YOLO26n vehicle detector (traffic.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class MethodSpec:
    """One ExecuTorch method: the module, its calibration/example inputs and the activation width."""

    name: str
    module: nn.Module
    samples: list[tuple[torch.Tensor, ...]] = field(default_factory=list)
    activation_bits: int = 8  # 8 or 16 (int16 activations, int8 weights)

    @property
    def example(self) -> tuple[torch.Tensor, ...]:
        return self.samples[0]


def get_methods() -> list[MethodSpec]:
    torch.manual_seed(0)
    # YOLO26n with the vehicle classes, for the Ethos-U55: the single method
    # `detect`, or stem, mid and head around the attention on the CPU.
    from traffic import get_traffic_methods

    return get_traffic_methods()
