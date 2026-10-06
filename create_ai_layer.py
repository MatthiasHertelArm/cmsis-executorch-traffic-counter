#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Create the AI layer of the CMSIS solution from its MLOps information.

    python create_ai_layer.py <solution>.cbuild-mlops.yml

Step 2 of the three-step flow:

    cbuild setup <solution>.csolution.yml --active <target>   # writes *.cbuild-mlops.yml
    python create_ai_layer.py <solution>.cbuild-mlops.yml     # this script
    cbuild <solution>.csolution.yml --active <target>         # compile and link

The *.cbuild-mlops.yml is what CMSIS-Toolbox generates from the `mlops:` node
of the csolution. This script reads the NPU and Vela settings from it, exports
every method that model/model.py declares (quantize, delegate to Ethos-U,
compile with Vela) into one ExecuTorch program and writes the complete AI
layer into the directory of the clayer named under `model.clayer`:

    ai_layer.clayer.yml   runtime, kernel utilities and registration, Ethos-U
                          backend and the operator components the exported
                          program actually uses
    model_pte.c / .h      the ExecuTorch program as a C array
    model_io.h            per method: input/output shapes, dtypes and the
                          quantization scale / zero point of each tensor
    model.pte             the program itself, for inspection

The methods take and return quantized tensors (int8 or int16): the float
quantize / dequantize boundary that the Arm backend normally leaves on the CPU
is removed with ExecuTorch's quantize-IO passes, and the parameters the CPU
needs to produce and consume those tensors go into model_io.h.

The script runs itself in the solution's .venv (see setup_venv.py) when it is
started with an interpreter that has no torch.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = "PyTorch::ExecuTorch"
SYMBOL = "model_pte"


def run_in_venv() -> None:
    """Re-run under .venv when torch is not importable from this interpreter."""
    try:
        import torch  # noqa: F401
    except ImportError:
        venv = HERE / ".venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # sys.prefix is the venv directory when running inside it (comparing
        # interpreter paths does not work: venv symlinks resolve to the base).
        if not python.is_file() or Path(sys.prefix).resolve() == venv.resolve():
            sys.exit(
                f"torch is not installed for {sys.executable}.\n"
                "Create the venv first: ./setup_venv.sh (Linux/macOS) or setup_venv.bat (Windows)"
            )
        sys.exit(subprocess.run([str(python), __file__, *sys.argv[1:]]).returncode)


def pack_root() -> Path:
    """The CMSIS pack root: $CMSIS_PACK_ROOT, else cpackget's default."""
    if env := os.environ.get("CMSIS_PACK_ROOT"):
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Arm/Packs"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "arm/packs"


def executorch_version(mlops_file: Path) -> str:
    """The ExecuTorch pack version cbuild setup resolved, from <solution>.cbuild-pack.yml.

    The file is a lock file that keeps earlier resolutions, so an unversioned
    selector can still point at an older pack; the entry selected by the
    csolution's exact pin (PyTorch::ExecuTorch@<version>) is the one in use.
    """
    import yaml

    pack_file = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".cbuild-pack.yml"))
    # After a version bump the lock file still lists the old pin as resolved
    # by its own selector: the csolution's pin decides (the lock file may be a
    # link into the solution directory).
    for csolution in pack_file.resolve().parent.glob("*.csolution.yml"):
        if pinned := re.search(rf"{re.escape(PACK)}@(\d+\.\d+\.\d+\S*)", csolution.read_text()):
            return pinned.group(1)
    fallback = None
    for entry in yaml.safe_load(pack_file.read_text())["cbuild-pack"]["resolved-packs"]:
        name, _, version = entry["resolved-pack"].partition("@")
        selectors = entry.get("selected-by-pack", [])
        if name != PACK or not selectors:
            continue
        if f"{PACK}@{version}" in selectors:
            return version
        fallback = fallback or version
    if fallback:
        return fallback
    sys.exit(f"{pack_file}: {PACK} is not among the resolved packs")


def executorch_pack(version: str) -> Path:
    """Directory of the installed ExecuTorch pack of that version."""
    vendor, _, pack = PACK.partition("::")
    return pack_root() / vendor / pack / version


def compile_spec(mlops: dict, mlops_dir: Path):
    """EthosUCompileSpec from the npu: and vela: nodes of the cbuild-mlops.yml."""
    from executorch.backends.arm.ethosu import EthosUCompileSpec

    npu = mlops.get("npu")
    if not npu:
        sys.exit("the solution's mlops: node names no NPU; this example needs an Ethos-U")
    vela = mlops.get("vela", {})
    options = vela.get("options", "")

    def option(name: str) -> str | None:
        found = re.search(rf"--{name}[= ](\S+)", options)
        return found.group(1) if found else None

    target = option("accelerator-config") or f"{npu['type'].lower()}-{npu.get('macs', 256)}"
    kwargs = {
        "target": target,
        "system_config": option("system-config"),
        "memory_mode": option("memory-mode"),
    }
    if vela.get("ini"):
        kwargs["config_ini"] = str(mlops_dir / vela["ini"])
    print(f"[ai_layer] Vela: {kwargs}")
    return EthosUCompileSpec(**kwargs)


def quantize_method(spec, method):
    """PT2E-quantize one method's module for the Ethos-U: int8, or int16 activations with int8 weights."""
    import torch
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_a16w8_quantization_config,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    graph = torch.export.export(method.module, method.example).module()
    quantizer = EthosUQuantizer(spec)
    if method.activation_bits == 16:
        quantizer.set_global(get_symmetric_a16w8_quantization_config(is_per_channel=True))
    else:
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in method.samples:
            prepared(*sample)  # calibrate
    return convert_pt2e(prepared)


def export_program(spec) -> tuple[bytes, dict]:
    """Export every method of model/model.py into one .pte with quantized IO; returns the bytes and the IO description."""
    import torch
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes.quantize_io_pass import quantize_input, quantize_output

    sys.path.insert(0, str(HERE / "model"))
    from model import get_methods

    methods = get_methods()
    programs = {m.name: torch.export.export(quantize_method(spec, m), m.example) for m in methods}
    edge = to_edge_transform_and_lower(
        programs,
        partitioner={name: [EthosUPartitioner(spec)] for name in programs},
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    # Move the float<->int boundary out of the program: the methods then take
    # and return the quantized tensors the delegate works on, and the CPU side
    # quantizes / dequantizes with the parameters recorded here.
    io = {}
    for method in methods:
        program = edge.exported_program(method.name)
        inputs, outputs = [], []
        for index, example in enumerate(method.example):
            if not example.is_floating_point():
                # An integer input (the indices of a gather) is not quantized:
                # it goes to the delegate as it is. Scale 1, zero point 0.
                info = torch.iinfo(example.dtype)
                inputs.append(_tensor_desc(tuple(example.shape), example.dtype, 1.0, 0, info.min, info.max))
                continue
            scale, zp, qmin, qmax, dtype = quantize_input(program, index)
            inputs.append(_tensor_desc(tuple(example.shape), dtype, scale, zp, qmin, qmax))
        for index in range(len(program.graph_signature.user_outputs)):
            shape = _output_shape(program, index)
            scale, zp, qmin, qmax, dtype = quantize_output(program, index)
            outputs.append(_tensor_desc(shape, dtype, scale, zp, qmin, qmax))
        io[method.name] = {"inputs": inputs, "outputs": outputs}

    # Inputs stay the caller's buffers: without this the runtime plans memory
    # for every input and memcpys the caller's tensors into it on each call
    # (the 507 kB image of the traffic counter). Outputs stay planned.
    from executorch.exir.passes import MemoryPlanningPass

    program = edge.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=False,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )
    return bytes(program.buffer), io


def _output_shape(program, index: int) -> tuple[int, ...]:
    node = list(program.graph.output_node().args[0])[index]
    return tuple(node.meta["val"].shape)


def _tensor_desc(shape, dtype, scale, zp, qmin, qmax) -> dict:
    import torch

    ctype = {torch.int8: "int8_t", torch.int16: "int16_t", torch.uint8: "uint8_t", torch.int32: "int32_t"}[dtype]
    return {
        "shape": [int(d) for d in shape],
        "dtype": str(dtype).replace("torch.", ""),
        "ctype": ctype,
        "scale": float(scale),
        "zero_point": int(zp),
        "qmin": int(qmin),
        "qmax": int(qmax),
    }


def components(pte: bytes, pack: Path) -> tuple[list[str], list[str]]:
    """Runtime, kernel utils and registration, backend, plus one operator component per operator the .pte uses.

    The pack's "Extension Tensor" is not selected: its tensor_ptr_maker.cpp
    needs std::random_device, which the LLVM embedded toolchain lacks; the
    runner wraps its input in a TensorImpl instead.
    """
    pdsc = next(pack.glob("*.pdsc"))
    available = set(re.findall(r'Csub="([^"]+)"', pdsc.read_text()))
    family = {"aten": "Portable", "quantized_decomposed": "Quantized", "cortex_m": "Cortex-M"}

    selected, unknown = set(), []
    for ns, op in sorted(set(re.findall(rb"(aten|quantized_decomposed|cortex_m)::(\w+)", pte))):
        ns, op = ns.decode(), op.decode()
        candidates = [
            f"{family[ns]} {op}",
            f"{family[ns]} {re.sub(r'_(per_tensor|per_channel|byte|copy)$', '', op)}",
        ]
        if match := next((c for c in candidates if c in available), None):
            selected.add(match)
        else:
            unknown.append(f"{ns}::{op}")
    if unknown:
        print(f"[ai_layer] warning: no component in {pack.name} for {unknown}", file=sys.stderr)

    return ["Runtime", "Kernel Utils", "Kernel Registration", "Backend EthosU"], sorted(selected)


def c_array(pte: bytes) -> str:
    rows = [", ".join(f"0x{b:02x}" for b in pte[i : i + 16]) for i in range(0, len(pte), 16)]
    return (
        "// Generated by create_ai_layer.py -- do not edit.\n"
        f"__attribute__((aligned(16))) const unsigned char {SYMBOL}[] = {{\n  "
        + ",\n  ".join(rows)
        + f"\n}};\nconst unsigned long {SYMBOL}_size = sizeof({SYMBOL});\n"
    )


HEADER = f"""// Generated by create_ai_layer.py -- do not edit.
#pragma once
#ifdef __cplusplus
extern "C" {{
#endif
extern const unsigned char {SYMBOL}[];
extern const unsigned long {SYMBOL}_size;
#ifdef __cplusplus
}}
#endif
"""


def io_header(io: dict) -> str:
    """model_io.h: one macro block per method tensor, C-preprocessor friendly."""
    lines = [
        "// Generated by create_ai_layer.py -- do not edit.",
        "// Quantized IO of every method in model_pte: shapes, C types and the",
        "// affine quantization q = round(x / SCALE) + ZERO_POINT the CPU applies.",
        "#pragma once",
        "",
    ]
    for name, desc in io.items():
        prefix = f"MODEL_{name.upper()}"
        lines.append(f'#define {prefix}_METHOD "{name}"')
        lines.append(f"#define {prefix}_NUM_INPUTS {len(desc['inputs'])}")
        lines.append(f"#define {prefix}_NUM_OUTPUTS {len(desc['outputs'])}")
        for kind in ("inputs", "outputs"):
            for index, t in enumerate(desc[kind]):
                p = f"{prefix}_{kind[:-1].upper()}{index}"
                lines.append(f"#define {p}_NDIM {len(t['shape'])}")
                lines.append(f"#define {p}_SHAPE {{{', '.join(str(d) for d in t['shape'])}}}")
                lines.append(f"#define {p}_NUMEL {_numel(t['shape'])}")
                lines.append(f"#define {p}_CTYPE {t['ctype']}")
                lines.append(f"#define {p}_DTYPE_{t['dtype'].upper()} 1")
                lines.append(f"#define {p}_SCALE {t['scale']!r}f")
                lines.append(f"#define {p}_ZERO_POINT {t['zero_point']}")
                lines.append(f"#define {p}_QMIN {t['qmin']}")
                lines.append(f"#define {p}_QMAX {t['qmax']}")
        lines.append("")
    return "\n".join(lines)


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def clayer(mlops: dict, runtime: list[str], operators: list[str], mlops_file: Path, version: str) -> str:
    lines = [
        f"# Generated by create_ai_layer.py from {mlops_file.name} -- do not edit.",
        f"# Re-run `python create_ai_layer.py {mlops_file.name}` after changing",
        "# model/model.py or the csolution's mlops: node.",
        "layer:",
        "  type: AI",
        f"  description: {mlops.get('description', mlops['model'].get('name', 'AI layer'))}",
        "",
        "  packs:",
        f"    - pack: {PACK}@{version}",
        "",
        "  define:",
        "    - ET_LOG_ENABLED: 0",
        "",
        "  add-path:",
        "    - .",
        "",
        "  components:",
        *[f"    - component: Machine Learning:ExecuTorch:{c}" for c in runtime],
        *[f"    - component: Machine Learning:ExecuTorch Operators:{c}" for c in operators],
        "",
        "  groups:",
        f"    - group: {mlops['model'].get('name', 'Model')}",
        "      files:",
        f"        - file: ./{SYMBOL}.c",
        f"        - file: ./{SYMBOL}.h",
        "        - file: ./model_io.h",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".cbuild-mlops.yml"):
        sys.exit(f"usage: {Path(__file__).name} <solution>.cbuild-mlops.yml")
    run_in_venv()

    import yaml

    mlops_file = Path(sys.argv[1]).resolve()
    mlops = yaml.safe_load(mlops_file.read_text())["cbuild-mlops"]
    layer_file = mlops_file.parent / mlops["model"]["clayer"]
    layer_dir = layer_file.parent

    pte, io = export_program(compile_spec(mlops, mlops_file.parent))
    version = executorch_version(mlops_file)
    runtime, operators = components(pte, executorch_pack(version))

    layer_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "model.pte").write_bytes(pte)
    (layer_dir / f"{SYMBOL}.c").write_text(c_array(pte), newline="\n")
    (layer_dir / f"{SYMBOL}.h").write_text(HEADER, newline="\n")
    (layer_dir / "model_io.h").write_text(io_header(io), newline="\n")
    layer_file.write_text(clayer(mlops, runtime, operators, mlops_file, version), newline="\n")

    for name, desc in io.items():
        for kind in ("inputs", "outputs"):
            for index, t in enumerate(desc[kind]):
                print(
                    f"[ai_layer] {name} {kind[:-1]}{index}: {t['dtype']}{t['shape']} "
                    f"scale={t['scale']:.6g} zp={t['zero_point']}"
                )
    print(f"[ai_layer] {len(pte)} byte program, operators: {operators}")
    print(f"[ai_layer] wrote {layer_file}")


if __name__ == "__main__":
    main()
