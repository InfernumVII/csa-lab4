"""Golden tests using pytest-golden for the processor toolchain."""

import os
import tempfile
from typing import Any

import pytest

from src.isa import write_binary, write_debug
from src.simulator import Simulator, parse_input
from src.translator import translate


def _run_golden(source: str, input_text: str, max_ticks: int = 5000000, log_limit: int = 100) -> dict[str, Any]:
    code, entry, irq, ast_str, str_inits = translate(source)
    source_loc = len([line for line in source.split("\n") if line.strip()])

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "target.bin")
        inp_path = os.path.join(tmpdir, "input.txt")
        hex_path = os.path.join(tmpdir, "target.bin.hex")

        write_binary(code, entry, irq, bin_path)
        write_debug(code, entry, irq, hex_path)
        with open(inp_path, "w", encoding="utf-8") as f:
            f.write(input_text)

        with open(bin_path, "rb") as f:
            binary_data = f.read()
        with open(hex_path, encoding="utf-8") as f:
            code_hex = f.read()

        input_schedule = parse_input(inp_path)
        sim = Simulator(code, entry, irq, input_schedule, str_inits)
        sim._max_ticks = max_ticks
        sim._log_limit = log_limit
        result = sim.run()

        log_text = "\n".join(sim.log[:log_limit]) + "\nEOF"

        dmem_nonzero = []
        for i, v in enumerate(sim.dmem):
            if v != 0:
                dmem_nonzero.append(f"{i:04d}: {v}")
        data_mem_dump = "\n".join(dmem_nonzero) if dmem_nonzero else "(empty)"

    stdout_text = f"source LoC: {source_loc} code instr: {len(code)}\n============================================================\n{result}\nticks: {sim.tick}\n"

    return {
        "out_code": binary_data,
        "out_code_hex": code_hex.rstrip(),
        "out_ast": ast_str,
        "out_data_mem": data_mem_dump,
        "out_stdout": stdout_text,
        "out_log": log_text,
    }


@pytest.mark.golden_test("golden/*.yml")
def test_golden(golden: Any) -> None:
    source = golden["in_source"]
    input_text = golden["in_stdin"]
    outputs = _run_golden(source, input_text)

    assert outputs["out_code"] == golden.out["out_code"]
    assert outputs["out_code_hex"] == golden.out["out_code_hex"]
    assert outputs["out_ast"] == golden.out["out_ast"]
    assert outputs["out_data_mem"] == golden.out["out_data_mem"]
    assert outputs["out_stdout"] == golden.out["out_stdout"]
    assert outputs["out_log"] == golden.out["out_log"]
