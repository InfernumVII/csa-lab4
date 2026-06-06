import os
import tempfile
import io
import contextlib
from typing import Any

import pytest

from src.isa import write_binary, write_debug, encode, disassemble, MASK32, INPUT_DATA_ADDR
from src.machine import DataMemory, Datapath, InstructionMemory, ControlUnit, Simulator, parse_schedule
from src.translator import translate


def _run_golden(source: str, input_text: str, max_ticks: int = 5000000, log_limit: int = 100) -> dict[str, Any]:
    code, data_section, entry, irq, ast_str = translate(source)
    source_loc = len([line for line in source.split("\n") if line.strip()])

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "target.bin")
        inp_path = os.path.join(tmpdir, "input.txt")
        hex_path = os.path.join(tmpdir, "target.bin.hex")

        write_binary(code, data_section, entry, irq, bin_path)
        write_debug(code, data_section, entry, irq, hex_path)
        with open(inp_path, "w", encoding="utf-8") as f:
            f.write(input_text)

        with open(bin_path, "rb") as f:
            binary_data = f.read()
        with open(hex_path, encoding="utf-8") as f:
            code_hex = f.read()

        input_schedule = parse_schedule(input_text)

        dp = Datapath()
        imem = InstructionMemory(4096)

        disasm_map = {}
        addr = 0
        for instr in code:
            disasm_map[addr] = disassemble(instr, addr).split(" - ")[-1]
            for w in encode(instr):
                imem.imem[addr] = w
                addr += 1

        for d_addr, d_val in data_section:
            dp.dmem.mem[d_addr] = d_val & MASK32

        cu = ControlUnit(dp, imem, entry, irq)
        sim = Simulator(dp, cu, input_schedule, disasm_map=disasm_map)

        log_lines = []

        with contextlib.redirect_stdout(io.StringIO()):
            while not cu.halted and cu.tick_count < max_ticks:
                while sim.schedule and cu.tick_count >= sim.schedule[0][0]:
                    tick, char = sim.schedule.pop(0)
                    val = 0 if char == "\\0" else ord(char)
                    dp.dmem.mem[INPUT_DATA_ADDR] = val
                    cu.irq_latch = True

                cu.tick()

                if cu.tick_count <= log_limit:
                    log_lines.append(sim.format_state())

        result = "".join(dp.dmem.output_buffer)

        log_text = "\n".join(log_lines)
        if cu.tick_count > log_limit:
            log_text += "\nEOF"

        dmem_nonzero = []
        for i, v in enumerate(dp.dmem.mem):

            if v != 0 and i < 4000:
                dmem_nonzero.append(f"{i:04d}: {v}")
        data_mem_dump = "\n".join(dmem_nonzero) if dmem_nonzero else "(empty)"

    stdout_text = f"source LoC: {source_loc} code instr: {len(code)}\n============================================================\n{result}\nticks: {cu.tick_count}\n"

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
