"""Processor simulator: CISC Harvard hardwired tick-accurate with trap I/O and vector extension.

DataPath -- data memory, registers, flags, ALU, I/O (passive).
ControlUnit -- word-by-word fetch, decode, signal generation (hardwired FSM).
"""

import sys

from src.isa import (
    DATA_MEM_SIZE,
    EXEC_TICKS,
    INPUT_AVAIL_ADDR,
    INPUT_DATA_ADDR,
    MASK32,
    OUTPUT_DATA_ADDR,
    SIGN32,
    STACK_START,
    VECTOR_SIZE,
    Instruction,
    Opcode,
    decode_instruction_from_words,
    disassemble,
    encode,
    read_binary,
    word_count_from_header,
)

FLAG_Z = 0
FLAG_N = 1
FLAG_C = 2
FLAG_V = 3
FLAG_I = 4


def to_signed(v: int) -> int:
    v = v & MASK32
    return v - 0x100000000 if v & SIGN32 else v


def to_u32(v: int) -> int:
    return v & MASK32


def u32_add(a: int, b: int) -> tuple[int, bool, bool]:
    r = a + b
    carry = r > MASK32
    result = r & MASK32
    overflow = (to_signed(a) + to_signed(b)) != to_signed(result)
    return result, carry, overflow


def u32_sub(a: int, b: int) -> tuple[int, bool, bool]:
    r = a - b
    carry = a < b
    result = r & MASK32
    overflow = (to_signed(a) - to_signed(b)) != to_signed(result)
    return result, carry, overflow


class DataPath:
    dmem: list[int]
    regs: list[int]
    sp: int
    flags: list[bool]
    vregs: list[list[int]]
    output: list[str]
    input_schedule: list[tuple[int, int]]
    input_idx: int
    pending_irq: bool
    in_irq: bool

    def __init__(self, input_schedule: list[tuple[int, int]], str_inits: list[tuple[int, list[int]]]) -> None:
        self.dmem = [0] * DATA_MEM_SIZE
        for sa, vals in str_inits:
            for i, v in enumerate(vals):
                self.dmem[sa + i] = to_u32(v)
        self.regs = [0] * 8
        self.sp = STACK_START
        self.regs[7] = STACK_START
        self.flags = [False] * 5
        self.vregs = [[0] * VECTOR_SIZE for _ in range(4)]
        self.output = []
        self.input_schedule = sorted(input_schedule, key=lambda x: x[0])
        self.input_idx = 0
        self.pending_irq = False
        self.in_irq = False

    def signal_latch_reg(self, reg: int, value: int) -> None:
        self.regs[reg] = to_u32(value) if reg < 7 else value
        if reg == 7:
            self.sp = value

    def signal_set_flags(self, result: int, carry: bool = False, overflow: bool = False) -> None:
        r = to_u32(result)
        self.flags[FLAG_Z] = r == 0
        self.flags[FLAG_N] = bool(r & SIGN32)
        self.flags[FLAG_C] = carry
        self.flags[FLAG_V] = overflow

    def signal_wr(self, addr: int, value: int) -> None:
        val = to_u32(value)
        if addr == OUTPUT_DATA_ADDR:
            ch = val & 0xFF
            if ch != 0:
                self.output.append(chr(ch))
        elif 0 <= addr < DATA_MEM_SIZE:
            self.dmem[addr] = val

    def signal_rd(self, addr: int) -> int:
        if 0 <= addr < DATA_MEM_SIZE:
            return self.dmem[addr]
        return 0

    def signal_push(self, val: int) -> None:
        self.sp -= 1
        self.dmem[self.sp] = to_u32(val)
        self.regs[7] = self.sp

    def signal_pop(self) -> int:
        val = self.dmem[self.sp]
        self.sp += 1
        self.regs[7] = self.sp
        return val

    def encode_flags(self) -> int:
        w = 0
        for i in range(5):
            if self.flags[i]:
                w |= 1 << i
        return w

    def decode_flags(self, w: int) -> None:
        for i in range(5):
            self.flags[i] = bool(w & (1 << i))

    def signal_check_irq(self, tick: int) -> None:
        if self.pending_irq or self.in_irq or self.dmem[INPUT_AVAIL_ADDR]:
            return
        while self.input_idx < len(self.input_schedule):
            t, ch = self.input_schedule[self.input_idx]
            if t <= tick:
                self.dmem[INPUT_DATA_ADDR] = ord(ch) if isinstance(ch, str) else ch
                self.dmem[INPUT_AVAIL_ADDR] = 1
                self.pending_irq = True
                self.input_idx += 1
                break
            else:
                break

    def zero(self) -> bool:
        return self.flags[FLAG_Z]

    def negative(self) -> bool:
        return self.flags[FLAG_N]

    def carry_flag(self) -> bool:
        return self.flags[FLAG_C]

    def overflow(self) -> bool:
        return self.flags[FLAG_V]

    def interrupts_enabled(self) -> bool:
        return self.flags[FLAG_I]


class ControlUnit:
    dp: DataPath
    imem: list[int]
    pc: int
    irq_handler: int
    tick: int
    halted: bool
    _phase: str
    _fetch_words: list[int]
    _fetch_needed: int
    _instr_start: int
    _current_instr: Instruction | None
    _exec_step: int
    _exec_total: int
    _log: list[str]
    _max_ticks: int
    _log_limit: int

    def __init__(self, instructions: list[Instruction], entry_point: int, irq_handler: int, dp: DataPath) -> None:
        self.dp = dp
        self.imem = []
        for instr in instructions:
            for word in self._encode_for_imem(instr):
                self.imem.append(word)
        self.pc = entry_point
        self.irq_handler = irq_handler
        self.tick = 0
        self.halted = False
        self._phase = "fetch"
        self._fetch_words = []
        self._fetch_needed = 0
        self._instr_start = 0
        self._current_instr = None
        self._exec_step = 0
        self._exec_total = 0
        self._log = []
        self._max_ticks = 1000000
        self._log_limit = 0

    @staticmethod
    def _encode_for_imem(instr: Instruction) -> list[int]:
        return encode(instr)

    def _log_state(self, instr: Instruction | None, phase_mark: str = "") -> None:
        if self._log_limit and len(self._log) >= self._log_limit:
            return
        dp = self.dp
        fstr = "".join("1" if dp.flags[i] else "0" for i in range(5))
        irq_mark = " [IRQ]" if dp.in_irq else ""
        addr = self._instr_start if self._phase != "fetch" or self._fetch_words else self.pc
        step = self._exec_step if self._phase == "exec" else len(self._fetch_words) - 1
        regs_str = f"R0={dp.regs[0]} R1={dp.regs[1]} R2={dp.regs[2]} R3={dp.regs[3]} " \
                   f"R4={dp.regs[4]} R5={dp.regs[5]} R6={dp.regs[6]} SP={dp.sp}"
        if instr is not None:
            mnem = disassemble(instr, self._instr_start)
            self._log.append(f"TICK:{self.tick:5} PC:{addr:4}/{step} {regs_str} F={fstr} {mnem}{irq_mark}")
        else:
            mark = phase_mark or ("fetch" if self._phase == "fetch" else "")
            self._log.append(f"TICK:{self.tick:5} PC:{addr:4}/{step} {regs_str} F={fstr} [{mark}]{irq_mark}")

    def _handle_irq(self) -> None:
        dp = self.dp
        dp.signal_push(self.pc)
        dp.signal_push(dp.encode_flags())
        dp.flags[FLAG_I] = False
        dp.in_irq = True
        self.pc = self.irq_handler
        dp.pending_irq = False
        self._phase = "fetch"
        self._fetch_words = []
        self._current_instr = None

    def run(self) -> str:
        while not self.halted and self.tick < self._max_ticks:
            if self._phase == "fetch" and not self._fetch_words:
                self.dp.signal_check_irq(self.tick)
                if self.dp.interrupts_enabled() and self.dp.pending_irq and not self.dp.in_irq:
                    self._log_state(None, "IRQ")
                    self._handle_irq()
                    self.tick += 1
                    continue

            if self._phase == "fetch":
                self._do_fetch_tick()
            else:
                self._do_exec_tick()

            self.tick += 1

        return "".join(self.dp.output)

    def _do_fetch_tick(self) -> None:
        if self.pc >= len(self.imem):
            self.halted = True
            return

        word = self.imem[self.pc]
        self._fetch_words.append(word)
        self.pc += 1

        if len(self._fetch_words) == 1:
            opcode = Opcode((word >> 24) & 0xFF)
            imm16 = word & 0xFFFF
            self._fetch_needed = word_count_from_header(opcode, imm16)
            self._instr_start = self.pc - 1

        self._log_state(None, "fetch")

        if len(self._fetch_words) >= self._fetch_needed:
            self._current_instr = decode_instruction_from_words(self._fetch_words)
            if self._current_instr is None:
                self.halted = True
                return
            if self._current_instr.opcode == Opcode.POLY:
                self._exec_total = len(self._current_instr.pairs)
            else:
                self._exec_total = EXEC_TICKS.get(int(self._current_instr.opcode), 0)
            self._exec_step = 0
            self._phase = "exec"
            if self._exec_total == 0:
                next_pc = self.pc
                instr = self._current_instr
                assert instr is not None
                new_pc = self._execute(instr, next_pc)
                self.pc = new_pc
                if not self.halted:
                    self._log_state(self._current_instr)
                    self._phase = "fetch"
                    self._fetch_words = []
                    self._current_instr = None

    def _do_exec_tick(self) -> None:
        self._log_state(self._current_instr)
        self._exec_step += 1
        if self._exec_step >= self._exec_total:
            next_pc = self.pc
            instr = self._current_instr
            assert instr is not None
            new_pc = self._execute(instr, next_pc)
            self.pc = new_pc
            if not self.halted:
                self._phase = "fetch"
                self._fetch_words = []
                self._current_instr = None

    def _execute(self, instr: Instruction, next_pc: int) -> int:
        op = instr.opcode
        r1 = instr.reg1
        r2 = instr.reg2
        imm = instr.imm
        dp = self.dp
        pc = next_pc

        if op == Opcode.HLT:
            self.halted = True

        elif op == Opcode.JMP:
            pc = imm

        elif op == Opcode.JZ:
            if dp.zero():
                pc = imm

        elif op == Opcode.JNZ:
            if not dp.zero():
                pc = imm

        elif op == Opcode.JG:
            if not dp.zero() and not dp.negative():
                pc = imm

        elif op == Opcode.JL:
            if dp.negative():
                pc = imm

        elif op == Opcode.JGE:
            if not dp.negative():
                pc = imm

        elif op == Opcode.JLE:
            if dp.zero() or dp.negative():
                pc = imm

        elif op == Opcode.JC:
            if dp.carry_flag():
                pc = imm

        elif op == Opcode.CALL:
            dp.signal_push(next_pc)
            pc = imm

        elif op == Opcode.RET:
            pc = dp.signal_pop()

        elif op == Opcode.IRET:
            fw = dp.signal_pop()
            dp.decode_flags(fw)
            pc = dp.signal_pop()
            dp.in_irq = False

        elif op == Opcode.STI:
            dp.flags[FLAG_I] = True

        elif op == Opcode.MOV:
            dp.signal_latch_reg(r1, dp.regs[r2])

        elif op == Opcode.MOVSP:
            dp.signal_latch_reg(r1, dp.sp)

        elif op == Opcode.LDSP:
            dp.sp = dp.regs[r1]
            dp.regs[7] = dp.sp

        elif op == Opcode.GETFLAGS:
            dp.signal_latch_reg(r1, dp.encode_flags())

        elif op == Opcode.SETFLAGS:
            dp.decode_flags(dp.regs[r1])

        elif op == Opcode.LD:
            addr = imm
            dp.signal_latch_reg(r1, dp.signal_rd(addr))

        elif op == Opcode.ST:
            addr = imm
            dp.signal_wr(addr, dp.regs[r1])

        elif op == Opcode.LDI:
            dp.signal_latch_reg(r1, to_u32(imm))

        elif op == Opcode.PUSH:
            dp.signal_push(dp.regs[r1])

        elif op == Opcode.POP:
            dp.signal_latch_reg(r1, dp.signal_pop())

        elif op == Opcode.LD_IND:
            addr = dp.regs[r2]
            dp.signal_latch_reg(r1, dp.signal_rd(addr))

        elif op == Opcode.ST_IND:
            addr = dp.regs[r1]
            val = dp.regs[r2]
            dp.signal_wr(addr, val)

        elif op == Opcode.ADD:
            a, b = dp.regs[r1], dp.regs[r2]
            res, carry, overflow = u32_add(a, b)
            dp.signal_latch_reg(r1, res)
            dp.signal_set_flags(res, carry, overflow)

        elif op == Opcode.SUB:
            a, b = dp.regs[r1], dp.regs[r2]
            res, carry, overflow = u32_sub(a, b)
            dp.signal_latch_reg(r1, res)
            dp.signal_set_flags(res, carry, overflow)

        elif op == Opcode.MUL:
            a, b = to_signed(dp.regs[r1]), to_signed(dp.regs[r2])
            res = a * b
            dp.signal_latch_reg(r1, to_u32(res))
            dp.signal_set_flags(to_u32(res))

        elif op == Opcode.DIV:
            a, b = to_signed(dp.regs[r1]), to_signed(dp.regs[r2])
            if b == 0:
                dp.signal_latch_reg(r1, 0)
                dp.signal_set_flags(0)
            else:
                res = int(a / b) if (a < 0) != (b < 0) and a % b != 0 else a // b
                dp.signal_latch_reg(r1, to_u32(res))
                dp.signal_set_flags(to_u32(res))

        elif op == Opcode.MOD:
            a, b = to_signed(dp.regs[r1]), to_signed(dp.regs[r2])
            if b == 0:
                dp.signal_latch_reg(r1, 0)
                dp.signal_set_flags(0)
            else:
                res = a % b
                dp.signal_latch_reg(r1, to_u32(res))
                dp.signal_set_flags(to_u32(res))

        elif op == Opcode.INC:
            val = dp.regs[r1]
            res, carry, overflow = u32_add(val, 1)
            dp.signal_latch_reg(r1, res)
            dp.signal_set_flags(res, carry, overflow)

        elif op == Opcode.DEC:
            val = dp.regs[r1]
            res, carry, overflow = u32_sub(val, 1)
            dp.signal_latch_reg(r1, res)
            dp.signal_set_flags(res, carry, overflow)

        elif op == Opcode.ADDI:
            a = dp.regs[r1]
            b = imm
            res, carry, overflow = u32_add(a, to_u32(b))
            dp.signal_latch_reg(r1, res)
            dp.signal_set_flags(res, carry, overflow)

        elif op == Opcode.SUBI:
            a = dp.regs[r1]
            b = imm
            res, carry, overflow = u32_sub(a, to_u32(b))
            dp.signal_latch_reg(r1, res)
            dp.signal_set_flags(res, carry, overflow)

        elif op == Opcode.MULI:
            a, b = to_signed(dp.regs[r1]), imm
            res = a * b
            dp.signal_latch_reg(r1, to_u32(res))
            dp.signal_set_flags(to_u32(res))

        elif op == Opcode.CMP:
            a, b = dp.regs[r1], dp.regs[r2]
            _, carry, overflow = u32_sub(a, b)
            dp.signal_set_flags(to_u32(a - b), carry, overflow)

        elif op == Opcode.CMPI:
            a = dp.regs[r1]
            b = to_u32(imm)
            _, carry, overflow = u32_sub(a, b)
            dp.signal_set_flags(to_u32(a - b), carry, overflow)

        elif op == Opcode.CMPM:
            a = dp.regs[r1]
            addr = imm
            b = dp.signal_rd(addr)
            _, carry, overflow = u32_sub(a, b)
            dp.signal_set_flags(to_u32(a - b), carry, overflow)

        elif op == Opcode.NOT:
            dp.signal_latch_reg(r1, to_u32(~dp.regs[r1]))
            dp.signal_set_flags(dp.regs[r1])

        elif op == Opcode.AND:
            dp.signal_latch_reg(r1, dp.regs[r1] & dp.regs[r2])
            dp.signal_set_flags(dp.regs[r1])

        elif op == Opcode.OR:
            dp.signal_latch_reg(r1, dp.regs[r1] | dp.regs[r2])
            dp.signal_set_flags(dp.regs[r1])

        elif op == Opcode.ADDM:
            a = dp.regs[r1]
            addr = imm
            b = dp.signal_rd(addr)
            res, carry, overflow = u32_add(a, b)
            dp.signal_latch_reg(r1, res)
            dp.signal_set_flags(res, carry, overflow)

        elif op == Opcode.SUBM:
            a = dp.regs[r1]
            addr = imm
            b = dp.signal_rd(addr)
            res, carry, overflow = u32_sub(a, b)
            dp.signal_latch_reg(r1, res)
            dp.signal_set_flags(res, carry, overflow)

        elif op == Opcode.MULM:
            a = to_signed(dp.regs[r1])
            addr = imm
            b = to_signed(dp.signal_rd(addr))
            dp.signal_latch_reg(r1, to_u32(a * b))
            dp.signal_set_flags(dp.regs[r1])

        elif op == Opcode.POLY:
            for ci, xi in instr.pairs:
                c_val = to_signed(dp.signal_rd(ci))
                x_val = to_signed(dp.signal_rd(xi))
                res = to_signed(dp.regs[r1]) + c_val * x_val
                dp.signal_latch_reg(r1, to_u32(res))
            dp.signal_set_flags(dp.regs[r1])

        elif op == Opcode.VLOAD:
            vn = r1
            addr = imm
            for i in range(VECTOR_SIZE):
                dp.vregs[vn][i] = dp.signal_rd(addr + i)

        elif op == Opcode.VSTORE:
            vn = r1
            addr = imm
            for i in range(VECTOR_SIZE):
                a = addr + i
                if 0 <= a < DATA_MEM_SIZE:
                    dp.dmem[a] = dp.vregs[vn][i]

        elif op == Opcode.VADD:
            for i in range(VECTOR_SIZE):
                dp.vregs[r1][i] = to_u32(dp.vregs[r1][i] + dp.vregs[r2][i])

        elif op == Opcode.VSUB:
            for i in range(VECTOR_SIZE):
                dp.vregs[r1][i] = to_u32(dp.vregs[r1][i] - dp.vregs[r2][i])

        elif op == Opcode.VMUL:
            for i in range(VECTOR_SIZE):
                dp.vregs[r1][i] = to_u32(to_signed(dp.vregs[r1][i]) * to_signed(dp.vregs[r2][i]))

        elif op == Opcode.VDIV:
            for i in range(VECTOR_SIZE):
                d = dp.vregs[r2][i]
                if d != 0:
                    dp.vregs[r1][i] = to_u32(to_signed(dp.vregs[r1][i]) // to_signed(d))
                else:
                    dp.vregs[r1][i] = 0

        elif op == Opcode.VCMP:
            for i in range(VECTOR_SIZE):
                diff = to_signed(dp.vregs[r1][i]) - to_signed(dp.vregs[r2][i])
                if diff != 0:
                    dp.signal_set_flags(to_u32(diff))
                    break
            else:
                dp.signal_set_flags(0)

        elif op == Opcode.VSET:
            for i in range(VECTOR_SIZE):
                dp.vregs[r1][i] = to_u32(imm)

        elif op == Opcode.VSCALAR:
            idx = imm & 3
            dp.vregs[r1][idx] = dp.regs[r2]

        elif op == Opcode.VGET:
            idx = imm & 3
            dp.signal_latch_reg(r1, dp.vregs[r2][idx])

        return pc


class Simulator:
    cu: ControlUnit
    _max_ticks: int
    _log_limit: int

    def __init__(
        self,
        instructions: list[Instruction],
        entry_point: int,
        irq_handler: int,
        input_schedule: list[tuple[int, int]],
        str_inits: list[tuple[int, list[int]]],
    ) -> None:
        dp = DataPath(input_schedule, str_inits)
        self.cu = ControlUnit(instructions, entry_point, irq_handler, dp)
        self._max_ticks = 1000000
        self._log_limit = 0

    def run(self) -> str:
        self.cu._max_ticks = self._max_ticks
        self.cu._log_limit = self._log_limit
        return self.cu.run()

    @property
    def tick(self) -> int:
        return self.cu.tick

    @property
    def halted(self) -> bool:
        return self.cu.halted

    @property
    def log(self) -> list[str]:
        return self.cu._log

    @property
    def imem(self) -> dict[int, Instruction]:
        return {}

    @property
    def dmem(self) -> list[int]:
        return self.cu.dp.dmem


def parse_input(filename: str) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            tick = int(parts[0])
            ch = parts[1] if len(parts) > 1 else ""
            val = ord(ch) if len(ch) == 1 and ch not in "0123456789" else int(ch) if ch else 0
            schedule.append((tick, val))
    return schedule


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: simulator.py <binary> <input_file>", file=sys.stderr)
        sys.exit(1)

    bin_file = sys.argv[1]
    input_file = sys.argv[2]

    instructions, entry_point, irq_handler = read_binary(bin_file)
    input_schedule = parse_input(input_file)

    dp = DataPath(input_schedule, [])
    cu = ControlUnit(instructions, entry_point, irq_handler, dp)
    cu._log_limit = 10000
    result = cu.run()

    print(result, end="", flush=True)

    base = bin_file.rsplit(".", 1)[0]
    with open(base + ".log", "w", encoding="utf-8") as f:
        f.write(f"Output: {result!r}\n")
        f.write(f"Ticks: {cu.tick}\n")
        f.write(f"Halted: {cu.halted}\n")
        f.write("\n".join(cu._log))


if __name__ == "__main__":
    main()
