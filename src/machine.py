import sys

from src.isa import (
    DATA_MEM_SIZE, INPUT_DATA_ADDR, MASK16, MASK32, OUTPUT_DATA_ADDR,
    SIGN32, STACK_START, TWO_WORD_OPCODES, SIGN_EXTEND_32_OPCODES,
    Opcode, decode_word0, read_binary, sign_extend_32,
)

R0, R1, R2, R3, R4, R5, FP, SP = 0, 1, 2, 3, 4, 5, 6, 7


class ALU:
    def __init__(self):
        self.a = self.b = self.res = 0
        self.out_z = self.out_n = self.out_c = False
        self.z = self.n = self.c = False

    def set_a(self, val: int) -> None: self.a = val & MASK32
    def set_b(self, val: int) -> None: self.b = val & MASK32

    def _calc_flags(self, update_c: bool = False, new_c: bool = False) -> None:
        self.out_z = (self.res == 0)
        self.out_n = (self.res & SIGN32) != 0
        self.out_c = new_c if update_c else self.c

    def latch_flags(self) -> None:
        self.z, self.n, self.c = self.out_z, self.out_n, self.out_c

    def signal_add(self) -> None:
        raw = self.a + self.b
        self.res = raw & MASK32
        self._calc_flags(update_c=True, new_c=raw > MASK32)

    def signal_sub(self) -> None:
        self.res = (self.a - self.b) & MASK32
        self._calc_flags(update_c=True, new_c=self.a < self.b)

    def signal_mul(self) -> None:
        self.res = (self.a * self.b) & MASK32
        self._calc_flags()

    def signal_div(self) -> None:
        self.res = (self.a // self.b) & MASK32 if self.b != 0 else 0
        self._calc_flags()

    def signal_mod(self) -> None:
        self.res = (self.a % self.b) & MASK32 if self.b != 0 else 0
        self._calc_flags()

    def signal_and(self) -> None:
        self.res = (self.a & self.b) & MASK32
        self._calc_flags()

    def signal_or(self) -> None:
        self.res = (self.a | self.b) & MASK32
        self._calc_flags()

    def signal_not(self) -> None:
        self.res = (~self.a) & MASK32
        self._calc_flags()

    def signal_inc(self) -> None:
        self.res = (self.a + 1) & MASK32
        self._calc_flags()

    def signal_dec(self) -> None:
        self.res = (self.a - 1) & MASK32
        self._calc_flags()


class Registers:
    def __init__(self):
        self.regs = [0] * 8
        self.regs[SP] = STACK_START
        self.outrs1 = self.outrs2 = 0

    def signal_rs1_out(
        self, register: int) -> None: self.outrs1 = self.regs[register]

    def signal_rs2_out(
        self, register: int) -> None: self.outrs2 = self.regs[register]
    def latch_reg(self, data: int,
                  regNum: int) -> None: self.regs[regNum] = data & MASK32


class DataMemory:
    def __init__(self, size: int):
        self.mem = [0] * size
        self.out = 0
        self.output_buffer: list[str] = []

    def signal_read(self, addr: int) -> None:
        self.out = self.mem[addr & MASK32]

    def signal_write(self, addr: int, data: int) -> None:
        addr &= MASK32
        self.mem[addr] = data & MASK32
        if addr == OUTPUT_DATA_ADDR:
            char = chr(data & 0xFF)
            print(char, end="", flush=True)
            self.output_buffer.append(char)


class InstructionMemory:
    def __init__(self, size: int):
        self.imem = [0] * size
        self.out = 0

    def load(self, words: list[int]) -> None:
        for i, w in enumerate(words):
            self.imem[i] = w

    def signal_read(self, addr: int) -> None:
        self.out = self.imem[addr & MASK32]


class Datapath:
    def __init__(self):
        self.alu = ALU()
        self.regs = Registers()
        self.dmem = DataMemory(DATA_MEM_SIZE)


class ControlUnit:
    def __init__(self, dp: Datapath, imem: InstructionMemory, entry: int, irq_handler: int):
        self.dp = dp
        self.imem = imem
        self.irq_handler = irq_handler

        self.pc = entry
        self.ir = self.tr = self.sc = self.ic = 0

        self.irq_latch = self.interrupts_enabled = self.handling_irq = self.in_isr = self.halted = False
        self.tick_count = 0
        self.current_action = "INIT"
        self.instr_pc = entry

        self.state_pc = entry
        self.state_ir = self.state_tr = self.state_sc = self.state_ic = 0
        self.state_regs = [0] * 8
        self.state_regs[SP] = STACK_START
        self.state_z = self.state_n = self.state_c = self.state_in_isr = False

    def tick(self) -> None:
        if self.halted:
            return
        self.tick_count += 1

        self.state_pc, self.state_ir, self.state_tr = self.pc, self.ir, self.tr
        self.state_sc, self.state_ic = self.sc, self.ic
        self.state_regs = list(self.dp.regs.regs)
        self.state_z, self.state_n, self.state_c = self.dp.alu.z, self.dp.alu.n, self.dp.alu.c
        self.state_in_isr = self.in_isr

        next_sc, next_ic, next_pc, next_ir, next_tr = self.sc, self.ic, self.pc, self.ir, self.tr
        action = "IDLE"

        if self.sc == 0:
            if self.interrupts_enabled and self.irq_latch:
                self.handling_irq = True
                self.interrupts_enabled = self.irq_latch = False
                self.in_isr = True
                self.dp.regs.signal_rs1_out(SP)
                self.dp.alu.set_a(self.dp.regs.outrs1)
                self.dp.alu.signal_dec()
                self.dp.regs.latch_reg(self.dp.alu.res, SP)
                action = "IRQ_DEC_SP"
                next_sc = 1
            else:
                self.instr_pc = self.pc
                self.imem.signal_read(self.pc)
                next_ir = self.imem.out
                next_pc = (self.pc + 1) & MASK32
                action = "FETCH_W0"
                next_sc = 1

        elif self.handling_irq:
            if self.sc == 1:
                self.dp.regs.signal_rs1_out(SP)
                self.dp.dmem.signal_write(self.dp.regs.outrs1, self.pc)
                self.dp.alu.set_a(self.dp.regs.outrs1)
                self.dp.alu.signal_dec()
                self.dp.regs.latch_reg(self.dp.alu.res, SP)
                action = "IRQ_PUSH_PC"
                next_sc = 2
            elif self.sc == 2:
                self.dp.regs.signal_rs1_out(SP)
                flags = (1 if self.dp.alu.z else 0) | (
                    2 if self.dp.alu.n else 0) | (4 if self.dp.alu.c else 0)
                self.dp.dmem.signal_write(self.dp.regs.outrs1, flags)
                next_pc = self.irq_handler
                action = "IRQ_PUSH_FLAGS"
                self.handling_irq = False
                next_sc = 0

        else:
            opcode, rs1, rs2, imm16 = decode_word0(self.ir)

            if opcode == Opcode.POLY:
                if self.sc == 1:
                    next_ic = imm16
                    next_sc = 2
                elif self.sc == 2:
                    if self.ic == 0:
                        next_sc = 0
                        action = "POLY_DONE"
                    else:
                        self.imem.signal_read(self.pc)
                        self.dp.dmem.signal_read(
                            (self.imem.out >> 16) & MASK16)
                        next_tr = self.dp.dmem.out
                        action = "POLY_FETCH_PAIR1"
                        next_sc = 3
                elif self.sc == 3:
                    self.imem.signal_read(self.pc)
                    self.dp.dmem.signal_read(self.imem.out & MASK16)
                    self.dp.alu.set_a(self.tr)
                    self.dp.alu.set_b(self.dp.dmem.out)
                    self.dp.alu.signal_mul()
                    next_tr = self.dp.alu.res
                    action = "POLY_FETCH_PAR2 AND MUL"
                    next_sc = 4
                elif self.sc == 4:
                    self.dp.regs.signal_rs1_out(rs1)
                    self.dp.alu.set_a(self.tr)
                    self.dp.alu.set_b(self.dp.regs.outrs1)
                    self.dp.alu.signal_add()
                    self.dp.regs.latch_reg(self.dp.alu.res, rs1)
                    next_ic = self.ic - 1
                    next_pc = (self.pc + 1) & MASK32
                    action = "POLY_ALU_ADD"
                    next_sc = 2

            elif opcode in TWO_WORD_OPCODES:
                if self.sc == 1:
                    self.imem.signal_read(self.pc)

                    val = self.imem.out
                    imm = sign_extend_32(
                        val) if opcode in SIGN_EXTEND_32_OPCODES else val

                    if opcode == Opcode.LDI:
                        self.dp.regs.latch_reg(imm, rs1)
                    elif opcode in (Opcode.ADDI, Opcode.SUBI, Opcode.MULI, Opcode.CMPI):
                        self.dp.regs.signal_rs1_out(rs1)
                        self.dp.alu.set_a(self.dp.regs.outrs1)
                        self.dp.alu.set_b(imm)
                        if opcode == Opcode.ADDI:
                            self.dp.alu.signal_add()
                        elif opcode == Opcode.SUBI:
                            self.dp.alu.signal_sub()
                        elif opcode == Opcode.MULI:
                            self.dp.alu.signal_mul()
                        elif opcode == Opcode.CMPI:
                            self.dp.alu.signal_sub()

                        self.dp.alu.latch_flags()
                        if opcode != Opcode.CMPI:
                            self.dp.regs.latch_reg(self.dp.alu.res, rs1)

                    next_pc = (self.pc + 1) & MASK32
                    action = f"EXEC_{opcode.name}"
                    next_sc = 0

            else:
                if self.sc == 1:
                    if opcode == Opcode.PUSH:
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.alu.set_a(self.dp.regs.outrs1)
                        self.dp.alu.signal_dec()
                        self.dp.regs.latch_reg(self.dp.alu.res, SP)
                        action = "EXEC_DEC_SP"
                        next_sc = 2
                    elif opcode in (Opcode.POP, Opcode.RET):
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.dmem.signal_read(self.dp.regs.outrs1)
                        if opcode == Opcode.POP:
                            self.dp.regs.latch_reg(self.dp.dmem.out, rs1)
                        else:
                            next_pc = self.dp.dmem.out
                        action = "EXEC_POP_MEM"
                        next_sc = 2
                    elif opcode == Opcode.IRET:
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.dmem.signal_read(self.dp.regs.outrs1)
                        flags = self.dp.dmem.out
                        self.dp.alu.z, self.dp.alu.n, self.dp.alu.c = bool(
                            flags & 1), bool(flags & 2), bool(flags & 4)
                        self.dp.alu.out_z, self.dp.alu.out_n, self.dp.alu.out_c = self.dp.alu.z, self.dp.alu.n, self.dp.alu.c
                        action = "IRET_POP_FLAGS"
                        next_sc = 2
                    elif opcode == Opcode.CALL:
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.alu.set_a(self.dp.regs.outrs1)
                        self.dp.alu.signal_dec()
                        self.dp.regs.latch_reg(self.dp.alu.res, SP)
                        action = "EXEC_DEC_SP"
                        next_sc = 2
                    else:
                        if opcode == Opcode.HLT:
                            self.halted = True
                        elif opcode == Opcode.MOV:
                            self.dp.regs.signal_rs2_out(rs2)
                            self.dp.regs.latch_reg(self.dp.regs.outrs2, rs1)
                        elif opcode == Opcode.LD_IND:
                            self.dp.regs.signal_rs2_out(rs2)
                            self.dp.dmem.signal_read(self.dp.regs.outrs2)
                            self.dp.regs.latch_reg(self.dp.dmem.out, rs1)
                        elif opcode == Opcode.ST_IND:
                            self.dp.regs.signal_rs1_out(rs1)
                            self.dp.regs.signal_rs2_out(rs2)
                            self.dp.dmem.signal_write(
                                self.dp.regs.outrs1, self.dp.regs.outrs2)
                        elif opcode in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD, Opcode.AND, Opcode.OR, Opcode.CMP):
                            self.dp.regs.signal_rs1_out(rs1)
                            self.dp.alu.set_a(self.dp.regs.outrs1)
                            self.dp.regs.signal_rs2_out(rs2)
                            self.dp.alu.set_b(self.dp.regs.outrs2)
                            if opcode == Opcode.ADD:
                                self.dp.alu.signal_add()
                            elif opcode == Opcode.SUB:
                                self.dp.alu.signal_sub()
                            elif opcode == Opcode.MUL:
                                self.dp.alu.signal_mul()
                            elif opcode == Opcode.DIV:
                                self.dp.alu.signal_div()
                            elif opcode == Opcode.MOD:
                                self.dp.alu.signal_mod()
                            elif opcode == Opcode.AND:
                                self.dp.alu.signal_and()
                            elif opcode == Opcode.OR:
                                self.dp.alu.signal_or()
                            elif opcode == Opcode.CMP:
                                self.dp.alu.signal_sub()
                            self.dp.alu.latch_flags()
                            if opcode != Opcode.CMP:
                                self.dp.regs.latch_reg(self.dp.alu.res, rs1)
                        elif opcode in (Opcode.INC, Opcode.DEC, Opcode.NOT):
                            self.dp.regs.signal_rs1_out(rs1)
                            self.dp.alu.set_a(self.dp.regs.outrs1)
                            if opcode == Opcode.INC:
                                self.dp.alu.signal_inc()
                            elif opcode == Opcode.DEC:
                                self.dp.alu.signal_dec()
                            elif opcode == Opcode.NOT:
                                self.dp.alu.signal_not()
                            self.dp.alu.latch_flags()
                            self.dp.regs.latch_reg(self.dp.alu.res, rs1)
                        elif opcode == Opcode.STI:
                            self.interrupts_enabled = True

                        elif opcode == Opcode.JMP:
                            next_pc = imm16
                        elif opcode == Opcode.JZ:
                            next_pc = imm16 if self.dp.alu.z else self.pc
                        elif opcode == Opcode.JNZ:
                            next_pc = imm16 if not self.dp.alu.z else self.pc
                        elif opcode == Opcode.JG:
                            next_pc = imm16 if not self.dp.alu.z and not self.dp.alu.n else self.pc
                        elif opcode == Opcode.JL:
                            next_pc = imm16 if self.dp.alu.n else self.pc
                        elif opcode == Opcode.JGE:
                            next_pc = imm16 if not self.dp.alu.n else self.pc
                        elif opcode == Opcode.JLE:
                            next_pc = imm16 if self.dp.alu.z or self.dp.alu.n else self.pc
                        elif opcode == Opcode.JC:
                            next_pc = imm16 if self.dp.alu.c else self.pc

                        elif opcode == Opcode.LD:
                            self.dp.dmem.signal_read(imm16)
                            self.dp.regs.latch_reg(self.dp.dmem.out, rs1)
                        elif opcode == Opcode.ST:
                            self.dp.regs.signal_rs1_out(rs1)
                            self.dp.dmem.signal_write(
                                imm16, self.dp.regs.outrs1)
                        elif opcode in (Opcode.ADDM, Opcode.SUBM, Opcode.MULM, Opcode.DIVM, Opcode.MODM, Opcode.ANDM, Opcode.ORM, Opcode.CMPM):
                            self.dp.dmem.signal_read(imm16)
                            self.dp.alu.set_b(self.dp.dmem.out)
                            self.dp.regs.signal_rs1_out(rs1)
                            self.dp.alu.set_a(self.dp.regs.outrs1)
                            if opcode == Opcode.ADDM:
                                self.dp.alu.signal_add()
                            elif opcode == Opcode.SUBM:
                                self.dp.alu.signal_sub()
                            elif opcode == Opcode.MULM:
                                self.dp.alu.signal_mul()
                            elif opcode == Opcode.DIVM:
                                self.dp.alu.signal_div()
                            elif opcode == Opcode.MODM:
                                self.dp.alu.signal_mod()
                            elif opcode == Opcode.ANDM:
                                self.dp.alu.signal_and()
                            elif opcode == Opcode.ORM:
                                self.dp.alu.signal_or()
                            elif opcode == Opcode.CMPM:
                                self.dp.alu.signal_sub()
                            self.dp.alu.latch_flags()
                            if opcode != Opcode.CMPM:
                                self.dp.regs.latch_reg(self.dp.alu.res, rs1)

                        action = f"EXEC_{opcode.name}"
                        next_sc = 0

                elif self.sc == 2:
                    if opcode == Opcode.PUSH:
                        self.dp.regs.signal_rs1_out(rs1)
                        self.dp.regs.signal_rs2_out(SP)
                        self.dp.dmem.signal_write(
                            self.dp.regs.outrs2, self.dp.regs.outrs1)
                        action = "EXEC_PUSH"
                        next_sc = 0
                    elif opcode in (Opcode.POP, Opcode.RET):
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.alu.set_a(self.dp.regs.outrs1)
                        self.dp.alu.signal_inc()
                        self.dp.regs.latch_reg(self.dp.alu.res, SP)
                        action = "EXEC_INC_SP"
                        next_sc = 0
                    elif opcode == Opcode.IRET:
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.alu.set_a(self.dp.regs.outrs1)
                        self.dp.alu.signal_inc()
                        self.dp.regs.latch_reg(self.dp.alu.res, SP)
                        action = "IRET_INC_SP1"
                        next_sc = 3
                    elif opcode == Opcode.CALL:
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.dmem.signal_write(self.dp.regs.outrs1, self.pc)
                        next_pc = imm16
                        action = "EXEC_CALL"
                        next_sc = 0

                elif self.sc == 3:
                    if opcode == Opcode.IRET:
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.dmem.signal_read(self.dp.regs.outrs1)
                        next_pc = self.dp.dmem.out
                        action = "IRET_POP_PC"
                        next_sc = 4

                elif self.sc == 4:
                    if opcode == Opcode.IRET:
                        self.dp.regs.signal_rs1_out(SP)
                        self.dp.alu.set_a(self.dp.regs.outrs1)
                        self.dp.alu.signal_inc()
                        self.dp.regs.latch_reg(self.dp.alu.res, SP)
                        self.interrupts_enabled, self.in_isr = True, False
                        action = "IRET_INC_SP2"
                        next_sc = 0

        self.sc, self.ic, self.pc, self.ir, self.tr = next_sc, next_ic, next_pc, next_ir, next_tr
        self.current_action = action


class Simulator:
    def __init__(self, datapath: Datapath, cu: ControlUnit, schedule: list[tuple[int, str]], disasm_map: dict[int, str] | None = None):
        self.dp = datapath
        self.cu = cu
        self.schedule = schedule
        self.disasm_map = disasm_map if disasm_map is not None else {}

    def run(self) -> None:
        print("\n--- Начало симуляции ---")
        while not self.cu.halted:
            while self.schedule and self.cu.tick_count >= self.schedule[0][0]:
                tick, char = self.schedule.pop(0)
                self.dp.dmem.mem[INPUT_DATA_ADDR] = 0 if char == "\\0" else ord(
                    char)
                self.cu.irq_latch = True
            self.cu.tick()
            self.print_state()
        print("\n--- Симуляция завершена ---")
        print(f"\n[OUTPUT]: {''.join(self.dp.dmem.output_buffer)}")
        print(f"[TICKS]:  {self.cu.tick_count}")

    def format_state(self) -> str:

        regs = self.cu.state_regs
        regs_str = f"R0:{regs[0]:08X} R1:{regs[1]:08X} R2:{regs[2]:08X} FP:{regs[6]:04X} SP:{regs[7]:04X}"
        #flags = f"{'N' if self.cu.state_n else '-'}{'Z' if self.cu.state_z else '-'}{'C' if self.cu.state_c else '-'}"
        #alu_str = f"ALU[res:{self.dp.alu.res:08X} {flags}]"

        ir_str = f"IR:{self.cu.state_ir:08X} TR:{self.cu.state_tr:08X}"
        fsm_str = f"SC:{self.cu.state_sc} IC:{self.cu.state_ic}"
        mode_str = "[ISR ]" if self.cu.state_in_isr else "[MAIN]"
        action = f"{self.cu.current_action:<16}"

        if self.cu.current_action.startswith("IRQ"):
            asm_str = "( hardware irq )"
        elif self.cu.state_sc == 0 and self.cu.current_action == "FETCH_W0":
            asm_str = self.disasm_map.get(self.cu.state_pc, "NOP / ???")
        else:
            asm_str = self.disasm_map.get(self.cu.instr_pc, "NOP / ???")

        return f"T:{self.cu.tick_count:04d} {mode_str} | PC:{self.cu.state_pc:04d} | {fsm_str} | {ir_str} | {regs_str} | {action} | {asm_str:<18}"

    def print_state(self) -> None:

        print(self.format_state())


def parse_schedule(text: str) -> list[tuple[int, str]]:
    schedule = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        schedule.append((int(parts[0]), parts[1] if len(parts) > 1 else " "))
    return sorted(schedule, key=lambda x: x[0])


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python machine.py <binary_file> [input_schedule_file]")
        sys.exit(1)

    schedule_text = open(
        sys.argv[2], "r", encoding="utf-8").read() if len(sys.argv) == 3 else ""
    instructions, data_section, entry_point, irq_handler = read_binary(
        sys.argv[1])

    dp = Datapath()
    imem = InstructionMemory(4096)

    from src.isa import encode, disassemble
    addr, disasm_map = 0, {}
    for instr in instructions:
        disasm_map[addr] = disassemble(instr, addr).split(" - ")[-1]
        for w in encode(instr):
            imem.imem[addr] = w
            addr += 1

    for d_addr, d_val in data_section:
        dp.dmem.mem[d_addr] = d_val & MASK32

    cu = ControlUnit(dp, imem, entry_point, irq_handler)
    Simulator(dp, cu, parse_schedule(schedule_text), disasm_map).run()


if __name__ == "__main__":
    main()
