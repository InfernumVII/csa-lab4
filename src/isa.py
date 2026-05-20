"""Instruction Set Architecture definitions for CISC processor (Harvard, hardwired, tick-accurate)."""

from dataclasses import dataclass
from enum import IntEnum

DATA_MEM_SIZE = 4096
INSTR_MEM_SIZE = 4096
WORD_SIZE = 32
BYTES_PER_WORD = 4
VECTOR_SIZE = 4

INPUT_BUF_READ_IDX = 0
INPUT_BUF_WRITE_IDX = 1
INPUT_BUFFER_BASE = 2
INPUT_BUFFER_WORDS = 256

INPUT_DATA_ADDR = DATA_MEM_SIZE - 4
INPUT_AVAIL_ADDR = DATA_MEM_SIZE - 3
OUTPUT_DATA_ADDR = DATA_MEM_SIZE - 2
OUTPUT_READY_ADDR = DATA_MEM_SIZE - 1
STACK_START = DATA_MEM_SIZE - 5

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF
MASK8 = 0xFF
MASK4 = 0xF
SIGN32 = 0x80000000
SIGN16 = 0x8000


class Opcode(IntEnum):
    HLT = 0x00
    JMP = 0x02
    JZ = 0x03
    JNZ = 0x04
    JG = 0x05
    JL = 0x06
    JGE = 0x07
    JLE = 0x08
    JC = 0x0E
    CALL = 0x09
    RET = 0x0A
    IRET = 0x0B
    STI = 0x0C

    MOV = 0x10
    LD = 0x11
    ST = 0x12
    LDI = 0x13
    PUSH = 0x14
    POP = 0x15
    LD_IND = 0x16
    ST_IND = 0x17
    MOVSP = 0x18
    LDSP = 0x19
    GETFLAGS = 0x1A
    SETFLAGS = 0x1B

    ADD = 0x20
    SUB = 0x21
    MUL = 0x22
    DIV = 0x23
    MOD = 0x24
    INC = 0x25
    DEC = 0x26
    ADDI = 0x27
    SUBI = 0x28
    MULI = 0x29
    CMP = 0x2A
    CMPI = 0x2B

    NOT = 0x33
    AND = 0x30
    OR = 0x31

    ADDM = 0x40
    SUBM = 0x41
    MULM = 0x42
    CMPM = 0x43

    POLY = 0x52

    VLOAD = 0x60
    VSTORE = 0x61
    VADD = 0x62
    VSUB = 0x63
    VMUL = 0x64
    VDIV = 0x65
    VCMP = 0x66
    VSET = 0x67
    VSCALAR = 0x68
    VGET = 0x69


ONE_WORD_OPCODES = frozenset({
    Opcode.HLT, Opcode.RET, Opcode.IRET, Opcode.STI,
    Opcode.MOV, Opcode.PUSH, Opcode.POP, Opcode.LD_IND, Opcode.ST_IND,
    Opcode.MOVSP, Opcode.LDSP, Opcode.GETFLAGS, Opcode.SETFLAGS,
    Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
    Opcode.INC, Opcode.DEC,
    Opcode.CMP,
    Opcode.NOT, Opcode.AND, Opcode.OR,
    Opcode.VADD, Opcode.VSUB, Opcode.VMUL, Opcode.VDIV, Opcode.VCMP,
    Opcode.VSCALAR, Opcode.VGET,
})

TWO_WORD_OPCODES = frozenset({
    Opcode.JMP, Opcode.JZ, Opcode.JNZ, Opcode.JG, Opcode.JL, Opcode.JGE, Opcode.JLE,
    Opcode.JC,
    Opcode.CALL,
    Opcode.LD, Opcode.ST, Opcode.LDI,
    Opcode.ADDI, Opcode.SUBI, Opcode.MULI, Opcode.CMPI,
    Opcode.ADDM, Opcode.SUBM, Opcode.MULM, Opcode.CMPM,
    Opcode.VLOAD, Opcode.VSTORE, Opcode.VSET,
})

SIGN_EXTEND_32_OPCODES = frozenset({
    Opcode.LDI, Opcode.ADDI, Opcode.SUBI, Opcode.MULI, Opcode.CMPI, Opcode.VSET,
})

EXEC_TICKS: dict[int, int] = {
    Opcode.HLT: 0,
    Opcode.JMP: 0,
    Opcode.JZ: 0,
    Opcode.JNZ: 0,
    Opcode.JG: 0,
    Opcode.JL: 0,
    Opcode.JGE: 0,
    Opcode.JLE: 0,
    Opcode.JC: 0,
    Opcode.CALL: 1,
    Opcode.RET: 2,
    Opcode.IRET: 2,
    Opcode.STI: 0,
    Opcode.MOV: 1,
    Opcode.LD: 2,
    Opcode.ST: 2,
    Opcode.LDI: 2,
    Opcode.PUSH: 1,
    Opcode.POP: 2,
    Opcode.LD_IND: 2,
    Opcode.ST_IND: 2,
    Opcode.MOVSP: 0,
    Opcode.LDSP: 0,
    Opcode.GETFLAGS: 0,
    Opcode.SETFLAGS: 0,
    Opcode.ADD: 1,
    Opcode.SUB: 1,
    Opcode.MUL: 2,
    Opcode.DIV: 3,
    Opcode.MOD: 3,
    Opcode.INC: 0,
    Opcode.DEC: 0,
    Opcode.ADDI: 2,
    Opcode.SUBI: 2,
    Opcode.MULI: 3,
    Opcode.CMP: 1,
    Opcode.CMPI: 2,
    Opcode.NOT: 0,
    Opcode.AND: 1,
    Opcode.OR: 1,
    Opcode.ADDM: 3,
    Opcode.SUBM: 3,
    Opcode.MULM: 4,
    Opcode.CMPM: 3,
    Opcode.VLOAD: 5,
    Opcode.VSTORE: 5,
    Opcode.VADD: 2,
    Opcode.VSUB: 2,
    Opcode.VMUL: 3,
    Opcode.VDIV: 4,
    Opcode.VCMP: 2,
    Opcode.VSET: 3,
    Opcode.VSCALAR: 1,
    Opcode.VGET: 1,
}

TICK_COUNTS: dict[int, int] = {}
for _op, _et in EXEC_TICKS.items():
    _wc = 1 if _op in ONE_WORD_OPCODES else 2
    TICK_COUNTS[_op] = _wc + _et


def word_count_from_header(opcode: Opcode, imm16: int) -> int:
    if opcode == Opcode.POLY:
        return 1 + (imm16 & MASK16)
    if opcode in ONE_WORD_OPCODES:
        return 1
    return 2


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    reg1: int = 0
    reg2: int = 0
    imm: int = 0
    pairs: tuple[tuple[int, int], ...] = ()

    def word_count(self) -> int:
        if self.opcode == Opcode.POLY:
            return 1 + len(self.pairs)
        if self.opcode in ONE_WORD_OPCODES:
            return 1
        return 2

    def byte_size(self) -> int:
        return self.word_count() * BYTES_PER_WORD


def encode(instr: Instruction) -> list[int]:
    w0 = (int(instr.opcode) << 24) | ((instr.reg1 & MASK4) << 20) | ((instr.reg2 & MASK4) << 16) | (instr.imm & MASK16)
    if instr.opcode == Opcode.POLY:
        words = [w0 & MASK32]
        for ci, xi in instr.pairs:
            words.append(((ci & MASK16) << 16) | (xi & MASK16))
        return words
    if instr.opcode in ONE_WORD_OPCODES:
        return [w0 & MASK32]
    w1 = instr.imm & MASK32
    return [w0 & MASK32, w1]


def decode_word0(w: int) -> tuple[Opcode, int, int, int]:
    opcode = Opcode((w >> 24) & MASK8)
    reg1 = (w >> 20) & MASK4
    reg2 = (w >> 16) & MASK4
    imm16 = w & MASK16
    return opcode, reg1, reg2, imm16


def sign_extend_32(val: int) -> int:
    val = val & MASK32
    if val & SIGN32:
        return val - (MASK32 + 1)
    return val


def sign_extend_16(val: int) -> int:
    val = val & MASK16
    if val & SIGN16:
        return val - (MASK16 + 1)
    return val


def to_unsigned32(val: int) -> int:
    return val & MASK32


REG_NAMES = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]


def disassemble(instr: Instruction, addr: int) -> str:
    op = instr.opcode
    r1 = REG_NAMES[instr.reg1] if instr.reg1 < len(REG_NAMES) else f"R{instr.reg1}"
    r2 = REG_NAMES[instr.reg2] if instr.reg2 < len(REG_NAMES) else f"R{instr.reg2}"
    imm = instr.imm

    mnemonics: dict[int, str] = {
        Opcode.HLT: "hlt",
        Opcode.JMP: f"jmp {imm}",
        Opcode.JZ: f"jz {imm}",
        Opcode.JNZ: f"jnz {imm}",
        Opcode.JG: f"jg {imm}",
        Opcode.JL: f"jl {imm}",
        Opcode.JGE: f"jge {imm}",
        Opcode.JLE: f"jle {imm}",
        Opcode.JC: f"jc {imm}",
        Opcode.CALL: f"call {imm}",
        Opcode.RET: "ret",
        Opcode.IRET: "iret",
        Opcode.STI: "sti",
        Opcode.MOV: f"mov {r1}, {r2}",
        Opcode.LD: f"ld {r1}, [{imm}]",
        Opcode.ST: f"st [{imm}], {r1}",
        Opcode.LDI: f"ldi {r1}, {imm}",
        Opcode.PUSH: f"push {r1}",
        Opcode.POP: f"pop {r1}",
        Opcode.LD_IND: f"ld {r1}, [{r2}]",
        Opcode.ST_IND: f"st [{r1}], {r2}",
        Opcode.MOVSP: f"movsp {r1}",
        Opcode.LDSP: f"ldsp {r1}",
        Opcode.GETFLAGS: f"getflags {r1}",
        Opcode.SETFLAGS: f"setflags {r1}",
        Opcode.ADD: f"add {r1}, {r2}",
        Opcode.SUB: f"sub {r1}, {r2}",
        Opcode.MUL: f"mul {r1}, {r2}",
        Opcode.DIV: f"div {r1}, {r2}",
        Opcode.MOD: f"mod {r1}, {r2}",
        Opcode.INC: f"inc {r1}",
        Opcode.DEC: f"dec {r1}",
        Opcode.ADDI: f"add {r1}, {imm}",
        Opcode.SUBI: f"sub {r1}, {imm}",
        Opcode.MULI: f"mul {r1}, {imm}",
        Opcode.CMP: f"cmp {r1}, {r2}",
        Opcode.CMPI: f"cmp {r1}, {imm}",
        Opcode.NOT: f"not {r1}",
        Opcode.AND: f"and {r1}, {r2}",
        Opcode.OR: f"or {r1}, {r2}",
        Opcode.ADDM: f"add {r1}, [{imm}]",
        Opcode.SUBM: f"sub {r1}, [{imm}]",
        Opcode.MULM: f"mul {r1}, [{imm}]",
        Opcode.CMPM: f"cmp {r1}, [{imm}]",
        Opcode.POLY: f"poly {r1}, {len(instr.pairs)} terms",
        Opcode.VLOAD: f"vload V{instr.reg1}, [{imm}]",
        Opcode.VSTORE: f"vstore [{imm}], V{instr.reg1}",
        Opcode.VADD: f"vadd V{instr.reg1}, V{instr.reg2}",
        Opcode.VSUB: f"vsub V{instr.reg1}, V{instr.reg2}",
        Opcode.VMUL: f"vmul V{instr.reg1}, V{instr.reg2}",
        Opcode.VDIV: f"vdiv V{instr.reg1}, V{instr.reg2}",
        Opcode.VCMP: f"vcmp V{instr.reg1}, V{instr.reg2}",
        Opcode.VSET: f"vset V{instr.reg1}, {imm}",
        Opcode.VSCALAR: f"vscalar V{instr.reg1}, {r2}, {imm}",
        Opcode.VGET: f"vget {r1}, V{instr.reg2}, {imm}",
    }

    mnemonic = mnemonics.get(int(op), f"??? (op={int(op)})")
    words = encode(instr)
    hexcode = "".join(f"{w:08x}" for w in words)
    return f"{addr:04d} - {hexcode} - {mnemonic}"


def write_binary(instructions: list[Instruction], entry_point: int, irq_handler: int, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(0x414C4731.to_bytes(4, "little"))
        f.write(entry_point.to_bytes(4, "little"))
        f.write(irq_handler.to_bytes(4, "little"))
        for instr in instructions:
            for word in encode(instr):
                f.write(word.to_bytes(4, "little"))


def write_debug(instructions: list[Instruction], entry_point: int, irq_handler: int, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"entry_point: {entry_point}\n")
        f.write(f"irq_handler: {irq_handler}\n")
        addr = 0
        for instr in instructions:
            f.write(disassemble(instr, addr) + "\n")
            addr += instr.word_count()


def decode_instruction_from_words(words: list[int]) -> Instruction:
    w0 = words[0]
    opcode, reg1, reg2, imm16 = decode_word0(w0)

    if opcode == Opcode.POLY:
        n = imm16 & MASK16
        pairs_list: list[tuple[int, int]] = []
        for i in range(n):
            pw = words[1 + i]
            ci = (pw >> 16) & MASK16
            xi = pw & MASK16
            pairs_list.append((ci, xi))
        return Instruction(opcode, reg1, reg2, n, tuple(pairs_list))

    if opcode in TWO_WORD_OPCODES and len(words) >= 2:
        w1 = words[1]
        imm = sign_extend_32(w1) if opcode in SIGN_EXTEND_32_OPCODES else w1
    else:
        imm = imm16

    return Instruction(opcode, reg1, reg2, imm)


def read_binary(filename: str) -> tuple[list[Instruction], int, int]:
    with open(filename, "rb") as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], "little")
    assert magic == 0x414C4731, f"Invalid binary file: bad magic {magic:#x}"
    entry_point = int.from_bytes(data[4:8], "little")
    irq_handler = int.from_bytes(data[8:12], "little")

    raw_words: list[int] = []
    offset = 12
    while offset + 4 <= len(data):
        raw_words.append(int.from_bytes(data[offset:offset + 4], "little"))
        offset += 4

    instructions: list[Instruction] = []
    wi = 0
    while wi < len(raw_words):
        w0 = raw_words[wi]
        opcode = Opcode((w0 >> 24) & MASK8)
        imm16 = w0 & MASK16
        wc = word_count_from_header(opcode, imm16)
        chunk = raw_words[wi:wi + wc]
        instr = decode_instruction_from_words(chunk)
        instructions.append(instr)
        wi += wc

    return instructions, entry_point, irq_handler
