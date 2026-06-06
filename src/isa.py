from dataclasses import dataclass
from enum import IntEnum

DATA_MEM_SIZE = 4096
INSTR_MEM_SIZE = 4096
WORD_SIZE = 32
BYTES_PER_WORD = 4
VECTOR_SIZE = 4

STACK_START = DATA_MEM_SIZE - 3
INPUT_DATA_ADDR = DATA_MEM_SIZE - 2
OUTPUT_DATA_ADDR = DATA_MEM_SIZE - 1


MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF
MASK8 = 0xFF
MASK4 = 0xF
SIGN32 = 0x80000000
SIGN16 = 0x8000

R0, R1, R2, R3, R4, R5, FP, SP = 0, 1, 2, 3, 4, 5, 6, 7

REG_NAMES = ["R0", "R1", "R2", "R3", "R4", "R5", "FP", "SP"]


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
    DIVM = 0x44
    MODM = 0x45
    ANDM = 0x46
    ORM = 0x47
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
    Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
    Opcode.INC, Opcode.DEC, Opcode.CMP, Opcode.NOT, Opcode.AND, Opcode.OR,

    Opcode.JMP, Opcode.JZ, Opcode.JNZ, Opcode.JG, Opcode.JL, Opcode.JGE, Opcode.JLE, Opcode.JC,
    Opcode.CALL,
    Opcode.LD, Opcode.ST,
    Opcode.ADDM, Opcode.SUBM, Opcode.MULM, Opcode.CMPM,
    Opcode.DIVM, Opcode.MODM, Opcode.ANDM, Opcode.ORM,
})

TWO_WORD_OPCODES = frozenset({
    Opcode.LDI, Opcode.ADDI, Opcode.SUBI, Opcode.MULI, Opcode.CMPI
})

SIGN_EXTEND_32_OPCODES = frozenset({
    Opcode.LDI, Opcode.ADDI, Opcode.SUBI, Opcode.MULI, Opcode.CMPI
})


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

    def byte_size(self) -> int: return self.word_count() * BYTES_PER_WORD


def encode(instr: Instruction) -> list[int]:
    w0 = (int(instr.opcode) << 24) | ((instr.reg1 & MASK4) << 20) | (
        (instr.reg2 & MASK4) << 16) | (instr.imm & MASK16)
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
    return opcode, (w >> 20) & MASK4, (w >> 16) & MASK4, w & MASK16


def sign_extend_32(val: int) -> int:
    val = val & MASK32
    return val - (MASK32 + 1) if val & SIGN32 else val


def sign_extend_16(val: int) -> int:
    val = val & MASK16
    return val - (MASK16 + 1) if val & SIGN16 else val


def to_unsigned32(val: int) -> int:
    return val & MASK32


def disassemble(instr: Instruction, addr: int) -> str:
    op = instr.opcode
    r1 = REG_NAMES[instr.reg1] if instr.reg1 < len(
        REG_NAMES) else f"R{instr.reg1}"
    r2 = REG_NAMES[instr.reg2] if instr.reg2 < len(
        REG_NAMES) else f"R{instr.reg2}"
    imm = instr.imm

    mnemonics: dict[int, str] = {
        Opcode.HLT: "hlt", Opcode.JMP: f"jmp {imm}", Opcode.JZ: f"jz {imm}",
        Opcode.JNZ: f"jnz {imm}", Opcode.JG: f"jg {imm}", Opcode.JL: f"jl {imm}",
        Opcode.JGE: f"jge {imm}", Opcode.JLE: f"jle {imm}", Opcode.JC: f"jc {imm}",
        Opcode.CALL: f"call {imm}", Opcode.RET: "ret", Opcode.IRET: "iret", Opcode.STI: "sti",
        Opcode.MOV: f"mov {r1}, {r2}", Opcode.LD: f"ld {r1}, [{imm}]", Opcode.ST: f"st [{imm}], {r1}",
        Opcode.LDI: f"ldi {r1}, {imm}", Opcode.PUSH: f"push {r1}", Opcode.POP: f"pop {r1}",
        Opcode.LD_IND: f"ld {r1}, [{r2}]", Opcode.ST_IND: f"st [{r1}], {r2}",
        Opcode.ADD: f"add {r1}, {r2}", Opcode.SUB: f"sub {r1}, {r2}", Opcode.MUL: f"mul {r1}, {r2}",
        Opcode.DIV: f"div {r1}, {r2}", Opcode.MOD: f"mod {r1}, {r2}", Opcode.INC: f"inc {r1}",
        Opcode.DEC: f"dec {r1}", Opcode.ADDI: f"add {r1}, {imm}", Opcode.SUBI: f"sub {r1}, {imm}",
        Opcode.MULI: f"mul {r1}, {imm}", Opcode.CMP: f"cmp {r1}, {r2}", Opcode.CMPI: f"cmp {r1}, {imm}",
        Opcode.NOT: f"not {r1}", Opcode.AND: f"and {r1}, {r2}", Opcode.OR: f"or {r1}, {r2}",
        Opcode.ADDM: f"add {r1}, [{imm}]", Opcode.SUBM: f"sub {r1}, [{imm}]", Opcode.MULM: f"mul {r1}, [{imm}]",
        Opcode.CMPM: f"cmp {r1}, [{imm}]", Opcode.DIVM: f"div {r1}, [{imm}]", Opcode.MODM: f"mod {r1}, [{imm}]",
        Opcode.ANDM: f"and {r1}, [{imm}]", Opcode.ORM: f"or {r1}, [{imm}]",
        Opcode.POLY: f"poly {r1}, {len(instr.pairs)} terms",
    }
    mnemonic = mnemonics.get(int(op), f"??? (op={int(op)})")
    hexcode = "".join(f"{w:08x}" for w in encode(instr))
    return f"{addr:04d} - {hexcode} - {mnemonic}"


def write_binary(instructions: list[Instruction], data_section: list[tuple[int, int]], entry_point: int, irq_handler: int, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(0x414C4731.to_bytes(4, "little"))
        f.write(entry_point.to_bytes(4, "little"))
        f.write(irq_handler.to_bytes(4, "little"))
        f.write(len(data_section).to_bytes(4, "little"))
        for addr, val in data_section:
            f.write((addr & MASK32).to_bytes(4, "little"))
            f.write((val & MASK32).to_bytes(4, "little"))
        for instr in instructions:
            for word in encode(instr):
                f.write(word.to_bytes(4, "little"))


def write_debug(instructions: list[Instruction], data_section: list[tuple[int, int]], entry_point: int, irq_handler: int, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"entry_point: {entry_point}\n")
        f.write(f"irq_handler: {irq_handler}\n")
        if data_section:
            f.write("\n--- DATA SECTION ---\n")
            for addr, val in data_section:
                f.write(f"[{addr:04d}] = {val} ({val & MASK32:#010x})\n")
        f.write("\n--- CODE SECTION ---\n")
        addr = 0
        for instr in instructions:
            f.write(disassemble(instr, addr) + "\n")
            addr += instr.word_count()


def decode_instruction_from_words(words: list[int]) -> Instruction:
    w0 = words[0]
    opcode, reg1, reg2, imm16 = decode_word0(w0)
    if opcode == Opcode.POLY:
        n = imm16 & MASK16
        pairs_list = [((pw >> 16) & MASK16, pw & MASK16)
                      for pw in words[1:1+n]]
        return Instruction(opcode, reg1, reg2, n, tuple(pairs_list))
    if opcode in TWO_WORD_OPCODES and len(words) >= 2:
        w1 = words[1]
        imm = sign_extend_32(w1) if opcode in SIGN_EXTEND_32_OPCODES else w1
    else:
        imm = imm16
    return Instruction(opcode, reg1, reg2, imm)


def read_binary(filename: str) -> tuple[list[Instruction], list[tuple[int, int]], int, int]:
    with open(filename, "rb") as f:
        data = f.read()
    assert int.from_bytes(data[0:4], "little") == 0x414C4731, "Invalid magic"
    entry_point = int.from_bytes(data[4:8], "little")
    irq_handler = int.from_bytes(data[8:12], "little")
    data_len = int.from_bytes(data[12:16], "little")

    data_section = []
    offset = 16
    for _ in range(data_len):
        data_section.append((int.from_bytes(data[offset:offset + 4], "little"),
                             int.from_bytes(data[offset + 4:offset + 8], "little", signed=True)))
        offset += 8

    raw_words = [int.from_bytes(data[i:i+4], "little")
                 for i in range(offset, len(data), 4)]
    instructions = []
    wi = 0
    while wi < len(raw_words):
        opcode = Opcode((raw_words[wi] >> 24) & MASK8)
        wc = word_count_from_header(opcode, raw_words[wi] & MASK16)
        instructions.append(
            decode_instruction_from_words(raw_words[wi:wi + wc]))
        wi += wc

    return instructions, data_section, entry_point, irq_handler
