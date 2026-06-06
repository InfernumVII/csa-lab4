import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

from src.isa import (
    INPUT_DATA_ADDR,
    OUTPUT_DATA_ADDR,
    STACK_START,
    Instruction,
    Opcode,
    R0, R1, R2, R3, R4, R5, FP, SP,
    write_binary,
    write_debug,
)

DATA_ALLOC_START = 0


class TT(Enum):
    NUM = auto()
    STR = auto()
    CHR = auto()
    IDENT = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    ASSIGN = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMI = auto()
    COMMA = auto()
    EOF = auto()


@dataclass
class Token:
    type: TT
    value: str | int


ESCAPE_MAP = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}
CHAR_ESCAPE = {"n": 10, "t": 9, "\\": 92, "'": 39, "0": 0}

TWO_CHAR_MAP = {"==": TT.EQ, "!=": TT.NEQ, "<=": TT.LE,
                ">=": TT.GE, "&&": TT.AND, "||": TT.OR}
ONE_CHAR_MAP = {
    "+": TT.PLUS, "-": TT.MINUS, "*": TT.STAR, "/": TT.SLASH, "%": TT.PERCENT,
    "<": TT.LT, ">": TT.GT, "=": TT.ASSIGN, "!": TT.NOT,
    "(": TT.LPAREN, ")": TT.RPAREN, "{": TT.LBRACE, "}": TT.RBRACE,
    "[": TT.LBRACKET, "]": TT.RBRACKET, ";": TT.SEMI, ",": TT.COMMA,
}


class Lexer:
    def __init__(self, source: str) -> None:
        self.src = source
        self.pos = 0
        self.tokens: list[Token] = []
        self._tokenize()

    def _tokenize(self) -> None:
        while self.pos < len(self.src):
            c = self.src[self.pos]

            if c.isspace():
                self.pos += 1
            elif c == "/" and self._peek(1) == "/":
                self._skip_comment()
            elif c.isdigit():
                self._read_number()
            elif c == '"':
                self._read_string()
            elif c == "'":
                self._read_char()
            elif c.isalpha() or c == "_":
                self._read_ident()
            else:
                self._read_operator()

        self.tokens.append(Token(TT.EOF, ""))

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.src[idx] if idx < len(self.src) else ""

    def _skip_comment(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos] != "\n":
            self.pos += 1

    def _read_number(self) -> None:
        start = self.pos
        while self.pos < len(self.src) and self.src[self.pos].isdigit():
            self.pos += 1
        self.tokens.append(Token(TT.NUM, int(self.src[start:self.pos])))

    def _read_string(self) -> None:
        self.pos += 1
        chars: list[str] = []
        while self.pos < len(self.src) and self.src[self.pos] != '"':
            if self.src[self.pos] == "\\":
                self.pos += 1
                chars.append(ESCAPE_MAP.get(
                    self.src[self.pos], self.src[self.pos]))
            else:
                chars.append(self.src[self.pos])
            self.pos += 1
        self.pos += 1
        self.tokens.append(Token(TT.STR, "".join(chars)))

    def _read_char(self) -> None:
        self.pos += 1
        if self.src[self.pos] == "\\":
            self.pos += 1
            val = CHAR_ESCAPE.get(self.src[self.pos], ord(self.src[self.pos]))
        else:
            val = ord(self.src[self.pos])
        self.pos += 2
        self.tokens.append(Token(TT.CHR, val))

    def _read_ident(self) -> None:
        start = self.pos
        while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == "_"):
            self.pos += 1
        self.tokens.append(Token(TT.IDENT, self.src[start:self.pos]))

    def _read_operator(self) -> None:
        op2 = self.src[self.pos: self.pos + 2]
        if op2 in TWO_CHAR_MAP:
            self.tokens.append(Token(TWO_CHAR_MAP[op2], op2))
            self.pos += 2
        else:
            c = self.src[self.pos]
            if c in ONE_CHAR_MAP:
                self.tokens.append(Token(ONE_CHAR_MAP[c], c))
                self.pos += 1
            else:
                raise SyntaxError(f"Unexpected char: {c!r} at {self.pos}")


class ASTNode:
    pass


@dataclass
class NumLit(ASTNode):
    value: int


@dataclass
class StrLit(ASTNode):
    value: str


@dataclass
class ChrLit(ASTNode):
    value: int


@dataclass
class VarRef(ASTNode):
    name: str


@dataclass
class ArrAccess(ASTNode):
    name: str
    index: ASTNode


@dataclass
class ArrayInit(ASTNode):
    elements: list[ASTNode]


@dataclass
class BinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode


@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode


@dataclass
class Call(ASTNode):
    name: str
    args: list[ASTNode]


@dataclass
class VarDecl(ASTNode):
    name: str
    init: ASTNode
    arr_size: int | None = None


@dataclass
class Assign(ASTNode):
    target: ASTNode
    value: ASTNode


@dataclass
class IfStmt(ASTNode):
    cond: ASTNode
    then_body: list[ASTNode]
    else_body: list[ASTNode] | None = None


@dataclass
class WhileStmt(ASTNode):
    cond: ASTNode
    body: list[ASTNode]


@dataclass
class FuncDef(ASTNode):
    name: str
    params: list[str]
    body: list[ASTNode]


@dataclass
class ReturnStmt(ASTNode):
    value: ASTNode | None = None


@dataclass
class ExprStmt(ASTNode):
    expr: ASTNode


@dataclass
class IrqDef(ASTNode):
    name: str
    body: list[ASTNode]


def dump_ast(node: ASTNode | list[ASTNode], indent: int = 0) -> str:
    p = "  " * indent
    if isinstance(node, list):
        return "\n".join(dump_ast(x, indent) for x in node)
    match node:
        case NumLit(v): return f"{p}NumLit({v})"
        case StrLit(v): return f'{p}StrLit("{v}")'
        case ChrLit(v): return f"{p}ChrLit({v!r})"
        case VarRef(n): return f"{p}VarRef({n})"
        case ArrAccess(n, i): return f"{p}ArrAccess({n}):\n{dump_ast(i, indent+1)}"
        case ArrayInit(elems): return f"{p}ArrayInit:\n" + "\n".join(dump_ast(e, indent+1) for e in elems)
        case BinOp(op, l, r): return f"{p}BinOp({op}):\n{dump_ast(l, indent+1)}\n{dump_ast(r, indent+1)}"
        case UnaryOp(op, x): return f"{p}UnaryOp({op}):\n{dump_ast(x, indent+1)}"
        case Call(n, args):
            args_str = "\n".join(dump_ast(a, indent+1) for a in args)
            return f"{p}Call({n}):\n{args_str}" if args_str else f"{p}Call({n})"
        case VarDecl(n, i, s):
            arr = f"[{s}]" if s else ""
            return f"{p}VarDecl({n}{arr}):\n{dump_ast(i, indent+1)}"
        case Assign(t, v): return f"{p}Assign:\n{dump_ast(t, indent+1)}\n{dump_ast(v, indent+1)}"
        case IfStmt(c, t, e):
            res = f"{p}IfStmt:\n{dump_ast(c, indent+1)}\n{p}  then:\n{dump_ast(t, indent+2)}"
            if e:
                res += f"\n{p}  else:\n{dump_ast(e, indent+2)}"
            return res
        case WhileStmt(c, b): return f"{p}WhileStmt:\n{dump_ast(c, indent+1)}\n{dump_ast(b, indent+1)}"
        case FuncDef(n, params, b): return f"{p}FuncDef({n}, {params}):\n{dump_ast(b, indent+1)}"
        case ReturnStmt(v): return f"{p}ReturnStmt:\n{dump_ast(v, indent+1)}" if v else f"{p}ReturnStmt"
        case ExprStmt(e): return f"{p}ExprStmt:\n{dump_ast(e, indent+1)}"
        case IrqDef(n, b): return f"{p}IrqDef({n}):\n{dump_ast(b, indent+1)}"
        case _: return f"{p}{node!r}"


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _cur(self) -> Token: return self.tokens[self.pos]
    def _is(self, tt: TT) -> bool: return self._cur().type == tt

    def _is_kw(
        self, kw: str) -> bool: return self._cur().type == TT.IDENT and self._cur().value == kw

    def _eat(self, tt: TT) -> Token:
        t = self._cur()
        if t.type != tt:
            raise SyntaxError(f"Expected {tt}, got {t.type} ({t.value!r})")
        self.pos += 1
        return t

    def parse(self) -> list[ASTNode]:
        stmts = self._stmts()
        self._eat(TT.EOF)
        return stmts

    def _stmts(self) -> list[ASTNode]:
        stmts = []
        while not self._is(TT.RBRACE) and not self._is(TT.EOF):
            stmts.append(self._stmt())
        return stmts

    def _stmt(self) -> ASTNode:
        if self._is_kw("var"):
            return self._var_decl()
        if self._is_kw("if"):
            return self._if_stmt()
        if self._is_kw("while"):
            return self._while_stmt()
        if self._is_kw("func"):
            return self._func_def()
        if self._is_kw("return"):
            return self._return_stmt()

        if self._is_kw("print"):
            return self._builtin_macro("__print_str")
        if self._is_kw("printnum"):
            return self._builtin_macro("__print_num")
        if self._is_kw("printu"):
            return self._builtin_macro("__print_u")
        if self._is_kw("putc"):
            return self._builtin_macro("__put_char")

        if self._is_kw("irq"):
            self.pos += 1
            name = str(self._eat(TT.IDENT).value)
            return IrqDef(name, self._block())

        expr = self._expr()
        if self._is(TT.ASSIGN):
            self._eat(TT.ASSIGN)
            val = self._expr()
            self._eat(TT.SEMI)
            return Assign(expr, val)

        self._eat(TT.SEMI)
        return ExprStmt(expr)

    def _builtin_macro(self, func_name: str) -> ExprStmt:
        self.pos += 1
        self._eat(TT.LPAREN)
        expr = self._expr()
        self._eat(TT.RPAREN)
        self._eat(TT.SEMI)
        return ExprStmt(Call(func_name, [expr]))

    def _var_decl(self) -> VarDecl:
        self.pos += 1
        name = str(self._eat(TT.IDENT).value)
        arr_size = None

        if self._is(TT.LBRACKET):
            self._eat(TT.LBRACKET)
            arr_size = int(self._eat(TT.NUM).value)
            self._eat(TT.RBRACKET)

        init = NumLit(0)
        if self._is(TT.ASSIGN):
            self._eat(TT.ASSIGN)
            if self._is(TT.LBRACE):
                self.pos += 1
                elems = []
                if not self._is(TT.RBRACE):
                    elems.append(self._expr())
                    while self._is(TT.COMMA):
                        self._eat(TT.COMMA)
                        elems.append(self._expr())
                self._eat(TT.RBRACE)
                init = ArrayInit(elems)
            else:
                init = self._expr()

        self._eat(TT.SEMI)
        return VarDecl(name, init, arr_size)

    def _if_stmt(self) -> IfStmt:
        self.pos += 1
        self._eat(TT.LPAREN)
        cond = self._expr()
        self._eat(TT.RPAREN)

        then_body = self._block()
        else_body = None
        if self._is_kw("else"):
            self.pos += 1
            else_body = self._block()

        return IfStmt(cond, then_body, else_body)

    def _while_stmt(self) -> WhileStmt:
        self.pos += 1
        self._eat(TT.LPAREN)
        cond = self._expr()
        self._eat(TT.RPAREN)
        return WhileStmt(cond, self._block())

    def _func_def(self) -> FuncDef:
        self.pos += 1
        name = str(self._eat(TT.IDENT).value)
        self._eat(TT.LPAREN)

        params: list[str] = []
        if not self._is(TT.RPAREN):
            params.append(str(self._eat(TT.IDENT).value))
            while self._is(TT.COMMA):
                self._eat(TT.COMMA)
                params.append(str(self._eat(TT.IDENT).value))

        self._eat(TT.RPAREN)
        return FuncDef(name, params, self._block())

    def _return_stmt(self) -> ReturnStmt:
        self.pos += 1
        val = None if self._is(TT.SEMI) else self._expr()
        self._eat(TT.SEMI)
        return ReturnStmt(val)

    def _block(self) -> list[ASTNode]:
        self._eat(TT.LBRACE)
        stmts = self._stmts()
        self._eat(TT.RBRACE)
        return stmts

    def _expr(self) -> ASTNode: return self._or()

    def _or(self) -> ASTNode:
        n = self._and()
        while self._is(TT.OR):
            self.pos += 1
            n = BinOp("||", n, self._and())
        return n

    def _and(self) -> ASTNode:
        n = self._eq()
        while self._is(TT.AND):
            self.pos += 1
            n = BinOp("&&", n, self._eq())
        return n

    def _eq(self) -> ASTNode:
        n = self._cmp()
        while self._cur().type in (TT.EQ, TT.NEQ):
            op = str(self._eat(self._cur().type).value)
            n = BinOp(op, n, self._cmp())
        return n

    def _cmp(self) -> ASTNode:
        n = self._add()
        while self._cur().type in (TT.LT, TT.GT, TT.LE, TT.GE):
            op = str(self._eat(self._cur().type).value)
            n = BinOp(op, n, self._add())
        return n

    def _add(self) -> ASTNode:
        n = self._mul()
        while self._cur().type in (TT.PLUS, TT.MINUS):
            op = str(self._eat(self._cur().type).value)
            n = BinOp(op, n, self._mul())
        return n

    def _mul(self) -> ASTNode:
        n = self._unary()
        while self._cur().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            op = str(self._eat(self._cur().type).value)
            n = BinOp(op, n, self._unary())
        return n

    def _unary(self) -> ASTNode:
        if self._is(TT.MINUS):
            self.pos += 1
            return UnaryOp("-", self._unary())
        if self._is(TT.NOT):
            self.pos += 1
            return UnaryOp("!", self._unary())
        return self._primary()

    def _primary(self) -> ASTNode:
        if self._is(TT.NUM):
            return NumLit(int(self._eat(TT.NUM).value))
        if self._is(TT.STR):
            return StrLit(str(self._eat(TT.STR).value))
        if self._is(TT.CHR):
            return ChrLit(int(self._eat(TT.CHR).value))

        if self._is(TT.LPAREN):
            self.pos += 1
            e = self._expr()
            self._eat(TT.RPAREN)
            return e

        if self._is(TT.IDENT):
            name = str(self._eat(TT.IDENT).value)
            if self._is(TT.LPAREN):
                self.pos += 1
                args: list[ASTNode] = []
                if not self._is(TT.RPAREN):
                    args.append(self._expr())
                    while self._is(TT.COMMA):
                        self.pos += 1
                        args.append(self._expr())
                self._eat(TT.RPAREN)
                return Call(name, args)
            if self._is(TT.LBRACKET):
                self.pos += 1
                idx = self._expr()
                self._eat(TT.RBRACKET)
                return ArrAccess(name, idx)
            return VarRef(name)

        raise SyntaxError(f"Unexpected: {self._cur()}")


class VarLoc(Enum):
    GLOBAL = auto()
    LOCAL = auto()


class CodeGen:
    CMP_OPS = frozenset({"==", "!=", "<", ">", "<=", ">="})

    ARITH_OPS: ClassVar[dict[str, Opcode]] = {
        "+": Opcode.ADD, "-": Opcode.SUB, "*": Opcode.MUL,
        "/": Opcode.DIV, "%": Opcode.MOD,
        "&&": Opcode.AND, "||": Opcode.OR,
    }
    CISC_MEM: ClassVar[dict[str, Opcode]] = {
        "+": Opcode.ADDM, "-": Opcode.SUBM, "*": Opcode.MULM,
        "/": Opcode.DIVM, "%": Opcode.MODM, "&&": Opcode.ANDM, "||": Opcode.ORM
    }
    CMP_DIRECT: ClassVar[dict[str, Opcode]] = {
        ">": Opcode.JG, "<": Opcode.JL, ">=": Opcode.JGE,
        "<=": Opcode.JLE, "==": Opcode.JZ, "!=": Opcode.JNZ,
    }

    def __init__(self) -> None:
        self.code: list[Instruction] = []
        self.addr = 0
        self.data_addr = DATA_ALLOC_START
        self.labels: dict[str, int] = {}
        self.fixups: list[tuple[int, str]] = []
        self.last_label_addr: int = -1

        self.globals: dict[str, tuple[int, int]] = {}
        self.locals: dict[str, tuple[int, int]] = {}
        self.static_locals: dict[str, tuple[int, int]] = {}
        self.local_offset = 0
        self.in_func = False

        self.strings: dict[str, int] = {}
        self.funcs: dict[str, int] = {}
        self.constants: dict[int, int] = {}
        self.used_funcs: set[str] = set()

        self.str_inits: list[tuple[int, list[int]]] = []
        self.const_inits: list[tuple[int, int]] = []
        self.global_inits: list[tuple[int, int]] = []

        self.irq_addr = 0
        self.entry = 0
        self._lbl_counter = 0

    def _lbl(self, prefix: str) -> str:
        self._lbl_counter += 1
        return f"{prefix}_{self._lbl_counter}"

    def _alloc(self, size: int = 1) -> int:
        addr = self.data_addr
        self.data_addr += size
        return addr

    def _str(self, s: str) -> int:
        if s not in self.strings:
            packed = []
            for i in range(0, len(s), 4):
                chunk = s[i:i+4]
                val = 0
                for j, c in enumerate(chunk):
                    val |= (ord(c) & 0xFF) << (j * 8)
                packed.append(val)
            addr = self._alloc(1 + len(packed))
            self.strings[s] = addr
            self.str_inits.append((addr, [len(s)] + packed))
        return self.strings[s]

    def _const(self, val: int) -> int:
        if val not in self.constants:
            addr = self._alloc(1)
            self.constants[val] = addr
            self.const_inits.append((addr, val))
        return self.constants[val]

    def _get_var(self, name: str) -> tuple[VarLoc, int, int]:
        if self.in_func:
            if name in self.locals:
                offset, size = self.locals[name]
                return VarLoc.LOCAL, offset, size
            if name in self.static_locals:
                addr, size = self.static_locals[name]
                return VarLoc.GLOBAL, addr, size
        if name in self.globals:
            addr, size = self.globals[name]
            return VarLoc.GLOBAL, addr, size
        raise ValueError(f"Unknown variable: {name}")

    def _get_var_loc_safe(self, name: str) -> VarLoc:
        if self.in_func:
            if name in self.locals:
                return VarLoc.LOCAL
            if name in self.static_locals:
                return VarLoc.GLOBAL
        return VarLoc.GLOBAL if name in self.globals else VarLoc.LOCAL

    def _needs_io(self, e: ASTNode) -> bool:
        match e:
            case Call(n, args): return n in ("getc", "getnum") or any(self._needs_io(a) for a in args)
            case BinOp(_, l, r): return self._needs_io(l) or self._needs_io(r)
            case UnaryOp(_, x): return self._needs_io(x)
            case ArrAccess(_, idx): return self._needs_io(idx)
            case ArrayInit(elems): return any(self._needs_io(el) for el in elems)
            case _: return False

    def _emit(self, op: Opcode, r1: int = 0, r2: int = 0, imm: int = 0, pairs: tuple = ()) -> None:
        if self.code and self.last_label_addr != self.addr:
            prev = self.code[-1]
            if op == Opcode.LD and prev.opcode == Opcode.ST and prev.reg1 == r1 and prev.imm == imm:
                return
            if op == Opcode.LDI and prev.opcode == Opcode.LDI and prev.reg1 == r1 and prev.imm == imm:
                return
            if op in (Opcode.ADDI, Opcode.SUBI) and imm == 0:
                return
            if op == Opcode.MOV and r1 == r2:
                return

            if op == Opcode.ADDI and prev.opcode == Opcode.ADDI and prev.reg1 == r1:
                self.code[-1] = Instruction(Opcode.ADDI, r1, 0, prev.imm + imm)
                return
            if op == Opcode.SUBI and prev.opcode == Opcode.SUBI and prev.reg1 == r1:
                self.code[-1] = Instruction(Opcode.SUBI, r1, 0, prev.imm + imm)
                return

        if op == Opcode.ADDI and imm == 1:
            op = Opcode.INC
            imm = 0
        elif op == Opcode.SUBI and imm == 1:
            op = Opcode.DEC
            imm = 0

        inst = Instruction(op, r1, r2, imm, pairs)
        self.code.append(inst)
        self.addr += inst.word_count()

    def _jlabel(self, name: str) -> None:
        self.labels[name] = self.addr
        self.last_label_addr = self.addr

    def _jmp(self, op: Opcode, lbl: str, r1: int = 0) -> None:
        idx = len(self.code)
        self._emit(op, r1, 0, 0)
        self.fixups.append((idx, lbl))

    def _resolve(self) -> None:
        for idx, lbl in self.fixups:
            t = self.labels[lbl]
            old = self.code[idx]
            self.code[idx] = Instruction(old.opcode, old.reg1, old.reg2, t)

    def _load_const(self, reg: int, val: int) -> None:
        self._emit(Opcode.LDI, reg, 0, val)

    def _emit_var_addr(self, name: str, reg: int = 1) -> None:
        loc, val, _ = self._get_var(name)
        if loc == VarLoc.GLOBAL:
            self._load_const(reg, val)
        else:
            self._emit(Opcode.MOV, reg, FP)
            if val == 1:
                self._emit(Opcode.INC, reg)
            elif val > 0:
                self._emit(Opcode.ADDI, reg, 0, val)
            elif val == -1:
                self._emit(Opcode.DEC, reg)
            elif val < 0:
                self._emit(Opcode.SUBI, reg, 0, -val)

    def _emit_load_var(self, name: str, reg: int = 0) -> None:
        loc, val, _ = self._get_var(name)
        if loc == VarLoc.GLOBAL:
            self._emit(Opcode.LD, reg, 0, val)
        else:
            self._emit_var_addr(name, reg)
            self._emit(Opcode.LD_IND, reg, reg)

    def _collect_globals(self, stmts: list[ASTNode], is_top_level: bool = True) -> None:
        for s in stmts:
            match s:
                case VarDecl(name, init, arr_size):
                    size = arr_size or (len(init.elements)
                                        if isinstance(init, ArrayInit) else 1)
                    a = self._alloc(size)
                    self.globals[name] = (a, size)

                    if is_top_level:
                        if isinstance(init, (NumLit, ChrLit)):
                            self.global_inits.append((a, init.value))
                        elif isinstance(init, StrLit):
                            self.global_inits.append(
                                (a, self._str(init.value)))
                        elif isinstance(init, ArrayInit):
                            for i, item in enumerate(init.elements):
                                if isinstance(item, (NumLit, ChrLit)):
                                    self.global_inits.append(
                                        (a + i, item.value))
                                elif isinstance(item, StrLit):
                                    self.global_inits.append(
                                        (a + i, self._str(item.value)))

                case IfStmt(_, then_b, else_b):
                    self._collect_globals(then_b, False)
                    if else_b:
                        self._collect_globals(else_b, False)
                case WhileStmt(_, body):
                    self._collect_globals(body, False)

    def _collect_static_locals(self, stmts: list[ASTNode]) -> None:
        for s in stmts:
            match s:
                case VarDecl(name, init, arr_size):
                    size = arr_size or (len(init.elements)
                                        if isinstance(init, ArrayInit) else 1)
                    a = self._alloc(size)
                    self.static_locals[name] = (a, size)

                    if isinstance(init, ArrayInit):
                        for i, item in enumerate(init.elements):
                            if isinstance(item, (NumLit, ChrLit)):
                                self.global_inits.append((a + i, item.value))
                            elif isinstance(item, StrLit):
                                self.global_inits.append(
                                    (a + i, self._str(item.value)))
                case IfStmt(_, then_b, else_b):
                    self._collect_static_locals(then_b)
                    if else_b:
                        self._collect_static_locals(else_b)
                case WhileStmt(_, body):
                    self._collect_static_locals(body)

    def _collect_constants(self, stmts: list[ASTNode], is_top_level: bool = True) -> None:
        for s in stmts:
            match s:
                case VarDecl(_, init, _):
                    if not (is_top_level and isinstance(init, (NumLit, ChrLit, StrLit, ArrayInit))):
                        self._collect_const_expr(init)
                case Assign(target, value):
                    self._collect_const_expr(target) if hasattr(
                        target, 'index') else None
                    self._collect_const_expr(value)
                case IfStmt(cond, then_b, else_b):
                    self._collect_const_expr(cond)
                    self._collect_constants(then_b, False)
                    if else_b:
                        self._collect_constants(else_b, False)
                case WhileStmt(cond, body):
                    self._collect_const_expr(cond)
                    self._collect_constants(body, False)
                case FuncDef(_, _, body) | IrqDef(_, body):
                    self._collect_constants(body, False)
                case ReturnStmt(val):
                    if val:
                        self._collect_const_expr(val)
                case ExprStmt(expr):
                    self._collect_const_expr(expr)

    def _collect_const_expr(self, e: ASTNode) -> None:
        match e:
            case NumLit(v) | ChrLit(v): self._const(v)
            case StrLit(v): self._const(self._str(v))
            case BinOp(_, l, r):
                self._collect_const_expr(l)
                self._collect_const_expr(r)
            case UnaryOp(_, op): self._collect_const_expr(op)
            case Call(_, args):
                for a in args:
                    self._collect_const_expr(a)
            case ArrAccess(name, idx):
                self._collect_const_expr(idx)
            case ArrayInit(elems):
                for el in elems:
                    self._collect_const_expr(el)

    def _get_used_functions(self, stmts: list[ASTNode]) -> set[str]:
        used = set()
        builtins = {"getc": "__get_char",
                    "getnum": "__get_num", "putc": "__put_char"}

        def visit(n: ASTNode):
            match n:
                case Call(name, args):
                    fname = builtins.get(name, name)
                    if fname not in used:
                        used.add(fname)
                        for f in stmts:
                            if isinstance(f, FuncDef) and f.name == fname:
                                for st in f.body:
                                    visit(st)
                    for a in args:
                        visit(a)
                case ExprStmt(e): visit(e)
                case BinOp(_, l, r): 
                    visit(l)
                    visit(r)
                case UnaryOp(_, o): visit(o)
                case ArrAccess(_, idx): visit(idx)
                case Assign(t, v): 
                    visit(t)
                    visit(v)
                case IfStmt(c, t, e):
                    visit(c)
                    [visit(s) for s in t]
                    if e:
                        [visit(s) for s in e]
                case WhileStmt(c, b):
                    visit(c)
                    [visit(s) for s in b]
                case ReturnStmt(v): visit(v) if v else None
                case VarDecl(_, init, _): visit(init) if init else None
                case IrqDef(_, b): [visit(s) for s in b]
                case ArrayInit(elems):
                    for el in elems:
                        visit(el)

        for s in stmts:
            if not isinstance(s, FuncDef):
                visit(s)

        if "__get_num" in used:
            used.add("__get_char")
        return used

    def generate(self, stmts: list[ASTNode]) -> tuple[list[Instruction], list[tuple[int, int]], int, int]:
        self._collect_globals(stmts, True)
        self._collect_constants(stmts, True)
        self.used_funcs = self._get_used_functions(stmts)

        has_irq = any(isinstance(s, IrqDef) for s in stmts)
        if has_irq:
            self._gen_irq(stmts)

        if "__print_str" in self.used_funcs:
            self._gen_builtin_print_str()
        if "__print_num" in self.used_funcs:
            self._gen_builtin_print_num()
        if "__print_u" in self.used_funcs:
            self._gen_builtin_print_u()
        if "__get_char" in self.used_funcs:
            self._gen_builtin_get_char()
        if "__put_char" in self.used_funcs:
            self._gen_builtin_put_char()
        if "__get_num" in self.used_funcs:
            self._gen_builtin_get_num()

        for s in stmts:
            if isinstance(s, FuncDef) and not s.name.startswith("__") and s.name in self.used_funcs:
                self._gen_func(s)

        self.entry = self.addr
        self._emit(Opcode.LDI, SP, 0, STACK_START)
        self._emit(Opcode.MOV, FP, SP)

        for s in stmts:
            if isinstance(s, VarDecl) and not self._needs_io(s.init):
                if isinstance(s.init, (NumLit, ChrLit, StrLit)):
                    continue
                if isinstance(s.init, ArrayInit) and all(isinstance(el, (NumLit, ChrLit, StrLit)) for el in s.init.elements):
                    continue
                self._gen_stmt(s)

        if has_irq:
            self._emit(Opcode.STI)

        for s in stmts:
            if (isinstance(s, VarDecl) and self._needs_io(s.init)) or not isinstance(s, (FuncDef, IrqDef, VarDecl)):
                self._gen_stmt(s)

        self._emit(Opcode.HLT)
        self._resolve()

        data_section: list[tuple[int, int]] = []
        data_section.extend(self.const_inits)
        data_section.extend(self.global_inits)
        for addr, vals in self.str_inits:
            for i, v in enumerate(vals):
                data_section.append((addr + i, v))

        data_section.sort(key=lambda item: item[0])

        return self.code, data_section, self.entry, self.irq_addr

    def _gen_array_addr(self, name: str, idx: ASTNode, dest_reg: int) -> None:
        loc, val, size = self._get_var(name)

        offset_imm = 0
        actual_idx = idx
        if isinstance(idx, BinOp) and isinstance(idx.right, NumLit):
            if idx.op == "+":
                offset_imm = idx.right.value
                actual_idx = idx.left
            elif idx.op == "-":
                offset_imm = -idx.right.value
                actual_idx = idx.left

        self._gen_expr(actual_idx, dest_reg)

        if loc == VarLoc.GLOBAL and size > 1:
            total_offset = val + offset_imm
            if total_offset != 0:
                self._emit(Opcode.ADDI, dest_reg, 0, total_offset)
        else:
            next_reg = dest_reg + 1
            if next_reg <= R4:
                if size == 1:
                    self._emit_load_var(name, next_reg)
                else:
                    self._emit_var_addr(name, next_reg)

                if offset_imm > 0:
                    self._emit(Opcode.ADDI, dest_reg, 0, offset_imm)
                elif offset_imm < 0:
                    self._emit(Opcode.SUBI, dest_reg, 0, -offset_imm)

                self._emit(Opcode.ADD, dest_reg, next_reg)
            else:
                self._emit(Opcode.PUSH, dest_reg)
                if size == 1:
                    self._emit_load_var(name, dest_reg)
                else:
                    self._emit_var_addr(name, dest_reg)
                self._emit(Opcode.MOV, R5, dest_reg)
                self._emit(Opcode.POP, dest_reg)

                if offset_imm > 0:
                    self._emit(Opcode.ADDI, dest_reg, 0, offset_imm)
                elif offset_imm < 0:
                    self._emit(Opcode.SUBI, dest_reg, 0, -offset_imm)

                self._emit(Opcode.ADD, dest_reg, R5)

    def _gen_stmt(self, s: ASTNode) -> None:
        match s:
            case VarDecl(name, init, _):
                loc, val, _ = self._get_var(name)
                if isinstance(init, ArrayInit):
                    for i, item in enumerate(init.elements):
                        if not isinstance(item, (NumLit, ChrLit, StrLit)):
                            self._gen_expr(item, R0)
                            if loc == VarLoc.GLOBAL:
                                self._emit(Opcode.ST, R0, 0, val + i)
                            else:
                                self._emit_var_addr(name, R1)
                                if i == 1:
                                    self._emit(Opcode.INC, R1)
                                elif i > 0:
                                    self._emit(Opcode.ADDI, R1, 0, i)
                                self._emit(Opcode.ST_IND, R1, R0)
                    return

                if self.in_func:
                    self._gen_expr(init, R0)
                    if loc == VarLoc.GLOBAL:
                        self._emit(Opcode.ST, R0, 0, val)
                    else:
                        self._emit_var_addr(name, R1)
                        self._emit(Opcode.ST_IND, R1, R0)
                else:
                    self._gen_expr(init, R0)
                    self._emit(Opcode.ST, R0, 0, val)

            case Assign(target, value):
                self._gen_assign(target, value)

            case IfStmt(cond, then_b, else_b):
                self._gen_if(cond, then_b, else_b)

            case WhileStmt(cond, body):
                self._gen_while(cond, body)

            case ReturnStmt(val):
                if val:
                    self._gen_expr(val, R0)
                else:
                    self._load_const(R0, 0)

                if self.in_func:
                    self._emit(Opcode.MOV, SP, FP)
                    self._emit(Opcode.POP, FP)
                self._emit(Opcode.RET)

            case ExprStmt(expr):
                self._gen_expr(expr, R0)

    def _gen_assign(self, target: ASTNode, value: ASTNode) -> None:
        if isinstance(target, ArrAccess):
            self._gen_expr(value, R0)
            self._gen_array_addr(target.name, target.index, R1)
            self._emit(Opcode.ST_IND, R1, R0)

        elif isinstance(target, VarRef):
            self._gen_expr(value, R0)
            loc, val, _ = self._get_var(target.name)
            if loc == VarLoc.GLOBAL:
                self._emit(Opcode.ST, R0, 0, val)
            else:
                self._emit_var_addr(target.name, R1)
                self._emit(Opcode.ST_IND, R1, R0)

    def _gen_cond_jump(self, cond: ASTNode, false_lbl: str) -> None:
        if isinstance(cond, BinOp) and cond.op in self.CMP_OPS:
            left, right = cond.left, cond.right
            reg = R0

            if isinstance(right, VarRef) and self._get_var_loc_safe(right.name) == VarLoc.GLOBAL:
                self._gen_expr(left, reg)
                _, val, _ = self._get_var(right.name)
                self._emit(Opcode.CMPM, reg, 0, val)
            elif isinstance(right, (NumLit, ChrLit)):
                self._gen_expr(left, reg)
                self._emit(Opcode.CMPM, reg, 0, self._const(right.value))
            else:
                r_right = self._eval_two(left, right, reg)
                self._emit(Opcode.CMP, reg, r_right)

            inv_jmp = {
                "==": Opcode.JNZ, "!=": Opcode.JZ,
                "<": Opcode.JGE, ">": Opcode.JLE,
                "<=": Opcode.JG, ">=": Opcode.JL
            }
            self._jmp(inv_jmp[cond.op], false_lbl)
        else:
            self._gen_expr(cond, R0)
            self._emit(Opcode.CMPM, R0, 0, self._const(0))
            self._jmp(Opcode.JZ, false_lbl)

    def _gen_if(self, cond: ASTNode, then_b: list[ASTNode], else_b: list[ASTNode] | None) -> None:
        l_else = self._lbl("ife")
        l_end = self._lbl("ifd")

        self._gen_cond_jump(cond, l_else)

        for st in then_b:
            self._gen_stmt(st)

        if else_b:
            self._jmp(Opcode.JMP, l_end)
            self._jlabel(l_else)
            for st in else_b:
                self._gen_stmt(st)
            self._jlabel(l_end)
        else:
            self._jlabel(l_else)

    def _gen_while(self, cond: ASTNode, body: list[ASTNode]) -> None:
        l_start = self._lbl("whs")
        l_end = self._lbl("whe")

        self._jlabel(l_start)
        self._gen_cond_jump(cond, l_end)

        for st in body:
            self._gen_stmt(st)
        self._jmp(Opcode.JMP, l_start)
        self._jlabel(l_end)

    def _eval_two(self, first: ASTNode, second: ASTNode, reg: int) -> int:
        self._gen_expr(first, reg)
        next_reg = reg + 1

        if next_reg <= R4:
            self._gen_expr(second, next_reg)
            return next_reg
        else:
            self._emit(Opcode.PUSH, reg)
            self._gen_expr(second, reg)
            self._emit(Opcode.MOV, R5, reg)
            self._emit(Opcode.POP, reg)
            return R5

    def _gen_expr(self, e: ASTNode, reg: int) -> None:
        match e:
            case NumLit(v) | ChrLit(v): self._load_const(reg, v)
            case StrLit(v): self._load_const(reg, self._str(v))
            case VarRef(n): self._emit_load_var(n, reg)
            case BinOp(op, l, r): self._gen_binop(op, l, r, reg)
            case UnaryOp(op, x): self._gen_unary(op, x, reg)
            case Call(n, args): self._gen_call(n, args, reg)
            case ArrAccess(n, idx):
                self._gen_array_addr(n, idx, reg)
                self._emit(Opcode.LD_IND, reg, reg)

    def _collect_poly_terms(self, e: ASTNode) -> list[tuple[VarRef, VarRef]] | None:
        match e:
            case BinOp("+", l, r):
                left = self._collect_poly_terms(l)
                right = self._collect_poly_terms(r)
                if left is not None and right is not None:
                    return left + right
                return None
            case BinOp("*", VarRef() as l, VarRef() as r):
                return [(l, r)]
            case _:
                return None

    def _gen_binop(self, op: str, left: ASTNode, right: ASTNode, reg: int) -> None:
        if op in self.CMP_OPS:
            self._gen_cmp(op, left, right, reg)
            return

        if op == "+":
            terms = self._collect_poly_terms(BinOp(op, left, right))
            if terms and len(terms) >= 2:
                if all(self._get_var_loc_safe(c.name) == VarLoc.GLOBAL and
                       self._get_var_loc_safe(x.name) == VarLoc.GLOBAL for c, x in terms):

                    self._load_const(reg, 0)
                    pairs = tuple(
                        (self._get_var(c.name)[1], self._get_var(x.name)[1]) for c, x in terms)
                    self._emit(Opcode.POLY, reg, 0, len(pairs), pairs=pairs)
                    return

        if isinstance(right, VarRef) and op in self.CISC_MEM:
            loc, val, _ = self._get_var(right.name)
            if loc == VarLoc.GLOBAL:
                self._gen_expr(left, reg)
                self._emit(self.CISC_MEM[op], reg, 0, val)
                return

        if isinstance(right, (NumLit, ChrLit)):
            self._gen_expr(left, reg)
            if op == "+":
                if right.value == 1:
                    self._emit(Opcode.INC, reg)
                else:
                    self._emit(Opcode.ADDI, reg, 0, right.value)
                return
            if op == "-":
                if right.value == 1:
                    self._emit(Opcode.DEC, reg)
                else:
                    self._emit(Opcode.SUBI, reg, 0, right.value)
                return
            if op in self.CISC_MEM:
                self._emit(self.CISC_MEM[op], reg, 0, self._const(right.value))
                return

        r_right = self._eval_two(left, right, reg)
        if op in self.ARITH_OPS:
            self._emit(self.ARITH_OPS[op], reg, r_right)

    def _gen_cmp(self, op: str, left: ASTNode, right: ASTNode, reg: int) -> None:
        if isinstance(right, VarRef) and self._get_var_loc_safe(right.name) == VarLoc.GLOBAL:
            self._gen_expr(left, reg)
            _, val, _ = self._get_var(right.name)
            self._emit(Opcode.CMPM, reg, 0, val)
        elif isinstance(right, (NumLit, ChrLit)):
            self._gen_expr(left, reg)
            self._emit(Opcode.CMPM, reg, 0, self._const(right.value))
        else:
            r_right = self._eval_two(left, right, reg)
            self._emit(Opcode.CMP, reg, r_right)

        l_end = self._lbl("cb")
        self._emit(Opcode.LDI, reg, 0, 1)
        self._jmp(self.CMP_DIRECT[op], l_end)
        self._emit(Opcode.LDI, reg, 0, 0)
        self._jlabel(l_end)

    def _gen_unary(self, op: str, operand: ASTNode, reg: int) -> None:
        self._gen_expr(operand, reg)
        if op == "-":
            self._emit(Opcode.NOT, reg)
            self._emit(Opcode.INC, reg)
        elif op == "!":
            self._emit(Opcode.CMPM, reg, 0, self._const(0))
            l_end = self._lbl("un")
            self._emit(Opcode.LDI, reg, 0, 0)
            self._jmp(Opcode.JNZ, l_end)
            self._emit(Opcode.LDI, reg, 0, 1)
            self._jlabel(l_end)

    def _resolve_vec_addr(self, arg: ASTNode) -> int:
        match arg:
            case VarRef(name):
                loc, val, _ = self._get_var(name)
                return val if loc == VarLoc.GLOBAL else 0
            case NumLit(v): return v
            case _: return 0

    def _gen_call(self, name: str, args: list[ASTNode], reg: int) -> None:
        builtins = {"getc": "__get_char",
                    "getnum": "__get_num", "putc": "__put_char"}
        vec_ops = {"vadd": Opcode.VADD, "vsub": Opcode.VSUB,
                   "vmul": Opcode.VMUL, "vdiv": Opcode.VDIV, "vcmp": Opcode.VCMP}

        if name in vec_ops:
            if reg == 0:
                if len(args) == 2:
                    r_right = self._eval_two(args[0], args[1], 0)
                    if r_right != 1:
                        self._emit(Opcode.MOV, 1, r_right)
                self._emit(vec_ops[name], 0, 1)
            else:
                self._emit(Opcode.PUSH, 0)
                self._emit(Opcode.PUSH, 1)
                if len(args) == 2:
                    r_right = self._eval_two(args[0], args[1], 0)
                    if r_right != 1:
                        self._emit(Opcode.MOV, 1, r_right)
                self._emit(vec_ops[name], 0, 1)
                self._emit(Opcode.POP, 1)
                self._emit(Opcode.POP, 0)
            return

        if name == "vload":
            vn = args[0].value if isinstance(args[0], NumLit) else 0
            self._emit(Opcode.VLOAD, vn, 0, self._resolve_vec_addr(args[1]))
            return

        if name == "vstore":
            addr = self._resolve_vec_addr(args[0])
            vn = args[1].value if isinstance(args[1], NumLit) else 0
            self._emit(Opcode.VSTORE, vn, 0, addr)
            return

        if name == "vset":
            vn = args[0].value if isinstance(args[0], NumLit) else 0
            if isinstance(args[1], NumLit):
                self._emit(Opcode.VSET, vn, 0, args[1].value)
            else:
                self._gen_expr(args[1], reg)
                for i in range(4):
                    self._emit(Opcode.VSCALAR, vn, reg, i)
            return

        if name == "vscalar":
            vn = args[0].value if isinstance(args[0], NumLit) else 0
            idx = args[2].value if len(
                args) > 2 and isinstance(args[2], NumLit) else 0
            self._gen_expr(args[1], reg)
            self._emit(Opcode.VSCALAR, vn, reg, idx)
            return

        if name == "vget":
            vn = args[0].value if isinstance(args[0], NumLit) else 0
            idx = args[1].value if isinstance(args[1], NumLit) else 0
            self._emit(Opcode.VGET, reg, vn, idx)
            return

        if name == "carry":
            l_end = self._lbl("carr")
            self._emit(Opcode.LDI, reg, 0, 1)
            self._jmp(Opcode.JC, l_end)
            self._emit(Opcode.LDI, reg, 0, 0)
            self._jlabel(l_end)
            return

        for i in range(reg):
            self._emit(Opcode.PUSH, i)

        fname = builtins.get(name, name)
        for arg in args:
            if isinstance(arg, VarRef) and self._get_var(arg.name)[2] > 1:
                self._emit_var_addr(arg.name, R0)
            else:
                self._gen_expr(arg, R0)
            self._emit(Opcode.PUSH, R0)

        self._jmp(Opcode.CALL, fname)
        if len(args) > 0:
            self._emit(Opcode.ADDI, SP, 0, len(args))

        if reg != R0:
            self._emit(Opcode.MOV, reg, R0)

        for i in reversed(range(reg)):
            self._emit(Opcode.POP, i)

    def _gen_func(self, f: FuncDef) -> None:
        self.funcs[f.name] = self.addr
        self._jlabel(f.name)
        self.in_func = True
        self.locals.clear()
        self.static_locals.clear()
        self.local_offset = 0

        N = len(f.params)
        for i, p in enumerate(f.params):
            self.locals[p] = (1 + (N - i), 1)

        self._collect_static_locals(f.body)

        self._emit(Opcode.PUSH, FP)
        self._emit(Opcode.MOV, FP, SP)
        if self.local_offset > 0:
            self._emit(Opcode.SUBI, SP, 0, self.local_offset)

        for s in f.body:
            self._gen_stmt(s)

        if not f.body or not isinstance(f.body[-1], ReturnStmt):
            self._load_const(R0, 0)
            self._emit(Opcode.MOV, SP, FP)
            self._emit(Opcode.POP, FP)
            self._emit(Opcode.RET)

        self.in_func = False

    def _gen_irq(self, stmts: list[ASTNode]) -> None:
        self.irq_addr = self.addr
        for r in [R0, R1, R2, R3, R4, R5, FP]:
            self._emit(Opcode.PUSH, r)

        self.in_func = True
        self.locals.clear()
        self.static_locals.clear()
        self.local_offset = 0

        for s in stmts:
            if isinstance(s, IrqDef):
                self._collect_static_locals(s.body)

        self._emit(Opcode.MOV, FP, SP)
        if self.local_offset > 0:
            self._emit(Opcode.SUBI, SP, 0, self.local_offset)

        for s in stmts:
            if isinstance(s, IrqDef):
                for st in s.body:
                    self._gen_stmt(st)

        self._emit(Opcode.MOV, SP, FP)
        self.in_func = False

        for r in [FP, R5, R4, R3, R2, R1, R0]:
            self._emit(Opcode.POP, r)
        self._emit(Opcode.IRET)

    def _gen_builtin_print_str(self) -> None:
        self.funcs["__print_str"] = self.addr
        self._jlabel("__print_str")
        self._emit(Opcode.MOV, R0, SP)
        self._emit(Opcode.INC, R0)
        self._emit(Opcode.LD_IND, R0, R0)

        ll, ld, lskip = self._lbl("psl"), self._lbl("psd"), self._lbl("psskip")

        self._emit(Opcode.MOV, R1, R0)
        self._emit(Opcode.LD_IND, R2, R1)
        self._emit(Opcode.INC, R1)
        self._load_const(R4, 4)

        self._jlabel(ll)
        self._emit(Opcode.CMPM, R2, 0, self._const(0))
        self._jmp(Opcode.JZ, ld)

        self._emit(Opcode.CMPM, R4, 0, self._const(4))
        self._jmp(Opcode.JNZ, lskip)

        self._emit(Opcode.LD_IND, R3, R1)
        self._emit(Opcode.INC, R1)
        self._load_const(R4, 0)

        self._jlabel(lskip)
        self._emit(Opcode.MOV, R5, R3)
        self._load_const(R0, 256)
        self._emit(Opcode.MOD, R5, R0)
        self._emit(Opcode.ST, R5, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.DIV, R3, R0)

        self._emit(Opcode.INC, R4)
        self._emit(Opcode.DEC, R2)
        self._jmp(Opcode.JMP, ll)

        self._jlabel(ld)
        self._emit(Opcode.RET)

    def _gen_builtin_print_num(self) -> None:
        self.funcs["__print_num"] = self.addr
        self._jlabel("__print_num")
        self._emit(Opcode.MOV, R0, SP)
        self._emit(Opcode.INC, R0)
        self._emit(Opcode.LD_IND, R0, R0)

        lp, lnz, le, lo, ld = self._lbl("pnp"), self._lbl(
            "pnn"), self._lbl("pne"), self._lbl("pno"), self._lbl("pnd")

        self._emit(Opcode.CMPM, R0, 0, self._const(0))
        self._jmp(Opcode.JGE, lp)
        self._emit(Opcode.PUSH, R0)
        self._load_const(R1, 45)
        self._emit(Opcode.ST, R1, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.POP, R0)
        self._emit(Opcode.NOT, R0)
        self._emit(Opcode.INC, R0)
        self._jlabel(lp)
        self._emit(Opcode.CMPM, R0, 0, self._const(0))
        self._jmp(Opcode.JNZ, lnz)
        self._load_const(R1, 48)
        self._emit(Opcode.ST, R1, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.RET)
        self._jlabel(lnz)
        self._load_const(R5, 0)
        self._jlabel(le)
        self._emit(Opcode.CMPM, R0, 0, self._const(0))
        self._jmp(Opcode.JZ, lo)
        self._emit(Opcode.MOV, R1, R0)
        self._load_const(R3, 10)
        self._emit(Opcode.MOD, R1, R3)
        self._emit(Opcode.ADDM, R1, 0, self._const(48))
        self._emit(Opcode.PUSH, R1)
        self._emit(Opcode.INC, R5)
        self._load_const(R3, 10)
        self._emit(Opcode.DIV, R0, R3)
        self._jmp(Opcode.JMP, le)
        self._jlabel(lo)
        self._emit(Opcode.CMPM, R5, 0, self._const(0))
        self._jmp(Opcode.JZ, ld)
        self._emit(Opcode.POP, R1)
        self._emit(Opcode.ST, R1, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.DEC, R5)
        self._jmp(Opcode.JMP, lo)
        self._jlabel(ld)
        self._emit(Opcode.RET)

    def _gen_builtin_print_u(self) -> None:
        self.funcs["__print_u"] = self.addr
        self._jlabel("__print_u")
        self._emit(Opcode.MOV, R0, SP)
        self._emit(Opcode.INC, R0)
        self._emit(Opcode.LD_IND, R0, R0)

        lnz = self._lbl("punz")
        self._emit(Opcode.CMPM, R0, 0, self._const(0))
        self._jmp(Opcode.JNZ, lnz)
        self._load_const(R1, 48)
        self._emit(Opcode.ST, R1, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.RET)
        self._jlabel(lnz)
        self._load_const(R4, 0)
        powers = [1000000000, 100000000, 10000000,
                  1000000, 100000, 10000, 1000, 100, 10, 1]
        for p in powers:
            ls, ds, sk, nx = self._lbl("pus"), self._lbl(
                "pud"), self._lbl("pusk"), self._lbl("punx")
            self._load_const(R5, 0)
            self._jlabel(ls)
            self._load_const(R1, p)
            self._emit(Opcode.CMP, R0, R1)
            self._jmp(Opcode.JC, ds)
            self._emit(Opcode.SUB, R0, R1)
            self._emit(Opcode.INC, R5)
            self._jmp(Opcode.JMP, ls)
            self._jlabel(ds)
            self._emit(Opcode.CMPM, R5, 0, self._const(0))
            self._jmp(Opcode.JZ, sk)
            self._load_const(R4, 1)
            self._jlabel(sk)
            self._emit(Opcode.CMPM, R4, 0, self._const(0))
            self._jmp(Opcode.JZ, nx)
            self._emit(Opcode.ADDM, R5, 0, self._const(48))
            self._emit(Opcode.ST, R5, 0, OUTPUT_DATA_ADDR)
            self._jlabel(nx)
        self._emit(Opcode.RET)

    def _gen_builtin_get_char(self) -> None:
        self.funcs["__get_char"] = self.addr
        self._jlabel("__get_char")
        self._emit(Opcode.LD, R0, 0, INPUT_DATA_ADDR)
        self._emit(Opcode.RET)

    def _gen_builtin_put_char(self) -> None:
        self.funcs["__put_char"] = self.addr
        self._jlabel("__put_char")
        self._emit(Opcode.MOV, R0, SP)
        self._emit(Opcode.INC, R0)
        self._emit(Opcode.LD_IND, R0, R0)
        self._emit(Opcode.ST, R0, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.RET)

    def _gen_builtin_get_num(self) -> None:
        self.funcs["__get_num"] = self.addr
        self._jlabel("__get_num")
        ls, la, ld = self._lbl("gns"), self._lbl("gna"), self._lbl("gnd")

        self._jlabel(ls)
        self._jmp(Opcode.CALL, "__get_char")
        self._emit(Opcode.CMPM, R0, 0, self._const(48))
        self._jmp(Opcode.JL, ls)
        self._emit(Opcode.CMPM, R0, 0, self._const(57))
        self._jmp(Opcode.JG, ls)
        self._emit(Opcode.SUBM, R0, 0, self._const(48))
        self._load_const(R1, 0)
        self._jlabel(la)
        self._load_const(R2, 10)
        self._emit(Opcode.MUL, R1, R2)
        self._emit(Opcode.ADD, R1, R0)
        self._emit(Opcode.PUSH, R1)
        self._jmp(Opcode.CALL, "__get_char")
        self._emit(Opcode.POP, R1)
        self._emit(Opcode.CMPM, R0, 0, self._const(48))
        self._jmp(Opcode.JL, ld)
        self._emit(Opcode.CMPM, R0, 0, self._const(57))
        self._jmp(Opcode.JG, ld)
        self._emit(Opcode.SUBM, R0, 0, self._const(48))
        self._jmp(Opcode.JMP, la)
        self._jlabel(ld)
        self._emit(Opcode.MOV, R0, R1)
        self._emit(Opcode.RET)


def translate(source: str) -> tuple[list[Instruction], list[tuple[int, int]], int, int, str]:
    lexer = Lexer(source)
    parser = Parser(lexer.tokens)
    ast = parser.parse()
    ast_str = dump_ast(ast)
    gen = CodeGen()
    code, data_section, entry, irq = gen.generate(ast)
    return code, data_section, entry, irq, ast_str


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: translator.py <source> <output>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        source = f.read()

    source_loc = len([line for line in source.split("\n") if line.strip()])
    code, data_section, entry, irq, ast_str = translate(source)

    write_binary(code, data_section, entry, irq, sys.argv[2])

    base = sys.argv[2].rsplit(".", 1)[0]
    write_debug(code, data_section, entry, irq, base + ".asm")

    with open(base + ".ast", "w", encoding="utf-8") as f:
        f.write(ast_str)

    print(f"source LoC: {source_loc} code instr: {len(code)}")


if __name__ == "__main__":
    main()
