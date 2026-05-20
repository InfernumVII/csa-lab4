"""Translator: Alg language -> binary machine code for CISC processor."""

import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

from src.isa import (
    INPUT_AVAIL_ADDR,
    INPUT_BUF_READ_IDX,
    INPUT_BUF_WRITE_IDX,
    INPUT_BUFFER_BASE,
    INPUT_DATA_ADDR,
    OUTPUT_DATA_ADDR,
    Instruction,
    Opcode,
    write_binary,
    write_debug,
)

DATA_ALLOC_START = INPUT_BUFFER_BASE + 256


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
TWO_CHAR_OPS = {"==", "!=", "<=", ">=", "&&", "||"}
ONE_CHAR_OPS = {"<", ">", "=", "!", "+", "-", "*", "/", "%", "(", ")", "{", "}", "[", "]", ";", ","}
TWO_CHAR_MAP = {"==": TT.EQ, "!=": TT.NEQ, "<=": TT.LE, ">=": TT.GE, "&&": TT.AND, "||": TT.OR}
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
            if c in " \t\r\n":
                self.pos += 1
            elif c == "/" and self.pos + 1 < len(self.src) and self.src[self.pos + 1] == "/":
                while self.pos < len(self.src) and self.src[self.pos] != "\n":
                    self.pos += 1
            elif c.isdigit():
                s = self.pos
                while self.pos < len(self.src) and self.src[self.pos].isdigit():
                    self.pos += 1
                self.tokens.append(Token(TT.NUM, int(self.src[s:self.pos])))
            elif c == '"':
                self.pos += 1
                chars: list[str] = []
                while self.pos < len(self.src) and self.src[self.pos] != '"':
                    if self.src[self.pos] == "\\":
                        self.pos += 1
                        chars.append(ESCAPE_MAP.get(self.src[self.pos], self.src[self.pos]))
                    else:
                        chars.append(self.src[self.pos])
                    self.pos += 1
                self.pos += 1
                self.tokens.append(Token(TT.STR, "".join(chars)))
            elif c == "'":
                self.pos += 1
                if self.src[self.pos] == "\\":
                    self.pos += 1
                    val = CHAR_ESCAPE.get(self.src[self.pos], ord(self.src[self.pos]))
                else:
                    val = ord(self.src[self.pos])
                self.pos += 2
                self.tokens.append(Token(TT.CHR, val))
            elif c.isalpha() or c == "_":
                s = self.pos
                while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == "_"):
                    self.pos += 1
                self.tokens.append(Token(TT.IDENT, self.src[s:self.pos]))
            else:
                for op in ["==", "!=", "<=", ">=", "&&", "||"]:
                    if self.src[self.pos:self.pos + 2] == op:
                        self.tokens.append(Token(TWO_CHAR_MAP[op], op))
                        self.pos += 2
                        break
                else:
                    if c in ONE_CHAR_MAP:
                        self.tokens.append(Token(ONE_CHAR_MAP[c], c))
                        self.pos += 1
                    else:
                        raise SyntaxError(f"Unexpected char: {c!r} at {self.pos}")
        self.tokens.append(Token(TT.EOF, ""))


# --- AST ---

@dataclass
class NumLit:
    value: int

@dataclass
class StrLit:
    value: str

@dataclass
class ChrLit:
    value: int

@dataclass
class VarRef:
    name: str

@dataclass
class ArrAccess:
    name: str
    index: object

@dataclass
class BinOp:
    op: str
    left: object
    right: object

@dataclass
class UnaryOp:
    op: str
    operand: object

@dataclass
class Call:
    name: str
    args: list[object]

@dataclass
class VarDecl:
    name: str
    init: object
    arr_size: int | None = None

@dataclass
class Assign:
    target: object
    value: object

@dataclass
class IfStmt:
    cond: object
    then_body: list[object]
    else_body: list[object] | None = None

@dataclass
class WhileStmt:
    cond: object
    body: list[object]

@dataclass
class FuncDef:
    name: str
    params: list[str]
    body: list[object]

@dataclass
class ReturnStmt:
    value: object | None = None

@dataclass
class ExprStmt:
    expr: object

@dataclass
class IrqDef:
    name: str
    body: list[object]


def ast_to_str(node: object, indent: int = 0) -> str:
    p = "  " * indent
    if isinstance(node, NumLit):
        return f"{p}NumLit({node.value})"
    if isinstance(node, StrLit):
        return f'{p}StrLit("{node.value}")'
    if isinstance(node, ChrLit):
        return f"{p}ChrLit({node.value!r})"
    if isinstance(node, VarRef):
        return f"{p}VarRef({node.name})"
    if isinstance(node, ArrAccess):
        return f"{p}ArrAccess({node.name}):\n{ast_to_str(node.index, indent+1)}"
    if isinstance(node, BinOp):
        return f"{p}BinOp({node.op}):\n{ast_to_str(node.left, indent+1)}\n{ast_to_str(node.right, indent+1)}"
    if isinstance(node, UnaryOp):
        return f"{p}UnaryOp({node.op}):\n{ast_to_str(node.operand, indent+1)}"
    if isinstance(node, Call):
        a = "\n".join(ast_to_str(x, indent+1) for x in node.args)
        return f"{p}Call({node.name}):\n{a}" if a else f"{p}Call({node.name})"
    if isinstance(node, VarDecl):
        s = f"{p}VarDecl({node.name}"
        if node.arr_size is not None:
            s += f"[{node.arr_size}]"
        return s + f"):\n{ast_to_str(node.init, indent+1)}"
    if isinstance(node, Assign):
        return f"{p}Assign:\n{ast_to_str(node.target, indent+1)}\n{ast_to_str(node.value, indent+1)}"
    if isinstance(node, IfStmt):
        s = f"{p}IfStmt:\n{ast_to_str(node.cond, indent+1)}\n{p}  then:\n"
        s += "\n".join(ast_to_str(x, indent+2) for x in node.then_body)
        if node.else_body:
            s += f"\n{p}  else:\n" + "\n".join(ast_to_str(x, indent+2) for x in node.else_body)
        return s
    if isinstance(node, WhileStmt):
        s = f"{p}WhileStmt:\n{ast_to_str(node.cond, indent+1)}\n"
        s += "\n".join(ast_to_str(x, indent+1) for x in node.body)
        return s
    if isinstance(node, FuncDef):
        s = f"{p}FuncDef({node.name}, {node.params}):\n"
        s += "\n".join(ast_to_str(x, indent+1) for x in node.body)
        return s
    if isinstance(node, ReturnStmt):
        return f"{p}ReturnStmt:\n{ast_to_str(node.value, indent+1)}" if node.value is not None else f"{p}ReturnStmt"
    if isinstance(node, ExprStmt):
        return f"{p}ExprStmt:\n{ast_to_str(node.expr, indent+1)}"
    if isinstance(node, IrqDef):
        return f"{p}IrqDef({node.name}):\n" + "\n".join(ast_to_str(x, indent+1) for x in node.body)
    if isinstance(node, list):
        return "\n".join(ast_to_str(x, indent) for x in node)
    return f"{p}{node!r}"


# --- Parser ---

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, tt: TT) -> Token:
        t = self._cur()
        if t.type != tt:
            raise SyntaxError(f"Expected {tt}, got {t.type} ({t.value!r})")
        self.pos += 1
        return t

    def _is(self, tt: TT) -> bool:
        return self._cur().type == tt

    def _is_kw(self, kw: str) -> bool:
        return self._cur().type == TT.IDENT and self._cur().value == kw

    def parse(self) -> list[object]:
        stmts = self._stmts()
        self._eat(TT.EOF)
        return stmts

    def _stmts(self) -> list[object]:
        r: list[object] = []
        while not self._is(TT.RBRACE) and not self._is(TT.EOF):
            r.append(self._stmt())
        return r

    def _stmt(self) -> object:
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
            self.pos += 1
            self._eat(TT.LPAREN)
            e = self._expr()
            self._eat(TT.RPAREN)
            self._eat(TT.SEMI)
            return ExprStmt(Call("__print_str", [e]))
        if self._is_kw("printnum"):
            self.pos += 1
            self._eat(TT.LPAREN)
            e = self._expr()
            self._eat(TT.RPAREN)
            self._eat(TT.SEMI)
            return ExprStmt(Call("__print_num", [e]))
        if self._is_kw("printu"):
            self.pos += 1
            self._eat(TT.LPAREN)
            e = self._expr()
            self._eat(TT.RPAREN)
            self._eat(TT.SEMI)
            return ExprStmt(Call("__print_u", [e]))
        if self._is_kw("putc"):
            self.pos += 1
            self._eat(TT.LPAREN)
            e = self._expr()
            self._eat(TT.RPAREN)
            self._eat(TT.SEMI)
            return ExprStmt(Call("__put_char", [e]))
        if self._is_kw("irq"):
            self.pos += 1
            name = str(self._eat(TT.IDENT).value)
            return IrqDef(name, self._block())
        e = self._expr()
        if self._is(TT.ASSIGN):
            self._eat(TT.ASSIGN)
            v = self._expr()
            self._eat(TT.SEMI)
            return Assign(e, v)
        self._eat(TT.SEMI)
        return ExprStmt(e)

    def _var_decl(self) -> VarDecl:
        self.pos += 1
        name = str(self._eat(TT.IDENT).value)
        arr = None
        if self._is(TT.LBRACKET):
            self._eat(TT.LBRACKET)
            arr = int(self._eat(TT.NUM).value)
            self._eat(TT.RBRACKET)
        init: object = NumLit(0)
        if self._is(TT.ASSIGN):
            self._eat(TT.ASSIGN)
            init = self._expr()
        self._eat(TT.SEMI)
        return VarDecl(name, init, arr)

    def _if_stmt(self) -> IfStmt:
        self.pos += 1
        self._eat(TT.LPAREN)
        cond = self._expr()
        self._eat(TT.RPAREN)
        then = self._block()
        els = None
        if self._is_kw("else"):
            self.pos += 1
            els = self._block()
        return IfStmt(cond, then, els)

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
        v = None if self._is(TT.SEMI) else self._expr()
        self._eat(TT.SEMI)
        return ReturnStmt(v)

    def _block(self) -> list[object]:
        self._eat(TT.LBRACE)
        r = self._stmts()
        self._eat(TT.RBRACE)
        return r

    def _expr(self) -> object:
        return self._or()

    def _or(self) -> object:
        n = self._and()
        while self._is(TT.OR):
            self.pos += 1
            n = BinOp("||", n, self._and())
        return n

    def _and(self) -> object:
        n = self._eq()
        while self._is(TT.AND):
            self.pos += 1
            n = BinOp("&&", n, self._eq())
        return n

    def _eq(self) -> object:
        n = self._cmp()
        while self._cur().type in (TT.EQ, TT.NEQ):
            op = str(self._eat(self._cur().type).value)
            n = BinOp(op, n, self._cmp())
        return n

    def _cmp(self) -> object:
        n = self._add()
        while self._cur().type in (TT.LT, TT.GT, TT.LE, TT.GE):
            op = str(self._eat(self._cur().type).value)
            n = BinOp(op, n, self._add())
        return n

    def _add(self) -> object:
        n = self._mul()
        while self._cur().type in (TT.PLUS, TT.MINUS):
            op = str(self._eat(self._cur().type).value)
            n = BinOp(op, n, self._mul())
        return n

    def _mul(self) -> object:
        n = self._unary()
        while self._cur().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            op = str(self._eat(self._cur().type).value)
            n = BinOp(op, n, self._unary())
        return n

    def _unary(self) -> object:
        if self._is(TT.MINUS):
            self.pos += 1
            return UnaryOp("-", self._unary())
        if self._is(TT.NOT):
            self.pos += 1
            return UnaryOp("!", self._unary())
        return self._primary()

    def _primary(self) -> object:
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
                args: list[object] = []
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


# --- Code Generator ---
# Convention: ST [addr], Rn => _emit(Opcode.ST, Rn, 0, addr) -- reg1=register, imm=address

R0, R1, R2, R3, R4, R5, R6, R7 = 0, 1, 2, 3, 4, 5, 6, 7


class CodeGen:
    def __init__(self) -> None:
        self.code: list[Instruction] = []
        self.addr = 0
        self.data_addr = DATA_ALLOC_START
        self.labels: dict[str, int] = {}
        self.fixups: list[tuple[int, str]] = []
        self.vars: dict[str, int] = {}
        self.var_sizes: dict[str, int] = {}
        self.strings: dict[str, int] = {}
        self.funcs: dict[str, int] = {}
        self._n = 0
        self.irq_addr = 0
        self.entry = 0
        self.str_inits: list[tuple[int, list[int]]] = []

    def _lbl(self, p: str) -> str:
        self._n += 1
        return f"{p}_{self._n}"

    def _alloc(self, n: int = 1) -> int:
        a = self.data_addr
        self.data_addr += n
        return a

    def _var(self, name: str, size: int = 1) -> int:
        if name not in self.vars:
            self.vars[name] = self._alloc(size)
            self.var_sizes[name] = size
        return self.vars[name]

    def _str(self, s: str) -> int:
        if s not in self.strings:
            a = self._alloc(1 + len(s))
            self.strings[s] = a
            self.str_inits.append((a, [len(s)] + [ord(c) for c in s]))
        return self.strings[s]

    def _emit(self, op: Opcode, r1: int = 0, r2: int = 0, imm: int = 0, pairs: tuple[tuple[int, int], ...] = ()) -> None:
        self.code.append(Instruction(op, r1, r2, imm, pairs))
        self.addr += self.code[-1].word_count()

    def _jlabel(self, name: str) -> None:
        self.labels[name] = self.addr

    def _jmp(self, op: Opcode, lbl: str, r1: int = 0) -> None:
        idx = len(self.code)
        self._emit(op, r1, 0, 0)
        self.fixups.append((idx, lbl))

    def _resolve(self) -> None:
        for idx, lbl in self.fixups:
            t = self.labels[lbl]
            old = self.code[idx]
            self.code[idx] = Instruction(old.opcode, old.reg1, old.reg2, t)

    def _addr_of(self, name: str, scope: str = "") -> int:
        key = f"{scope}.{name}" if scope else name
        return self.vars.get(key, self.vars.get(name, self._var(name)))

    def _needs_io(self, e: object) -> bool:
        if isinstance(e, Call):
            if e.name in ("getc", "getnum"):
                return True
            for a in e.args:
                if self._needs_io(a):
                    return True
        if isinstance(e, BinOp):
            return self._needs_io(e.left) or self._needs_io(e.right)
        if isinstance(e, UnaryOp):
            return self._needs_io(e.operand)
        if isinstance(e, ArrAccess):
            return self._needs_io(e.index)
        return False

    def generate(self, stmts: list[object]) -> tuple[list[Instruction], int, int, list[tuple[int, list[int]]]]:
        self._collect(stmts)
        self._collect_strings(stmts)
        self._gen_irq(stmts)
        self._gen_print_str()
        self._gen_print_num()
        self._gen_print_u()
        self._gen_get_char()
        self._gen_put_char()
        self._gen_get_num()
        for s in stmts:
            if isinstance(s, FuncDef) and not s.name.startswith("__"):
                self._gen_func(s)
        self.entry = self.addr
        self._gen_str_init()
        for s in stmts:
            if isinstance(s, VarDecl) and not self._needs_io(s.init):
                self._stmt(s)
        self._emit(Opcode.STI)
        for s in stmts:
            if (isinstance(s, VarDecl) and self._needs_io(s.init)) or not isinstance(s, (FuncDef, IrqDef, VarDecl)):
                self._stmt(s)
        self._emit(Opcode.HLT)
        self._resolve()
        return self.code, self.entry, self.irq_addr, self.str_inits

    def _collect(self, stmts: list[object], scope: str = "") -> None:
        for s in stmts:
            if isinstance(s, VarDecl):
                k = f"{scope}.{s.name}" if scope else s.name
                self._var(k, s.arr_size or 1)
            elif isinstance(s, FuncDef):
                for p in s.params:
                    self._var(f"{s.name}.{p}")
                self._collect(s.body, s.name)
            elif isinstance(s, IfStmt):
                self._collect(s.then_body, scope)
                if s.else_body:
                    self._collect(s.else_body, scope)
            elif isinstance(s, (WhileStmt, IrqDef)):
                self._collect(s.body, scope)

    def _collect_strings(self, stmts: list[object], scope: str = "") -> None:
        for s in stmts:
            if isinstance(s, VarDecl):
                self._collect_str_expr(s.init, scope)
            elif isinstance(s, Assign):
                self._collect_str_expr(s.value, scope)
            elif isinstance(s, IfStmt):
                self._collect_str_expr(s.cond, scope)
                self._collect_strings(s.then_body, scope)
                if s.else_body:
                    self._collect_strings(s.else_body, scope)
            elif isinstance(s, WhileStmt):
                self._collect_str_expr(s.cond, scope)
                self._collect_strings(s.body, scope)
            elif isinstance(s, FuncDef):
                self._collect_strings(s.body, s.name)
            elif isinstance(s, ReturnStmt):
                if s.value is not None:
                    self._collect_str_expr(s.value, scope)
            elif isinstance(s, ExprStmt):
                self._collect_str_expr(s.expr, scope)
            elif isinstance(s, IrqDef):
                self._collect_strings(s.body, scope)

    def _collect_str_expr(self, e: object, scope: str = "") -> None:
        if isinstance(e, StrLit):
            self._str(e.value)
        elif isinstance(e, BinOp):
            self._collect_str_expr(e.left, scope)
            self._collect_str_expr(e.right, scope)
        elif isinstance(e, UnaryOp):
            self._collect_str_expr(e.operand, scope)
        elif isinstance(e, Call):
            for a in e.args:
                self._collect_str_expr(a, scope)
        elif isinstance(e, ArrAccess):
            self._collect_str_expr(e.index, scope)

    def _gen_str_init(self) -> None:
        for addr, vals in self.str_inits:
            for i, v in enumerate(vals):
                self._emit(Opcode.LDI, R0, 0, v)
                self._emit(Opcode.ST, R0, 0, addr + i)

    def _gen_irq(self, stmts: list[object]) -> None:
        self.irq_addr = self.addr
        self._emit(Opcode.PUSH, R0)
        self._emit(Opcode.PUSH, R1)
        self._emit(Opcode.PUSH, R2)
        self._emit(Opcode.LD, R0, 0, INPUT_DATA_ADDR)
        self._emit(Opcode.LD, R1, 0, INPUT_BUF_WRITE_IDX)
        self._emit(Opcode.MOV, R2, R1)
        self._emit(Opcode.ADDI, R2, 0, INPUT_BUFFER_BASE)
        self._emit(Opcode.ST_IND, R2, R0)
        self._emit(Opcode.INC, R1)
        self._emit(Opcode.ST, R1, 0, INPUT_BUF_WRITE_IDX)
        self._emit(Opcode.LDI, R0, 0, 0)
        self._emit(Opcode.ST, R0, 0, INPUT_AVAIL_ADDR)
        for s in stmts:
            if isinstance(s, IrqDef):
                for st in s.body:
                    if not isinstance(st, VarDecl):
                        self._stmt(st)
        self._emit(Opcode.POP, R2)
        self._emit(Opcode.POP, R1)
        self._emit(Opcode.POP, R0)
        self._emit(Opcode.IRET)

    def _gen_print_str(self) -> None:
        self.funcs["__print_str"] = self.addr
        self._jlabel("__print_str")
        ll = self._lbl("psl")
        ld = self._lbl("psd")
        self._emit(Opcode.PUSH, R0)
        self._emit(Opcode.LD_IND, R1, R0)
        self._emit(Opcode.INC, R0)
        self._jlabel(ll)
        self._emit(Opcode.CMPI, R1, 0, 0)
        self._jmp(Opcode.JZ, ld)
        self._emit(Opcode.LD_IND, R2, R0)
        self._emit(Opcode.ST, R2, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.INC, R0)
        self._emit(Opcode.DEC, R1)
        self._jmp(Opcode.JMP, ll)
        self._jlabel(ld)
        self._emit(Opcode.POP, R0)
        self._emit(Opcode.RET)

    def _gen_print_num(self) -> None:
        self.funcs["__print_num"] = self.addr
        self._jlabel("__print_num")
        lp = self._lbl("pnp")
        lnz = self._lbl("pnn")
        le = self._lbl("pne")
        lo = self._lbl("pno")
        ld = self._lbl("pnd")
        self._emit(Opcode.CMPI, R0, 0, 0)
        self._jmp(Opcode.JGE, lp)
        self._emit(Opcode.PUSH, R0)
        self._emit(Opcode.LDI, R1, 0, 45)
        self._emit(Opcode.ST, R1, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.POP, R0)
        self._emit(Opcode.NOT, R0)
        self._emit(Opcode.INC, R0)
        self._jlabel(lp)
        self._emit(Opcode.CMPI, R0, 0, 0)
        self._jmp(Opcode.JNZ, lnz)
        self._emit(Opcode.LDI, R1, 0, 48)
        self._emit(Opcode.ST, R1, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.RET)
        self._jlabel(lnz)
        self._emit(Opcode.LDI, R5, 0, 0)
        self._jlabel(le)
        self._emit(Opcode.CMPI, R0, 0, 0)
        self._jmp(Opcode.JZ, lo)
        self._emit(Opcode.MOV, R1, R0)
        self._emit(Opcode.LDI, R3, 0, 10)
        self._emit(Opcode.MOD, R1, R3)
        self._emit(Opcode.ADDI, R1, 0, 48)
        self._emit(Opcode.PUSH, R1)
        self._emit(Opcode.INC, R5)
        self._emit(Opcode.LDI, R3, 0, 10)
        self._emit(Opcode.DIV, R0, R3)
        self._jmp(Opcode.JMP, le)
        self._jlabel(lo)
        self._emit(Opcode.CMPI, R5, 0, 0)
        self._jmp(Opcode.JZ, ld)
        self._emit(Opcode.POP, R1)
        self._emit(Opcode.ST, R1, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.DEC, R5)
        self._jmp(Opcode.JMP, lo)
        self._jlabel(ld)
        self._emit(Opcode.RET)

    def _gen_print_u(self) -> None:
        self.funcs["__print_u"] = self.addr
        self._jlabel("__print_u")
        lnz = self._lbl("punz")
        self._emit(Opcode.CMPI, R0, 0, 0)
        self._jmp(Opcode.JNZ, lnz)
        self._emit(Opcode.LDI, R1, 0, 48)
        self._emit(Opcode.ST, R1, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.RET)
        self._jlabel(lnz)
        self._emit(Opcode.LDI, R4, 0, 0)
        powers = [1000000000, 100000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10, 1]
        for p in powers:
            ls = self._lbl("pus")
            ds = self._lbl("pud")
            sk = self._lbl("pusk")
            nx = self._lbl("punx")
            self._emit(Opcode.LDI, R5, 0, 0)
            self._jlabel(ls)
            self._emit(Opcode.LDI, R1, 0, p)
            self._emit(Opcode.CMP, R0, R1)
            self._jmp(Opcode.JC, ds)
            self._emit(Opcode.SUB, R0, R1)
            self._emit(Opcode.INC, R5)
            self._jmp(Opcode.JMP, ls)
            self._jlabel(ds)
            self._emit(Opcode.CMPI, R5, 0, 0)
            self._jmp(Opcode.JZ, sk)
            self._emit(Opcode.LDI, R4, 0, 1)
            self._jlabel(sk)
            self._emit(Opcode.CMPI, R4, 0, 0)
            self._jmp(Opcode.JZ, nx)
            self._emit(Opcode.ADDI, R5, 0, 48)
            self._emit(Opcode.ST, R5, 0, OUTPUT_DATA_ADDR)
            self._jlabel(nx)
        self._emit(Opcode.RET)

    def _gen_get_char(self) -> None:
        self.funcs["__get_char"] = self.addr
        self._jlabel("__get_char")
        ls = self._lbl("gcs")
        self._jlabel(ls)
        self._emit(Opcode.LD, R0, 0, INPUT_BUF_READ_IDX)
        self._emit(Opcode.LD, R2, 0, INPUT_BUF_WRITE_IDX)
        self._emit(Opcode.CMP, R0, R2)
        self._jmp(Opcode.JGE, ls)
        self._emit(Opcode.MOV, R2, R0)
        self._emit(Opcode.ADDI, R2, 0, INPUT_BUFFER_BASE)
        self._emit(Opcode.LD_IND, R0, R2)
        self._emit(Opcode.PUSH, R0)
        self._emit(Opcode.LD, R0, 0, INPUT_BUF_READ_IDX)
        self._emit(Opcode.INC, R0)
        self._emit(Opcode.ST, R0, 0, INPUT_BUF_READ_IDX)
        self._emit(Opcode.POP, R0)
        self._emit(Opcode.RET)

    def _gen_put_char(self) -> None:
        self.funcs["__put_char"] = self.addr
        self._jlabel("__put_char")
        self._emit(Opcode.ST, R0, 0, OUTPUT_DATA_ADDR)
        self._emit(Opcode.RET)

    def _gen_get_num(self) -> None:
        self.funcs["__get_num"] = self.addr
        self._jlabel("__get_num")
        ls = self._lbl("gns")
        la = self._lbl("gna")
        ld = self._lbl("gnd")
        self._jlabel(ls)
        self._jmp(Opcode.CALL, "__get_char")
        self._emit(Opcode.CMPI, R0, 0, 48)
        self._jmp(Opcode.JL, ls)
        self._emit(Opcode.CMPI, R0, 0, 57)
        self._jmp(Opcode.JG, ls)
        self._emit(Opcode.SUBI, R0, 0, 48)
        self._emit(Opcode.LDI, R1, 0, 0)
        self._jlabel(la)
        self._emit(Opcode.LDI, R2, 0, 10)
        self._emit(Opcode.MUL, R1, R2)
        self._emit(Opcode.ADD, R1, R0)
        self._emit(Opcode.PUSH, R1)
        self._jmp(Opcode.CALL, "__get_char")
        self._emit(Opcode.POP, R1)
        self._emit(Opcode.CMPI, R0, 0, 48)
        self._jmp(Opcode.JL, ld)
        self._emit(Opcode.CMPI, R0, 0, 57)
        self._jmp(Opcode.JG, ld)
        self._emit(Opcode.SUBI, R0, 0, 48)
        self._jmp(Opcode.JMP, la)
        self._jlabel(ld)
        self._emit(Opcode.MOV, R0, R1)
        self._emit(Opcode.RET)

    def _gen_func(self, f: FuncDef) -> None:
        self.funcs[f.name] = self.addr
        self._jlabel(f.name)
        if f.params:
            a = self.vars[f"{f.name}.{f.params[0]}"]
            self._emit(Opcode.ST, R0, 0, a)
        if len(f.params) > 1:
            a2 = self.vars[f"{f.name}.{f.params[1]}"]
            self._emit(Opcode.ST, R2, 0, a2)
        for s in f.body:
            self._stmt(s, f.name)
        if not f.body or not isinstance(f.body[-1], ReturnStmt):
            self._emit(Opcode.LDI, R0, 0, 0)
            self._emit(Opcode.RET)

    def _stmt(self, s: object, scope: str = "") -> None:
        if isinstance(s, VarDecl):
            a = self._addr_of(s.name, scope)
            self._expr(s.init, scope)
            self._emit(Opcode.ST, R0, 0, a)
        elif isinstance(s, Assign):
            self._assign(s, scope)
        elif isinstance(s, IfStmt):
            self._if(s, scope)
        elif isinstance(s, WhileStmt):
            self._while(s, scope)
        elif isinstance(s, ReturnStmt):
            if s.value is not None:
                self._expr(s.value, scope)
            else:
                self._emit(Opcode.LDI, R0, 0, 0)
            self._emit(Opcode.RET)
        elif isinstance(s, ExprStmt):
            self._expr(s.expr, scope)
        elif isinstance(s, (FuncDef, IrqDef)):
            pass

    def _assign(self, s: Assign, scope: str) -> None:
        if isinstance(s.target, ArrAccess):
            self._expr(s.value, scope)
            self._emit(Opcode.PUSH, R0)
            self._expr(s.target.index, scope)
            resolved = self._addr_of(s.target.name, scope)
            key = f"{scope}.{s.target.name}" if scope else s.target.name
            if self.var_sizes.get(key, self.var_sizes.get(s.target.name, 1)) == 1:
                self._emit(Opcode.PUSH, R0)
                self._emit(Opcode.LD, R0, 0, resolved)
                self._emit(Opcode.MOV, R1, R0)
                self._emit(Opcode.POP, R0)
                self._emit(Opcode.ADD, R0, R1)
            else:
                self._emit(Opcode.ADDI, R0, 0, resolved)
            self._emit(Opcode.POP, R1)
            self._emit(Opcode.ST_IND, R0, R1)
        elif isinstance(s.target, VarRef):
            a = self._addr_of(s.target.name, scope)
            self._expr(s.value, scope)
            self._emit(Opcode.ST, R0, 0, a)

    def _if(self, s: IfStmt, scope: str) -> None:
        le = self._lbl("ife")
        lend = self._lbl("ifd")
        self._expr(s.cond, scope)
        self._emit(Opcode.CMPI, R0, 0, 0)
        self._jmp(Opcode.JZ, le)
        for st in s.then_body:
            self._stmt(st, scope)
        if s.else_body:
            self._jmp(Opcode.JMP, lend)
            self._jlabel(le)
            for st in s.else_body:
                self._stmt(st, scope)
            self._jlabel(lend)
        else:
            self._jlabel(le)

    def _while(self, s: WhileStmt, scope: str) -> None:
        ls = self._lbl("whs")
        le = self._lbl("whe")
        self._jlabel(ls)
        self._expr(s.cond, scope)
        self._emit(Opcode.CMPI, R0, 0, 0)
        self._jmp(Opcode.JZ, le)
        for st in s.body:
            self._stmt(st, scope)
        self._jmp(Opcode.JMP, ls)
        self._jlabel(le)

    def _expr(self, e: object, scope: str = "") -> None:
        if isinstance(e, (NumLit, ChrLit)):
            self._emit(Opcode.LDI, R0, 0, e.value)
        elif isinstance(e, StrLit):
            self._emit(Opcode.LDI, R0, 0, self._str(e.value))
        elif isinstance(e, VarRef):
            self._emit(Opcode.LD, R0, 0, self._addr_of(e.name, scope))
        elif isinstance(e, ArrAccess):
            self._expr(e.index, scope)
            resolved = self._addr_of(e.name, scope)
            key = f"{scope}.{e.name}" if scope else e.name
            if self.var_sizes.get(key, self.var_sizes.get(e.name, 1)) == 1:
                self._emit(Opcode.PUSH, R0)
                self._emit(Opcode.LD, R0, 0, resolved)
                self._emit(Opcode.MOV, R1, R0)
                self._emit(Opcode.POP, R0)
                self._emit(Opcode.ADD, R0, R1)
                self._emit(Opcode.LD_IND, R0, R0)
            else:
                self._emit(Opcode.ADDI, R0, 0, resolved)
                self._emit(Opcode.LD_IND, R0, R0)
        elif isinstance(e, BinOp):
            self._binop(e, scope)
        elif isinstance(e, UnaryOp):
            self._unary(e, scope)
        elif isinstance(e, Call):
            self._call(e, scope)

    CMP_OPS = frozenset({"==", "!=", "<", ">", "<=", ">="})
    ARITH_OPS: ClassVar[dict[str, Opcode]] = {
        "+": Opcode.ADD, "-": Opcode.SUB, "*": Opcode.MUL,
        "/": Opcode.DIV, "%": Opcode.MOD,
        "&&": Opcode.AND, "||": Opcode.OR,
    }
    CISC_MEM: ClassVar[dict[str, Opcode]] = {"+": Opcode.ADDM, "-": Opcode.SUBM, "*": Opcode.MULM}
    CISC_IMM: ClassVar[dict[str, Opcode]] = {"+": Opcode.ADDI, "-": Opcode.SUBI, "*": Opcode.MULI}
    CMP_DIRECT: ClassVar[dict[str, Opcode]] = {
        ">": Opcode.JG, "<": Opcode.JL, ">=": Opcode.JGE,
        "<=": Opcode.JLE, "==": Opcode.JZ, "!=": Opcode.JNZ,
    }

    def _collect_poly_terms(self, e: object) -> list[tuple[VarRef, VarRef]] | None:
        if isinstance(e, BinOp) and e.op == "+":
            left = self._collect_poly_terms(e.left)
            right = self._collect_poly_terms(e.right)
            if left is not None and right is not None:
                return left + right
            return None
        if isinstance(e, BinOp) and e.op == "*":
            if isinstance(e.left, VarRef) and isinstance(e.right, VarRef):
                return [(e.left, e.right)]
            return None
        return None

    def _binop(self, e: BinOp, scope: str) -> None:
        op = e.op
        if op in self.CMP_OPS:
            self._expr(e.left, scope)
            if isinstance(e.right, VarRef):
                self._emit(Opcode.CMPM, R0, 0, self._addr_of(e.right.name, scope))
            elif isinstance(e.right, NumLit):
                self._emit(Opcode.CMPI, R0, 0, e.right.value)
            else:
                self._emit(Opcode.PUSH, R0)
                self._expr(e.right, scope)
                self._emit(Opcode.MOV, R1, R0)
                self._emit(Opcode.POP, R0)
                self._emit(Opcode.CMP, R0, R1)
            self._cmp_bool(op)
            return

        if op == "+":
            terms = self._collect_poly_terms(e)
            if terms is not None and len(terms) >= 2:
                self._emit(Opcode.LDI, R0, 0, 0)
                pairs = tuple(
                    (self._addr_of(c.name, scope), self._addr_of(x.name, scope))
                    for c, x in terms
                )
                self._emit(Opcode.POLY, R0, 0, len(pairs), pairs=pairs)
                return

        if isinstance(e.right, VarRef) and op in self.CISC_MEM:
            self._expr(e.left, scope)
            self._emit(self.CISC_MEM[op], R0, 0, self._addr_of(e.right.name, scope))
            return

        if isinstance(e.right, NumLit) and op in self.CISC_IMM:
            self._expr(e.left, scope)
            self._emit(self.CISC_IMM[op], R0, 0, e.right.value)
            return

        self._expr(e.left, scope)
        self._emit(Opcode.PUSH, R0)
        self._expr(e.right, scope)
        self._emit(Opcode.MOV, R1, R0)
        self._emit(Opcode.POP, R0)
        if op in self.ARITH_OPS:
            self._emit(self.ARITH_OPS[op], R0, R1)

    def _cmp_bool(self, op: str) -> None:
        le = self._lbl("cb")
        self._emit(Opcode.LDI, R0, 0, 1)
        self._jmp(self.CMP_DIRECT[op], le)
        self._emit(Opcode.LDI, R0, 0, 0)
        self._jlabel(le)

    def _unary(self, e: UnaryOp, scope: str) -> None:
        self._expr(e.operand, scope)
        if e.op == "-":
            self._emit(Opcode.NOT, R0)
            self._emit(Opcode.INC, R0)
        elif e.op == "!":
            self._emit(Opcode.CMPI, R0, 0, 0)
            le = self._lbl("un")
            self._emit(Opcode.LDI, R0, 0, 0)
            self._jmp(Opcode.JNZ, le)
            self._emit(Opcode.LDI, R0, 0, 1)
            self._jlabel(le)

    def _resolve_vec_addr(self, arg: object, scope: str) -> int:
        if isinstance(arg, VarRef):
            return self._addr_of(arg.name, scope)
        if isinstance(arg, NumLit):
            return arg.value
        return 0

    def _call(self, e: Call, scope: str = "") -> None:
        builtins = {"getc": "__get_char", "getnum": "__get_num", "putc": "__put_char"}
        vec_ops = {"vadd", "vsub", "vmul", "vdiv", "vcmp"}
        vec_load = "vload"
        vec_store = "vstore"
        vec_set = "vset"
        vec_scalar = "vscalar"
        vec_get = "vget"

        if e.name in vec_ops:
            if e.args:
                self._expr(e.args[0], scope)
                self._emit(Opcode.MOV, R1, R0)
                self._expr(e.args[1], scope)
            op_map = {"vadd": Opcode.VADD, "vsub": Opcode.VSUB, "vmul": Opcode.VMUL, "vdiv": Opcode.VDIV, "vcmp": Opcode.VCMP}
            self._emit(op_map[e.name], 0, 1)
            return

        if e.name == vec_load:
            vn = int(e.args[0].value) if isinstance(e.args[0], NumLit) else 0
            addr = self._resolve_vec_addr(e.args[1], scope)
            self._emit(Opcode.VLOAD, vn, 0, addr)
            return

        if e.name == vec_store:
            addr = self._resolve_vec_addr(e.args[0], scope)
            vn = int(e.args[1].value) if isinstance(e.args[1], NumLit) else 0
            self._emit(Opcode.VSTORE, vn, 0, addr)
            return

        if e.name == vec_set:
            vn = int(e.args[0].value) if isinstance(e.args[0], NumLit) else 0
            if isinstance(e.args[1], NumLit):
                self._emit(Opcode.VSET, vn, 0, e.args[1].value)
            else:
                self._expr(e.args[1], scope)
                self._emit(Opcode.VSCALAR, vn, R0, 0)
                self._emit(Opcode.VSCALAR, vn, R0, 1)
                self._emit(Opcode.VSCALAR, vn, R0, 2)
                self._emit(Opcode.VSCALAR, vn, R0, 3)
            return

        if e.name == vec_scalar:
            vn = int(e.args[0].value) if isinstance(e.args[0], NumLit) else 0
            idx = int(e.args[2].value) if len(e.args) > 2 and isinstance(e.args[2], NumLit) else 0
            self._expr(e.args[1], scope)
            self._emit(Opcode.VSCALAR, vn, R0, idx)
            return

        if e.name == vec_get:
            vn = int(e.args[0].value) if isinstance(e.args[0], NumLit) else 0
            idx = int(e.args[1].value) if isinstance(e.args[1], NumLit) else 0
            self._emit(Opcode.VGET, R0, vn, idx)
            return

        if e.name == "carry":
            le = self._lbl("carr")
            self._emit(Opcode.LDI, R0, 0, 1)
            self._jmp(Opcode.JC, le)
            self._emit(Opcode.LDI, R0, 0, 0)
            self._jlabel(le)
            return

        fname = builtins.get(e.name, e.name)

        if e.args:
            first = e.args[0]
            if isinstance(first, VarRef) and self.var_sizes.get(first.name, 1) > 1:
                self._emit(Opcode.LDI, R0, 0, self._addr_of(first.name, scope))
            else:
                self._expr(first, scope)
            if len(e.args) > 1:
                self._emit(Opcode.MOV, R1, R0)
                self._expr(e.args[1], scope)
                self._emit(Opcode.MOV, R2, R0)
                self._emit(Opcode.MOV, R0, R1)

        self._jmp(Opcode.CALL, fname)


def translate(source: str) -> tuple[list[Instruction], int, int, str, list[tuple[int, list[int]]]]:
    lexer = Lexer(source)
    parser = Parser(lexer.tokens)
    ast = parser.parse()
    ast_str = ast_to_str(ast)
    gen = CodeGen()
    code, entry, irq, str_inits = gen.generate(ast)
    return code, entry, irq, ast_str, str_inits


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: translator.py <source> <output>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        source = f.read()
    source_loc = len([line for line in source.split("\n") if line.strip()])
    code, entry, irq, ast_str, _str_inits = translate(source)
    write_binary(code, entry, irq, sys.argv[2])
    base = sys.argv[2].rsplit(".", 1)[0]
    write_debug(code, entry, irq, base + ".asm")
    with open(base + ".ast", "w", encoding="utf-8") as f:
        f.write(ast_str)
    print(f"source LoC: {source_loc} code instr: {len(code)}")


if __name__ == "__main__":
    main()
