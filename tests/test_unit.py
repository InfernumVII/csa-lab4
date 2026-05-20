"""Unit tests for lexer, parser, encode/decode, DataPath, and ControlUnit."""

import tempfile

import pytest

from src.isa import (
    OUTPUT_DATA_ADDR,
    TICK_COUNTS,
    TWO_WORD_OPCODES,
    Instruction,
    Opcode,
    decode_instruction_from_words,
    decode_word0,
    encode,
    read_binary,
    sign_extend_16,
    sign_extend_32,
    to_unsigned32,
    write_binary,
)
from src.simulator import (
    FLAG_C,
    FLAG_Z,
    ControlUnit,
    DataPath,
    Simulator,
    parse_input,
    to_signed,
    to_u32,
    u32_add,
    u32_sub,
)
from src.translator import (
    TT,
    ArrAccess,
    Assign,
    BinOp,
    Call,
    ExprStmt,
    FuncDef,
    IfStmt,
    IrqDef,
    Lexer,
    NumLit,
    Parser,
    ReturnStmt,
    UnaryOp,
    VarDecl,
    WhileStmt,
    ast_to_str,
    translate,
)

# --- Lexer ---


class TestLexer:
    def test_numbers(self) -> None:
        tokens = Lexer("42 0 999").tokens
        assert tokens[0].type == TT.NUM and tokens[0].value == 42
        assert tokens[1].type == TT.NUM and tokens[1].value == 0
        assert tokens[2].type == TT.NUM and tokens[2].value == 999

    def test_identifiers(self) -> None:
        tokens = Lexer("foo bar_baz x1").tokens
        assert tokens[0].type == TT.IDENT and tokens[0].value == "foo"
        assert tokens[1].type == TT.IDENT and tokens[1].value == "bar_baz"
        assert tokens[2].type == TT.IDENT and tokens[2].value == "x1"

    def test_string_escapes(self) -> None:
        tokens = Lexer(r'"hello\nworld\t"').tokens
        assert tokens[0].type == TT.STR
        assert tokens[0].value == "hello\nworld\t"

    def test_string_double_quote_escape(self) -> None:
        tokens = Lexer(r'"say \"hi\""').tokens
        assert tokens[0].type == TT.STR
        assert tokens[0].value == 'say "hi"'

    def test_char_literal(self) -> None:
        tokens = Lexer("'A' '\\n' '0'").tokens
        assert tokens[0].type == TT.CHR and tokens[0].value == 65
        assert tokens[1].type == TT.CHR and tokens[1].value == 10
        assert tokens[2].type == TT.CHR and tokens[2].value == 48

    def test_single_char_operators(self) -> None:
        tokens = Lexer("+ - * / % < > = ! ( ) { } [ ] ; ,").tokens
        types = [t.type for t in tokens[:-1]]
        expected = [
            TT.PLUS, TT.MINUS, TT.STAR, TT.SLASH, TT.PERCENT,
            TT.LT, TT.GT, TT.ASSIGN, TT.NOT,
            TT.LPAREN, TT.RPAREN, TT.LBRACE, TT.RBRACE,
            TT.LBRACKET, TT.RBRACKET, TT.SEMI, TT.COMMA,
        ]
        assert types == expected

    def test_two_char_operators(self) -> None:
        tokens = Lexer("== != <= >= && ||").tokens
        types = [t.type for t in tokens[:-1]]
        assert types == [TT.EQ, TT.NEQ, TT.LE, TT.GE, TT.AND, TT.OR]

    def test_comment_stripping(self) -> None:
        tokens = Lexer("var x = 1; // comment\nvar y = 2;").tokens
        idents = [t.value for t in tokens if t.type == TT.IDENT]
        assert idents == ["var", "x", "var", "y"]

    def test_whitespace_ignored(self) -> None:
        tokens = Lexer("  1  \n  2  \t  3  ").tokens
        nums = [t.value for t in tokens if t.type == TT.NUM]
        assert nums == [1, 2, 3]

    def test_eof_token(self) -> None:
        tokens = Lexer("42").tokens
        assert tokens[-1].type == TT.EOF

    def test_invalid_char_raises(self) -> None:
        with pytest.raises(SyntaxError, match="Unexpected char"):
            Lexer("@invalid")


# --- Parser / AST ---


class TestParser:
    def test_var_decl_with_init(self) -> None:
        ast = Parser(Lexer("var x = 42;").tokens).parse()
        assert len(ast) == 1
        assert isinstance(ast[0], VarDecl) and ast[0].name == "x"
        assert isinstance(ast[0].init, NumLit) and ast[0].init.value == 42

    def test_var_decl_array(self) -> None:
        ast = Parser(Lexer("var buf[64];").tokens).parse()
        assert isinstance(ast[0], VarDecl) and ast[0].arr_size == 64

    def test_var_decl_no_init(self) -> None:
        ast = Parser(Lexer("var x;").tokens).parse()
        assert isinstance(ast[0], VarDecl) and isinstance(ast[0].init, NumLit) and ast[0].init.value == 0

    def test_if_else(self) -> None:
        ast = Parser(Lexer("if (1) { var x = 1; } else { var y = 2; }").tokens).parse()
        assert isinstance(ast[0], IfStmt)
        assert ast[0].else_body is not None
        assert len(ast[0].then_body) == 1
        assert len(ast[0].else_body) == 1

    def test_while(self) -> None:
        ast = Parser(Lexer("while (i < 10) { i = i + 1; }").tokens).parse()
        assert isinstance(ast[0], WhileStmt)

    def test_func_def(self) -> None:
        ast = Parser(Lexer("func add(a, b) { return a + b; }").tokens).parse()
        assert isinstance(ast[0], FuncDef)
        assert ast[0].name == "add"
        assert ast[0].params == ["a", "b"]

    def test_func_no_params(self) -> None:
        ast = Parser(Lexer("func f() { return 1; }").tokens).parse()
        assert isinstance(ast[0], FuncDef) and ast[0].params == []

    def test_return_void(self) -> None:
        ast = Parser(Lexer("return;").tokens).parse()
        assert isinstance(ast[0], ReturnStmt) and ast[0].value is None

    def test_operator_precedence(self) -> None:
        ast = Parser(Lexer("var r = 2 + 3 * 4;").tokens).parse()
        init = ast[0].init
        assert isinstance(init, BinOp) and init.op == "+"
        assert isinstance(init.right, BinOp) and init.right.op == "*"

    def test_paren_override_precedence(self) -> None:
        ast = Parser(Lexer("var r = (2 + 3) * 4;").tokens).parse()
        init = ast[0].init
        assert isinstance(init, BinOp) and init.op == "*"
        assert isinstance(init.left, BinOp) and init.left.op == "+"

    def test_comparison_produces_binop(self) -> None:
        ast = Parser(Lexer("var x = a < b;").tokens).parse()
        init = ast[0].init
        assert isinstance(init, BinOp) and init.op == "<"

    def test_logical_ops(self) -> None:
        ast = Parser(Lexer("var x = a && b || c;").tokens).parse()
        init = ast[0].init
        assert isinstance(init, BinOp) and init.op == "||"

    def test_unary_negation(self) -> None:
        ast = Parser(Lexer("var x = -5;").tokens).parse()
        init = ast[0].init
        assert isinstance(init, UnaryOp) and init.op == "-"

    def test_unary_not(self) -> None:
        ast = Parser(Lexer("var x = !0;").tokens).parse()
        init = ast[0].init
        assert isinstance(init, UnaryOp) and init.op == "!"

    def test_array_access(self) -> None:
        ast = Parser(Lexer("buf[i] = 5;").tokens).parse()
        assert isinstance(ast[0], Assign) and isinstance(ast[0].target, ArrAccess)

    def test_call_with_args(self) -> None:
        ast = Parser(Lexer("printnum(x);").tokens).parse()
        assert isinstance(ast[0], ExprStmt)
        assert isinstance(ast[0].expr, Call) and ast[0].expr.name == "__print_num"

    def test_irq_def(self) -> None:
        ast = Parser(Lexer("irq handler { count = count + 1; }").tokens).parse()
        assert isinstance(ast[0], IrqDef) and ast[0].name == "handler"

    def test_ast_to_str_readable(self) -> None:
        ast = Parser(Lexer("var x = 1 + 2;").tokens).parse()
        s = ast_to_str(ast)
        assert "VarDecl(x)" in s
        assert "BinOp(+)" in s
        assert "NumLit(1)" in s
        assert "NumLit(2)" in s


# --- Encode / Decode ---


class TestEncodeDecode:
    def test_one_word_roundtrip(self) -> None:
        instr = Instruction(Opcode.ADD, 1, 2, 0)
        words = encode(instr)
        assert len(words) == 1
        opcode, r1, r2, _imm16 = decode_word0(words[0])
        assert opcode == Opcode.ADD and r1 == 1 and r2 == 2

    def test_two_word_roundtrip(self) -> None:
        instr = Instruction(Opcode.LDI, 0, 0, 4294967295)
        words = encode(instr)
        assert len(words) == 2

    def test_binary_file_roundtrip(self) -> None:
        code = [
            Instruction(Opcode.HLT),
            Instruction(Opcode.LDI, 0, 0, 42),
            Instruction(Opcode.ADD, 0, 1, 0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/test.bin"
            write_binary(code, 5, 0, path)
            instructions, entry, irq = read_binary(path)
            assert entry == 5 and irq == 0
            assert len(instructions) == 3
            assert instructions[0].opcode == Opcode.HLT
            assert instructions[1].opcode == Opcode.LDI and instructions[1].imm == 42
            assert instructions[2].opcode == Opcode.ADD

    def test_all_opcodes_have_tick_count(self) -> None:
        for op in Opcode:
            if op == Opcode.POLY:
                continue
            assert int(op) in TICK_COUNTS, f"Missing tick count for {op.name}"

    def test_two_word_opcodes_set(self) -> None:
        assert Opcode.JMP in TWO_WORD_OPCODES
        assert Opcode.LD in TWO_WORD_OPCODES
        assert Opcode.VSET in TWO_WORD_OPCODES
        assert Opcode.ADD not in TWO_WORD_OPCODES

    def test_sign_extend_16(self) -> None:
        assert sign_extend_16(0xFFFF) == -1
        assert sign_extend_16(0x7FFF) == 0x7FFF
        assert sign_extend_16(0x8000) == -32768
        assert sign_extend_16(0) == 0

    def test_sign_extend_32(self) -> None:
        assert sign_extend_32(0xFFFFFFFF) == -1
        assert sign_extend_32(0x7FFFFFFF) == 0x7FFFFFFF
        assert sign_extend_32(0x80000000) == -2147483648
        assert sign_extend_32(0) == 0

    def test_to_unsigned32(self) -> None:
        assert to_unsigned32(-1) == 0xFFFFFFFF
        assert to_unsigned32(42) == 42
        assert to_unsigned32(0x100000000 + 5) == 5


# --- DataPath ---


class TestDataPath:
    def _make_dp(self, schedule: list[tuple[int, int]] | None = None) -> DataPath:
        return DataPath(schedule or [], [])

    def test_signal_wr_rd(self) -> None:
        dp = self._make_dp()
        dp.signal_wr(300, 42)
        assert dp.signal_rd(300) == 42

    def test_signal_wr_output(self) -> None:
        dp = self._make_dp()
        dp.signal_wr(OUTPUT_DATA_ADDR, 65)
        assert dp.output == ["A"]
        assert dp.signal_rd(OUTPUT_DATA_ADDR) == 0

    def test_signal_latch_reg(self) -> None:
        dp = self._make_dp()
        dp.signal_latch_reg(0, 99)
        assert dp.regs[0] == 99

    def test_signal_push_pop(self) -> None:
        dp = self._make_dp()
        dp.signal_push(42)
        dp.signal_push(77)
        assert dp.signal_pop() == 77
        assert dp.signal_pop() == 42

    def test_signal_set_flags_zero(self) -> None:
        dp = self._make_dp()
        dp.signal_set_flags(0)
        assert dp.zero() and not dp.negative() and not dp.carry_flag()

    def test_signal_set_flags_negative(self) -> None:
        dp = self._make_dp()
        dp.signal_set_flags(0x80000000)
        assert dp.negative() and not dp.zero()

    def test_signal_set_flags_carry(self) -> None:
        dp = self._make_dp()
        dp.signal_set_flags(0, carry=True)
        assert dp.carry_flag()

    def test_signal_check_irq(self) -> None:
        dp = self._make_dp([(5, 65)])
        dp.signal_check_irq(10)
        assert dp.pending_irq
        assert dp.dmem[0xFFC] == 65
        assert dp.dmem[0xFFD] == 1

    def test_signal_check_irq_not_yet(self) -> None:
        dp = self._make_dp([(100, 65)])
        dp.signal_check_irq(50)
        assert not dp.pending_irq

    def test_encode_decode_flags(self) -> None:
        dp = self._make_dp()
        dp.flags = [True, False, True, False, True]
        w = dp.encode_flags()
        dp.flags = [False] * 5
        dp.decode_flags(w)
        assert dp.flags == [True, False, True, False, True]

    def test_dmem_initialized_from_str_inits(self) -> None:
        dp = DataPath([], [(300, [3, 65, 66, 67])])
        assert dp.dmem[300] == 3
        assert dp.dmem[301] == 65


# --- ControlUnit ---


class TestControlUnit:
    def _make_cu(self, code: list[Instruction], entry: int = 0, schedule: list[tuple[int, int]] | None = None) -> ControlUnit:
        dp = DataPath(schedule or [], [])
        return ControlUnit(code, entry, 0, dp)

    def test_hlt(self) -> None:
        cu = self._make_cu([Instruction(Opcode.HLT)])
        cu._log_limit = 100
        cu.run()
        assert cu.halted

    def test_ldi(self) -> None:
        cu = self._make_cu([Instruction(Opcode.LDI, 0, 0, 42), Instruction(Opcode.HLT)])
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 42

    def test_add(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 10),
            Instruction(Opcode.LDI, 1, 0, 20),
            Instruction(Opcode.ADD, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 30

    def test_sub(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 50),
            Instruction(Opcode.LDI, 1, 0, 20),
            Instruction(Opcode.SUB, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 30

    def test_mul(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 6),
            Instruction(Opcode.LDI, 1, 0, 7),
            Instruction(Opcode.MUL, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 42

    def test_div(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 0xFFFFFFFF),
            Instruction(Opcode.LDI, 1, 0, 2),
            Instruction(Opcode.DIV, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert to_signed(cu.dp.regs[0]) == 0

    def test_mod(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 0xFFFFFFFF),
            Instruction(Opcode.LDI, 1, 0, 2),
            Instruction(Opcode.MOD, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert to_signed(cu.dp.regs[0]) == 1

    def test_jz_taken(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.CMPI, 0, 0, 0),
            Instruction(Opcode.JZ, 0, 0, 4),
            Instruction(Opcode.LDI, 0, 0, 99),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 0

    def test_jz_not_taken(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 5),
            Instruction(Opcode.CMPI, 0, 0, 0),
            Instruction(Opcode.JZ, 0, 0, 4),
            Instruction(Opcode.LDI, 0, 0, 99),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 99

    def test_call_ret(self) -> None:
        code = [
            Instruction(Opcode.CALL, 0, 0, 3),
            Instruction(Opcode.HLT),
            Instruction(Opcode.LDI, 0, 0, 77),
            Instruction(Opcode.RET),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 77

    def test_push_pop(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 42),
            Instruction(Opcode.PUSH, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.POP, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 42

    def test_st_ld(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 123),
            Instruction(Opcode.ST, 0, 0, 300),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LD, 0, 0, 300),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 123

    def test_addi(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 10),
            Instruction(Opcode.ADDI, 0, 0, 5),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 15

    def test_cmp_sets_flags(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 5),
            Instruction(Opcode.LDI, 1, 0, 10),
            Instruction(Opcode.CMP, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.negative()
        assert not cu.dp.zero()

    def test_inc_dec(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 10),
            Instruction(Opcode.INC, 0),
            Instruction(Opcode.DEC, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 10

    def test_addm(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 10),
            Instruction(Opcode.LDI, 1, 0, 20),
            Instruction(Opcode.ST, 1, 0, 300),
            Instruction(Opcode.ADDM, 0, 0, 300),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 30

    def test_logical_ops(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 0xFF),
            Instruction(Opcode.LDI, 1, 0, 0x0F),
            Instruction(Opcode.AND, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 0x0F

    def test_not_op(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.NOT, 0),
            Instruction(Opcode.HLT),
        ]
        cu = self._make_cu(code)
        cu._log_limit = 100
        cu.run()
        assert cu.dp.regs[0] == 0xFFFFFFFF

    def test_vload_vstore_vadd(self) -> None:
        dp = DataPath([], [])
        for i in range(4):
            dp.dmem[300 + i] = (i + 1) * 10
            dp.dmem[400 + i] = (i + 1) * 100
        code = [
            Instruction(Opcode.VLOAD, 0, 0, 300),
            Instruction(Opcode.VLOAD, 1, 0, 400),
            Instruction(Opcode.VADD, 0, 1, 0),
            Instruction(Opcode.VSTORE, 0, 0, 500),
            Instruction(Opcode.HLT),
        ]
        cu = ControlUnit(code, 0, 0, dp)
        cu._log_limit = 100
        cu.run()
        assert dp.dmem[500] == 110
        assert dp.dmem[501] == 220
        assert dp.dmem[502] == 330
        assert dp.dmem[503] == 440

    def test_irq_handling(self) -> None:
        dp = DataPath([(1, 65)], [])
        irq_code = [
            Instruction(Opcode.PUSH, 0),
            Instruction(Opcode.LD, 0, 0, 0xFFC),
            Instruction(Opcode.ST, 0, 0, 300),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.ST, 0, 0, 0xFFD),
            Instruction(Opcode.POP, 0),
            Instruction(Opcode.IRET),
        ]
        irq_end_addr = sum(i.word_count() for i in irq_code)
        main_code = [
            Instruction(Opcode.STI),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LD, 0, 0, 300),
            Instruction(Opcode.ST, 0, 0, 0xFFE),
            Instruction(Opcode.HLT),
        ]
        full_code = irq_code + main_code
        cu = ControlUnit(full_code, irq_end_addr, 0, dp)
        cu._max_ticks = 200
        cu._log_limit = 200
        cu.run()
        assert dp.dmem[300] == 65
        assert dp.output == ["A"]

    def test_nested_irq_blocked(self) -> None:
        dp = DataPath([(1, 65), (5, 66)], [])
        irq_code = [
            Instruction(Opcode.PUSH, 0),
            Instruction(Opcode.LD, 0, 0, 0xFFC),
            Instruction(Opcode.ST, 0, 0, 300),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.ST, 0, 0, 0xFFD),
            Instruction(Opcode.POP, 0),
            Instruction(Opcode.IRET),
        ]
        irq_end_addr = sum(i.word_count() for i in irq_code)
        main_code = [
            Instruction(Opcode.STI),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.STI),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LD, 0, 0, 300),
            Instruction(Opcode.ST, 0, 0, 0xFFE),
            Instruction(Opcode.HLT),
        ]
        full_code = irq_code + main_code
        cu = ControlUnit(full_code, irq_end_addr, 0, dp)
        cu._max_ticks = 500
        cu._log_limit = 500
        cu.run()
        assert dp.dmem[300] in (65, 66)
        assert len(dp.output) == 1
        irq_entries = [line for line in cu._log if "PC:" in line and "[IRQ]" in line and ("push" in line or "fetch]" in line) and cu._log.index(line) < 5]
        assert len(irq_entries) >= 1

    def test_per_tick_logging(self) -> None:
        cu = self._make_cu([Instruction(Opcode.LDI, 0, 0, 42), Instruction(Opcode.HLT)])
        cu._log_limit = 100
        cu.run()
        assert len(cu._log) > 2
        assert "TICK:" in cu._log[0]
        assert "PC:" in cu._log[0]


# --- ALU helpers ---


class TestALU:
    def test_u32_add_no_overflow(self) -> None:
        res, carry, overflow = u32_add(10, 20)
        assert res == 30 and not carry and not overflow

    def test_u32_add_carry(self) -> None:
        res, carry, overflow = u32_add(0xFFFFFFFF, 1)
        assert res == 0 and carry and not overflow

    def test_u32_add_overflow(self) -> None:
        _res, carry, overflow = u32_add(0x7FFFFFFF, 1)
        assert not carry and overflow

    def test_u32_sub_no_borrow(self) -> None:
        res, carry, _overflow = u32_sub(20, 10)
        assert res == 10 and not carry

    def test_u32_sub_borrow(self) -> None:
        res, carry, _overflow = u32_sub(0, 1)
        assert res == 0xFFFFFFFF and carry

    def test_to_signed(self) -> None:
        assert to_signed(0xFFFFFFFF) == -1
        assert to_signed(0x7FFFFFFF) == 0x7FFFFFFF
        assert to_signed(0) == 0

    def test_to_u32(self) -> None:
        assert to_u32(-1) == 0xFFFFFFFF
        assert to_u32(42) == 42


# --- Translator integration (end-to-end via translate()) ---


class TestTranslate:
    def test_ast_verification_readable(self) -> None:
        _code, _entry, _irq, ast_str, _str_inits = translate("var x = 1 + 2;\nprintnum(x);\n")
        assert "VarDecl(x)" in ast_str
        assert "BinOp(+)" in ast_str
        assert "NumLit(1)" in ast_str
        assert "NumLit(2)" in ast_str
        assert "Call(__print_num)" in ast_str
        lines = ast_str.strip().split("\n")
        assert len(lines) >= 4

    def test_ast_verification_poly(self) -> None:
        source = "var a = 1;\nvar b = 2;\nvar c = 3;\nvar d = 4;\nvar r = a*b + c*d;\n"
        _code, _entry, _irq, ast_str, _str_inits = translate(source)
        assert "VarDecl(a)" in ast_str
        assert "VarDecl(r)" in ast_str
        assert "BinOp(+)" in ast_str
        assert "BinOp(*)" in ast_str

    def test_hello_produces_code(self) -> None:
        code, entry, irq, ast_str, _str_inits = translate('print("Hello\\n");')
        assert len(code) > 0
        assert entry > 0
        assert irq >= 0
        assert "StrLit(" in ast_str

    def test_prob1_output(self) -> None:
        code, entry, irq, _ast_str, str_inits = translate(
            "var max = 0;\nvar i = 999;\nwhile (i >= 100) {\n    var j = 999;\nwhile (j >= i) {\n"
            "        var p = i * j;\n        if (p <= max) {\n            j = 0;\n        } else {\n"
            "            var t = p;\n            var rev = 0;\n            while (t != 0) {\n"
            "                rev = rev * 10 + t % 10;\n                t = t / 10;\n            }\n"
            "            if (rev == p) {\n                max = p;\n            }\n        }\n"
            "        j = j - 1;\n    }\n    i = i - 1;\n}\nprintnum(max);\nputc('\\n');\n"
        )
        sim = Simulator(code, entry, irq, [(0, 88)], str_inits)
        sim._max_ticks = 5000000
        result = sim.run()
        assert result == "906609\n"

    def test_cisc_addm_emitted(self) -> None:
        code, _e, _i, _a, _s = translate("var x = 1;\nvar y = x + 2;\n")
        opcodes = [instr.opcode for instr in code]
        assert Opcode.ADDM in opcodes or Opcode.ADDI in opcodes

    def test_cisc_addi_emitted(self) -> None:
        code, _e, _i, _a, _s = translate("var x = 10;\nvar y = x + 5;\n")
        opcodes = [instr.opcode for instr in code]
        assert Opcode.ADDI in opcodes

    def test_cisc_cmpm_emitted(self) -> None:
        code, _e, _i, _a, _s = translate("var x = 1;\nvar y = 2;\nvar z = x < y;\n")
        opcodes = [instr.opcode for instr in code]
        assert Opcode.CMPM in opcodes or Opcode.CMPI in opcodes

    def test_poly_emitted_for_sum_of_products(self) -> None:
        source = "var a = 1;\nvar b = 2;\nvar c = 3;\nvar d = 4;\nvar e = 5;\nvar f = 6;\nvar r = a*b + c*d + e*f;\n"
        code, _e, _i, _a, _s = translate(source)
        opcodes = [instr.opcode for instr in code]
        assert Opcode.POLY in opcodes


# --- Special registers ---


class TestSpecialRegs:
    def test_movsp(self) -> None:
        dp = DataPath([], [])
        dp.sp = 100
        dp.regs[7] = 100
        code = [Instruction(Opcode.MOVSP, 1)]
        cu = ControlUnit(code, 0, 0, dp)
        cu._max_ticks = 10
        cu.run()
        assert dp.regs[1] == 100

    def test_ldsp(self) -> None:
        dp = DataPath([], [])
        dp.regs[1] = 200
        code = [Instruction(Opcode.LDSP, 1)]
        cu = ControlUnit(code, 0, 0, dp)
        cu._max_ticks = 10
        cu.run()
        assert dp.sp == 200
        assert dp.regs[7] == 200

    def test_getflags_setflags(self) -> None:
        dp = DataPath([], [])
        dp.flags[FLAG_Z] = True
        dp.flags[FLAG_C] = True
        code = [
            Instruction(Opcode.GETFLAGS, 1),
            Instruction(Opcode.LDI, 2, 0, 0),
            Instruction(Opcode.SETFLAGS, 2),
            Instruction(Opcode.HLT),
        ]
        cu = ControlUnit(code, 0, 0, dp)
        cu._max_ticks = 20
        cu.run()
        assert dp.regs[1] == 0b00101


# --- POLY ---


class TestPoly:
    def test_poly_encode_decode(self) -> None:
        instr = Instruction(Opcode.POLY, 0, 0, 3, ((260, 270), (261, 271), (262, 272)))
        assert instr.word_count() == 4
        words = encode(instr)
        assert len(words) == 4
        decoded = decode_instruction_from_words(words)
        assert decoded.opcode == Opcode.POLY
        assert decoded.pairs == ((260, 270), (261, 271), (262, 272))
        assert decoded.imm == 3

    def test_poly_execution(self) -> None:
        dp = DataPath([], [])
        dp.dmem[260] = 2
        dp.dmem[270] = 3
        dp.dmem[261] = 4
        dp.dmem[271] = 5
        dp.regs[0] = 10
        code = [
            Instruction(Opcode.POLY, 0, 0, 2, ((260, 270), (261, 271))),
            Instruction(Opcode.HLT),
        ]
        cu = ControlUnit(code, 0, 0, dp)
        cu._max_ticks = 30
        cu.run()
        assert dp.regs[0] == 10 + 2 * 3 + 4 * 5

    def test_poly_one_term_falls_back(self) -> None:
        source = "var a = 3;\nvar b = 4;\nvar r = a * b;\n"
        code, _e, _i, _a, _s = translate(source)
        opcodes = [instr.opcode for instr in code]
        assert Opcode.POLY not in opcodes
        assert Opcode.MULM in opcodes or Opcode.MUL in opcodes


# --- Word-by-word fetch ---


class TestWordFetch:
    def test_fetch_produces_log_entries(self) -> None:
        dp = DataPath([], [])
        code = [Instruction(Opcode.LDI, 0, 0, 42), Instruction(Opcode.HLT)]
        cu = ControlUnit(code, 0, 0, dp)
        cu._max_ticks = 20
        cu._log_limit = 100
        cu.run()
        fetch_logs = [line for line in cu._log if "[fetch]" in line]
        assert len(fetch_logs) > 0

    def test_two_word_instruction_fetches_two_words(self) -> None:
        dp = DataPath([], [])
        code = [Instruction(Opcode.LDI, 0, 0, 42), Instruction(Opcode.HLT)]
        cu = ControlUnit(code, 0, 0, dp)
        cu._max_ticks = 20
        cu._log_limit = 100
        cu.run()
        ldi_fetch_logs = [line for line in cu._log if "[fetch]" in line and "PC:   0" in line]
        assert len(ldi_fetch_logs) == 2


# --- parse_input ---


class TestParseInput:
    def test_tick_char_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/inp.txt"
            with open(path, "w") as f:
                f.write("0 65\n100 66\n")
            result = parse_input(path)
            assert result == [(0, 65), (100, 66)]

    def test_comment_lines_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/inp.txt"
            with open(path, "w") as f:
                f.write("# comment\n0 65\n")
            result = parse_input(path)
            assert result == [(0, 65)]

    def test_empty_lines_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/inp.txt"
            with open(path, "w") as f:
                f.write("\n0 65\n\n")
            result = parse_input(path)
            assert result == [(0, 65)]
