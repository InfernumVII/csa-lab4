import tempfile

import pytest

from src.isa import (
    OUTPUT_DATA_ADDR,
    TWO_WORD_OPCODES,
    Instruction,
    Opcode,
    decode_word0,
    encode,
    read_binary,
    sign_extend_32,
    write_binary,
)
from src.machine import (
    ALU,
    ControlUnit,
    DataMemory,
    Datapath,
    InstructionMemory,
    Registers,
    Simulator,
    parse_schedule,
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
    translate,
)


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
        assert isinstance(ast[0], VarDecl) and isinstance(
            ast[0].init, NumLit) and ast[0].init.value == 0

    def test_if_else(self) -> None:
        ast = Parser(
            Lexer("if (1) { var x = 1; } else { var y = 2; }").tokens).parse()
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
        assert isinstance(ast[0], Assign) and isinstance(
            ast[0].target, ArrAccess)

    def test_call_with_args(self) -> None:
        ast = Parser(Lexer("printnum(x);").tokens).parse()
        assert isinstance(ast[0], ExprStmt)
        assert isinstance(
            ast[0].expr, Call) and ast[0].expr.name == "__print_num"

    def test_irq_def(self) -> None:
        ast = Parser(
            Lexer("irq handler { count = count + 1; }").tokens).parse()
        assert isinstance(ast[0], IrqDef) and ast[0].name == "handler"


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
        data = [(10, 99)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/test.bin"
            write_binary(code, data, 5, 0, path)
            instructions, out_data, entry, irq = read_binary(path)
            assert entry == 5 and irq == 0
            assert out_data == [(10, 99)]
            assert len(instructions) == 3
            assert instructions[0].opcode == Opcode.HLT
            assert instructions[1].opcode == Opcode.LDI and instructions[1].imm == 42
            assert instructions[2].opcode == Opcode.ADD

    def test_sign_extend_32(self) -> None:
        assert sign_extend_32(0xFFFFFFFF) == -1
        assert sign_extend_32(0x7FFFFFFF) == 0x7FFFFFFF
        assert sign_extend_32(0x80000000) == -2147483648
        assert sign_extend_32(0) == 0


class TestALU:
    def test_alu_add_no_overflow(self) -> None:
        alu = ALU()
        alu.set_a(10)
        alu.set_b(20)
        alu.signal_add()
        alu.latch_flags()
        assert alu.res == 30
        assert not alu.c and not alu.z and not alu.n

    def test_alu_add_carry(self) -> None:
        alu = ALU()
        alu.set_a(0xFFFFFFFF)
        alu.set_b(1)
        alu.signal_add()
        alu.latch_flags()
        assert alu.res == 0
        assert alu.c and alu.z and not alu.n

    def test_alu_sub_borrow(self) -> None:
        alu = ALU()
        alu.set_a(0)
        alu.set_b(1)
        alu.signal_sub()
        alu.latch_flags()
        assert alu.res == 0xFFFFFFFF
        assert alu.c and alu.n and not alu.z


class TestRegisters:
    def test_latch_and_out(self) -> None:
        regs = Registers()
        regs.latch_reg(42, 1)
        regs.signal_rs1_out(1)
        assert regs.outrs1 == 42


class TestDataMemory:
    def test_read_write(self) -> None:
        dmem = DataMemory(8192)
        dmem.signal_write(300, 99)
        dmem.signal_read(300)
        assert dmem.out == 99

    def test_write_output(self) -> None:
        dmem = DataMemory(8192)
        dmem.signal_write(OUTPUT_DATA_ADDR, 65)
        assert dmem.output_buffer == ["A"]


class TestControlUnit:
    def _make_sim(self, code: list[Instruction], entry: int = 0, irq: int = 0, schedule: list[tuple[int, str]] | None = None) -> Simulator:
        dp = Datapath()
        imem = InstructionMemory(1024)
        addr = 0
        for instr in code:
            for w in encode(instr):
                imem.imem[addr] = w
                addr += 1
        cu = ControlUnit(dp, imem, entry, irq)
        return Simulator(dp, cu, schedule or [])

    def test_hlt(self) -> None:
        sim = self._make_sim([Instruction(Opcode.HLT)])
        sim.run()
        assert sim.cu.halted

    def test_ldi(self) -> None:
        sim = self._make_sim(
            [Instruction(Opcode.LDI, 0, 0, 42), Instruction(Opcode.HLT)])
        sim.run()
        assert sim.dp.regs.regs[0] == 42

    def test_add(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 10),
            Instruction(Opcode.LDI, 1, 0, 20),
            Instruction(Opcode.ADD, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 30

    def test_sub(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 50),
            Instruction(Opcode.LDI, 1, 0, 20),
            Instruction(Opcode.SUB, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 30

    def test_mul(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 6),
            Instruction(Opcode.LDI, 1, 0, 7),
            Instruction(Opcode.MUL, 0, 1, 0),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 42

    def test_jz_taken(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.CMPI, 0, 0, 0),
            Instruction(Opcode.JZ, 0, 0, 8),
            Instruction(Opcode.LDI, 0, 0, 99),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 0

    def test_jz_not_taken(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 5),
            Instruction(Opcode.CMPI, 0, 0, 0),
            Instruction(Opcode.JZ, 0, 0, 8),
            Instruction(Opcode.LDI, 0, 0, 99),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 99

    def test_call_ret(self) -> None:
        code = [
            Instruction(Opcode.CALL, 0, 0, 2),
            Instruction(Opcode.HLT),
            Instruction(Opcode.LDI, 0, 0, 77),
            Instruction(Opcode.RET),
        ]
        sim = self._make_sim(code)
        sim.run()
        print(sim.dp.regs.regs)
        assert sim.dp.regs.regs[0] == 77

    def test_push_pop(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 42),
            Instruction(Opcode.PUSH, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.POP, 0),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 42

    def test_st_ld(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 123),
            Instruction(Opcode.ST, 0, 0, 300),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LD, 0, 0, 300),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 123

    def test_addi(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 10),
            Instruction(Opcode.ADDI, 0, 0, 5),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 15

    def test_inc_dec(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 10),
            Instruction(Opcode.INC, 0),
            Instruction(Opcode.DEC, 0),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 10

    def test_not_op(self) -> None:
        code = [
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.NOT, 0),
            Instruction(Opcode.HLT),
        ]
        sim = self._make_sim(code)
        sim.run()
        assert sim.dp.regs.regs[0] == 0xFFFFFFFF

    def test_irq_handling(self) -> None:
        irq_code = [
            Instruction(Opcode.PUSH, 0),
            Instruction(Opcode.LDI, 0, 0, 65),
            Instruction(Opcode.ST, 0, 0, 300),
            Instruction(Opcode.POP, 0),
            Instruction(Opcode.IRET),
        ]
        main_code = [
            Instruction(Opcode.STI),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.LDI, 0, 0, 0),
            Instruction(Opcode.HLT),
        ]
        irq_end_addr = sum(i.word_count() for i in irq_code)
        sim = self._make_sim(irq_code + main_code,
                             entry=irq_end_addr, irq=0, schedule=[(2, 'A')])
        sim.run()
        assert sim.dp.dmem.mem[300] == 65


class TestPoly:
    def test_poly_encode_decode(self) -> None:
        instr = Instruction(Opcode.POLY, 0, 0, 3,
                            ((260, 270), (261, 271), (262, 272)))
        assert instr.word_count() == 4
        words = encode(instr)
        assert len(words) == 4
        op, r1, r2, imm = decode_word0(words[0])
        assert op == Opcode.POLY
        assert imm == 3

    def test_poly_execution(self) -> None:
        dp = Datapath()
        dp.dmem.mem[260] = 2
        dp.dmem.mem[270] = 3
        dp.dmem.mem[261] = 4
        dp.dmem.mem[271] = 5
        dp.regs.regs[0] = 10

        code = [
            Instruction(Opcode.POLY, 0, 0, 2, ((260, 270), (261, 271))),
            Instruction(Opcode.HLT),
        ]
        imem = InstructionMemory(1024)
        addr = 0
        for instr in code:
            for w in encode(instr):
                imem.imem[addr] = w
                addr += 1
        cu = ControlUnit(dp, imem, 0, 0)
        sim = Simulator(dp, cu, [])
        sim.run()

        assert dp.regs.regs[0] == 36


class TestParseSchedule:
    def test_tick_char_format(self) -> None:
        result = parse_schedule("0 A\n100 B\n")
        assert result == [(0, 'A'), (100, 'B')]

    def test_empty_lines_ignored(self) -> None:
        result = parse_schedule("\n0 A\n\n")
        assert result == [(0, 'A')]


class TestTranslate:
    def test_ast_verification_readable(self) -> None:
        _code, _data, _entry, _irq, ast_str = translate(
            "var x = 1 + 2;\nprintnum(x);\n")
        assert "VarDecl(x)" in ast_str
        assert "BinOp(+)" in ast_str
        assert "NumLit(1)" in ast_str
        assert "NumLit(2)" in ast_str
        assert "Call(__print_num)" in ast_str
        lines = ast_str.strip().split("\n")
        assert len(lines) >= 4

    def test_ast_verification_poly(self) -> None:
        source = "var a = 1;\nvar b = 2;\nvar c = 3;\nvar d = 4;\nvar r = a*b + c*d;\n"
        _code, _data, _entry, _irq, ast_str = translate(source)
        assert "VarDecl(a)" in ast_str
        assert "VarDecl(r)" in ast_str
        assert "BinOp(+)" in ast_str
        assert "BinOp(*)" in ast_str

    def test_hello_produces_code(self) -> None:
        code, data, entry, irq, ast_str = translate('print("Hello\\n");')
        assert len(code) > 0
        assert entry >= 0
        assert irq >= 0
        assert "StrLit(" in ast_str

    def test_prob1_output(self) -> None:
        code, data_section, entry, irq, _ast_str = translate(
            "var max = 0;\nvar i = 999;\nwhile (i >= 100) {\n    var j = 999;\nwhile (j >= i) {\n"
            "        var p = i * j;\n        if (p <= max) {\n            j = 0;\n        } else {\n"
            "            var t = p;\n            var rev = 0;\n            while (t != 0) {\n"
            "                rev = rev * 10 + t % 10;\n                t = t / 10;\n            }\n"
            "            if (rev == p) {\n                max = p;\n            }\n        }\n"
            "        j = j - 1;\n    }\n    i = i - 1;\n}\nprintnum(max);\nputc('\\n');\n"
        )
        dp = Datapath()
        for addr, val in data_section:
            dp.dmem.mem[addr] = val

        imem = InstructionMemory(4096)
        addr = 0
        for instr in code:
            for w in encode(instr):
                imem.imem[addr] = w
                addr += 1

        cu = ControlUnit(dp, imem, entry, irq)
        sim = Simulator(dp, cu, [])

        sim.run()

        assert "".join(dp.dmem.output_buffer) == "906609\n"

    def test_cisc_addm_emitted(self) -> None:
        code, _data, _e, _i, _a = translate("var x = 1;\nvar y = x + 2;\n")
        opcodes = [instr.opcode for instr in code]
        assert Opcode.ADDM in opcodes or Opcode.ADDI in opcodes

    def test_cisc_addi_emitted(self) -> None:
        code, _data, _e, _i, _a = translate("var x = 10;\nvar y = x + 5;\n")
        opcodes = [instr.opcode for instr in code]
        assert Opcode.ADDM in opcodes or Opcode.ADDI in opcodes or Opcode.ADD in opcodes

    def test_cisc_cmpm_emitted(self) -> None:
        code, _data, _e, _i, _a = translate(
            "var x = 1;\nvar y = 2;\nvar z = x < y;\n")
        opcodes = [instr.opcode for instr in code]
        assert Opcode.CMPM in opcodes or Opcode.CMPI in opcodes

    def test_poly_emitted_for_sum_of_products(self) -> None:
        source = "var a = 1;\nvar b = 2;\nvar c = 3;\nvar d = 4;\nvar e = 5;\nvar f = 6;\nvar r = a*b + c*d + e*f;\n"
        code, _data, _e, _i, _a = translate(source)
        opcodes = [instr.opcode for instr in code]
        assert Opcode.POLY in opcodes

    def test_poly_one_term_falls_back(self) -> None:
        source = "var a = 3;\nvar b = 4;\nvar r = a * b;\n"
        code, _data, _e, _i, _a = translate(source)
        opcodes = [instr.opcode for instr in code]
        assert Opcode.POLY not in opcodes
        assert Opcode.MULM in opcodes or Opcode.MUL in opcodes
