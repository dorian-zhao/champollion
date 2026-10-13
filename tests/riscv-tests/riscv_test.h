// User-mode test environment for riscv-tests, replacing the upstream machine-mode
// env (riscv-test-env). Tests run as static Linux executables under our interpreter:
//
// - Entry is `_start`; no CSR setup, no trap vectors, no tohost/fromhost.
// - Pass: exit(0). Fail: exit(TESTNUM), the number of the first failing test case.
//   TEST_PASSFAIL only reaches RVTEST_FAIL with TESTNUM != 0, so 0 always means pass.
//
// Exit is the Linux `exit` syscall (a7 = 93), so the harness only has to recognize
// one ecall, and the same binaries also run unmodified under qemu-riscv64.

#ifndef CHAMPOLLION_RISCV_TEST_H
#define CHAMPOLLION_RISCV_TEST_H

#define RVTEST_RV64U
#define RVTEST_RV32U
#define RVTEST_RV64UF
#define RVTEST_RV64UD

#define TESTNUM gp

#define RVTEST_CODE_BEGIN \
        .text;            \
        .globl _start;    \
_start:

#define RVTEST_CODE_END unimp

#define RVTEST_PASS   \
        fence;        \
        li a0, 0;     \
        li a7, 93;    \
        ecall

#define RVTEST_FAIL      \
        fence;           \
        mv a0, TESTNUM;  \
        li a7, 93;       \
        ecall

#define EXTRA_DATA

#define RVTEST_DATA_BEGIN \
        EXTRA_DATA;       \
        .data;            \
        .balign 16;

#define RVTEST_DATA_END

#endif
