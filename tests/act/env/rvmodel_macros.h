// User-mode DUT environment for the ACT tests (riscv-arch-test): the test is an
// ordinary static Linux executable. It is already in U-mode at entry, has no M-mode
// CSRs beyond the user-visible ones, and can print through Linux syscalls. HALT
// dumps the signature region to stdout (one write) and exits 0/1, which is what the
// runner in tests/fixtures/act compares against the committed qemu-riscv64 baseline.
//
// tests that need machine mode or catching traps (Misalign, Zicsr, Zicntr, ...) are
// not part of the built set; see tests/act/dirs.txt.
#ifndef _RVMODEL_MACROS_H
#define _RVMODEL_MACROS_H

#define RVMODEL_DATA_SECTION

/* No boot work: the process starts valid. */
/* RVMODEL_BOOT left undefined */

/* Already in U-mode: skip the bare-metal mode switch... */
#define RVMODEL_BOOT_TO_UMODE
/* ...and the default M-mode boot (it writes mstatus and installs mtvec handlers;
   INIT_FLOAT_VECTOR_STATE runs even with this defined, so the DUT config must not
   claim F/D/vector support, which is exactly the user-mode situation anyway). */
#define RVMODEL_BOOT_TO_MMODE

/* Called as RVMODEL_IO_INIT(r1, r2, r3): the parameter list must be written out, or
   the arguments survive into the assembly. */
#define RVMODEL_IO_INIT(_R1, _R2, _R3)

/* Failure messages are not needed: the exit status and signature carry the result. */
#define RVMODEL_IO_WRITE_STR(_R1, _R2, _R3, _STR_PTR)

#define RVMODEL_FENCEI fence.i

/* write(1, begin_signature, end_signature - begin_signature); exit(rc) */
#define RVMODEL_HALT(_RC)                \
    li a0, 1;                            \
    la a1, begin_signature;              \
    la a2, end_signature;                \
    sub a2, a2, a1;                      \
    li a7, 64;                           \
    ecall;                               \
    li a0, (_RC);                        \
    li a7, 93;                           \
    ecall

#define RVMODEL_HALT_PASS \
    RVMODEL_HALT(0)

#define RVMODEL_HALT_FAIL \
    RVMODEL_HALT(1)

#endif
