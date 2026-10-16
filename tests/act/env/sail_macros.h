// Shadows the framework's sail_macros.h for user-mode runs: the original replaces the
// DUT's halt/IO macros with the bare-metal tohost/HTIF spin (watched only by the Sail
// reference model). Keeping the DUT macros (Linux syscalls) lets the same signature-mode
// ELF run under qemu-riscv64 and rvrun and dump its signature for a byte-for-byte diff.
// (Copied framework headers shadow ours by same-directory include lookup, so build.sh
// overlays this file into the env directory it compiles against.)
#ifndef _SAIL_MACROS_H
#define _SAIL_MACROS_H
#endif
