// Three-instruction rotates, the shape RV64GC compilers emit because the base ISA has
// no rotate instruction. rvrun's JIT fuses the two shifts and the OR into one arm64
// rotate; the interpreter must agree, including on the sequences that only look like a
// rotate.
#include <stdint.h>
#include <stdio.h>

static int failures;

static void check_equal(const char *name, uint64_t got, uint64_t want) {
    if (got != want) {
        printf("FAIL %s: got %016llx want %016llx\n", name,
               (unsigned long long)got, (unsigned long long)want);
        failures++;
    }
}

static inline uint64_t rot_slli_then_srli(uint64_t x) {
    uint64_t t, y;
    __asm__ volatile("slli %[t], %[x], 1\n\t"
                     "srli %[y], %[x], 63\n\t"
                     "or   %[y], %[t], %[y]"
                     : [t] "=&r"(t), [y] "=&r"(y)
                     : [x] "r"(x));
    return y;
}

static inline uint64_t rot_srli_then_slli(uint64_t x) {
    uint64_t t;
    __asm__ volatile("slli %[t], %[x], 57\n\t"
                     "srli %[x], %[x], 7\n\t"
                     "or   %[x], %[x], %[t]"
                     : [t] "=&r"(t), [x] "+r"(x));
    return x;
}

static inline uint64_t rot_third_destination(uint64_t x) {
    uint64_t t, u, y;
    __asm__ volatile("srli %[t], %[x], 32\n\t"
                     "slli %[u], %[x], 32\n\t"
                     "or   %[y], %[t], %[u]"
                     : [t] "=&r"(t), [u] "=&r"(u), [y] "=&r"(y)
                     : [x] "r"(x));
    return y;
}

static inline uint64_t rot_slli_first(uint64_t x) {
    uint64_t t, u, y;
    __asm__ volatile("slli %[t], %[x], 13\n\t"
                     "srli %[u], %[x], 51\n\t"
                     "or   %[y], %[u], %[t]"
                     : [t] "=&r"(t), [u] "=&r"(u), [y] "=&r"(y)
                     : [x] "r"(x));
    return y;
}

// The OR reuses the first shift's destination, so the second shift's result (in the
// source register) is the one that has to be kept for the guest register file.
static inline uint64_t rot_keep_source(uint64_t x, uint64_t *kept) {
    uint64_t t;
    __asm__ volatile("slli %[t], %[x], 32\n\t"
                     "srli %[x], %[x], 32\n\t"
                     "or   %[t], %[t], %[x]"
                     : [t] "=&r"(t), [x] "+r"(x));
    *kept = x;
    return t;
}

// The first shift overwrites the source, so the second shifts the result, not the
// original value. This is not a rotate and must not be fused.
static inline uint64_t shifts_with_dependency(uint64_t x) {
    uint64_t t;
    __asm__ volatile("srli %[x], %[x], 32\n\t"
                     "slli %[t], %[x], 32\n\t"
                     "or   %[x], %[t], %[x]"
                     : [t] "=&r"(t), [x] "+r"(x));
    return x;
}

static int64_t sext32(uint32_t v) {
    return (int32_t)v;
}

// The SHA-256 shape: 32-bit shifts, with the next rotate's first shift scheduled
// between this rotate's second shift and its OR.
static inline int64_t rotw_slli_then_srli(uint64_t x) {
    int64_t t, y;
    __asm__ volatile("sllw %[t], %[x], 26\n\t"
                     "srlw %[y], %[x], 6\n\t"
                     "or   %[y], %[t], %[y]"
                     : [t] "=&r"(t), [y] "=&r"(y)
                     : [x] "r"(x));
    return y;
}

static inline void rotw_pair(uint64_t x, int64_t *a, int64_t *b) {
    int64_t t0, t1, t2, t3;
    __asm__ volatile("sllw %[t0], %[x], 26\n\t"
                     "srlw %[t1], %[x], 6\n\t"
                     "sllw %[t2], %[x], 21\n\t"
                     "or   %[t1], %[t1], %[t0]\n\t"
                     "srlw %[t3], %[x], 11\n\t"
                     "or   %[t3], %[t3], %[t2]"
                     : [t0] "=&r"(t0), [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3)
                     : [x] "r"(x));
    *a = t1;
    *b = t3;
}

// The add reads the first shift's result before the OR: the window must not fuse this,
// because the skipped shift would leave the add reading a stale value.
static inline int64_t rotw_read_dep(uint64_t x, int64_t *out) {
    int64_t t, u, y;
    __asm__ volatile("sllw %[t], %[x], 26\n\t"
                     "addiw %[u], %[t], 1\n\t"
                     "srlw %[y], %[x], 6\n\t"
                     "or   %[y], %[t], %[y]"
                     : [t] "=&r"(t), [u] "=&r"(u), [y] "=&r"(y)
                     : [x] "r"(x));
    *out = u;
    return y;
}

// The add replaces the first shift's result before the OR, so this is not a rotate.
static inline int64_t rotw_write_dep(uint64_t x) {
    int64_t t, y;
    __asm__ volatile("sllw %[t], %[x], 26\n\t"
                     "addiw %[t], %[t], 1\n\t"
                     "srlw %[y], %[x], 6\n\t"
                     "or   %[y], %[t], %[y]"
                     : [t] "=&r"(t), [y] "=&r"(y)
                     : [x] "r"(x));
    return y;
}

static inline int64_t rotw_source_dependency(uint64_t x) {
    int64_t t;
    __asm__ volatile("srlw %[x], %[x], 6\n\t"
                     "sllw %[t], %[x], 26\n\t"
                     "or   %[x], %[t], %[x]"
                     : [t] "=&r"(t), [x] "+r"(x));
    return x;
}

int main(void) {
    static const uint64_t values[] = {
        0, 1, 2, 0x0123456789abcdef, 0x8000000000000000, 0xffffffffffffffff,
        0xdeadbeefcafebabe, 0x5555555555555555,
    };
    for (unsigned i = 0; i < sizeof values / sizeof values[0]; i++) {
        uint64_t x = values[i];
        check_equal("slli then srli", rot_slli_then_srli(x), x << 1 | x >> 63);
        check_equal("srli then slli", rot_srli_then_slli(x), x << 57 | x >> 7);
        check_equal("third destination", rot_third_destination(x), x << 32 | x >> 32);
        check_equal("slli first", rot_slli_first(x), x << 13 | x >> 51);
        uint64_t kept;
        check_equal("keep source rotate", rot_keep_source(x, &kept), x << 32 | x >> 32);
        check_equal("keep source value", kept, x >> 32);
        uint64_t y = x >> 32;
        check_equal("dependency", shifts_with_dependency(x), y | y << 32);
        check_equal("word rotate", rotw_slli_then_srli(x),
                    sext32((uint32_t)x << 26 | (uint32_t)x >> 6));
        int64_t a, b;
        rotw_pair(x, &a, &b);
        check_equal("word pair first", a, sext32((uint32_t)x << 26 | (uint32_t)x >> 6));
        check_equal("word pair second", b, sext32((uint32_t)x << 21 | (uint32_t)x >> 11));
        int64_t u;
        check_equal("word read dependency", rotw_read_dep(x, &u),
                    sext32((uint32_t)x << 26 | (uint32_t)x >> 6));
        check_equal("word read value", u, sext32(((uint32_t)x << 26) + 1));
        int64_t t = sext32((uint32_t)x << 26);
        check_equal("word write dependency", rotw_write_dep(x),
                    sext32((uint32_t)(t + 1) | (uint32_t)x >> 6));
        int64_t w = sext32((uint32_t)x >> 6);
        check_equal("word source dependency", rotw_source_dependency(x),
                    sext32((uint32_t)sext32((uint32_t)w << 26) | (uint32_t)w));
    }
    if (failures == 0) {
        printf("ok rotate sequences\n");
    }
    return failures != 0;
}
