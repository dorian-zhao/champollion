// ARM64 encodings the JIT emits, in the order crates/champollion-rv/tests/encode_arm64.rs
// produces them with the encoder. Assembled by build.sh (zig, aarch64) into arm64.bin.
add x0, x1, x2
add w3, w4, w5
add x6, x7, x8, lsl #3
sub x9, x10, x11
sub w12, w13, w14
cmp x15, x16
and x17, x19, x20
orr x21, x22, x23
eor x24, x25, x26
mov x27, x28
mov w0, w1
lsl x2, x3, x4
lsr x5, x6, x7
asr x8, x9, x10
lsl w11, w12, w13
lsr w14, w15, w16
asr w17, w19, w20
mul x0, x1, x2
mul w3, w4, w5
msub x6, x7, x8, x9
msub w10, w11, w12, w13
smulh x14, x15, x16
umulh x17, x19, x20
sdiv x21, x22, x23
udiv x24, x25, x26
sdiv w27, w28, w29
udiv w0, w1, w2
csel x3, x4, x5, lt
cset x6, eq
cset x7, lo
add x8, x9, #4095
add w10, w11, #1
add sp, sp, #16
sub x12, x13, #2048
sub w14, w15, #7
cmp x16, #0
and x17, x19, #0x3fffffffff
orr x20, xzr, #0xff00
eor x21, x22, #0x5555555555555555
lsl x0, x1, #12
lsr x2, x3, #63
asr x4, x5, #1
lsl w6, w7, #31
lsr w8, w9, #4
asr w10, w11, #31
sxtw x12, w13
movz x14, #0x1234, lsl #16
movk x15, #0xffff, lsl #48
movn x16, #0x5
ldrb w0, [x1, x2]
ldrh w3, [x4, x5]
ldr w6, [x7, x8]
ldr x9, [x10, x11]
ldrsb x12, [x13, x14]
ldrsh x15, [x16, x17]
ldrsw x19, [x20, x21]
strb w22, [x23, x24]
strh w25, [x26, x27]
str w28, [x29, x0]
str x1, [x2, x3]
ldr x4, [x5, #32760]
str x6, [x7, #8]
ldr w8, [x9, #4]
str w10, [x11, #16380]
stp x29, x30, [sp, #-16]!
ldp x29, x30, [sp], #16
br x16
blr x17
ret
nop
brk #0x1234
dmb ish
