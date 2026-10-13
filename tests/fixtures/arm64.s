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
mvn x2, x3
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
csinv x0, x1, xzr, ne
csinv w2, w3, wzr, ne
cset x7, lo
add x8, x9, #4095
add w10, w11, #1
add sp, sp, #16
sub x12, x13, #2048
sub w14, w15, #7
cmp x16, #0
cmp w17, #5
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
tst x0, #7
cmp w1, w2
ldaddal x3, x4, [x5]
ldclral w6, w7, [x8]
ldeoral x9, x10, [x11]
ldsetal w12, w13, [x14]
ldsmaxal x15, x16, [x17]
ldsminal w19, w20, [x21]
ldumaxal x22, x23, [x24]
lduminal w25, w26, [x27]
swpal x0, xzr, [x1]
casal x2, x3, [x4]
casal w5, w6, [x7]
ldr d16, [x28, #256]
str d17, [x28, #504]
ldr s18, [x1, #4]
str s19, [x2, #8]
fmov s16, w14
fmov d17, x15
fmov w14, s18
fmov x15, d19
fadd s16, s17, s18
fsub d16, d17, d18
fmul s19, s20, s21
fdiv d22, d23, d24
fsqrt s16, s17
fsqrt d18, d19
fmadd s16, s17, s18, s19
fmsub d20, d21, d22, d23
fnmadd s24, s25, s26, s27
fnmsub d28, d29, d30, d31
fcmp s16, s17
fcmpe d18, d19
fcvt d16, s17
fcvt s18, d19
fcvtns w0, s16
fcvtau x1, d17
fcvtps x2, s18
fcvtmu w3, d19
fcvtzs x4, d20
fcvtzu w5, s21
scvtf s16, w0
ucvtf d17, x1
scvtf d18, x2
ucvtf s19, w3
mrs x0, fpcr
msr fpcr, x1
mrs x2, fpsr
msr fpsr, x3
rbit w4, w5
cmn w6, #1
and w7, w8, #0x1f
ldr d20, [x27, x17]
str d21, [x27, x17]
ldr s22, [x27, x17]
str s23, [x27, x17]
fmov d1, d2
fneg d3, d4
fabs d5, d6
stp d8, d9, [sp, #-16]!
ldp d14, d15, [sp], #16
ldrb w0, [x17, #4095]
ldrh w1, [x17, #8190]
ldr w2, [x17, #16380]
ldr x3, [x17, #32760]
ldrsb x4, [x17, #1]
ldrsh x5, [x17, #2]
ldrsw x6, [x17, #4]
ldurb w7, [x17, #-256]
ldurh w8, [x17, #-2]
ldur w9, [x17, #-4]
ldur x10, [x17, #-8]
ldursb x11, [x17, #-1]
ldursh x12, [x17, #-3]
ldursw x13, [x17, #255]
ldur x19, [x17, #7]
strb w20, [x17, #4095]
strh w21, [x17, #2]
str w22, [x17, #4]
str x23, [x17, #8]
sturb w24, [x17, #-1]
sturh w25, [x17, #-2]
stur w26, [x17, #-4]
stur x0, [x17, #-256]
ldr s1, [x17, #4]
ldr d2, [x17, #8]
ldur s3, [x17, #-4]
ldur d4, [x17, #-8]
str s5, [x17, #16380]
str d6, [x17, #32760]
stur s7, [x17, #-4]
stur d8, [x17, #3]
