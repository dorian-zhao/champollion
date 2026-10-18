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
sub x16, x27, #20, lsl #12
add x13, x14, #20, lsl #12
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
adrp x14, #4096
adrp x0, #-8192
adrp x17, #0x4000000
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
pmull v18.1q, v16.1d, v17.1d
umov x0, v18.d[1]
extr x1, x2, x3, #1
zip1 v16.16b, v17.16b, v18.16b
shl v18.16b, v18.16b, #4
ushr v19.16b, v19.16b, #4
rev16 v20.16b, v21.16b
rev32 v22.16b, v23.16b
rev64 v24.16b, v25.16b
clz v26.16b, v27.16b
clz v28.8h, v29.8h
clz v30.4s, v31.4s
cnt v20.16b, v21.16b
uaddlp v22.8h, v23.16b
uaddlp v24.4s, v25.8h
uaddlp v26.2d, v27.4s
mov v28.b[0], v29.b[0]
mov v30.h[0], v31.h[0]
addv b28, v29.16b
addv h30, v31.8h
addv s28, v29.4s
addp d30, v31.2d
mla v20.16b, v21.16b, v22.16b
mla v23.8h, v24.8h, v25.8h
mla v26.4s, v27.4s, v28.4s
sshll v20.8h, v21.8b, #0
ushll v22.8h, v23.8b, #0
sshll v24.4s, v25.4h, #0
ushll v26.4s, v27.4h, #0
sshll v28.2d, v29.2s, #0
ushll v30.2d, v31.2s, #0
movi v20.8h, #31
fadd s16, s17, s18
fsub d16, d17, d18
fmul s19, s20, s21
fdiv d22, d23, d24
fmin s16, s17, s18
fmax s19, s20, s21
fminnm s22, s23, s24
fmaxnm s25, s26, s27
fmin d16, d17, d18
fmax d19, d20, d21
fminnm d22, d23, d24
fmaxnm d25, d26, d27
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
mrs x16, cntvct_el0
mrs x17, cntfrq_el0
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
bic x0, x1, x2
orn x3, x4, x5
eon x6, x7, x8
clz x9, x10
clz w11, w12
rbit x13, x14
rev x15, x16
rorv x17, x19, x20
rorv w21, w22, w23
ror x24, x25, #26
ror w27, w28, #29
add x0, x1, w2, uxtw
add x3, x4, w5, uxtw #1
add x6, x7, w8, uxtw #2
add x9, x10, w11, uxtw #3
cnt v16.16b, v17.16b
addv b18, v19.16b
cmeq v20.16b, v21.16b, #0
mvn v22.16b, v23.16b
rbit v16.16b, v16.16b
rbit v3.16b, v5.16b
ldr q0, [x1]
ldr q2, [x3, #32]
str q4, [x5]
str q6, [x7, #48]
ldr q25, [x2, x3]
str q26, [x2, x3]
add v0.16b, v1.16b, v2.16b
add v3.8h, v4.8h, v5.8h
add v6.4s, v7.4s, v8.4s
add v9.2d, v10.2d, v11.2d
sub v0.16b, v1.16b, v2.16b
sub v3.8h, v4.8h, v5.8h
sub v6.4s, v7.4s, v8.4s
sub v9.2d, v10.2d, v11.2d
and v12.16b, v13.16b, v14.16b
orr v15.16b, v16.16b, v17.16b
eor v18.16b, v19.16b, v20.16b
dup v21.16b, w1
dup v22.8h, w2
dup v23.4s, w3
dup v24.2d, x4
cmeq v3.16b, v4.16b, v5.16b
movi v5.16b, #1
ext v2.16b, v0.16b, v0.16b, #8
ushr v0.16b, v1.16b, #3
ushr v2.8h, v3.8h, #5
ushr v4.4s, v5.4s, #7
ushr v6.2d, v7.2d, #9
sshr v8.16b, v9.16b, #3
sshr v10.8h, v11.8h, #5
sshr v12.4s, v13.4s, #7
sshr v14.2d, v15.2d, #9
ushl v16.16b, v17.16b, v18.16b
ushl v19.8h, v20.8h, v21.8h
ushl v22.4s, v23.4s, v24.4s
ushl v25.2d, v26.2d, v27.2d
sshl v0.16b, v1.16b, v2.16b
sshl v3.8h, v4.8h, v5.8h
sshl v6.4s, v7.4s, v8.4s
sshl v9.2d, v10.2d, v11.2d
mul v12.16b, v13.16b, v14.16b
mul v15.8h, v16.8h, v17.8h
mul v18.4s, v19.4s, v20.4s
ins v14.s[0], v19.s[0]
fadd v0.4s, v1.4s, v2.4s
fadd v3.2d, v4.2d, v5.2d
fsub v6.4s, v7.4s, v8.4s
fsub v9.2d, v10.2d, v11.2d
fmul v12.4s, v13.4s, v14.4s
fmul v15.2d, v16.2d, v17.2d
fdiv v18.4s, v19.4s, v20.4s
fdiv v21.2d, v22.2d, v23.2d
fmla v24.4s, v25.4s, v26.4s
fmla v27.2d, v28.2d, v29.2d
fmls v30.4s, v31.4s, v0.4s
fmls v1.2d, v2.2d, v3.2d
fneg v4.4s, v5.4s
fneg v6.2d, v7.2d
dup v8.4s, v9.s[0]
dup v10.2d, v11.d[0]
ldp q0, q1, [x2]
stp q3, q4, [x5, #48]
