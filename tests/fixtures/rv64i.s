# Every RV64I instruction form, one per line, in the order the decoder
# test (crates/champollion-rv/tests/decode_rv64i.rs) expects them.
# Assembled to rv64i.bin by build.sh; the raw words are the reference
# encodings the decoder is checked against.
.option norvc
lui a0, 0x12345
lui t1, 0xfffff
auipc t0, 1
auipc sp, 0x80000
jal ra, 2048
jal zero, -1048576
jalr ra, 16(a0)
jalr zero, -1(ra)
beq a0, a1, -4096
bne a0, a1, 4094
blt t0, t1, 8
bge t0, t1, -8
bltu s0, s1, 16
bgeu s0, s1, -16
lb a0, -2048(sp)
lh a1, -1(sp)
lw a2, 0(sp)
ld a3, 8(sp)
lbu a4, 1(gp)
lhu a5, 2(tp)
lwu a6, 2047(s0)
sb a0, -2048(sp)
sh a1, -1(sp)
sw a2, 0(sp)
sd a3, 2047(sp)
addi a0, a1, -2048
slti a0, a1, 2047
sltiu a0, a1, -1
xori a0, a1, -1
ori a0, a1, 0x7ff
andi a0, a1, 0xff
slli a0, a1, 63
srli a0, a1, 1
srai a0, a1, 63
add s2, s3, s4
sub s2, s3, s4
sll s2, s3, s4
slt s2, s3, s4
sltu s2, s3, s4
xor s2, s3, s4
srl s2, s3, s4
sra s2, s3, s4
or s2, s3, s4
and t4, t5, t6
addiw a0, a1, -1
slliw a0, a1, 31
srliw a0, a1, 0
sraiw a0, a1, 31
addw a0, a1, a2
subw a0, a1, a2
sllw a0, a1, a2
srlw a0, a1, a2
sraw a0, a1, a2
fence
fence rw, w
fence.tso
ecall
ebreak
