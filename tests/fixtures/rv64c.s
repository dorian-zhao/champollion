# C extension (integer forms), one instruction per line; see
# crates/champollion-rv/tests/decode.rs. Immediates hit field boundaries.
.option rvc
c.addi4spn s0, sp, 4
c.addi4spn a5, sp, 1020
c.lw a0, 0(a1)
c.lw s1, 124(a5)
c.ld a0, 0(a1)
c.ld a2, 248(s0)
c.sw a0, 124(a1)
c.sd a3, 248(a4)
c.nop
c.addi a0, -32
c.addi t6, 31
c.addiw a0, -1
c.li ra, 31
c.li a0, -32
c.addi16sp sp, -512
c.addi16sp sp, 496
c.lui a0, 1
c.lui t0, 0xfffff
c.lui s1, 0x1f
c.srli a0, 1
c.srli a5, 63
c.srai s0, 63
c.andi a1, -32
c.andi a1, 31
c.sub s0, s1
c.xor a0, a1
c.or a2, a3
c.and a4, a5
c.subw s0, a5
c.addw a0, a1
c.j -2048
c.j 2046
c.beqz a0, -256
c.bnez s1, 254
c.slli a0, 1
c.slli t6, 63
c.lwsp ra, 252(sp)
c.lwsp a0, 0(sp)
c.ldsp ra, 504(sp)
c.ldsp a0, 8(sp)
c.jr ra
c.mv a0, a1
c.ebreak
c.jalr t0
c.add a0, t6
c.swsp ra, 252(sp)
c.sdsp s0, 504(sp)
c.fld fa0, 248(a1)
c.fsd fs1, 0(a5)
c.fldsp ft0, 504(sp)
c.fsdsp fs11, 8(sp)
