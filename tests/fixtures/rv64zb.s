# Zba, Zbb, Zbs, Zicond, and the Zcb compressed forms, one instruction per line;
# see crates/champollion-rv/tests/decode.rs.
.option arch, +zba,+zbb,+zbs,+zicond,+zcb
# Zba
sh1add a0, a1, a2
sh2add a3, a4, a5
sh3add a6, a7, s0
sh1add.uw s1, s2, s3
sh2add.uw s4, s5, s6
sh3add.uw s7, s8, s9
add.uw s10, s11, a0
slli.uw a1, a2, 40
# Zbb register-register
andn a0, a1, a2
orn a3, a4, a5
xnor a6, a7, s0
max s7, s8, s9
maxu s10, s11, a0
min a1, a2, a3
minu a4, a5, a6
rol s9, s10, s11
ror a0, a1, a2
# Zbb unary and immediate
clz s1, s2
ctz s3, s4
cpop s5, s6
sext.b a7, s0
sext.h s1, s2
zext.h s3, s4
rev8 s5, s6
orc.b s7, s8
rori a3, a4, 60
rolw a5, a6, a7
rorw s0, s1, s2
roriw s3, s4, 25
# Zbs
bclr s5, s6, s7
bext s8, s9, s10
binv s11, a0, a1
bset a2, a3, a4
bclri a5, a6, 63
bexti a7, s0, 20
binvi s1, s2, 33
bseti s3, s4, 42
# Zicond
czero.eqz s5, s6, s7
czero.nez s8, s9, s10
# Zcb
c.lbu a0, 2(s1)
c.lhu a1, 2(s1)
c.sb a2, 3(s1)
c.sh a3, 2(s1)
c.zext.b a0
c.sext.b a1
c.zext.h a2
c.sext.h a3
c.zext.w a4
c.not a5
c.mul a0, a1
