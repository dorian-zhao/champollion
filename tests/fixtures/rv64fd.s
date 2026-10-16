# F/D data-movement subset, one instruction per line; see
# crates/champollion-rv/tests/decode.rs.
.option norvc
.option arch, +f,+d,+zfa
fli.s fa0, -1.0
fli.d fa1, 1.5
flw fa0, -4(sp)
fld fs11, 2040(a0)
fsw ft0, 0(a1)
fsd fa7, -2048(s0)
fmv.x.w a0, fa1
fmv.w.x ft1, t2
fmv.x.d a2, fs0
fmv.d.x fa5, zero
fsgnj.s fa0, fa1, fa2
fsgnjn.s fa0, fa1, fa2
fsgnjx.s fa0, fa1, fa2
fsgnj.d ft3, ft4, ft5
fsgnjn.d ft3, ft4, ft5
fsgnjx.d ft3, ft4, ft5
fadd.s fa0, fa1, fa2, rne
fsub.d ft0, ft1, ft2, rtz
fmul.s fs0, fs1, fs2, rdn
fdiv.d fa3, fa4, fa5, rup
fadd.d fa0, fa1, fa2, rmm
fmul.d fa0, fa1, fa2, dyn
fsqrt.s ft3, ft4
fsqrt.d ft3, ft4, rtz
fmadd.s fa0, fa1, fa2, fa3
fmsub.d fa0, fa1, fa2, fa3, rne
fnmsub.s ft0, ft1, ft2, ft3, rtz
fnmadd.d fs0, fs1, fs2, fs3
fmin.s fa0, fa1, fa2
fmax.d fa0, fa1, fa2
feq.s a0, fa1, fa2
flt.d a1, ft0, ft1
fle.s a2, fs0, fs1
fclass.d a3, fa0
fcvt.w.s a0, fa0, rtz
fcvt.wu.d a1, fa1, rne
fcvt.l.s a2, fa2
fcvt.lu.d a3, fa3, rmm
fcvt.s.w fa0, a0
fcvt.d.wu fa1, a1
fcvt.s.l fa2, a2, rdn
fcvt.d.lu fa3, a3
fcvt.s.d fa0, fa1
fcvt.d.s fa2, fa3
