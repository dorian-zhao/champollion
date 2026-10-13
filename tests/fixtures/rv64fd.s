# F/D data-movement subset, one instruction per line; see
# crates/champollion-rv/tests/decode.rs.
.option norvc
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
