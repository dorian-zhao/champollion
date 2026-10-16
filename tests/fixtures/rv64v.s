# The RVV subset (see crates/champollion-rv/src/decoder/vector.rs), one instruction
# per line; see crates/champollion-rv/tests/decode.rs.
.option arch, +v,+zvkb
vsetvli a0, a1, e8, m1, ta, ma
vsetvli a0, a1, e32, mf2, ta, ma
vsetvli a0, a1, e64, m1, ta, ma
vsetivli a0, 16, e8, m1, tu, mu
vsetivli a0, 2, e8, mf8, ta, ma
vsetvli a0, a1, e8, m1
vsetvl a0, a1, a2
vle8.v v1, (a0)
vle8ff.v v1, (a0)
vle16.v v1, (a0)
vle32.v v1, (a0)
vle64.v v1, (a0)
vse8.v v1, (a0)
vse16.v v1, (a0)
vse32.v v1, (a0)
vse64.v v1, (a0)
vle8.v v1, (a0), v0.t
vlse64.v v1, (a0), a1
vlm.v v0, (a0)
vl1re64.v v2, (a0)
vl1r.v v2, (a0)
vl2r.v v2, (a0)
vs1r.v v2, (a0)
vs2r.v v2, (a0)
vsext.vf8 v8, v10
vzext.vf4 v8, v10
vmv.v.i v1, 5
vmv.v.x v1, a0
vmv.s.x v1, a0
vmv.x.s a0, v1
vmv1r.v v2, v3
vid.v v1
vmseq.vi v1, v2, 3
vmseq.vx v1, v2, a0
vmsne.vv v1, v2, v3
vmsne.vi v1, v2, 3
vmsleu.vi v1, v2, 3
vmsgtu.vi v1, v2, 3
vmor.mm v1, v2, v3
vfirst.m a0, v1
vcpop.m a0, v1
vadd.vv v1, v2, v3
vadd.vx v1, v2, a0
vadd.vi v1, v2, 3
vadd.vv v1, v2, v3, v0.t
vmacc.vv v1, v2, v3
vmacc.vx v1, a0, v3
vmadd.vv v1, v2, v3
vmadd.vx v1, a0, v3
vxor.vv v1, v2, v3
vxor.vi v1, v2, 3
vor.vv v1, v2, v3
vand.vv v1, v2, v3
vand.vi v1, v2, 3
vandn.vv v1, v2, v3
vmul.vx v1, v2, a0
vrsub.vi v1, v2, 3
vsll.vi v1, v2, 3
vnsrl.wi v1, v2, 3
vslidedown.vi v1, v2, 3
vslideup.vi v1, v2, 3
vslide1down.vx v1, v2, a0
vrgather.vi v1, v2, 3
vrgather.vv v1, v2, v3
vmerge.vvm v1, v2, v3, v0
vredor.vs v1, v2, v3
vredand.vs v1, v2, v3
