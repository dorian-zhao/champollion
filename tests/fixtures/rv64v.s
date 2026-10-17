# The RVV subset (see crates/champollion-rv/src/decoder/vector.rs), one instruction
# per line; see crates/champollion-rv/tests/decode.rs.
.option arch, +v,+zvbb,+zvkb,+zvfhmin
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
vlm.v v1, (a1)
vsm.v v3, (a0)
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
vmand.mm v1, v2, v3
vmnand.mm v4, v5, v6
vmandn.mm v7, v8, v9
vmxor.mm v10, v11, v12
vmnor.mm v13, v14, v15
vmorn.mm v16, v17, v18
vmxnor.mm v19, v20, v21
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
vredsum.vs v1, v2, v3
vsub.vv v1, v2, v3
vsub.vx v1, v2, a0
vsrl.vv v1, v2, v3
vsrl.vx v1, v2, a0
vsrl.vi v1, v2, 3
vsra.vv v1, v2, v3
vsra.vx v1, v2, a0
vsra.vi v1, v2, 3
vmul.vv v1, v2, v3
vmslt.vv v1, v2, v3
vmslt.vx v1, v2, a0
vmsltu.vv v1, v2, v3
vmsle.vv v1, v2, v3
vmsle.vi v1, v2, 3
vmsgt.vi v1, v2, 3
vredxor.vs v1, v2, v3
vmerge.vim v1, v2, 3, v0
vmerge.vxm v1, v2, a0, v0
vcompress.vm v1, v2, v3
vwaddu.wv v1, v2, v3
vwaddu.wx v1, v2, a0
vnsrl.wv v1, v2, v3
vnsrl.wx v1, v2, a0
vnsra.wv v1, v2, v3
vnsra.wi v1, v2, 3
vluxei8.v v1, (a0), v2
vluxei16.v v1, (a0), v2
vluxei32.v v1, (a0), v2
vluxei64.v v1, (a0), v2
vror.vv v1, v2, v3
vror.vx v1, v2, a0
vror.vi v1, v2, 7
vror.vi v1, v2, 40
vrol.vv v1, v2, v3
vrol.vx v1, v2, a0
vrev8.v v1, v2
vbrev8.v v1, v2
vandn.vx v1, v2, a0
vlseg2e8.v v2, (a0)
vsseg2e64.v v4, (a1)
vlseg3e8.v v6, (a0)
vlseg8e16.v v8, (a0)
vlsseg8e32.v v16, (a0), a2
vssseg4e16.v v20, (a1), a3
vsuxei8.v v2, (a0), v4
vsuxei16.v v2, (a0), v4
vsuxei32.v v2, (a0), v4
vsuxei64.v v2, (a0), v4
vl2r.v v4, (a1)
vl4r.v v12, (a1)
vs4r.v v12, (a1)
vfadd.vv v1, v2, v3
vfadd.vf v1, v2, fa0
vfmacc.vv v1, v2, v3
vfmacc.vf v1, fa0, v2
vfmin.vv v1, v2, v3
vfsqrt.v v1, v2
vfclass.v v1, v2
vfcvt.x.f.v v1, v2
vfcvt.rtz.xu.f.v v1, v2
vfwcvt.f.f.v v1, v2
vfncvt.f.f.w v3, v4
vmfeq.vv v1, v2, v3
vmflt.vf v1, v2, fa0
vfredosum.vs v1, v2, v3
vfmv.v.f v1, fa0
vfmv.s.f v1, fa0
vfmv.f.s fa0, v1
vfmerge.vfm v1, v2, fa0, v0
vfslide1down.vf v1, v2, fa0
vfslide1up.vf v1, v2, fa0
