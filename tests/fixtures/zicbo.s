# Zicbop / Zicbom / Zicboz / Zawrs, one per line; see crates/champollion-rv/tests/decode.rs.
# prefetch hints decode as the ORI-to-x0 they are encoded as.
.option norvc
.option arch, +zicbop,+zicbom,+zicboz,+zawrs
prefetch.r 32(a0)
prefetch.w 64(a0)
prefetch.i 0(a0)
cbo.clean (a0)
cbo.flush (a0)
cbo.inval (a0)
cbo.zero (a0)
wrs.nto
wrs.sto
