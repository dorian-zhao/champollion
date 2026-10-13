# Zicsr on the user-mode CSRs we implement, one instruction per line; see
# crates/champollion-rv/tests/decode.rs.
.option norvc
csrrw a0, fcsr, a1
csrrs t0, fflags, zero
csrrc zero, frm, a2
csrrwi a0, frm, 3
csrrsi a1, fflags, 31
csrrci a2, fcsr, 1
csrrs a3, cycle, zero
csrrs a4, time, zero
csrrs a5, instret, zero
csrrc s0, instret, zero
