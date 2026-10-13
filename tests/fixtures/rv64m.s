# M extension, one instruction per line; see crates/champollion-rv/tests/decode.rs.
.option norvc
mul a0, a1, a2
mulh a0, a1, a2
mulhsu a0, a1, a2
mulhu a0, a1, a2
div s2, s3, s4
divu s2, s3, s4
rem t4, t5, t6
remu t4, t5, t6
mulw a0, a1, a2
divw a0, a1, a2
divuw a0, a1, a2
remw a0, a1, a2
remuw a0, a1, a2
