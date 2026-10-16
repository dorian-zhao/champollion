// DUT configuration for the user-mode ACT build, replacing the UDB-generated header.
// Only the extensions listed here are claimed; tests do not read these defines, but
// the environment does (see rvmodel_macros.h on why F/D must not be claimed).
#ifndef RVTEST_CONFIG_H
#define RVTEST_CONFIG_H

#define UDB_MXLEN 64
#define UDB_NUM_PMP_ENTRIES 0

#define I_SUPPORTED
#define M_SUPPORTED
#define A_SUPPORTED
#define ZMMUL_SUPPORTED
#define ZBA_SUPPORTED
#define ZBB_SUPPORTED
#define ZBS_SUPPORTED
#define ZICOND_SUPPORTED

// U-mode only: no M-mode CSRs, no S-mode.
#define U_SUPPORTED

#endif
