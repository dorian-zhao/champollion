// Accessors for SoftFloat's thread-local state, which Rust cannot name directly.
#include <stdint.h>
#include "platform.h"
#include "softfloat.h"

void champollion_sf_begin(uint_fast8_t rounding_mode) {
    softfloat_roundingMode = rounding_mode;
    softfloat_detectTininess = softfloat_tininess_afterRounding;
    softfloat_exceptionFlags = 0;
}

uint_fast8_t champollion_sf_flags(void) { return softfloat_exceptionFlags; }
