// A dynamically linked program (needs a musl sysroot to run; see tests/sysroot).
#include <stdio.h>
#include <math.h>
int main(int argc, char **argv) {
    (void)argv;
    printf("hello from a dynamically linked program, sqrt %.3f\n", sqrt((double)argc + 1));
    return 0;
}
