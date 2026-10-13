// Dereferences an unmapped address; rvrun must exit like a Linux process killed by SIGSEGV.
int main(void) {
    volatile int *p = (int *)0x10;
    return *p;
}
