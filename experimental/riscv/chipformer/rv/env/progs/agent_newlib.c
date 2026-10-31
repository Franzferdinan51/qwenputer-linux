// newlib syscall glue for the bare-metal agent build (xPack toolchain
// route; the Debian picolibc route keeps using env/sys.c instead).
// putchar is ecall a7=1, exit is ecall a7=93 - see rt.h.
#include <sys/stat.h>
#include <sys/types.h>

#include "rt.h"

int _write(int fd, const char *buf, int len) {
    (void)fd;
    for (int i = 0; i < len; i++) rv_putchar(buf[i]);
    return len;
}

int _read(int fd, char *buf, int len) {
    (void)fd; (void)buf; (void)len;
    return 0;
}

int _close(int fd) { (void)fd; return -1; }
int _lseek(int fd, int off, int whence) {
    (void)fd; (void)off; (void)whence;
    return 0;
}
int _fstat(int fd, struct stat *st) { (void)fd; (void)st; return -1; }
int _isatty(int fd) { (void)fd; return 1; }
int _kill(int pid, int sig) { (void)pid; (void)sig; return -1; }
int _getpid(void) { return 1; }

void *_sbrk(int incr) {
    extern char __heap_start;
    static char *brk;
    if (!brk) brk = &__heap_start;
    if (brk + incr > (char *)0x7000) return (void *)-1;
    char *old = brk;
    brk += incr;
    return old;
}

void _exit(int code) {
    for (;;) rv_ecall2(93, code, 0, 0);
}
