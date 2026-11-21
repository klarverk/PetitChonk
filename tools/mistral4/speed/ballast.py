"""P4 part B, B6's ballast (P4_benchmark_design.md section 5.4): commit, touch and lock N GiB of RAM so that the
PC behaves like one with that much less, then hold it until stdin closes or a line arrives.  Windows reuses
standby pages (the file cache) before anything locked, so the model's file cache keeps only what is left.

usage: <python> -I ballast.py <GiB>
prints  LOCKED <bytes> <available GiB after>  once held, or  FAILED <step> <winerror>  and exits 1.
No privilege beyond an ordinary user's: raising one's own working set ("Increase a process working set") is granted
to Users by default; VirtualLock needs no other.
"""
import ctypes
import ctypes.wintypes as wt
import sys

GiB = 1 << 30
CHUNK = 256 << 20
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE, PAGE_READWRITE = 0x1000, 0x2000, 0x8000, 0x04
QUOTA_LIMITS_HARDWS_MIN_ENABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x1, 0x8
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
k32.GetCurrentProcess.restype = wt.HANDLE
k32.SetProcessWorkingSetSizeEx.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD]


class MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_gib() -> float:
    m = MemStatus()
    m.dwLength = ctypes.sizeof(MemStatus)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / GiB


def fail(step: str) -> None:
    print(f"FAILED {step} {ctypes.get_last_error()}", flush=True)
    sys.exit(1)


def main(gib: float) -> None:
    size = int(gib * GiB) // CHUNK * CHUNK
    if not k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), size + GiB, size + 2 * GiB,
                                          QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE):
        fail("working_set")
    base = k32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    if not base:
        fail("alloc")
    for off in range(0, size, CHUNK):                  # touch every page, then lock the chunk
        ctypes.memset(base + off, 0xA5, CHUNK)
        if not k32.VirtualLock(base + off, CHUNK):
            fail(f"lock_at_{off // GiB}GiB")
    print(f"LOCKED {size} {available_gib():.2f}", flush=True)
    sys.stdin.readline()                               # held until the caller writes a line or closes stdin
    k32.VirtualFree(base, 0, MEM_RELEASE)
    print(f"RELEASED {available_gib():.2f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(float(sys.argv[1]))
