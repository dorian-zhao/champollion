"""Exercises CPython (Alpine's build, dynamically linked) under rvrun: C extension modules
over shared libraries, subprocesses, threads, signals, sockets, asyncio's epoll loop,
multiprocessing, and files. Prints one deterministic line per check, the same on any
POSIX host, so a native run defines the expected output (tests/sysroot/build.sh python
installs the riscv64 CPython it runs on)."""
import asyncio, bz2, ctypes, decimal, hashlib, json, lzma, os, re, signal, socket
import sqlite3, subprocess, sys, tempfile, threading, time, zlib
from collections import Counter

print("version", sys.version_info[:2] >= (3, 12), os.name)
print("json+re", json.dumps({"k": [1, 2]}), re.sub(r"(\d)", r"<\1>", "a1b2"))
print("decimal", decimal.Decimal(1) / decimal.Decimal(7))
db = sqlite3.connect(":memory:")
db.execute("create table t(x)")
db.executemany("insert into t values (?)", [(i,) for i in range(100)])
print("sqlite3", db.execute("select sum(x) from t").fetchone()[0])
data = b"champollion" * 100
print("hashlib", hashlib.sha256(b"abc").hexdigest()[:16])
print("compress", all(m.decompress(m.compress(data)) == data for m in (zlib, bz2, lzma)))
libc = ctypes.CDLL(None)
print("ctypes", libc.abs(-42), ctypes.sizeof(ctypes.c_void_p))
print("counter", Counter("mississippi").most_common(2))

out = subprocess.run(["sh", "-c", "echo from a child; exit 3"], capture_output=True, text=True)
print("subprocess", out.stdout.strip(), out.returncode)
print("pipeline", subprocess.check_output("printf 'b\\na\\n' | sort | head -1", shell=True, text=True).strip())

results = []
lock = threading.Lock()
def work(n):
    s = sum(range(n * 10000))
    with lock:
        results.append(s)
threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
for t in threads: t.start()
for t in threads: t.join()
print("threads", len(results), sum(results))

got = []
signal.signal(signal.SIGUSR1, lambda s, f: got.append(s))
os.kill(os.getpid(), signal.SIGUSR1)
print("signal", got == [signal.SIGUSR1])
try:
    signal.alarm(1)
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
    time.sleep(5)
    print("alarm", "missed")
except TimeoutError:
    print("alarm", "interrupted sleep")

srv = socket.create_server(("127.0.0.1", 0))
def serve():
    c, _ = srv.accept()
    c.sendall(c.recv(100).upper())
    c.close()
threading.Thread(target=serve).start()
cli = socket.create_connection(srv.getsockname())
cli.sendall(b"over tcp")
print("socket", cli.recv(100).decode())

async def main():
    async def echo(r, w):
        w.write((await r.readline()).upper()); await w.drain(); w.close()
    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(b"asyncio\n"); await w.drain()
    line = await r.readline()
    w.close(); server.close()
    results = await asyncio.gather(*(asyncio.sleep(0.01, i) for i in range(5)))
    return line.decode().strip(), sum(results)
print("asyncio", *asyncio.run(main()))

with tempfile.TemporaryDirectory() as d:
    for i in range(3):
        open(os.path.join(d, f"f{i}"), "w").write(str(i) * (i + 1))
    print("files", sorted(os.listdir(d)), sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)))

if __name__ == "__main__":
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(2) as pool:
        print("multiprocessing", pool.map(abs, [-1, -2, -3]))
