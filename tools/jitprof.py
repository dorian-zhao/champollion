#!/usr/bin/env python3
"""Render an rvrun JIT profile (RVRUN_PROF=FILE): disassemble the generated arm64 code of the
hottest blocks with llvm-mc and show each host instruction's share of the block's samples.

    RVRUN_PROF=/tmp/prof.txt rvrun prog args...
    tools/jitprof.py /tmp/prof.txt
"""
import subprocess
import sys

LLVM_MC = "/opt/homebrew/opt/llvm/bin/llvm-mc"
LLVM_CXXFILT = "/opt/homebrew/opt/llvm/bin/llvm-cxxfilt"


def demangle(names):
    out = subprocess.run([LLVM_CXXFILT], input="\n".join(names), capture_output=True, text=True)
    lines = out.stdout.splitlines()
    return lines if len(lines) == len(names) else names


def disassemble(words):
    data = " ".join(f"0x{b:02x}" for w in words for b in w.to_bytes(4, "little"))
    out = subprocess.run(
        [LLVM_MC, "--disassemble", "-triple=aarch64", "-mattr=+lse"],
        input=data, capture_output=True, text=True,
    ).stdout
    lines = [l.strip() for l in out.splitlines() if l.strip() and not l.strip().startswith(".")]
    return lines if len(lines) == len(words) else [f"<{w:08x}>" for w in words]


def flush(block_lines, words, counts):
    total = sum(counts) or 1
    for line in block_lines:
        print(line)
    for asm, c in zip(disassemble(words), counts):
        bar = "#" * round(40 * c / total)
        print(f"  {c:6d} {asm:<40} {bar}")


def main(path):
    lines = open(path).read().splitlines()
    symbols = [i for i, l in enumerate(lines) if l.startswith("  symbol ")]
    names = demangle([lines[i].split(" ", 4)[4] for i in symbols])
    for i, name in zip(symbols, names):
        lines[i] = f"  {int(lines[i].split()[1]):6d} {name}"
    block, words, counts = [], [], []
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("  host "):
            _, w, c = line.split()
            words.append(int(w, 16))
            counts.append(int(c))
            continue
        if words:
            flush(block, words, counts)
            block, words, counts = [], [], []
        block.append(line)
    flush(block, words, counts)


if __name__ == "__main__":
    main(sys.argv[1])
