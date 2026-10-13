#!/usr/bin/env python3
"""Render an rvrun JIT profile (RVRUN_PROF=FILE): disassemble the generated arm64 code of the
hottest blocks with llvm-mc and show each host instruction's share of the block's samples.

    RVRUN_PROF=/tmp/prof.txt rvrun prog args...
    tools/jitprof.py /tmp/prof.txt
"""
import subprocess
import sys

LLVM_MC = "/opt/homebrew/opt/llvm/bin/llvm-mc"


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
    block, words, counts = [], [], []
    for line in open(path):
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
