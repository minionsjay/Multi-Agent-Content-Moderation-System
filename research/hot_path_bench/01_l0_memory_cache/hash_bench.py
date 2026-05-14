#!/usr/bin/env python3
"""
Compare performance of SHA256 vs MD5 hashing.
Checks for SHA hardware acceleration (sha_ni flag on x86).
"""

import os
import sys
import time
import hashlib
import platform

def cpu_has_sha_ni():
    """Check if the CPU supports SHA-NI instructions (x86 only)."""
    if platform.machine().lower() not in ('x86_64', 'amd64', 'i386', 'i686'):
        return None  # not x86
    try:
        # On Linux, read /proc/cpuinfo for 'sha_ni' flag
        if sys.platform == 'linux':
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('flags'):
                        flags = line.strip().split(':')[1].split()
                        return 'sha_ni' in flags
        # On other platforms (Windows, macOS) could use cpuinfo or py-cpuinfo
        # For simplicity, fallback to None
        return None
    except:
        return None

def benchmark_hash(data_size_bytes, iterations=10000):
    """
    Benchmark SHA256 and MD5 for a given data size.
    Returns average time per hash in microseconds.
    """
    # Generate random data of the given size
    data = os.urandom(data_size_bytes)
    
    # Warm-up (ensure everything is loaded)
    _ = hashlib.sha256(data).digest()
    _ = hashlib.md5(data).digest()
    
    # Benchmark SHA256
    start = time.perf_counter()
    for _ in range(iterations):
        hashlib.sha256(data).digest()
    sha_time = (time.perf_counter() - start) / iterations * 1_000_000  # μs
    
    # Benchmark MD5
    start = time.perf_counter()
    for _ in range(iterations):
        hashlib.md5(data).digest()
    md5_time = (time.perf_counter() - start) / iterations * 1_000_000  # μs
    
    return sha_time, md5_time

def main():
    print("=" * 70)
    print("SHA256 vs MD5 Performance Test")
    print("=" * 70)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {platform.processor() or 'Unknown'}")
    
    sha_ni = cpu_has_sha_ni()
    if sha_ni is True:
        print("SHA-NI (hardware acceleration): **SUPPORTED**")
    elif sha_ni is False:
        print("SHA-NI (hardware acceleration): NOT supported")
    else:
        print("SHA-NI (hardware acceleration): Unknown (non-x86 or detection failed)")
    
    print("\nTesting with different data chunk sizes...")
    print(f"{'Size':>8} {'SHA256 (μs)':>12} {'MD5 (μs)':>12} {'Ratio (MD5/SHA256)':>20}")
    print("-" * 70)
    
    sizes = [64, 256, 1024, 4096, 16384, 65536]  # bytes
    iterations = 50000 if sizes[0] <= 256 else 10000 if sizes[2] <= 4096 else 2000
    
    for size in sizes:
        # Adjust iterations so test takes reasonable time
        iters = iterations
        if size > 16384:
            iters = max(500, iters // 4)
        sha, md5 = benchmark_hash(size, iters)
        ratio = md5 / sha  # >1 means MD5 slower than SHA256
        print(f"{size:8d} {sha:12.2f} {md5:12.2f} {ratio:19.3f}")
    
    print("\nInterpretation:")
    print("* Ratio > 1.0  -> SHA256 is faster than MD5 (likely due to hardware acceleration)")
    print("* Ratio < 1.0  -> SHA256 is slower than MD5 (pure software comparison)")
    print("* On x86 with SHA-NI, SHA256 often outperforms MD5 for moderate/large blocks.\n")

if __name__ == "__main__":
    main()