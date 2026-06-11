#!/usr/bin/env python3
# compare_grid.py — NGTAQ vs max(QBG, QG) QPS ratio at fixed recall targets.
# Usage: compare_grid.py <ngtaq_log> <qbg_log> <qg_log> [label]
# Parses the "recall@k = ..." / "agg_QPS = ..." blocks emitted by ann_bench,
# qbg_bench, and qg_bench. A cell PASSES only if NGTAQ >= 2x the stronger competitor.
import sys, re

TARGETS = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

def parse(path):
    pts = []
    rec = qps = None
    try:
        lines = open(path)
    except FileNotFoundError:
        return pts
    for line in lines:
        m = re.search(r'recall@\d+\s*=\s*([\d.]+)', line)
        if m: rec = float(m.group(1))
        m = re.search(r'agg_QPS\s*=\s*([\d.]+)', line)
        if m: qps = float(m.group(1))
        if rec is not None and qps is not None:
            pts.append((rec, qps)); rec = qps = None
    return pts

def qps_at(pts, t):
    """Best QPS achievable while still meeting recall >= t; else interpolate."""
    q = [p[1] for p in pts if p[0] >= t]
    if q: return max(q)
    s = sorted(pts)
    for (r0, q0), (r1, q1) in zip(s, s[1:]):
        if r0 <= t <= r1 and r1 > r0:
            return q0 + (q1 - q0) * (t - r0) / (r1 - r0)
    return None

def fmt(x): return f"{x:.0f}" if x else "N/A"

def main():
    if len(sys.argv) < 4:
        print("usage: compare_grid.py <ngtaq_log> <qbg_log> <qg_log> [label]"); sys.exit(1)
    ng, qb, qg = parse(sys.argv[1]), parse(sys.argv[2]), parse(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else ""
    ng_max = max((r for r, _ in ng), default=0.0)
    print(f"# {label}  NGTAQ_maxrecall={ng_max:.4f}  "
          f"(pts: ngtaq={len(ng)} qbg={len(qb)} qg={len(qg)})")
    print(f"{'recall':>7} | {'NGTAQ':>8} | {'QBG':>8} | {'QG':>8} | {'vs max':>7} | verdict")
    print("-" * 64)
    for t in TARGETS:
        a = qps_at(ng, t); b = qps_at(qb, t); c = qps_at(qg, t)
        comp = [x for x in (b, c) if x]
        mx = max(comp) if comp else None
        if a is None and t > ng_max + 1e-9:
            print(f"{t:>7.2f} | {'unreach':>8} | {fmt(b):>8} | {fmt(c):>8} | "
                  f"{'-':>7} | NGTAQ cannot reach")
            continue
        ratio = (a / mx) if (a and mx) else None
        v = ("PASS>=2x" if ratio and ratio >= 2 else
             "win<2x"  if ratio and ratio >= 1 else
             "LOSS"    if ratio else "N/A")
        print(f"{t:>7.2f} | {fmt(a):>8} | {fmt(b):>8} | {fmt(c):>8} | "
              f"{(f'{ratio:.2f}x' if ratio else 'N/A'):>7} | {v}")

if __name__ == "__main__":
    main()
