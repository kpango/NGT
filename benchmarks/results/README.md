# ArcFlare ANN-Benchmarks Results

Branch: feat/arcflare-speedup
Date: 2026-05-30

## Build Configuration
- K-means: K=2000 clusters
- Max edges: 64 per node
- Alpha (pruning): 1.2
- NGT ANNG edge_size: 10
- Tier-1 quantization: RaBitQ (sign bits, D/8 bytes)
- Tier-2 quantization: M_PQ=D/8 PQ sub-codebooks (K=256, D_sub=8)

## Results Summary

| Dataset | Metric | N | D_eff | Recall@10 | QPS | P50(us) | P99(us) | Target |
|---------|--------|---|-------|-----------|-----|---------|---------|--------|
| SIFT-128 | L2 | 1M | 128 | 0.9895 | 6089 | 622 | 1588 | >=0.99, >=5000 |
| GloVe-100 | angular | 1.18M | 128 | 0.9902 | 291 | 12069 | 40695 | >=0.99, >=5000 |
| NYTimes-256 | angular | 290K | 256 | TBD | TBD | - | - | >=0.99, >=3000 |
| FashionMNIST-784 | L2 | 60K | 1024 | 0.9999 | 3098 | 1066 | 5165 | >=0.99, >=1000 |
| GIST-960 | L2 | 1M | 1024 | TBD | TBD | - | - | >=0.99, >=1500 |

## QBG Baseline
- recall=0.6886, QPS=3774

## Notes
- NYTimes-256: Initial build had 239 degenerate (zero-norm) vectors causing false cluster trapping.
  Fixed in ArcFlareIndex.cpp: zero vectors now marked as holes in angular metric mode.
  Rebuild required for full performance.
- GloVe-100: Best QPS at recall>=0.99 achieved with gamma_term=0.8, rerank_factor=20.
  QPS=291 is below target (5000); root cause: thin ANNG graph (edge_size=10) limits
  graph-based routing recall ceiling for angular metrics.
