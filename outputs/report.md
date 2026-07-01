# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

## Your Turn Extensions

- Cache economics: break-even is 1.11 reads; observed average is 150.0 reads/prefix, so cache is enabled for 16/16 prefix groups.
- Reasoning budget: reasoning is 8.4% of traffic but 16.5% of optimized cost and 94.0% of inference Wh. A 10% cap is already satisfied; a 5% default cap saves about $12/month and 357,972 Wh/month.

_Figures are June-2026 as-of snapshots; re-baseline before acting._