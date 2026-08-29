# Benchmark result summaries

Commit only compact, reproducible summaries here. Large checkpoints, videos, raw simulator dumps and caches stay outside git.

Recommended committed artifacts:

- one JSON summary per promoted experiment,
- aggregate CSV for model/data comparisons,
- notes explaining promotion/rejection decisions.

Each summary should reference the run manifest created by `tools/experiment/write_run_manifest.py` and include its git SHA, dataset/eval manifest hashes, GPU, wall time and metrics.

Suggested result record:

```json
{
  "run_id": "pi05-r16-equal-data-001",
  "model": "pi05",
  "protocol": "equal_data",
  "track_like": "track1_like",
  "success_rate": 0.0,
  "policy_latency_ms": 0.0,
  "peak_vram_mib": 0,
  "train_wall_time_s": 0,
  "git_sha": "...",
  "dataset_manifest_sha256": "...",
  "eval_split_sha256": "..."
}
```
