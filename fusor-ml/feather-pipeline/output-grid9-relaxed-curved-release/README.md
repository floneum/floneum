# Feather Grid Pipeline

This report was generated from the full SVG corpus under `/tmp/feather-icons-repo/icons`. Each icon was parsed with `usvg`, flattened into absolute polylines, fitted to its tight geometry bounding box, quantized onto a `9x9` lattice, phase-aligned over a small translation search, and then filtered conservatively.

Frozen thresholds:

- mean symmetric error `<= 0.20` grid cells
- max symmetric error `<= 0.40` grid cells
- turn inflation `<= 1.35x`
- path length inflation `<= 1.50x`
- rendered overlap F1 `>= 0.83`
- rendered overlap IoU `>= 0.70`
- no introduced repeated internal lattice points beyond the source icon's own intersection topology
- no repeated edges
- no introduced self-intersections
- no disconnected or ambiguous closed-contour graphs
- circles and ellipses are rejected by default unless `--allow-curved-source` is used
- acceptance uses a mixed score that combines line distance with rendered stroke overlap
- a reproducible mixed-metric rescue can clear only soft failures (`mean_error`, `max_error`, `turn_inflation`, `render_f1`, `render_iou`, `circle_source_blacklist`, `ellipse_source_blacklist`) when rendered overlap is strong and topology remains clean

Run command:

```bash
cargo run -p fusor-feather-pipeline -- --input /path/to/feather/icons --output /path/to/report --grid 9
```

Current run summary: `130` accepted, `157` rejected, `45.3%` acceptance. Mixed-metric rescues applied: `129`.
