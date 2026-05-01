# Samid — Acoustic Drone Detection

Built for Defensathon 2026 (Saudi defense hackathon, Drones + AI track).

Passive acoustic drone detection, classification, and localisation.
Designed for cheap distributed microphone nodes running at the edge — no
cloud dependency.

## Trained model

Hosted on Hugging Face:
**https://huggingface.co/Rashidbm/samid-drone-detector**

```bash
pip install torch transformers soundfile sounddevice numpy
python scripts/standalone_inference.py --hub-id Rashidbm/samid-drone-detector
```

## Performance (held-out test, 18,032 samples)

| Metric    | Value  |
|-----------|--------|
| F1 @ 0.30 | 0.9974 |
| Precision | 1.0000 |
| Recall    | 0.9948 |
| PR-AUC    | 1.0000 |

Within-dataset numbers. Cross-dataset performance expected lower.

## Repo layout

- `src/` — model, training, inference, triangulation, decoy detection, open-set head
- `scripts/` — dataset download, training orchestrator, model push, standalone inference
- `pyproject.toml` — dependencies (uv-managed)

## License

MIT.
