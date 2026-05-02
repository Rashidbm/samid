# samid

Acoustic drone detector. AST fine-tuned on geronimobasso + NUS DroneAudioSet
with domain adaptation on real-world recordings.

Trained model: https://huggingface.co/Rashidbm/samid-drone-detector

## Run

```bash
pip install torch transformers soundfile sounddevice numpy scipy
python scripts/standalone_inference.py --wav clip.mp3
```

Live mic:

```bash
python scripts/standalone_inference.py
```

## Training

```bash
python scripts/download_geronimobasso.py
python -m src.train
python scripts/finetune_abdulrahman.py --abd-wav your_recording.wav
python scripts/push_to_hub.py --repo username/samid-drone-detector
```

## Layout

```
src/
  config.py       hyperparameters
  data.py         dataset, windowing, splits
  model.py        AST + binary head
  losses.py       focal loss
  metrics.py      F1 / PR-AUC / threshold sweep
  augment.py      codec / RIR / EQ / FilterAugment / Patchout / SpecAugment
  inference.py    median filter + consecutive-window aggregation
  train.py        training loop
scripts/
  download_geronimobasso.py
  finetune_abdulrahman.py
  push_to_hub.py
  standalone_inference.py
```

## License

MIT
