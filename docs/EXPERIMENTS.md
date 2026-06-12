# Experiments

COIFPM:

```bash
python scripts/run_coifpm_experiments.py
```

DRSRD1:

```bash
python scripts/run_drsrd1_experiments.py
```

One COIFPM scale:

```bash
python scripts/evaluate_bicubic.py --config configs/coifpm_4x.yaml
python scripts/train.py --config configs/coifpm_4x.yaml --models simplified_edsr
```

DRSRD1 manual run:

```bash
python scripts/evaluate_bicubic.py --config configs/drsrd1_4x.yaml
python scripts/train.py --config configs/drsrd1_4x.yaml --models srcnn fsrcnn edsr rcan
```
