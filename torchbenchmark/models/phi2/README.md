# Phi 2

- Install requirements
```bash
pip install -r requirements.txt
```
- Download hf model
```bash
python scripts/download.py
```

- Convert HF checkpoint to pure pytorch model
```bash
python scripts/convert_hf_checkpoint.py
```

- Run Inference
```bash
python generate.py --checkpoint_path checkpoints/microsoft/phi-2/model.pth --profile ./trace --precision float32
```