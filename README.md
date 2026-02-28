# Stage 3 (revamped): DeBERTa-v3-large + CV ensemble + OOF threshold + optional 7-category auxiliary supervision

The original reference repo relied on notebooks + `simpletransformers` and a Google Drive model download.
Those workflows are preserved under `legacy/`.

This revamp provides a reproducible CLI pipeline aligned to the spec:

* DeBERTa-v3-large encoder
* Binary head + auxiliary 7-category head (optional)
* 5-fold stratified CV
* OOF threshold tuning for **F1(PCL=1)**
* CV ensemble inference
* Strict output formatting for `dev.txt` / `test.txt`
* Dev F1 evaluation identical to the reference (`scripts/eval_dev_f1.py`)

## What you need in `data/`

Minimum (train + dev evaluation):

* `data/train_df.csv` (TSV: must include `par_id`, `text`, `label`, and ideally `target_flag`)
* `data/dev_df_2.csv` (TSV: must include `par_id`, `text`, `label`, `target_flag`)

Optional (aux labels):

* **Option A**: `data/raw/train_semeval_parids-labels.csv` and `data/raw/dev_semeval_parids-labels.csv`
  * columns: `par_id`, `label` where `label` is a 7-dim list string like `[1,0,0,0,0,0,0]`

* **Option B**: `data/raw/dontpatronizeme_categories.tsv` (span-level categories)
  * if Option A is missing, we aggregate spans -> paragraph multi-hot labels

Optional (test predictions):

* `data/task4_test.tsv`

## Run (RunPod)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install CUDA torch first (pick the right index-url for your pod), then:
pip install -r requirements.txt
pip install -e .

python scripts/env_check.py

# 1) Train 5-fold CV + tune OOF threshold
python scripts/train_cv.py --config configs/default.yaml

# 2) Predict dev/test (replace RUN_DIR with the printed one)
python scripts/predict.py --config configs/default.yaml --run_dir outputs/<RUN_DIR> --split both

# 3) Evaluate dev F1 exactly like the reference
python scripts/eval_dev_f1.py --dev_csv data/dev_df_2.csv --pred outputs/<RUN_DIR>/dev.txt
```

