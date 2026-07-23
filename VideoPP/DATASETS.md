# Datasets

We build the AVR benchmark by injecting synthetic anomalies into the **normal**
(training) clips of three video-anomaly-detection datasets. The original normal
clip is the ground-truth restoration target; the injection region is the GT mask.
ShanghaiTech's **test** split (real anomalies) is used only for qualitative demos.

Download each dataset and place it under `data/raw/` (or anywhere; pass the path
with `--raw`). The loaders locate a dataset by a marker directory name (below), so
the exact parent folder does not matter.

## 1. UCSD Ped2

```bash
mkdir -p data/raw && cd data/raw
wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
tar xzf UCSD_Anomaly_Dataset.tar.gz && cd -
```
Marker dir: `UCSDped2/` (loader reads `UCSDped2/Train/Train###/*.tif`).

## 2. CUHK Avenue

```bash
cd data/raw
wget http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip
unzip -q Avenue_Dataset.zip && cd -
```
Marker dir: `training_videos/` (loader reads `training_videos/*.avi`).

## 3. ShanghaiTech (via Kaggle CLI)

```bash
pip install kaggle   # then place kaggle.json in ~/.kaggle/
# normal (train) split — required for the benchmark
kaggle datasets download -d nikanvasei/shanghaitech-campus-dataset -p data/raw/shanghaitech --unzip
# test split (real anomalies) — optional, for scripts/case_real.py
kaggle datasets download -d nikanvasei/shanghaitech-campus-dataset-test -p data/raw/shanghaitech_test --unzip
```
Marker dir: `frames/` under `SHANGHAI/SHANGHAI_TRAIN/` (loader reads
`frames/<video>/*.jpg`). Test split: `SHANGHAI/SHANGHAI_Test/{frames,label}`.

## 4. Build a benchmark

```bash
# ped2 / avenue search recursively under data/raw
python scripts/prepare_dataset.py --dataset ped2   --raw data/raw --out data/ped2_avr   --clips 60 --T 16 --size 256
python scripts/prepare_dataset.py --dataset avenue --raw data/raw --out data/avenue_avr --clips 60 --T 16 --size 256
# shanghaitech: point --raw at its own folder
python scripts/prepare_dataset.py --dataset shanghaitech --raw data/raw/shanghaitech --out data/sht_avr --clips 60 --T 16 --size 256

# add --quick for a tiny 6-clip smoke benchmark
```

Each benchmark is a folder of `(anom.npy, clean.npy, mask.npy, meta.json)` triples
plus `manifest.json`, consumed by every `scripts/*.py`. Anomaly types injected:
`static_object`, `moving_object`, `local_appearance` (round-robin).

The experiment scripts in `experiments/` expect the 60-clip benchmarks named
`data/{ped2,avenue,sht}_night` — build those with the commands above, changing
`--out` accordingly.

## 5. Real-anomaly qualitative (ShanghaiTech test)

```bash
python scripts/case_real.py --video 02_0161 --prompt "a bicycle. a person riding a bicycle." --out runs/case_bike
```
Produces a side-by-side removal video + flicker curve on a real anomaly located
open-world by `grounded_sam2` (no GT mask needed).

## Note on baselines

No prior work reports restoration fidelity on these datasets under our
anomaly-injection protocol, so external baselines (ProPainter, E2FGVI, …) must be
run here, not copied. See `BASELINES.md`.
