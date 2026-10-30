# Dataset preparation

Run the commands from the repository root. Set `DATASET_ROOT` to the directory
used by the training option `--dataset_dir`.

```bash
export DATASET_ROOT=/path/to/dataset
mkdir -p "$DATASET_ROOT"
```

## 1. Extract pathology patches

- **AGGC:** Request WSIs and annotations from the [AGGC challenge](https://aggc22.grand-challenge.org/Schedule/).
- **CCTGS:** Apply through [METU-CCTGS](https://eakbas.github.io/metu-cctgs/).
  We cannot redistribute CCTGS due to access permissions.
- Configure and run `AGGC/extract_patch.ipynb` and `CCTGS/extract_patch.ipynb`
  using the lists in `WSI_Splits/`. Set the input and output paths in the
  notebooks. Save images and masks to `images/<split>` and `annotations/<split>`
  under `$DATASET_ROOT/aggc2022_10x_1024` and `$DATASET_ROOT/cctgs_10x_1024`.

| Dataset | Train pairs | Validation pairs | Test pairs |
|---|---:|---:|---:|
| AGGC | 47,512 | 1,097 | 24,923 |
| CCTGS | 11,279 | 629 | 1,502 |

## 2. Prepare CoNIC

Obtain the development data from the [CoNIC challenge](https://conic-challenge.grand-challenge.org/Data/).
If needed, use the [official extraction script](https://github.com/TissueImageAnalytics/CoNIC/blob/main/extract_patches.py)
to prepare `images.npy`, `labels.npy` and `patch_info.csv`. Run
`CoNIC/conic_proc.ipynb`, then place the resulting `data_split/images` and
`data_split/annotations` under `$DATASET_ROOT/conic`.

## 3. Prepare PathVQA images

Download the Git LFS parts, reconstruct the archive and verify its SHA-256:

```bash
mkdir -p "$DATASET_ROOT/pathvqa30k"
git lfs pull --include='data_preprocess_scripts/PathVQA/pathvqa_images.zip.part*'
cat data_preprocess_scripts/PathVQA/pathvqa_images.zip.part00 \
    data_preprocess_scripts/PathVQA/pathvqa_images.zip.part01 \
    data_preprocess_scripts/PathVQA/pathvqa_images.zip.part02 \
    > "$DATASET_ROOT/pathvqa30k/pathvqa_images.zip"
printf '%s  %s\n' '863780b6d184c5c08306399e2aafeaf4425ea5fb1e59bff5494a3d6fa219d29c' \
    "$DATASET_ROOT/pathvqa30k/pathvqa_images.zip" | sha256sum -c -
unzip "$DATASET_ROOT/pathvqa30k/pathvqa_images.zip" -d "$DATASET_ROOT/pathvqa30k"
```

The [PathVQA dataset reference](https://github.com/sarahESL/PathVQA) describes
the original data. The supplied TumorVQA JSONs already include its training
records; do not append them again.

## 4. Assemble TumorVQA and TumorVQA+

The supplied `train_vqa.json` files each contain 34,654 records, including
PathVQA. Standard segmentation patches omit 7,500 tumor-negative images used
by each version. Generate additional patches from training WSIs using the
later extraction cells in both notebooks, with the zero-mask filter disabled:

- AGGC: reset `root` and `anno` to the training sources and set `save_dir` to
  a separate output directory before running the later `train data` cell.
- CCTGS: configure `svs_roots` and `export_dir`, then run the final extraction
  function and training cell.

The cells use stride 1024. Setting stride to 512 while retaining negative
patches was also verified against all tumor-patch images in both VQA JSONs.
Use the output image directories in the commands below:

```bash
export AGGC_VQA_IMAGES=/path/to/aggc_vqa_images
export CCTGS_VQA_IMAGES=/path/to/cctgs_vqa_images
export PATHVQA_IMAGE_DIR="$DATASET_ROOT/pathvqa30k/images"

python data_preprocess_scripts/prepare_vqa.py \
  --annotations data_preprocess_scripts/TumorVQA_Annotations/train_vqa.json \
  --image-dir "$AGGC_VQA_IMAGES" --image-dir "$CCTGS_VQA_IMAGES" \
  --image-dir "$DATASET_ROOT/aggc2022_10x_1024/images" \
  --image-dir "$DATASET_ROOT/cctgs_10x_1024/images" \
  --image-dir "$PATHVQA_IMAGE_DIR" --output-dir "$DATASET_ROOT/vqa"

python data_preprocess_scripts/prepare_vqa.py \
  --annotations 'data_preprocess_scripts/TumorVQA+Annotations/train_vqa.json' \
  --image-dir "$AGGC_VQA_IMAGES" --image-dir "$CCTGS_VQA_IMAGES" \
  --image-dir "$DATASET_ROOT/aggc2022_10x_1024/images" \
  --image-dir "$DATASET_ROOT/cctgs_10x_1024/images" \
  --image-dir "$PATHVQA_IMAGE_DIR" --output-dir "$DATASET_ROOT/vqa_plus"
```

The script checks missing images and filename conflicts. Use `--vqa_data vqa`
or `--vqa_data vqa_plus` when training.

## 5. Assemble ReasonPath

```bash
python data_preprocess_scripts/ReasonPath_Annotations/collect_reason_path.py \
  --image-dir "$DATASET_ROOT/aggc2022_10x_1024/images" \
  --image-dir "$DATASET_ROOT/cctgs_10x_1024/images" \
  --mask-dir "$DATASET_ROOT/aggc2022_10x_1024/annotations" \
  --mask-dir "$DATASET_ROOT/cctgs_10x_1024/annotations" \
  --output-dir "$DATASET_ROOT/PathInstruct"
```

The output contains `data.json`, `images/` and `annotations/` per split:
965 train, 909 validation and 194 test records. Train and test are filtered;
validation is the original unfiltered split for separate evaluation. Train
with `--reason_seg_data 'PathInstruct|train'`.
