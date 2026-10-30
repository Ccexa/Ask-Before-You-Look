# Ask Before You Look

Code for **Ask Before You Look: A conversational multi-level pathology
segmentation framework**.

## Dataset construction

The model uses AGGC and CCTGS tissue patches, CoNIC cell patches, VQA data,
and ReasonPath annotations. Prepare the datasets below under a shared
`DATASET_ROOT`; the detailed commands and expected directory layout are in
[data_preprocess_scripts/steps.md](data_preprocess_scripts/steps.md).

### Obtaining the source datasets

- **AGGC:** Request the complete dataset, including test ground truth, through
  the [AGGC 2022 challenge data page](https://aggc22.grand-challenge.org/Schedule/).
- **METU-CCTGS:** We cannot redistribute the CCTGS dataset due to access
  permissions. Apply directly using the download request on the
  [METU-CCTGS dataset website](https://eakbas.github.io/metu-cctgs/).
- **CoNIC:** Obtain the development data from the [CoNIC challenge data page](https://conic-challenge.grand-challenge.org/Data/).
  The [official CoNIC repository](https://github.com/TissueImageAnalytics/CoNIC)
  provides data-processing code and examples.
- **PathVQA:** See the [PathVQA dataset reference](https://github.com/sarahESL/PathVQA)
  for the source data. Images must be prepared with the filenames referenced
  by the supplied training JSONs.

### Preparation steps

1. Obtain the original AGGC and CCTGS whole-slide images and annotations.
   Configure and run the [AGGC](data_preprocess_scripts/AGGC/extract_patch.ipynb)
   and [CCTGS](data_preprocess_scripts/CCTGS/extract_patch.ipynb) notebooks with the
   provided [WSI split lists](data_preprocess_scripts/WSI_Splits). They produce
   paired image patches and segmentation masks.
2. Prepare CoNIC from its original `images.npy`, `labels.npy`, and
   `patch_info.csv` using the [CoNIC notebook](data_preprocess_scripts/CoNIC/conic_proc.ipynb).
3. Prepare PathVQA images. The archive is stored in Git LFS as parts under
   `data_preprocess_scripts/PathVQA/`; download and reconstruct it as described
   in the dataset preparation guide.
4. Assemble TumorVQA and TumorVQA+ using
   [prepare_vqa.py](data_preprocess_scripts/prepare_vqa.py). The supplied training
   JSON files already include PathVQA records. Provide image directories for
   PathVQA and for **all** tumor patches referenced by the JSON. The later
   notebook cells extract additional VQA patches with stride 1024 and retain
   tumor-negative patches; see the preparation guide. The script reports
   missing images and filename conflicts.
5. Assemble ReasonPath with
   [collect_reason_path.py](data_preprocess_scripts/ReasonPath_Annotations/collect_reason_path.py),
   using the supplied train, validation and test JSONs and the matching image
   patches and masks. Train and test are filtered; validation is the original
   unfiltered split and can be used for separate evaluation.

The expected patch and annotation counts are documented in the detailed
[dataset preparation guide](data_preprocess_scripts/steps.md). Obtain the source
datasets and follow their respective access and usage terms.

## TODO

- We are exploring uploading the checkpoints to Hugging Face.
