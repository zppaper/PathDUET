# PathDUET

**Disentangle to Unify: Fusing Internal Data and External Semantics via Pathway Disentanglement for Imbalanced Multi-Omics**

PathDUET integrates mRNA, miRNA, DNA methylation and copy-number variation using shared pathway representations, omics-specific residual representations, fixed pathway semantic memory and branch-aware fusion.

![Overview of PathDUET](assets/figure2.png)

## Code and datasets

- **Code:** [https://github.com/zppaper/PathDUET](https://github.com/zppaper/PathDUET)
- **Datasets:** [https://zenodo.org/records/23833721](https://zenodo.org/records/23833721)
- **Dataset DOI:** [https://doi.org/10.5281/zenodo.23833721](https://doi.org/10.5281/zenodo.23833721)

## Directory setup

Download the code from GitHub and the datasets from the Zenodo record above. Extract the downloaded data and place `datasets/` at the repository root, alongside `train.py`:

```text
PathDUET/
├── README.md
├── requirements.txt
├── assets/
│   └── figure2.png
├── train.py
├── pathduet/
│   ├── __init__.py
│   ├── model.py
│   ├── losses.py
│   ├── training.py
│   ├── data.py
│   ├── runtime.py
│   └── constants.py
└── datasets/
    ├── stage/
    │   ├── COAD/
    │   ├── KIRP/
    │   ├── THCA/
    │   └── UCS/
    └── resources/
        ├── gene_pathway.csv
        └── semantic_memory.csv
```

Each cohort directory contains `mRNA.csv`, `miRNA.csv`, `dnam_gene.csv`, `cnv_gene.csv`, `label.csv` and `splits/`. To use another data location, specify `--data /path/to/datasets`.

Molecular matrices have samples in rows, `sample_id` first, and 8,074 ordered gene columns. Each seed directory contains five split CSVs and their saved training-fold scaler CSVs. The four cohorts share fixed resources for 297 pathways. All model inputs and semantic resources are local CSV files; no external language-model service is required.

## Run the main experiments

The reference environment uses Python 3.10.19 and PyTorch 2.7.0 with CUDA 12.8. From the repository root, install dependencies:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Use a PyTorch build suitable for your hardware. The script automatically selects CUDA when available, otherwise CPU; use `--device cpu` to select CPU explicitly.

Cancer-stage classification on COAD, KIRP, THCA and UCS:

```bash
python train.py --output results
```

This runs seeds 1028, 2024 and 3407 with five folds per cohort: 60 folds in total.

To run one cohort and seed:

```bash
python train.py --cohorts COAD --seeds 1028 --output results_coad
```
