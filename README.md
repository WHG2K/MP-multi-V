# MP-multi-V

Code for reproducing the numerical experiments in the paper on **multi-purchase assortment optimization with surrogate formulations**.

This repository implements surrogate-based optimization methods (SP, RSP) for assortment problems under various choice models (MNL, mixed MNL) and constraint types (cardinality, space), and compares them against brute-force optimal solutions.

## Project structure

```
MP-multi-V/
├── src/                        # Core library
│   ├── models.py               # Surrogate and original optimization models
│   ├── distributions.py        # Random utility distributions (Gumbel, Normal, etc.)
│   ├── heuristics.py           # ADXOPT, greedy, and revenue-ordered heuristics
│   ├── brute_force.py          # Brute-force optimizer for benchmarking
│   ├── InstanceGenerator.py    # Random instance generation
│   └── utils.py                # Shared helpers
├── scripts/
│   ├── generate/               # Instance generation scripts
│   ├── run/                    # Experiment runner scripts
│   └── plot/                   # Plotting and post-processing scripts
├── requirements.txt
├── .env.example                # Gurobi configuration template
├── LICENSE
└── README.md
```

## Installation

### 1. Clone and install Python dependencies

```bash
git clone <repository-url>
cd MP-multi-V
pip install -r requirements.txt
```

### 2. Set up Gurobi

This project uses [Gurobi](https://www.gurobi.com/) as the optimization solver.

1. Download the latest Gurobi Optimizer from the [official downloads page](https://www.gurobi.com/downloads/gurobi-software/) and install it on your machine.
2. Install the Gurobi Python interface:
   ```bash
   pip install gurobipy
   ```
3. Obtain a license file (`gurobi.lic`). Academic users can request a free license.

### 3. Configure environment variables

Copy the provided template and fill in your paths:

```bash
cp .env.example .env
```

Edit `.env` to point to your local Gurobi installation and license:

```bash
GUROBI_HOME=/path/to/gurobi1201/linux64
PATH=$GUROBI_HOME/bin:$PATH
LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
GRB_LICENSE_FILE=/path/to/gurobi.lic
```

## Reproducing the figures

Each figure in the paper is produced by a three-step pipeline: **(1) generate** problem instances, **(2) solve** each instance, and **(3) plot** the results. All scripts should be run from the project root directory.


> **Note on sample sizes.** The scripts are configured to generate **small test samples** for quick verification. The full datasets used in the paper are hosted on Dropbox (link below) — you can skip directly to the plotting step to regenerate the paper's figures without rerunning the experiments.

## Reproducing figures in the paper

First, download the solved data from the Dropbox link: https://www.dropbox.com/scl/fo/43ajhbekto3kq8z2ieznk/AGJhqDH3OT-KSXE6bHtMCQM?rlkey=lyppljcosnol9qnixqfax6oz6&st=9vcgg290&dl=0. Then create a folder called `paper data` in the project root to store all the data. Then run:

```bash
python scripts/plot/plots_paper.py
```

This will read the solved datasets from `./paper data/` and save all figures to `./paper data/outputs/`.



## Reproducing by Rerun Your Preferred Parameters

Each figure in the paper is produced by a three-step pipeline: (1) **generate** problem instances, (2) **solve** each instance, and (3) **plot** the results.

### Figures 2 and L2: SP vs. optimal assortment (multi-purchase setting)

```bash
python scripts/generate/generate_SP2OP.py
python scripts/run/run_SP2OP.py --data "./data/SP2OP/SP2OP_data.jsonl"
python scripts/plot/boxplot_SP2OP.py
```

Outputs: `boxplot_SP2OP_ind.pdf` (Figure 2), `boxplot_SP2OP_linear.pdf` (Figure L2) in `./data/SP2OP/`.

### Figures 3 and L3: RSP vs. SP solution quality

```bash
python scripts/generate/generate_RSP2SP.py
python scripts/run/run_RSP2SP.py --data "./data/RSP2SP/RSP2SP_data.jsonl"
python scripts/plot/boxplot_RSP2SP.py
```

Outputs in `./data/RSP2SP/`: `boxplot_RSP2SP_B_2_ind.pdf` (Figure 3a), `boxplot_RSP2SP_B_4_ind.pdf` (Figure 3b), `boxplot_RSP2SP_B_2_linear.pdf` (Figure L3a), `boxplot_RSP2SP_B_4_linear.pdf` (Figure L3b).

### Figure 4: SP vs. RSP runtime comparison

```bash
python scripts/generate/generate_RSP2SP_runtime.py
python scripts/run/run_RSP2SP_runtime.py --data "./data/RSP2SP/RSP2SP_RUNTIME_data.jsonl"
python scripts/plot/lineplot_RSP2SP_runtimes.py
```

Outputs in `./data/RSP2SP/`: `plot_runtims_rsp_sp_B_2.pdf` (Figure 4a), `plot_runtims_rsp_sp_B_4.pdf` (Figure 4b).

### Figures 5 and L4: SP vs. optimal under single-purchase MNL

```bash
python scripts/generate/generate_MNL.py
python scripts/run/run_MNL_cardinality.py --data "./data/MNL/MNL_data.jsonl"
python scripts/run/run_MNL_spaceconstr.py --data "./data/MNL/MNL_data.jsonl"
python scripts/plot/boxplot_MNL_cardinality.py
python scripts/plot/boxplot_MNL_spaceconstr.py
```

Outputs in `./data/MNL/`: `boxplot_MNL_cardinality_ind.pdf` (Figure 5a), `boxplot_MNL_spaceconstr_ind.pdf` (Figure 5b), `boxplot_MNL_cardinality_linear.pdf` (Figure L4a), `boxplot_MNL_spaceconstr_linear.pdf` (Figure L4b).

### Figure 6: SP vs. optimal under mixture MNL

```bash
python scripts/generate/generate_MixMNL.py
python scripts/run/run_MixMNL_cardinality.py --data "./data/mixMNL/MIXMNL_data.jsonl"
python scripts/run/run_MixMNL_spaceconstr.py --data "./data/mixMNL/MIXMNL_data.jsonl"
python scripts/plot/boxplot_mixMNL_cardinality.py
python scripts/plot/boxplot_mixMNL_spaceconstr.py
```

Outputs in `./data/mixMNL/`: `boxplot_mixmnl_cardinality.pdf` (Figure 6a), `boxplot_mixmnl_spaceconstr.pdf` (Figure 6b).

### Table 1: Robustness to misspecification

```bash
python scripts/generate/generate_SP_robust.py
python scripts/run/run_SP_robust.py --data "./data/misspecification/ROBUST_data.jsonl"
python scripts/plot/process_SP_robust.py
python scripts/plot/table_SP_robust.py
```

The post-processing step produces `./data/misspecification/ROBUST_data_processed.jsonl`, which is read by the table script.

### Figure 7: RSP(w) vs. SP(w) value curves

```bash
python scripts/plot/lineplot_RSPw_SPw.py
```

Outputs in `./data/RSPw2SPw/`: `RSPw2SPw_data.jsonl` (saved curves), `lineplot_RSP(w)_SP(w)_N_20.pdf` (Figure 7a), `lineplot_RSP(w)_SP(w)_N_30.pdf` (Figure 7b), `lineplot_RSP(w)_SP(w)_N_40.pdf` (Figure 7c), `lineplot_RSP(w)_SP(w)_N_50.pdf` (Figure 7d).

## License

This project is released under the [MIT License](LICENSE).
