## Setting up Gurobi

This project uses [Gurobi](https://www.gurobi.com/) as the optimization solver. Follow the steps below to install Gurobi, set up the Python interface, and configure the local environment.

### 1. Download Gurobi and install the Python interface

Download the latest Gurobi Optimizer from the [official downloads page](https://www.gurobi.com/downloads/gurobi-software/) and install it on your machine.

Then install the Gurobi Python interface into your Python environment:

```bash
pip install gurobipy
```

### 2. Obtain and install a license

Gurobi requires a license file (`gurobi.lic`) to run. Academic users can request a free license [here](https://www.gurobi.com/academia/academic-program-and-licenses/); commercial users can find licensing options [here](https://www.gurobi.com/solutions/licensing/). Follow the instructions to download and place the `gurobi.lic` file on your machine.

### 3. Configure environment variables via `.env`

Create a `.env` file in the project root that tells the code where Gurobi is installed and where your license lives:

```bash
cd /path/to/MP-multi-V
touch .env
nano .env   # or use your preferred editor
```

Paste the following into the file and adjust the paths to match your own Gurobi installation and license location:

```bash
GUROBI_HOME=/path/to/gurobi1201/linux64
PATH=$GUROBI_HOME/bin:$PATH
LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
GRB_LICENSE_FILE=/path/to/gurobi.lic
```

Save and exit (in `nano`: `Ctrl+O`, `Enter`, `Ctrl+X`).



## Reproducing Figures in the Paper

First, download the solved data from the dropbox link: (TODO). Then create a folder called ```paper data``` to store all the data. Then run


```bash
pyhton plots_paper.py
```






## Reproducing by Rerun Your Preferred Parameters

Each figure in the paper is produced by a three-step pipeline: (1) **generate** problem instances, (2) **solve** each instance, and (3) **plot** the results.

> **Note on sample sizes.** The scripts in this repository are configured to generate **small test samples** for quick reproduction. The full datasets used in the paper are hosted on Dropbox (link below) — you can skip directly to the plotting step if you'd like to regenerate the paper's figures without rerunning the experiments.
>
> *Dropbox link: [to be added]*

---

### Figures 2 and L2: SP vs. Optimal Assortment (Multi-purchase setting)

```bash
python generate_SP2OP.py                                    # generate instances
python run_SP2OP.py --data "./data/SP2OP/SP2OP_data.jsonl"  # solve instances
python boxplot_SP2OP.py                                     # plot results
```

Outputs: `boxplot_SP2OP_ind.pdf` (Figure 2), `boxplot_SP2OP_linear.pdf` (Figure L2) in `./data/SP2OP/`.

### Figures 3 and L3: RSP vs. SP Solution Quality

```bash
python generate_RSP2SP.py                                      # generate instances
python run_RSP2SP.py --data "./data/RSP2SP/RSP2SP_data.jsonl"  # solve instances
python boxplot_RSP2SP.py                                       # plot results
```

Outputs in `./data/RSP2SP/`: `boxplot_RSP2SP_B_2_ind.pdf` (Figure 3a), `boxplot_RSP2SP_B_4_ind.pdf` (Figure 3b), `boxplot_RSP2SP_B_2_linear.pdf` (Figure L3a), `boxplot_RSP2SP_B_4_linear.pdf` (Figure L3b).

### Figure 4: SP vs. RSP Runtime Comparison

```bash
python generate_RSP2SP_runtime.py                                              # generate instances
python run_RSP2SP_runtime.py --data "./data/RSP2SP/RSP2SP_RUNTIME_data.jsonl"  # solve instances
python lineplot_RSP2SP_runtims.py                                              # plot results
```

Outputs in `./data/RSP2SP/`: `plot_runtims_rsp_sp_B_2.pdf` (Figure 4a), `plot_runtims_rsp_sp_B_4.pdf` (Figure 4b).

### Figures 5 and L4: SP vs. Optimal under Single-Purchase MNL

```bash
python generate_MNL.py                                             # generate instances
python run_MNL_cardinality.py --data "./data/MNL/MNL_data.jsonl"   # solve (cardinality constraint)
python run_MNL_spaceconstr.py --data "./data/MNL/MNL_data.jsonl"   # solve (space constraint)
python boxplot_MNL_cardinality.py                                  # plot cardinality results
python boxplot_MNL_spaceconstr.py                                  # plot space constraint results
```

Outputs in `./data/MNL/`: `boxplot_MNL_cardinality_ind.pdf` (Figure 5a), `boxplot_MNL_spaceconstr_ind.pdf` (Figure 5b), `boxplot_MNL_cardinality_linear.pdf` (Figure L4a), `boxplot_MNL_spaceconstr_linear.pdf` (Figure L4b).

### Figure 6: SP vs. Optimal under Mixture MNL

```bash
python generate_MixMNL.py                                                  # generate instances
python run_MixMNL_cardinality.py --data "./data/mixMNL/MIXMNL_data.jsonl"  # solve (cardinality constraint)
python run_MixMNL_spaceconstr.py --data "./data/mixMNL/MIXMNL_data.jsonl"  # solve (space constraint)
python boxplot_mixMNL_cardinality.py                                       # plot cardinality results
python boxplot_mixMNL_spaceconstr.py                                       # plot space constraint results
```

Outputs in `./data/mixMNL/`: `boxplot_mixmnl_cardinality.pdf` (Figure 6a), `boxplot_mixmnl_spaceconstr.pdf` (Figure 6b).

### Table 1: Robustness to Misspecification

```bash
python generate_SP_robust.py                                                # generate instances
python run_SP_robust.py --data "./data/misspecification/ROBUST_data.jsonl"  # solve instances
python process_SP_robust.py                                                 # post-process (compute revenue gaps)
python table_SP_robust.py                                                   # print Table 1 to terminal
```

The post-processing step produces `./data/misspecification/ROBUST_data_processed.jsonl`, which is what `table_SP_robust.py` reads.

### Figure 7: RSP(w) vs. SP(w) Value Curves


```bash
python lineplot_RSPw_SPw.py  # compute RSP(w)/SP(w) curves and plot
```

Outputs in `./data/RSPw2SPw/`: `RSPw2SPw_data.jsonl` (saved curves), `lineplot_RSP(w)_SP(w)_N_20.pdf` (Figure 7a), `lineplot_RSP(w)_SP(w)_N_30.pdf` (Figure 7b), `lineplot_RSP(w)_SP(w)_N_40.pdf` (Figure 7c), `lineplot_RSP(w)_SP(w)_N_50.pdf` (Figure 7d).
