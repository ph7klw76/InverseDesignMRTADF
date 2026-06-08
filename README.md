# Inverse design MR-TADF Pipeline

**Reliability-gated inverse design and semiempirical descriptor workflows for multi-resonance TADF candidate prioritisation**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit-informational)
![Surrogates](https://img.shields.io/badge/Models-GPR%20%7C%20XGBoost%20%7C%20Hurdle%20SOC-purple)
![Validation](https://img.shields.io/badge/Validation-external%20%2F%20manual-orange)
![License](https://img.shields.io/badge/License-not%20specified-lightgrey)

---

## Repository summary

This repository contains a research-oriented Python workflow for **MR-TADF molecular candidate generation, surrogate prediction, reliability-gated ranking, and validation-queue export**.

The core script, `mr_tadf_bo_pipeline.py`, trains target-specific surrogate models from user-provided labelled molecular data, generates chemically filtered candidate structures by rule-based graph edits, scores them with uncertainty-aware acquisition logic, and writes ranked candidate tables for external quantum-chemical or experimental validation.

The repository also includes utilities for computing or parsing **sTDA-xTB excited-state descriptors** and **frontier-orbital overlap heuristics** used as optional feature blocks.

> **Important scope statement:** this pipeline is a **proposal and prioritisation engine**. It does **not** perform DFT, TD-DFT, spin-orbit coupling calculations, or experimental validation by itself. Validation is explicitly external/manual and can be fed back through CSV files.

---

## Why this repository exists

MR-TADF emitter design requires balancing multiple quantities that are difficult to optimise simultaneously:

- small or controlled `DeltaEST`
- suitable `T2-T1`
- non-negligible oscillator strength
- useful singlet energy window
- potentially useful SOC channels
- chemical plausibility and applicability-domain support

This codebase provides an implementation-oriented workflow for turning a labelled MR-TADF corpus into:

1. surrogate models for several photophysical targets,
2. rule-generated candidate molecules,
3. uncertainty- and applicability-domain-aware rankings,
4. exploitation / exploration / control queues for the next external validation round,
5. audit files documenting descriptor health, cross-validation behaviour, and ranking decisions.

---

## Graphical abstract

```mermaid id="wij0yb"
flowchart TD
    A["Labelled MR-TADF data<br/>Excel descriptors + target labels"] --> B["Name-based merge<br/>unit contract + leakage guard"]
    B --> C["Optional descriptor augmentation"]
    C --> C1["GFN2-xTB descriptors<br/>precomputed + candidate cache"]
    C --> C2["sTDA-xTB descriptors<br/>S1/T1/T2, fS1, config-weight cosines"]
    C --> C3["Heavy-atom position descriptors<br/>SMILES-only"]
    C --> C4["Frontier overlap heuristics<br/>xtb --molden + stda_overlap.py"]

    B --> D["Target-specific surrogate training"]
    C1 --> D
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E["Leave-family-out and<br/>leave-Murcko-core-out diagnostics"]
    D --> F["Operator-based candidate generation<br/>MR motif + chemistry filters"]
    F --> G["Candidate descriptor computation<br/>cached optional blocks"]
    G --> H["Prediction + uncertainty + AD scoring"]

    H --> I["Adaptive reliability-gated acquisition"]
    I --> J["Ranked candidates"]
    I --> K["Exploit / explore / control queues"]
    I --> L["Validation export"]

    L --> M["External QC / TD-DFT / experiment<br/>manual step outside this repository"]
    M --> N["Optional QC CSV feedback<br/>validation history + discrepancy report"]
```

---

## Repository scope

### Implemented in code

| Area | Implemented behaviour |
|---|---|
| Data loading | Reads descriptor and target Excel files, normalises `Name`, merges by `Name`, removes potentially leaky descriptor columns, handles duplicate-name conflicts. |
| Unit handling | Requires or verifies unit sidecars for labelled target/QC/benchmark files unless explicitly bypassed. |
| Surrogate modelling | Supports GPR, bagged XGBoost, stacked XGB+Ridge, target-specific feature selection, and special SOC handling. |
| SOC modelling | Default two-stage hurdle SOC model with classifier for active SOC and regressor on active rows; asinh transform remains available for SOC targets. |
| Candidate generation | Uses operator-based graph edits, MR motif filters, synthetic-accessibility heuristic, retrosynthetic pattern filter, charge/radical rejection, diversity controls. |
| Descriptor augmentation | Optional xTB, sTDA-xTB, heavy-atom-position, and frontier-overlap descriptor blocks. |
| Ranking | Computes uncertainty-aware acquisition scores, AD penalties, feasibility factors, final scores, diversity-selected shortlists, and queue labels. |
| Diagnostics | Writes cross-validation reports, feature-importance summaries, descriptor-health metadata, plots, reproducibility metadata, and validation-history JSON. |


---

## Main components

| File | Role |
|---|---|
| `mr_tadf_bo_pipeline_v21.py` | Main inverse-design pipeline. Loads training data, trains surrogates, generates candidates, computes descriptors, scores/ranks candidates, writes queues and diagnostics. |
| `stda_descriptors.py` | Runs `xtb4stda` and `stda` over `validatedxyz/`, derives broader sTDA excited-state descriptors, writes descriptor CSV, manifest, correlations, cache, and failures. |
| `parse_stda_xtb_important_descriptors.py` | Parses existing sTDA output folders into a compact numeric descriptor table focused on important excited-state features for SOC modelling. |
| `stda_overlap.py` | Parses modern `xtb --molden` files and computes frontier coefficient-overlap, charge-transfer-length, transition-dipole proxy, and ζ-weighted frontier-density descriptors. |
| `a5.sh` | SLURM example for a full v21 inverse-design run on a Linux cluster. |
| `run_v21_local.ps1` | Reduced local Windows/PowerShell example run. |
| `run_stda_xtb_important_folder.sh` | SLURM helper intended to loop over `.xyz` files, run xTB/sTDA, and parse important descriptors. See the usage note below. |
| `target5-new1_reordered.xlsx.units.json` | Unit contract sidecar for the target-label Excel file. |
| `final_updated_data.xlsx` | Example descriptor table used by the provided run scripts. |
| `target5-new1_reordered.xlsx` | Example target-label table used by the provided run scripts. |
| `xtb_descriptors.csv` | Example precomputed xTB descriptor table. |
| `stda_xtb_important_descriptors.csv` | Example precomputed compact sTDA-xTB descriptor table. |

---

## Combined workflow concept

The intended workflow is file-based rather than fully orchestrated:

```mermaid id="ovnsdd"
flowchart LR
    A["Geometry folder<br/>validatedxyz/*.xyz"] --> B["sTDA descriptor generation<br/>stda_descriptors.py or parser workflow"]
    B --> C["stda_xtb_important_descriptors.csv<br/>or stda_out/stda_descriptors.csv"]

    D["Base descriptors<br/>final_updated_data.xlsx"] --> F["mr_tadf_bo_pipeline_v21.py"]
    E["Target labels<br/>target5-new1_reordered.xlsx<br/>+ .units.json"] --> F
    C --> F
    G["Optional xTB descriptors<br/>xtb_descriptors.csv"] --> F

    F --> H["results_v21/all_scored.csv"]
    F --> I["results_v21/ranked_candidates.csv"]
    F --> J["results_v21/export_validation_queue.csv"]

    J --> K["External QC / TD-DFT / experiment"]
    K --> L["qc_results.csv<br/>manual feedback"]
    L --> F
```

The scripts can be used together, but they are not a single automated closed-loop active-learning system. The validation loop is closed manually by selecting candidates, running external calculations or measurements, and re-running the pipeline with `--qc_results_file`.

---

## Highlights

- **Adaptive objective logic:** `DeltaEST` and singlet-window feasibility are always active under `--objective adaptive`; `fOSC` and `T2-T1` are enabled only where target-specific evidence supports them.
- **SOC is treated cautiously:** default SOC modelling uses a hurdle model, but under the adaptive objective SOC is reported for screening rather than used as a hard optimisation driver.
- **Descriptor-health auditing:** optional descriptor blocks are tracked for real-value fraction; mostly imputed blocks are warned about or can be made fatal with `--strict_descriptors`.
- **Applicability-domain gating:** candidate scores include descriptor distance, fingerprint similarity, core similarity, and uncertainty terms.
- **External validation design:** output queues are explicitly labelled as candidates to compute or measure next, not as validated discoveries.
- **Deterministic output schema:** scored/ranked CSVs use a canonical column ordering with optional extras appended.

---

## Code-to-README validation note

This README is derived from inspection of the provided code files and example data names. Claims are limited to behaviour visible in the scripts.

Where the code implements only a heuristic, this README describes it as a heuristic. Where the code requires manual file handoff or external validation, this README states that explicitly. Where helper scripts reference files or arguments that were not included in the upload, those are listed under limitations and recommended additions.

---

## Data model

### Base descriptor file

The main pipeline expects an Excel descriptor file such as:

```text id="hdy9hw"
final_updated_data.xlsx
```

The loader renames the first two columns to:

```text id="8xx0j1"
Name
SMILES
```

The provided example file uses `Name` and `smile`; the loader renames the second column internally.

### Target file

The target Excel file is expected to contain a first column used as `Name` plus target columns including:

| Target column | Meaning in pipeline |
|---|---|
| `DeltaEST` | S1-T1 gap target |
| `T2-T1` | T2-T1 energy gap target |
| `T1-S1(SOC)` | SOC target for T1/S1 |
| `T2-S1(SOC)` | SOC target for T2/S1 |
| `Oscillator Strengths` | Oscillator strength target, reported as `OscStr` in output columns |
| `Singlets` | Singlet energy target |

The provided unit contract declares:

| Quantity | Unit |
|---|---|
| `DeltaEST` | eV |
| `T2-T1` | eV |
| `Singlets` | eV |
| `T1-S1(SOC)` | cm⁻¹ |
| `T2-S1(SOC)` | cm⁻¹ |
| `Oscillator Strengths` | dimensionless |

By default, labelled target/QC/benchmark files are expected to have hash-bound unit contract sidecars. Legacy bypass flags exist, but using them weakens provenance.

---

## Detailed workflow 1: main inverse-design pipeline

### Purpose

`mr_tadf_bo_pipeline_v21.py` trains surrogate models and proposes MR-TADF candidates for external validation.

### Core operations

1. Load base descriptors and targets.
2. Verify declared target units.
3. Remove descriptor columns that appear to leak target information.
4. Optionally merge precomputed xTB and sTDA-xTB descriptor CSVs by `Name`.
5. Optionally append SMILES-only heavy-atom-position descriptors.
6. Optionally compute frontier-overlap descriptors from `xtb --molden`.
7. Train independent target surrogates.
8. Run leave-scaffold-family-out and strict leave-Murcko-core-out diagnostics.
9. Generate candidate structures by graph edit operators.
10. Compute candidate descriptors and descriptor-health flags.
11. Predict target values and uncertainties.
12. Score with feasibility, AD, novelty, trust, and diversity logic.
13. Export ranked candidates and validation queues.
14. Optionally merge external QC results and append validation-history records.

### Surrogate model options

| Option | Behaviour |
|---|---|
| default | Bagged XGBoost-style ensemble path for standard targets. |
| `--use_gpr` | Uses Gaussian process regression for non-SOC targets. |
| `--gp_alpha` | Sets the GPR diagonal ridge/noise floor; default is designed to reduce overfitting. |
| `--enable_stacking` | Uses XGB bagging followed by a ridge meta-learner for non-SOC targets. Mutually exclusive with `--use_gpr`. |
| `--soc_hurdle` | Default-on two-stage SOC model. |
| `--no_soc_hurdle` | Falls back to the older single-regressor SOC path. |
| `--asinh_soc` / `--no_asinh_soc` | Enables/disables signed-log treatment for SOC targets in relevant paths. |

### Candidate-generation logic

The generator is operator-based. It includes:

- heteroatom substitutions such as C/N and chalcogen substitutions,
- optional bulky substituent operations,
- optional diaza edits,
- optional annulation,
- ν-DABNA-style BN annulation logic,
- P-chalcogenide insertion motifs,
- donor/graft operations.

Candidates are filtered for:

- allowed atom sets,
- MR-defining motifs,
- aromatic ring count,
- heavy-atom range,
- rotatable-bond cap with donor-graft relaxation,
- charge/radical/open-shell rejection,
- heuristic retrosynthetic feasibility,
- corpus-relative synthetic-accessibility score.

The synthetic-accessibility score is local-corpus-relative and Ertl-Schuffenhauer-inspired. It is **not** the canonical published SAscore.

### Adaptive objective logic

Under:

```bash id="ui0xaz"
--objective adaptive
```

the code applies the following policy:

| Axis | Role |
|---|---|
| `DeltaEST` | Always active. |
| `Singlets` | Always constrained to the singlet energy window. |
| `fOSC` | Optional; enabled only with sufficient target-specific local reliability evidence. |
| `T2-T1` | Optional; enabled only with sufficient target-specific local reliability evidence. |
| SOC targets | Predicted and reported, but deferred to external validation under the adaptive objective. |

The optional axes are gated by factors including:

- exact Murcko-core support,
- target-specific applicability-domain score,
- strict-core out-of-fold reliability,
- held-out exact-core MAE behaviour,
- configured thresholds such as `--adaptive_min_ad`.

### Main outputs

A typical output directory such as `results_v21/` may contain:

| Output | Meaning |
|---|---|
| `all_scored.csv` | Full scored candidate audit table. |
| `ranked_candidates.csv` | Ranked candidates after scoring and ranking filters. |
| `top_diverse_candidates.csv` | Diversity-selected shortlist. |
| `top50_candidates.csv` | Top 50 ranked candidates. |
| `queue_exploit.csv` | High-final-score AD-clean candidates. |
| `queue_explore.csv` | High-novelty AD-clean candidates. |
| `queue_control.csv` | Familiar/control candidates near seed/core chemistry. |
| `export_validation_queue.csv` | Compact file intended for external validation selection. |
| `topology_summary.csv` | Topological novelty and core descriptor summary. |
| `infeasible_candidates.csv` | Written when all candidates fail feasibility gates. |
| `metadata.json` | Run metadata, descriptor health, validation mode, ranking settings. |
| `publication_diagnostics.json` | Publication-style run summary. |
| `reproducibility_manifest.json` | Input/source/executable provenance summary. |
| `scaffold_stratified_cv.json` | Leave-scaffold-family-out diagnostics. |
| `core_split_cv.json` | Strict leave-Murcko-core-out diagnostics. |
| `adaptive_axis_training_only_core_cv.json` | Optional-axis reliability evidence when split calibration is used. |
| `residual_covariance.json` | Out-of-fold residual covariance, if enabled. |
| `uncertainty_error_correlation.json` | OOF uncertainty/error association diagnostics. |
| `feature_importance_summary.json` | Feature-importance summary where available. |
| `conformal_kappa.json` | Global conformal scale factors, if calibration is enabled. |
| `conformal_kappa_by_scaffold.json` | Per-scaffold conformal scale factors, if enabled. |
| `conformal_support.json` | Scope/support summary for conformal intervals. |
| `validation_history.json` | Persistent active-learning history across runs. |
| `validation_merged_results.csv` | Candidate shortlist with matched external QC results, if supplied. |
| `qc_discrepancy_summary.json` | QC-vs-surrogate discrepancy summary, if QC is supplied. |
| `benchmark_predictions.csv` | Benchmark predictions, if a benchmark file is supplied. |
| `benchmark_metrics.json` | Benchmark metrics when labelled benchmark targets are available. |
| `novelty_vs_AD.png` | Diagnostic plot. |
| `parent_core_vs_score.png` | Diagnostic plot. |
| `scaffold_family_barplot.png` | Diagnostic plot. |

---

## Detailed workflow 2: compact sTDA-xTB important descriptor parser

### Script

```text id="e8g63e"
parse_stda_xtb_important_descriptors.py
```

### Purpose

Parses existing sTDA work directories into a compact numeric descriptor CSV focused on excited-state features useful for downstream modelling, especially SOC-related surrogate features.

### Expected files per molecule work directory

```text id="5zvrnb"
tda_singlet.dat
tda_triplet.dat
stda_singlet.out
stda_triplet.out
```

### Descriptors produced

| Descriptor group | Columns |
|---|---|
| Vertical energies | `S1_stda_xtb_eV`, `T1_stda_xtb_eV`, `T2_stda_xtb_eV` |
| Derived gaps | `DeltaEST_stda_xtb_eV`, `T2_T1_stda_xtb_eV` |
| Oscillator strength | `fS1_stda_xtb` |
| Configuration character | `S1_T1_config_weight_cosine`, `S1_T2_config_weight_cosine`, mismatch columns |
| Configuration complexity | entropy, dominant-weight, and count-above-threshold columns |

The configuration-weight cosine values are similarities between normalised squared-amplitude configuration-weight distributions. They are **not** signed CI-vector overlaps, wavefunction overlaps, or transition-density overlaps.

### One-molecule command

```bash id="o5twik"
python parse_stda_xtb_important_descriptors.py \
  --name 44-OQAO \
  --workdir stda_xtb_test/44-OQAO \
  --out stda_xtb_important_descriptors.csv \
  --append \
  --ewin 10 \
  --xtb4stda /path/to/xtb4stda \
  --stda /path/to/stda
```

### Folder command

```bash id="2u7mjm"
python parse_stda_xtb_important_descriptors.py \
  --root stda_xtb_test \
  --out stda_xtb_important_descriptors.csv \
  --ewin 10 \
  --xtb4stda /path/to/xtb4stda \
  --stda /path/to/stda
```

### Outputs

| Output | Meaning |
|---|---|
| `stda_xtb_important_descriptors.csv` | Compact descriptor table. |
| `stda_xtb_important_descriptors.csv.manifest.json` | Provenance sidecar with parser, executable, source-output, and energy-window hashes/settings. |

### Usage note for `run_stda_xtb_important_folder.sh`

The provided SLURM helper is intended to loop over `validatedxyz/*.xyz`, run xTB/sTDA, and call the parser per molecule. However, the parser as provided requires:

```text id="8t99fw"
--ewin
--xtb4stda
--stda
```

The uploaded `run_stda_xtb_important_folder.sh` parser call does not pass those required arguments. Patch the parser call or use the direct parser commands above to avoid an argument error.

---

## Detailed workflow 3: broader sTDA-xTB descriptor generation

### Script

```text id="ixv732"
stda_descriptors.py
```

### Purpose

Runs `xtb4stda` and `stda` directly over an `.xyz` geometry folder, derives semiempirical excited-state descriptors, and computes correlations with target labels.

This script is broader than the compact parser workflow and writes its own output folder.

### Typical commands

Quick test:

```bash id="mskh87"
python stda_descriptors.py \
  --xyz_dir validatedxyz \
  --target target5-new1_reordered.xlsx \
  --out stda_out \
  --limit 5
```

Full run:

```bash id="bnkxak"
python stda_descriptors.py \
  --xyz_dir validatedxyz \
  --target target5-new1_reordered.xlsx \
  --out stda_out \
  --workers 6 \
  --omp_threads 2 \
  --ewin 10
```

With explicit executables:

```bash id="gjg4wv"
python stda_descriptors.py \
  --xyz_dir validatedxyz \
  --target target5-new1_reordered.xlsx \
  --out stda_out \
  --xtb4stda /path/to/xtb4stda \
  --stda /path/to/stda \
  --workers 6
```

### Outputs

| Output | Meaning |
|---|---|
| `stda_out/stda_descriptors.csv` | One descriptor row per successful molecule. |
| `stda_out/stda_descriptors.csv.manifest.json` | Descriptor provenance manifest. |
| `stda_out/stda_target_correlations.csv` | Pearson/Spearman correlations against available target labels. |
| `stda_out/stda_failures.txt` | Failed molecules and errors. |
| `stda_out/stda_cache/*.json` | Manifest-keyed per-molecule descriptor cache. |

### Interpretation

sTDA-xTB descriptors can provide inexpensive excited-state correlates such as vertical S1/T1/T2 energies and S1 oscillator strength. The script does **not** compute SOC matrix elements.

---

## Detailed workflow 4: frontier-overlap heuristics

### Script

```text id="7bmo4b"
stda_overlap.py
```

### Purpose

Parses Molden files produced by modern:

```bash id="aemmqs"
xtb --molden
```

and computes fast frontier-orbital screening descriptors.

### Descriptor families

| Family | Examples |
|---|---|
| Frontier overlap | `Lambda_HL`, `Lambda_HLp1` |
| Charge-transfer length | `dr_HL_ang`, `dr_HLp1_ang` |
| ζ-weighted SOC proxy | `soc_homo_zeta_rho`, `soc_lumo_zeta_rho`, `soc_geom_zeta_rho` |
| Heavy frontier density | `soc_heavy_frontier_frac` |
| Transition-dipole proxy | `mu_HL_eA`, `fosc_HL_proxy`, `qnorm_HL` |

### Critical limitation

These descriptors are atom-condensed MO-coefficient heuristics. Molden files do not provide the AO overlap matrix used by rigorous Mulliken/Löwdin population analysis, so these values should be interpreted as **screening features**, not quantitative orbital-overlap observables.

---

## Suggested repository layout

```text id="f4oes7"
.
├── README.md
├── mr_tadf_bo_pipeline_v21.py
├── stda_descriptors.py
├── parse_stda_xtb_important_descriptors.py
├── stda_overlap.py
├── descriptor_provenance.py              # required by descriptor manifest code; include in repo
├── xtb_descriptors.py                    # referenced by pipeline for xTB features; include if using xTB block
├── a5.sh
├── run_v21_local.ps1
├── run_stda_xtb_important_folder.sh
├── data/
│   ├── final_updated_data.xlsx
│   ├── target5-new1_reordered.xlsx
│   └── target5-new1_reordered.xlsx.units.json
├── descriptors/
│   ├── xtb_descriptors.csv
│   ├── xtb_descriptors.csv.manifest.json
│   ├── stda_xtb_important_descriptors.csv
│   └── stda_xtb_important_descriptors.csv.manifest.json
├── validatedxyz/
│   └── <Name>.xyz
├── results_v21/
│   └── ...
└── environment.yml
```

The provided upload includes descriptor CSVs but not their `.manifest.json` sidecars. The main pipeline can be run with the explicit legacy bypass flag `--allow_unverified_descriptor_csv`, but publication runs should regenerate or provide manifest sidecars.

---

## Suggested software environment

### Python packages

The scripts require the following non-standard Python packages:

```text id="vko4hw"
numpy
pandas
scipy
scikit-learn
xgboost
matplotlib
rdkit
openpyxl
```

`openpyxl` is needed by pandas for Excel input/output support.

Optional or context-dependent:

```text id="7ko9ej"
psutil
torch
selfies
```

`torch` and `selfies` are imported defensively in the main pipeline, but the deep generative stack is documented as removed in v18 and is not part of the implemented v21 workflow.

### Local repository modules

The provided scripts reference local modules that should be included for complete functionality:

```text id="htj3k5"
descriptor_provenance.py
xtb_descriptors.py
stda_overlap.py
parse_stda_xtb_important_descriptors.py
stda_descriptors.py
```

`descriptor_provenance.py` and `xtb_descriptors.py` are referenced by the code but were not among the uploaded `.py` files. Include them in the repository if using descriptor provenance checks and xTB descriptor augmentation.

### External executables

Optional descriptor workflows require external command-line tools:

| Executable | Used by |
|---|---|
| `xtb` | xTB descriptors, candidate geometry optimisation, Molden overlap descriptors |
| `xtb4stda` | sTDA-xTB wavefunction preparation |
| `stda` | sTDA singlet/triplet calculations |

The cluster scripts assume a conda environment named `inverteddesign` and a SLURM-based HPC system. Adjust those names for your local installation.

---

## Quick start

### 1. Minimal smoke run without optional quantum descriptors

This is the simplest way to test the main pipeline logic.

```bash id="lm861l"
python mr_tadf_bo_pipeline_v21.py \
  --data final_updated_data.xlsx \
  --target target5-new1_reordered.xlsx \
  --n_candidates 100 \
  --n_workers 4 \
  --attempts_per_worker 50000 \
  --n_ensemble 8 \
  --output results_smoke \
  --objective adaptive
```

This requires the target unit sidecar:

```text id="t1xmkh"
target5-new1_reordered.xlsx.units.json
```

### 2. Run with precomputed xTB and sTDA descriptor CSVs

Use this when descriptor manifests are available:

```bash id="t99r27"
python mr_tadf_bo_pipeline_v21.py \
  --data final_updated_data.xlsx \
  --target target5-new1_reordered.xlsx \
  --n_candidates 500 \
  --n_workers 4 \
  --n_ensemble 20 \
  --output results_v21 \
  --objective adaptive \
  --use_gpr \
  --gp_alpha 0.1 \
  --use_xtb_descriptors \
  --xtb_descriptors_file xtb_descriptors.csv \
  --use_stda_descriptors \
  --stda_descriptors_file stda_xtb_important_descriptors.csv \
  --soc_hurdle \
  --use_heavy_pos
```

For legacy descriptor CSVs without manifest sidecars:

```bash id="mazs9u"
python mr_tadf_bo_pipeline_v21.py \
  --data final_updated_data.xlsx \
  --target target5-new1_reordered.xlsx \
  --n_candidates 500 \
  --n_workers 4 \
  --n_ensemble 20 \
  --output results_v21 \
  --objective adaptive \
  --use_gpr \
  --gp_alpha 0.1 \
  --use_xtb_descriptors \
  --xtb_descriptors_file xtb_descriptors.csv \
  --use_stda_descriptors \
  --stda_descriptors_file stda_xtb_important_descriptors.csv \
  --soc_hurdle \
  --use_heavy_pos \
  --allow_unverified_descriptor_csv
```

Use the legacy bypass only when you understand that descriptor provenance is no longer established by the run.

### 3. Full SLURM example

```bash id="fkd0it"
sbatch a5.sh
```

The provided `a5.sh` enables:

- GPR with `--gp_alpha 0.1`,
- adaptive objective,
- xTB descriptors,
- sTDA descriptors,
- heavy-atom-position descriptors,
- overlap descriptors,
- candidate GFN2-xTB geometry optimisation,
- residual covariance,
- diversity limits.

This is a heavier run and assumes working `xtb`, `xtb4stda`, and `stda` paths.

### 4. Reduced local Windows example

```powershell id="downb7"
pwsh run_v21_local.ps1 60
```

To additionally request candidate geometry optimisation:

```powershell id="80dmta"
pwsh run_v21_local.ps1 60 -OptGeom
```

The local script uses hard-coded Windows paths. Edit `$PY` and `$XTB` before running on another system.

---

## Using the workflows together

### Step A: prepare labelled data

Place the base descriptor and target files in the working directory:

```text id="pe5bs6"
final_updated_data.xlsx
target5-new1_reordered.xlsx
target5-new1_reordered.xlsx.units.json
```

### Step B: prepare optional descriptor tables

Use one or more of:

```bash id="2pa24a"
python stda_descriptors.py --xyz_dir validatedxyz --out stda_out
```

or:

```bash id="s7eypg"
python parse_stda_xtb_important_descriptors.py \
  --root stda_xtb_test \
  --out stda_xtb_important_descriptors.csv \
  --ewin 10 \
  --xtb4stda /path/to/xtb4stda \
  --stda /path/to/stda
```

For xTB descriptors, the main script expects a producer named `xtb_descriptors.py` when `--use_xtb_descriptors` is active.

### Step C: run inverse design

```bash id="3tb0jc"
python mr_tadf_bo_pipeline_v21.py \
  --data final_updated_data.xlsx \
  --target target5-new1_reordered.xlsx \
  --output results_v21 \
  --objective adaptive \
  --use_xtb_descriptors \
  --xtb_descriptors_file xtb_descriptors.csv \
  --use_stda_descriptors \
  --stda_descriptors_file stda_xtb_important_descriptors.csv
```

### Step D: inspect validation queue

Prioritise candidates from:

```text id="43r727"
results_v21/export_validation_queue.csv
results_v21/queue_exploit.csv
results_v21/queue_explore.csv
results_v21/queue_control.csv
```

### Step E: run external validation manually

Run DFT, TD-DFT, SOC calculations, or experiments outside this repository.

### Step F: feed results back

Prepare a CSV with a `smiles` column and any target columns to merge, for example:

```text id="kgtam6"
smiles,DeltaEST,T2-T1,Oscillator Strengths,Singlets,T1-S1(SOC),T2-S1(SOC)
...
```

Then re-run with:

```bash id="43ke6c"
python mr_tadf_bo_pipeline_v21.py \
  --data final_updated_data.xlsx \
  --target target5-new1_reordered.xlsx \
  --qc_results_file qc_results.csv \
  --validation_history_file results_v21/validation_history.json \
  --output results_v21_next
```

QC files are also subject to unit-contract verification unless the legacy bypass is used.

---

## Reproducibility notes

The code includes several reproducibility and audit mechanisms:

- hash-bound unit contract checks for labelled files,
- descriptor CSV manifest verification,
- descriptor-health summaries,
- candidate descriptor-integrity gates,
- deterministic scored CSV schema,
- reproducibility manifest with input/source/executable records,
- leave-scaffold-family-out and strict leave-Murcko-core-out diagnostics,
- optional residual covariance from out-of-fold residuals,
- validation-history JSON across runs.

For publication use, keep the following files with each run:

```text id="hqwea5"
metadata.json
publication_diagnostics.json
reproducibility_manifest.json
scaffold_stratified_cv.json
core_split_cv.json
all_scored.csv
ranked_candidates.csv
export_validation_queue.csv
```

Also archive the exact input data, descriptor CSVs, descriptor manifests, unit contracts, and external executable versions.

---

## Confidence and decision logic

### Candidate confidence signals

The main ranked outputs include several columns that should be interpreted together:

| Column family | Interpretation |
|---|---|
| `pred_*` | Surrogate mean prediction. |
| `unc_*` | Model uncertainty estimate. |
| `AD_score` | Applicability-domain score combining descriptor distance, fingerprint/core similarity, and uncertainty. |
| `trust_parent`, `trust_train` | Similarity/trust penalties relative to parent and training chemistry. |
| `p_*` | Feasibility probability factors used by the acquisition/ranking logic. |
| `cei` | Constrained expected improvement-like acquisition value. |
| `final_score` | Integrated ranking score after acquisition, novelty, AD/trust, and other terms. |
| `validation_priority_score` | Score used for validation-priority export. |
| `queue` | Exploit, explore, control, or blank. |

### Queue interpretation

| Queue | Intended use |
|---|---|
| `exploit` | Highest-scoring AD-clean candidates for near-term validation. |
| `explore` | High-novelty candidates to expand chemical coverage. |
| `control` | Familiar candidates close to seed/core chemistry; useful for sanity-checking surrogate behaviour. |

Queue membership is not validation. It is a recommendation for external computation or measurement.

---

## Methodological contribution and interpretation

The codebase combines several methodological elements into a practical MR-TADF design workflow:

1. **Rule-based molecular proposal generation** within MR-like chemistry.
2. **Multi-target surrogate modelling** with target-specific reliability diagnostics.
3. **Descriptor augmentation** using low-cost xTB/sTDA-derived features.
4. **Cautious SOC modelling** through hurdle decomposition and heavy-atom-position descriptors.
5. **Adaptive acquisition** that avoids using poorly supported optional axes outside local reliability evidence.
6. **External validation handoff** through ranked queues and feedback CSV merging.

The most defensible interpretation of this repository is:

> a reproducible candidate-prioritisation workflow for deciding which MR-TADF molecules are worth validating next.

It should not be interpreted as a standalone discovery engine that validates final emitter performance.

---

## Example citation block

Replace the placeholders with the manuscript and repository metadata when available.

```bibtex id="fosquc"
@software{mr_tadf_pipeline_v21,
  title        = {MR-TADF Pipeline v21: Reliability-Gated Inverse Design and sTDA-xTB Descriptor Workflows},
  author       = {<Author list>},
  year         = {<Year>},
  version      = {v21},
  url          = {https://github.com/<user>/<repository>},
  note         = {Publication companion repository for MR-TADF candidate prioritisation}
}
```

For manuscript companion use:

```bibtex id="fch921"
@article{mr_tadf_inverse_design,
  title   = {<Manuscript title>},
  author  = {<Author list>},
  journal = {<Journal>},
  year    = {<Year>},
  doi     = {<DOI>}
}
```

---

## Recommended additions for publication readiness

Before public release, consider adding:

- `environment.yml` or `requirements.txt` with pinned package versions,
- `descriptor_provenance.py` if descriptor manifests are required,
- `xtb_descriptors.py` if xTB descriptor generation is part of the published workflow,
- descriptor CSV manifest sidecars for all precomputed descriptor CSVs,
- example `validatedxyz/` subset for smoke tests,
- a tiny synthetic dataset for CI that does not expose unpublished data,
- tests for data loading, unit contracts, descriptor alias migration, and parser behaviour,
- patched `run_stda_xtb_important_folder.sh` passing the parser’s required arguments,
- `LICENSE`,
- `CITATION.cff`,
- schema documentation for output CSV columns,
- a worked example notebook that reads `ranked_candidates.csv` and visualises queue trade-offs,
- a clear statement of external QC settings used in the associated paper.

---

## Limitations

- **No automatic validation:** the pipeline does not run DFT, TD-DFT, SOC, or experiments.
- **Surrogate predictions are corpus-dependent:** out-of-domain molecules may receive unreliable predictions even if they pass filters.
- **SOC remains difficult:** the hurdle model and screening scores are useful engineering devices, not a replacement for explicit SOC validation.
- **sTDA does not provide SOC:** sTDA descriptors are excited-state correlates, not SOC matrix elements.
- **Overlap descriptors are heuristic:** Molden parsing uses coefficient-condensed proxies without an AO overlap matrix.
- **Synthetic accessibility is local and heuristic:** the implemented score is corpus-relative and not a canonical SAscore.
- **Descriptor provenance must be complete for strict publication runs:** missing manifests require either regeneration or an explicit legacy bypass.
- **Some helper scripts are environment-specific:** SLURM scripts assume a particular cluster module/conda setup; the PowerShell script contains local Windows paths.
- **The uploaded folder-level sTDA SLURM helper needs patching:** it omits parser arguments required by the uploaded parser.
- **Deep generative models are not active:** older deep generative functionality is documented as removed and should not be cited as implemented in v21.

---

## Acknowledgments

This repository builds on widely used scientific Python and computational chemistry tooling, including RDKit, NumPy, pandas, SciPy, scikit-learn, XGBoost, matplotlib, xTB, xtb4stda, and sTDA.

The workflow is designed for transparent candidate prioritisation and should be paired with independent quantum-chemical or experimental validation before making materials claims.

---

## Maintainer note

When updating the repository, keep the README aligned with the code by checking:

1. whether new flags are implemented or only planned,
2. whether output files are always written or conditional,
3. whether validation remains external/manual,
4. whether descriptor manifests and unit contracts are required,
5. whether any helper script has drifted from parser CLI requirements.

For publication-facing releases, prefer conservative claims and include the exact input data, descriptor provenance, unit contracts, and run metadata needed to reproduce each reported table.
