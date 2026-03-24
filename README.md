# Uncertainty Quantification Prediction Project

This is the IUQ Final Project of Kaichen Shen, Moe Kyaw Thu, and Srikar Viswanatha. We have included our BNN model implementation, code for data preprocessing, datasets, notebooks for running the model on each individual stock, and code for SDE exploratory data analysis.

## Project Overview

This repository studies uncertainty in stock-price forecasting around GDPR by combining:

- **Bayesian Neural Networks (BNN)** for probabilistic prediction and uncertainty estimates.
- **Data preprocessing pipelines** to denoise and prepare stock time series.
- **Pre-/post-GDPR datasets** for multiple companies.
- **Exploratory SDE analysis** to compare drift/volatility behavior across GDPR periods.

## Repository Structure

### `Code/`
Main modeling code and experiment notebooks.

- `Code/bnn/`, `Code/sgmcmc_bayes_net/`, `Code/samplers/`, `Code/prior_mappers/`, `Code/gp/`, `Code/metrics/`, `Code/utils/`: core modules used by Bayesian and GP-style workflows.
- `Code/BNN_trained_prior_*.ipynb`: stock-specific notebooks (ASML, SAP, STM, Vodafone, Hexagon, Infineon, and Atos/Dassault/Ericcsson).
- `Code/exp/`: saved experiment artifacts/checkpoints.

### `BNN/`
Additional BNN-related work, including preliminary analysis and trained-prior notebook assets.

- `BNN/code/1-preliminary/`: early exploratory notebooks.
- `BNN/code/2-BNN_trained_prior/`: trained-prior notebook, sample CSVs, and installation notes.
- `BNN/optbnn/`: alternative notebook/data copies.

### `Datasets/`
All datasets and source materials.

- `Datasets/Raw Datasets Compustat/`: raw stock and index inputs.
- `Datasets/Pre_Post_GDPR_Datasets/`: pre/post GDPR stock datasets used in experiments.
- `Datasets/Processed Datasets Stoxx50/` and `Datasets/Processed Datasets Stoxx600/`: processed denoised datasets.
- `Datasets/10K Filings Reports/`: supporting filings and annual reports.

### `Preprocessing/`
Data-preparation code and generated denoised outputs.

- `Preprocessing/preprocess2.py`: script for regression-based denoising against ETF returns.
- `Preprocessing/Processed_Datasets_50/` and `Preprocessing/Processed_Datasets_600/`: denoised CSV outputs.

### `SDE/`
Stochastic Differential Equation exploratory analysis.

- `SDE/code/SDE.py`: computes drift/volatility summaries by GDPR phase.
- `SDE/gdpr_sde_output.csv`: exported SDE analysis results.

### `Figures/` and `HTML Results/`
Generated visualizations and notebook exports for report-ready review.

## Typical Workflow

1. **Prepare/inspect datasets** in `Datasets/` and `Preprocessing/`.
2. **Run preprocessing** (denoising) if needed.
3. **Run stock-specific BNN notebooks** in `Code/BNN_trained_prior_*.ipynb`.
4. **Review generated figures** in `Figures/` and notebook HTML in `HTML Results/`.
5. **Run SDE exploratory analysis** with `SDE/code/SDE.py` for GDPR-phase comparisons.

## Setup Notes

An installation note is available at:

- `BNN/code/2-BNN_trained_prior/installation.md`

That document references installing dependencies from an external Bayesian deep learning repository and using a Python virtual environment.

## Included Companies

The notebooks, datasets, and figures cover these companies:

- ASML
- ATOS
- Dassault
- Ericcsson
- Hexagon AB
- Infineon
- SAP
- STM
- Vodafone

## Outputs You Can Reproduce

- Predictive means and uncertainty (standard deviation) before/after GDPR.
- Per-company figures for model comparison.
- Drift/volatility summaries from SDE analysis around GDPR periods.

## Notes

- Some directories contain duplicated or intermediate artifacts (e.g., notebook exports and checkpoint files) retained for reproducibility.
- This project is organized as a research repository rather than a packaged Python library.
