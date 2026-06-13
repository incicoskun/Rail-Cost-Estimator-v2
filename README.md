# Data-Adaptive Rail Cost Estimator

A lightweight, machine learning-based Decision Support System designed to predict railway project costs during early conceptual planning phases, especially when final blueprints or detailed geological studies are not yet available. Developed as a graduation project at Galatasaray University by İnci Coşkun.

---

## Core Logic & Purpose

* **The Problem:** Early-stage rail budgeting suffers from a lack of architectural detail, often leading to massive cost overruns.
* **The Solution:** This system utilizes a Quantile Gradient Boosting Regressor to generate objective, data-driven median cost estimates based on primary physical inputs (length, stations, tunnel %, and transit mode).
* **Handling Data Imbalance:** To prevent model distortion in countries with very few historical projects, it implements Bayesian Smoothed Target Encoding. This automatically shrinks sparse regional data toward global empirical baselines.
* **Budget Boundaries & Breakdown:** Instead of a single static number, the system calibrates Out-of-Fold (OOF) error distributions to output reliable $Q_{10} - Q_{90}$ probability bounds.
* **Subsystem Disaggregation:** Leverages historical normalized median cost structures from the Federal Transit Administration (FTA) to safely break down total cost projections into 8 core engineering asset classes based on the selected transit mode.
* **Explainable AI (SHAP):** Integrates game-theoretic Shapley Additive exPlanations to visually dismantle the black-box nature of the model, quantifying the exact dollar impact of features like station density or tunnel percentage for executive transparency.

---

## Quick Start & Execution

### 1. Installation
Install the core production requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard (Streamlit UI)
To launch the interface in your local browser, run:
```bash
streamlit run app.py
```


**UI Features:**
* **Left Sidebar:** Input structural inputs (Country, City, Length, Tunnel %, Station Count, Project Years, and Transit Mode).
* **Main Dashboard:** Dynamic KPI blocks rendering median costs/km, total budget boundaries ($Q_{10}-Q_{90}$), FTA subsystem breakdown charts, and SHAP contribution plots.

### 3. Retrain the Pipeline (Optional)
If you update the source raw databases and want to rebuild the processing states, compute fold-calibrations, and overwrite the serialized model pickles, execute:
```bash
python train.py
```

---

## Performance Benchmarks

* **MdAPE (Median Absolute Percentage Error):** 19.4% 
* **Logarithmic $R^2$:** 0.532
* **Mean Bias Error:** -0.1% 
* **Success Rate ($\pm30\%$ Boundary):** 67.8% 

---

## Project Architecture

```text
Data-Adaptive-Rail-Cost-Estimator/
├── data/
│   ├── raw/             # Global rail costs & FTA summary matrices
│   └── processed/       # Cleaned, standardized, and imputed model-ready states
├── src/
│   ├── config/          # Centralized configuration tokens and hyperparameter constants
│   ├── data/            # ETL processors, parsing blocks, and data loaders
│   ├── inference/       # Prediction orchestration, FTA routing, and SHAP explainers
│   ├── model/           # Stratified OOF pipelines, trainers, and mathematical encoders
│   ├── pipeline/        # Orchestrator tying data, model, and inference workflows
│   └── constants.py     # Global project constants
├── .gitignore           # Version control exclusion rules
├── app.py               # Main Streamlit web application entry-point
├── train.py             # Global execution script for pipeline retraining
├── requirements.txt     # Production dependencies
└── *.pkl                # Serialized model weights, feature names, and memory packages
```
