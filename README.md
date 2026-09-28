# Banking Analytics & Customer Transaction Behaviour (DS_05)

An empirical end-to-end data science study and retail banking customer transaction behavior investigation.

---

## 📊 Overview

This repository contains code, analysis, serialized models, charts, and comprehensive reporting analyzing customer demographic profiles and multi-year transaction behavior across 2,000 retail banking accounts and 13.3+ million transactions.

- **Full Report**: [BANKING_ANALYTICS_REPORT.md](BANKING_ANALYTICS_REPORT.md)
- **Jupyter Notebook**: [DS_Day01_05_Banking_Customer_Transaction_Behaviour.ipynb](DS_Day01_05_Banking_Customer_Transaction_Behaviour.ipynb)

---

## 📁 Repository Structure

```text
├── BANKING_ANALYTICS_REPORT.md       # Comprehensive executive & technical report
├── DS_Day01_05_Banking_...ipynb      # Interactive Jupyter notebook with complete workflows
├── run_complete_pipeline.py          # End-to-end data pipeline script
├── aggregate_transactions.py         # Transaction aggregation script
├── build_notebook.py                 # Notebook generation script
├── inspect_data.py                   # Data inspection and validation script
├── execute_notebook.py               # Headless notebook execution runner
├── cleaned_banking_data.csv          # Merged & cleaned user and transaction metrics
├── customer_transactions_aggregated.csv # Per-customer behavioral aggregates
├── users_data.csv                    # User demographic and account dataset
├── monthly_channel_trends.csv        # 10-year monthly transaction trends
├── demo_predictions.json             # Demo customer predictions
├── feature_list.json                 # Model feature metadata
├── inspection_summary.json           # Data schema & statistical summary
├── knn_pipeline.pkl                  # Trained Scikit-Learn KNN classification pipeline
├── chart_*.png                       # High-resolution visualization figures
└── .gitignore                        # Git exclusion rules
```

> **Note on Data Files:**  
> The raw `transactions_data.csv` file (~1.25 GB, 13.3M rows) is excluded from this repository per GitHub's 100 MB file limit. All necessary cleaned and aggregated datasets (`cleaned_banking_data.csv`, `customer_transactions_aggregated.csv`, `users_data.csv`, `monthly_channel_trends.csv`) are included for immediate analysis, modeling, and visualization.

---

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

2. **Run Pipeline:**
   ```bash
   python run_complete_pipeline.py
   ```

3. **Explore Notebook:**
   ```bash
   jupyter notebook DS_Day01_05_Banking_Customer_Transaction_Behaviour.ipynb
   ```
