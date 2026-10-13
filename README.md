# InsightForge — Advanced EDA Dashboard (CodSoft Task 2)

A Streamlit-based exploratory analytics workspace that profiles datasets, visualizes distributions and relationships, flags potential outliers, and optionally trains a baseline classification model.

## Features
- Upload any CSV or use the built-in generated customer demo dataset
- Interactive categorical and numeric filters
- Executive overview and dataset profiling
- Missing-value, duplicate, unique-value, and type summaries
- Distribution plots and categorical counts
- Correlation matrix, scatterplots, and group comparisons
- IQR outlier detection with a box-plot inspector
- Optional Logistic Regression classification baseline with train/test split, weighted metrics, and confusion matrix
- Export filtered data, profile, outlier report, and Markdown findings summary

## Project structure
```text
CodSoft_Task2_EDA_Advanced/
├── app.py
├── requirements.txt
├── README.md
├── run_windows.bat
├── .gitignore
├── data/
├── reports/
└── visualizations/
```

## Requirements
Python 3.10 or newer is recommended.

## Run on Windows
1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal in that folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Then visit `http://localhost:8501`.

If PowerShell blocks activation:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

You can also double-click `run_windows.bat`.

## Dataset
No external dataset is required to start. The app generates a deterministic 600-row synthetic customer dataset for demonstration. Upload a real CSV from the sidebar for your actual analysis. The generated data is synthetic and must not be presented as real-world evidence.

## Important notes
- This tool is for exploratory analysis and learning, not automated decision-making.
- Correlation does not imply causation.
- Outliers are flagged for review, not automatically removed.
- Predictive scores depend on target/feature choice and data quality. Avoid leakage and use domain judgment.
