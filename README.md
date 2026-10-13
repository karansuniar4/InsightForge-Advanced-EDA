# InsightForge — Exploratory Data Analysis Dashboard

**CodSoft Data Analytics Internship | Task 2: Exploratory Data Analysis (EDA)**

## Project Overview

InsightForge is an interactive data analysis dashboard developed using Python, Pandas, NumPy, Plotly, and Streamlit. The project focuses on exploring datasets to identify trends, relationships, patterns, and potential outliers through statistical analysis and interactive visualizations.

The dashboard provides a user-friendly interface for examining data, understanding distributions, comparing variables, and summarizing analytical findings.

## Internship Task Objective

The objective of Task 2 is to perform Exploratory Data Analysis (EDA) on a dataset to identify:

* Trends and patterns in the data.
* Relationships between variables.
* Statistical summaries and distributions.
* Potential outliers and unusual observations.
* Insights that can support further data analysis.

This project demonstrates these techniques through an interactive Python application.

## Technologies Used

* **Python** — Core programming language.
* **Pandas** — Data manipulation and analysis.
* **NumPy** — Numerical operations.
* **Plotly** — Interactive charts and visualizations.
* **Streamlit** — Interactive web dashboard.
* **Scikit-learn** — Optional baseline classification and model evaluation.

## Key Features

### 1. Dataset Exploration

* Upload datasets in CSV format.
* Explore the built-in demonstration dataset.
* Preview records and filter available data.
* Examine dataset dimensions and column types.

### 2. Statistical Analysis

* Generate descriptive statistics.
* Examine numerical distributions.
* Review unique values and data types.
* Identify missing values and duplicate records.

### 3. Data Visualization

* Histograms for numerical variables.
* Bar charts for categorical variables.
* Correlation heatmaps.
* Scatter plots for examining relationships.
* Group comparisons and box plots.

### 4. Outlier Analysis

* Detect potential outliers using the Interquartile Range (IQR) method.
* Review affected variables and unusual observations.
* Visualize distributions using box plots.
* Export outlier analysis results.

### 5. Analytical Reports

* Export filtered datasets.
* Download data profiling results.
* Export outlier reports.
* Generate a Markdown summary of analytical findings.

### 6. Optional Classification Analysis

* Train a baseline Logistic Regression classifier when suitable data and a target variable are available.
* Evaluate performance using accuracy, precision, recall, and F1-score.
* Review the confusion matrix and classification report.

## Project Structure

```text
InsightForge-Advanced-EDA/
├── app.py
├── requirements.txt
├── README.md
├── run_windows.bat
├── .gitignore
├── data/
│   └── README.txt
├── reports/
│   └── README.txt
└── visualizations/
    └── README.txt
```

## Installation and Execution

### Prerequisites

* Python 3.10 or newer.
* Visual Studio Code or another Python editor.

### Step 1: Clone the repository

```bash
git clone https://github.com/karansuniar4/InsightForge-Advanced-EDA.git
cd InsightForge-Advanced-EDA
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

On Windows, you can also use the provided `run_windows.bat` file.

## Dataset Information

The application includes a reproducible, synthetic customer dataset for demonstration and supports uploading external CSV files.

The synthetic dataset is intended for testing and demonstrating EDA functionality. Its results should not be interpreted as findings about real customers or real-world business behavior.

## Expected Learning Outcomes

Through this project, the following practical skills are demonstrated:

* Loading and exploring datasets using Pandas.
* Calculating descriptive statistics.
* Examining distributions and relationships between variables.
* Creating interactive data visualizations.
* Identifying potential outliers.
* Summarizing analytical observations.
* Building a data analysis dashboard using Streamlit.

## Limitations

* The quality of findings depends on the input dataset.
* Outlier detection flags potential anomalies and does not automatically establish data errors.
* Correlation does not imply causation.
* The optional classification model is a baseline for educational analysis, not a production-ready predictive system.
* The demonstration dataset is synthetic.

## GitHub Repository

**Project Source Code:**
https://github.com/karansuniar4/InsightForge-Advanced-EDA

## Conclusion

InsightForge demonstrates an end-to-end exploratory data analysis workflow using Python and its data analytics ecosystem. By combining statistical summaries, interactive visualizations, outlier analysis, and report exports, the project provides a practical approach to identifying patterns and relationships in datasets.

Developed as part of the **CodSoft Data Analytics Internship — Task 2: Exploratory Data Analysis**.
