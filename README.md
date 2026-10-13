# InsightForge — Advanced EDA Dashboard

**CodSoft Data Analytics Internship — Task 2: Exploratory Data Analysis**

InsightForge is an interactive data analytics dashboard developed using Python, Pandas, NumPy, Plotly, and Streamlit. It helps users explore datasets, identify trends and relationships, visualize data distributions, detect potential outliers, and generate analytical reports.

The project demonstrates practical exploratory data analysis (EDA) techniques through an easy-to-use web interface, with support for CSV uploads and a built-in synthetic customer dataset.

## Features

### 1. Dataset Upload and Preview

* Upload CSV datasets for analysis.
* Use the built-in demonstration dataset.
* Preview and filter records.
* Explore numeric and categorical columns.
* Analyze filtered data interactively.

### 2. Executive Overview

* View key dataset statistics.
* Review the number of records and columns.
* Explore numerical summaries.
* Understand the general structure of the dataset.

### 3. Automated Data Profiling

* Inspect column names and data types.
* Identify missing values.
* Check duplicate records.
* Examine unique-value counts.
* Review descriptive statistics.
* Export the dataset profile.

### 4. Data Visualization

* Generate histograms for numeric variables.
* Explore categorical value counts.
* Visualize correlations using a heatmap.
* Investigate relationships using scatter plots.
* Compare groups across selected variables.
* Use interactive charts to explore patterns.

### 5. Outlier Detection

* Identify potential outliers using the Interquartile Range (IQR) method.
* Review outlier counts and affected records.
* Visualize distributions using box plots.
* Export an outlier report.

Outliers are flagged for investigation and are not automatically removed.

### 6. Optional Machine Learning Analysis

* Train a baseline Logistic Regression classification model.
* Split data into training and testing sets.
* Review accuracy, precision, recall, and F1-score.
* Inspect the confusion matrix and classification report.

Model evaluation depends on dataset quality, target selection, and the suitability of the selected features. This feature is intended for educational exploration, not production decision-making.

### 7. Reports and Data Export

* Download filtered data as CSV.
* Export dataset profiling results.
* Download outlier analysis reports.
* Generate a Markdown summary of analytical findings.

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

### File Descriptions

* `app.py` — Main Streamlit application containing the dashboard, analysis tools, visualizations, and exports.
* `requirements.txt` — Python dependencies required to run the project.
* `README.md` — Project documentation and setup instructions.
* `run_windows.bat` — Windows batch file for launching the application.
* `.gitignore` — Excludes selected local and generated files from Git.
* `data/` — Directory reserved for dataset-related files.
* `reports/` — Directory reserved for report-related files.
* `visualizations/` — Directory reserved for visualization-related files.

Generated reports and downloaded files may be saved through the application's download features rather than automatically stored in these directories.

## Requirements

* Python 3.10 or newer.
* Visual Studio Code or another Python editor.
* Git, if you want to manage or update the GitHub repository.
* An internet connection for installing dependencies, if they are not already installed.

## Installation and Setup

### Step 1: Open the project

1. Extract the project ZIP file.
2. Open Visual Studio Code.
3. Select **File → Open Folder**.
4. Choose the `InsightForge-Advanced-EDA` project folder.
5. Open **Terminal → New Terminal**.

### Step 2: Create a virtual environment

Run:

```powershell
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly in the commands below.

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the virtual environment is not activated:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 4: Run the dashboard

```powershell
python -m streamlit run app.py
```

Alternatively, use:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also double-click `run_windows.bat`.

### Step 5: Open the application

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

Keep the terminal open while using the application. Press `Ctrl+C` in the terminal to stop it.

## How to Use the Dashboard

1. Start the Streamlit application.
2. Use the built-in dataset or upload a CSV file.
3. Review the executive overview and key statistics.
4. Inspect column types, missing values, duplicates, and descriptive statistics.
5. Apply available filters to focus on relevant records.
6. Explore distributions, categorical counts, correlations, and scatter plots.
7. Review potential outliers with the IQR method.
8. Optionally run the classification baseline when a suitable target variable is available.
9. Export filtered data, profiles, outlier reports, and findings summaries.

The available analysis options depend on the columns and data types in the selected dataset.

## Dataset Information

InsightForge can generate a reproducible, synthetic customer dataset containing 600 rows for demonstration.

No external dataset is required to launch the application. Users can upload their own CSV files to explore other datasets.

**Important:** The generated customer data is synthetic and does not represent real-world observations. Use actual, appropriately authorized datasets when performing real-world analysis, and clearly distinguish synthetic results from real findings.

## Exploratory Data Analysis Workflow

The project supports the following general EDA workflow:

1. **Understand the dataset:** Examine its dimensions, column names, and data types.
2. **Assess data quality:** Review missing values, duplicate records, and unique values.
3. **Summarize variables:** Calculate descriptive statistics for numeric data and frequency counts for categorical data.
4. **Explore distributions:** Use histograms, category charts, and box plots.
5. **Investigate relationships:** Examine correlations, scatter plots, and group comparisons.
6. **Review potential outliers:** Apply the IQR method and investigate flagged observations.
7. **Summarize findings:** Export analytical summaries and relevant reports.
8. **Evaluate optional predictive analysis:** Train and assess a baseline classification model when appropriate.

EDA helps identify patterns and potential issues, but findings should always be interpreted in the context of the dataset.

## Testing the Application

Suggested test cases include:

* Launching the dashboard with its generated dataset.
* Uploading a valid CSV file.
* Exploring datasets containing numeric and categorical columns.
* Checking missing-value and duplicate summaries.
* Applying available filters.
* Viewing charts and correlation analysis.
* Reviewing potential outliers.
* Running the optional classification model with a suitable target.
* Downloading the available reports and filtered data.

Verify the downloaded files and review the results before using them in further analysis.

## GitHub Repository

**Repository:** https://github.com/karansuniar4/InsightForge-Advanced-EDA

### Update the README on GitHub

After replacing and saving `README.md` in VS Code, open the terminal in the project directory and run:

```powershell
git add README.md
git commit -m "Improve Task 2 EDA project README"
git push origin main
```

Refresh the GitHub repository to confirm the updated documentation appears on the main page.

## Data Privacy and Responsible Analysis

* Use synthetic or authorized datasets for testing.
* Do not upload confidential or personally identifiable information to a public repository.
* Keep private datasets and sensitive outputs out of Git.
* Review data quality before drawing conclusions.
* Do not interpret correlation as proof of causation.
* Treat detected outliers as candidates for investigation, not automatically invalid data.
* Validate machine-learning results before making any decisions based on them.

## Limitations

* The project is designed for learning and exploratory analysis.
* The included demonstration dataset is synthetic.
* Analysis quality depends on the dataset and selected variables.
* Outlier detection identifies potential anomalies but does not establish that a value is erroneous.
* Correlation does not establish causation.
* The optional classification model is a baseline and is not intended as a production-ready predictive system.
* The dashboard does not replace domain expertise or careful statistical validation.

## Project Summary

**InsightForge — Advanced Exploratory Data Analysis Dashboard**

Developed an interactive Python-based EDA dashboard using Pandas, NumPy, Plotly, and Streamlit. The application supports CSV uploads, automated data profiling, descriptive statistics, interactive visualizations, correlation analysis, group comparisons, IQR-based outlier detection, optional Logistic Regression classification, and downloadable analytical reports.

## Acknowledgment

Created as part of the **CodSoft Data Analytics Internship — Task 2**, focusing on Exploratory Data Analysis using Python and Pandas.
