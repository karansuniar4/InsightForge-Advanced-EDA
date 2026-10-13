from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
REPORT_DIR = APP_DIR / "reports"
VIZ_DIR = APP_DIR / "visualizations"
for folder in (DATA_DIR, REPORT_DIR, VIZ_DIR):
    folder.mkdir(exist_ok=True)

st.set_page_config(
    page_title="InsightForge | Advanced EDA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px 16px; border-radius: 14px; border: 1px solid rgba(128,128,128,.16);}
.hero {padding: 24px 26px; border-radius: 18px; background: linear-gradient(120deg, rgba(31,78,121,.22), rgba(94,53,177,.15)); border: 1px solid rgba(128,128,128,.25); margin-bottom: 18px;}
.small-muted {opacity: .72; font-size: .9rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>🔬 InsightForge</h1>
      <h3>Advanced Exploratory Data Analysis & Predictive Insights</h3>
     
      <p class="small-muted">Profile data quality, discover patterns, quantify relationships, detect anomalies, and optionally train a baseline model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

DEFAULT_DATA = DATA_DIR / "titanic.csv"

@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))

@st.cache_data(show_spinner=False)
def get_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 600
    age = np.clip(rng.normal(34, 13, n), 1, 85).round(1)
    income = np.clip(rng.lognormal(10.7, 0.45, n), 15000, 180000).round(0)
    tenure = rng.integers(1, 121, n)
    spend = np.clip(income * rng.uniform(0.08, 0.30, n), 1000, 50000).round(0)
    segment = rng.choice(["Consumer", "Corporate", "Small Business"], n, p=[0.55, 0.28, 0.17])
    region = rng.choice(["North", "South", "East", "West"], n)
    score = (
        -1.7 + 0.025 * (age - 35) + 0.018 * (tenure - 36)
        + 0.000018 * (income - 50000)
        + np.where(segment == "Corporate", 0.45, 0)
        + rng.normal(0, 0.8, n)
    )
    probability = 1 / (1 + np.exp(-score))
    retained = rng.binomial(1, probability)
    df = pd.DataFrame({
        "Customer_ID": [f"C{i:04d}" for i in range(1, n + 1)],
        "Age": age,
        "Annual_Income": income.astype(int),
        "Tenure_Months": tenure,
        "Annual_Spend": spend.astype(int),
        "Segment": segment,
        "Region": region,
        "Retained": retained,
    })
    # Add a few missing values and numeric extremes for quality-analysis demos.
    for col in ["Age", "Annual_Income", "Region"]:
        idx = rng.choice(n, size=8, replace=False)
        df.loc[idx, col] = np.nan
    df.loc[0, "Annual_Spend"] = int(df["Annual_Spend"].quantile(.99) * 3)
    return df

@st.cache_data(show_spinner=False)
def analyze_iqr(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    results = []
    for col in columns:
        series = pd.to_numeric(data[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(.25), series.quantile(.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < low) | (series > high)
        results.append({
            "Column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Lower Bound": low, "Upper Bound": high,
            "Outlier Count": int(mask.sum()),
            "Outlier %": round(float(mask.mean() * 100), 2),
        })
    return pd.DataFrame(results)

def make_download_csv(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")

# Sidebar: load and filter
st.sidebar.header("Workspace")
uploaded = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
use_demo = st.sidebar.toggle("Use generated demo dataset", value=False, help="Use a reproducible 600-row customer dataset if no CSV is uploaded.")
try:
    if uploaded is not None:
        df_raw = read_csv_bytes(uploaded.getvalue())
        source_name = uploaded.name
    elif DEFAULT_DATA.exists() and not use_demo:
        df_raw = pd.read_csv(DEFAULT_DATA)
        source_name = "data/titanic.csv"
    elif use_demo or not DEFAULT_DATA.exists():
        df_raw = get_demo_data()
        source_name = "Generated customer demo dataset"
    else:
        df_raw = get_demo_data()
        source_name = "Generated customer demo dataset"
except Exception as exc:
    st.error(f"Unable to read the selected dataset: {exc}")
    st.stop()

if df_raw.empty:
    st.error("The dataset has no rows. Please upload a non-empty CSV.")
    st.stop()

# Avoid using huge files without warning
if len(df_raw) > 500_000:
    st.warning("This dataset contains more than 500,000 rows. Some visualizations may be slower.")

st.sidebar.divider()
st.sidebar.subheader("Filters")
df = df_raw.copy()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

filter_cols = st.sidebar.multiselect(
    "Categorical columns to filter",
    options=categorical_cols,
    default=categorical_cols[:1],
)
for col in filter_cols:
    vals = sorted(df[col].dropna().astype(str).unique().tolist())
    if len(vals) > 100:
        vals = vals[:100]
        st.sidebar.caption(f"{col}: showing first 100 distinct values")
    chosen = st.sidebar.multiselect(f"{col} values", vals, default=vals, key=f"cat_{col}")
    df = df[df[col].isna() | df[col].astype(str).isin(chosen)]

if numeric_cols:
    chosen_num_filter = st.sidebar.selectbox("Optional numeric filter", ["None"] + numeric_cols)
    if chosen_num_filter != "None":
        values = pd.to_numeric(df[chosen_num_filter], errors="coerce").dropna()
        if not values.empty and values.min() < values.max():
            lo, hi = float(values.min()), float(values.max())
            bounds = st.sidebar.slider(
                f"{chosen_num_filter} range", min_value=lo, max_value=hi,
                value=(lo, hi), key=f"num_{chosen_num_filter}"
            )
            df = df[df[chosen_num_filter].between(bounds[0], bounds[1]) | df[chosen_num_filter].isna()]

if df.empty:
    st.warning("No rows match the current filters. Clear or broaden your filters.")
    st.stop()

# Global KPIs
n_rows, n_cols = df.shape
missing_cells = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows in view", f"{n_rows:,}")
m2.metric("Features", n_cols)
m3.metric("Missing cells", f"{missing_cells:,}")
m4.metric("Duplicate rows", f"{duplicate_rows:,}")
m5.metric("Numeric features", len(numeric_cols))

tabs = st.tabs([
    "Executive Overview", "Data Profiling", "Univariate Analysis",
    "Relationships", "Outlier Lab", "Predictive Baseline", "Export Center"
])

# 1 Overview
with tabs[0]:
    st.subheader("Dataset snapshot")
    left, right = st.columns([1.3, 1])
    with left:
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Data type mix")
        type_counts = df.dtypes.astype(str).value_counts().rename_axis("Data type").reset_index(name="Columns")
        fig = px.pie(type_counts, names="Data type", values="Columns", hole=.55, title="Feature types")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Quick notes")
        st.write(f"- Dataset source: **{source_name}**")
        st.write(f"- Numeric features: **{len(numeric_cols)}**")
        st.write(f"- Categorical features: **{len(categorical_cols)}**")
        st.write(f"- Exact duplicate rows: **{duplicate_rows}**")
    st.info("EDA is descriptive: patterns and correlations are useful clues, not proof of causation.")

# 2 Data profiling
with tabs[1]:
    st.subheader("Automated data profile")
    profile = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Non-null": df.notna().sum(),
        "Missing": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(2),
        "Unique Values": df.nunique(dropna=True),
    })
    profile.index.name = "Column"
    st.dataframe(profile, use_container_width=True)
    if numeric_cols:
        st.markdown("#### Numeric summary")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    if categorical_cols:
        st.markdown("#### Categorical summary")
        cat_profile = pd.DataFrame({
            "Unique Values": df[categorical_cols].nunique(),
            "Most Frequent": df[categorical_cols].mode().iloc[0] if not df[categorical_cols].mode().empty else np.nan,
            "Most Frequent Count": [df[c].value_counts(dropna=True).max() if df[c].notna().any() else 0 for c in categorical_cols],
        })
        st.dataframe(cat_profile, use_container_width=True)

# 3 Univariate
with tabs[2]:
    st.subheader("Explore one variable at a time")
    if numeric_cols:
        selected_num = st.selectbox("Numeric feature", numeric_cols, key="uni_num")
        series = pd.to_numeric(df[selected_num], errors="coerce").dropna()
        if not series.empty:
            fig = px.histogram(df, x=selected_num, marginal="box", title=f"Distribution of {selected_num}", nbins=35)
            st.plotly_chart(fig, use_container_width=True)
            st.write({
                "Mean": round(float(series.mean()), 3),
                "Median": round(float(series.median()), 3),
                "Std Dev": round(float(series.std()), 3) if len(series) > 1 else 0,
                "Skewness": round(float(series.skew()), 3) if len(series) > 2 else None,
            })
    if categorical_cols:
        selected_cat = st.selectbox("Categorical feature", categorical_cols, key="uni_cat")
        counts = df[selected_cat].fillna("Missing").astype(str).value_counts().head(25).rename_axis(selected_cat).reset_index(name="Count")
        fig = px.bar(counts.sort_values("Count"), x="Count", y=selected_cat, orientation="h", title=f"Top categories in {selected_cat}")
        st.plotly_chart(fig, use_container_width=True)

# 4 Relationships
with tabs[3]:
    st.subheader("Relationships and segmentation")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson correlation matrix")
        st.plotly_chart(fig, use_container_width=True)
        pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not pairs.empty:
            top_pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(8).reset_index()
            top_pairs.columns = ["Feature A", "Feature B", "Correlation"]
            top_pairs["Correlation"] = top_pairs["Correlation"].round(3)
            st.markdown("#### Strongest numeric relationships")
            st.dataframe(top_pairs, use_container_width=True, hide_index=True)
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis", numeric_cols, index=0, key="scatter_x")
        y_options = [c for c in numeric_cols if c != x_col]
        if y_options:
            y_col = st.selectbox("Y-axis", y_options, index=0, key="scatter_y")
            color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols, key="scatter_color")
            plot_df = df[[x_col, y_col] + ([color_col] if color_col != "None" else [])].dropna()
            if len(plot_df) > 10_000:
                plot_df = plot_df.sample(10_000, random_state=42)
                st.caption("Scatterplot sampled to 10,000 rows for performance.")
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=None if color_col == "None" else color_col, opacity=.7, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
    if categorical_cols and numeric_cols:
        st.markdown("#### Group comparison")
        g_col = st.selectbox("Group by category", categorical_cols, key="group_cat")
        metric_col = st.selectbox("Numeric metric", numeric_cols, key="group_metric")
        grouped = df.groupby(g_col, dropna=False)[metric_col].agg(["mean", "median", "count"]).reset_index()
        grouped = grouped.sort_values("mean", ascending=False).head(30)
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        fig = px.bar(grouped, x=g_col, y="mean", hover_data=["median", "count"], title=f"Mean {metric_col} by {g_col}")
        st.plotly_chart(fig, use_container_width=True)

# 5 Outlier lab
with tabs[4]:
    st.subheader("Outlier detection lab")
    st.caption("IQR is a screening rule; flagged values are not automatically errors.")
    if numeric_cols:
        outlier_report = analyze_iqr(df, numeric_cols)
        st.dataframe(outlier_report.round(3), use_container_width=True, hide_index=True)
        chosen_outlier = st.selectbox("Inspect feature", numeric_cols, key="outlier_col")
        series = pd.to_numeric(df[chosen_outlier], errors="coerce").dropna()
        if not series.empty:
            q1, q3 = series.quantile(.25), series.quantile(.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df.loc[df[chosen_outlier].notna() & ((df[chosen_outlier] < low) | (df[chosen_outlier] > high))]
            fig = px.box(df, y=chosen_outlier, points="outliers", title=f"Box plot: {chosen_outlier}")
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Potential outlier rows in current view: **{len(outliers)}**")
            if not outliers.empty:
                st.dataframe(outliers.head(100), use_container_width=True)
    else:
        st.info("No numeric features are available for IQR analysis.")

# 6 Predictive baseline
with tabs[5]:
    st.subheader("Optional classification baseline")
    st.caption("Use only when you have a meaningful categorical/binary target. This is a demonstration model, not a deployment-ready system.")
    if len(df) < 30:
        st.warning("At least 30 rows are recommended for a train/test demonstration.")
    if df.shape[1] < 2:
        st.warning("At least two columns are needed.")
    else:
        target = st.selectbox("Choose target column", df.columns.tolist(), key="model_target")
        possible_features = [c for c in df.columns if c != target]
        features = st.multiselect("Choose input features", possible_features, default=possible_features[:min(5, len(possible_features))], key="model_features")
        test_size = st.slider("Test set (%)", 15, 40, 20, 5, key="test_size")
        if st.button("Train baseline model", type="primary"):
            work = df[features + [target]].dropna(subset=[target]).copy()
            y = work[target]
            if not features:
                st.error("Select at least one input feature.")
            elif y.nunique() < 2:
                st.error("The target must contain at least two classes.")
            elif y.nunique() > 20 or y.dtype.kind in "fc" and y.nunique() > 20:
                st.error("This demo supports classification targets with at most 20 distinct classes. Choose a categorical or binary target.")
            elif len(work) < 30:
                st.error("Not enough labeled rows to train a useful demo model.")
            else:
                X = work[features]
                cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                num_features = [c for c in features if c not in cat_features]
                transformers = []
                if num_features:
                    transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_features))
                if cat_features:
                    transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat_features))
                preprocessor = ColumnTransformer(transformers=transformers)
                model = Pipeline([("preprocess", preprocessor), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
                stratify = y if y.value_counts().min() >= 2 else None
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42, stratify=stratify)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    a, b, c, d = st.columns(4)
                    a.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
                    b.metric("Precision (weighted)", f"{precision_score(y_test, preds, average='weighted', zero_division=0):.3f}")
                    c.metric("Recall (weighted)", f"{recall_score(y_test, preds, average='weighted', zero_division=0):.3f}")
                    d.metric("F1 (weighted)", f"{f1_score(y_test, preds, average='weighted', zero_division=0):.3f}")
                    labels = model.named_steps["classifier"].classes_
                    cm = confusion_matrix(y_test, preds, labels=labels)
                    fig = px.imshow(cm, text_auto=True, x=[str(x) for x in labels], y=[str(x) for x in labels], labels={"x":"Predicted","y":"Actual","color":"Count"}, title="Confusion matrix")
                    st.plotly_chart(fig, use_container_width=True)
                    st.code(classification_report(y_test, preds, zero_division=0), language="text")
                    st.success("Baseline trained successfully. Evaluate leakage, class balance, and domain context before interpreting results.")
                except Exception as exc:
                    st.error(f"Could not train this model with the selected columns: {exc}")

# 7 Export
with tabs[6]:
    st.subheader("Export center")
    st.download_button("Download filtered data CSV", make_download_csv(df), "filtered_data.csv", "text/csv")
    profile = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Count": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values,
    })
    st.download_button("Download data profile CSV", make_download_csv(profile), "data_profile.csv", "text/csv")
    if numeric_cols:
        outlier_report = analyze_iqr(df, numeric_cols)
        st.download_button("Download outlier report CSV", make_download_csv(outlier_report), "outlier_report.csv", "text/csv")
    st.markdown("#### Save a local report")
    report_text = f"""# EDA Report

- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Dataset: {source_name}
- Rows in current view: {len(df)}
- Columns: {df.shape[1]}
- Missing cells: {int(df.isna().sum().sum())}
- Duplicate rows: {int(df.duplicated().sum())}

## Notes
This report summarizes the current filtered view. Correlations do not establish causation. IQR outliers are candidates for review, not automatically data errors.
"""
    st.download_button("Download findings summary (Markdown)", report_text, "eda_findings.md", "text/markdown")

st.divider()
st.caption("InsightForge · CodSoft Task 2 · Exploratory Data Analysis")
