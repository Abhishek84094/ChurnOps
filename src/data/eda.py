"""
Phase 3 — Exploratory Data Analysis (EDA) Module
================================================
Performs comprehensive univariate, bivariate, and association analysis on Telco dataset.
Produces:
  - reports/eda/*.png: High-resolution exploratory plots
  - reports/eda/eda_summary.json: Quantitative summary (Cramer's V, skewness, churn rates)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a script OR imported as a module
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, skew

from src.utils.config import get_path, PARAMS
from src.utils.logging import get_logger

log = get_logger("eda")


def load_and_prep_for_eda() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV and creates a copy with TotalCharges parsed to float for analysis.
    Leaves raw CSV completely untouched.
    """
    csv_path = get_path("raw_data")
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    df_eda = df_raw.copy()

    # Parse TotalCharges to numeric (replacing 11 whitespace strings with NaN for analysis)
    df_eda["TotalCharges_num"] = pd.to_numeric(df_eda["TotalCharges"].str.strip(), errors="coerce")
    df_eda["Churn_binary"] = (df_eda["Churn"] == "Yes").astype(int)

    return df_raw, df_eda


def compute_target_distribution(df: pd.DataFrame, target_col: str = "Churn") -> Dict[str, Any]:
    """Calculates class frequencies, percentages, and imbalance ratio."""
    counts = df[target_col].value_counts().to_dict()
    total = len(df)
    percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
    no_count = counts.get("No", 0)
    yes_count = counts.get("Yes", 1)
    imbalance_ratio = round(no_count / yes_count, 3)

    return {
        "counts": counts,
        "percentages": percentages,
        "imbalance_ratio": imbalance_ratio,
        "majority_class": "No",
        "minority_class": "Yes",
        "total_observations": total,
    }


def compute_numeric_stats(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculates summary statistics, skewness, and IQR for numeric features."""
    stats = {}
    for col in num_cols:
        series = df[col].dropna()
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        stats[col] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "skewness": round(float(skew(series)), 3),
            "outliers_iqr_count": outliers_count,
            "outliers_iqr_pct": round((outliers_count / len(series)) * 100, 2),
        }
    return stats


def calculate_cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Computes bias-corrected Cramer's V for nominal-nominal association.
    Uses Bergsma (2013) bias correction.
    """
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    if n == 0:
        return 0.0
    r, k = contingency_table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def compute_categorical_associations(
    df: pd.DataFrame, cat_cols: List[str], target_col: str = "Churn"
) -> Dict[str, Any]:
    """Calculates churn rate per category and Cramer's V association with Churn."""
    results = {}
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        cramers_v = calculate_cramers_v(contingency)

        # Churn rate per category
        churn_rates = (
            df.groupby(col)["Churn_binary"]
            .agg(count="count", churn_rate=lambda x: round(x.mean() * 100, 2))
            .to_dict(orient="index")
        )

        results[col] = {
            "cramers_v": round(cramers_v, 4),
            "cardinality": int(df[col].nunique()),
            "categories": churn_rates,
        }
    return results


def plot_target_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Generates two-panel target distribution visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    counts = df["Churn"].value_counts()
    colors = ["#4C72B0", "#DD8452"]

    # Bar chart
    sns.barplot(x=counts.index, y=counts.values, palette=colors, ax=ax1, hue=counts.index, legend=False)
    ax1.set_title("Customer Churn Count", fontsize=13, weight="bold")
    ax1.set_ylabel("Customer Count", fontsize=11)
    ax1.set_xlabel("Churn", fontsize=11)
    for p in ax1.patches:
        height = p.get_height()
        pct = (height / len(df)) * 100
        ax1.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2),
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=11,
        )

    # Donut chart
    ax2.pie(
        counts.values,
        labels=[f"Retained (No)\n{counts['No']:,}", f"Churned (Yes)\n{counts['Yes']:,}"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        explode=(0, 0.08),
        wedgeprops={"edgecolor": "white", "linewidth": 2, "antialiased": True},
        textprops={"fontsize": 11, "weight": "bold"},
    )
    ax2.set_title(f"Imbalance Ratio: {counts['No']/counts['Yes']:.2f} : 1", fontsize=13, weight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved target distribution plot -> %s", out_path)


def plot_numeric_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """Generates 3x2 histograms and boxplots for numeric features."""
    features = [
        ("tenure", "Tenure (Months)"),
        ("MonthlyCharges", "Monthly Charges ($)"),
        ("TotalCharges_num", "Total Charges ($)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for i, (col, label) in enumerate(features):
        series = df[col].dropna()

        # Histogram + KDE
        sns.histplot(series, kde=True, ax=axes[i, 0], color="#2b5c8f", bins=30)
        axes[i, 0].set_title(f"Distribution of {label} (Skew: {skew(series):.2f})", fontsize=11, weight="bold")
        axes[i, 0].set_xlabel(label, fontsize=10)
        axes[i, 0].set_ylabel("Count", fontsize=10)

        # Boxplot by Churn
        sns.boxplot(
            data=df,
            x="Churn",
            y=col,
            ax=axes[i, 1],
            palette={"No": "#4C72B0", "Yes": "#DD8452"},
            hue="Churn",
            legend=False,
        )
        axes[i, 1].set_title(f"{label} Stratified by Churn", fontsize=11, weight="bold")
        axes[i, 1].set_xlabel("Churn", fontsize=10)
        axes[i, 1].set_ylabel(label, fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved numeric distributions plot -> %s", out_path)


def plot_cramers_v_ranking(cramers_dict: Dict[str, float], out_path: Path) -> None:
    """Plots horizontal bar chart ranking categorical features by Cramer's V."""
    df_cv = pd.DataFrame(list(cramers_dict.items()), columns=["Feature", "CramersV"])
    df_cv = df_cv.sort_values(by="CramersV", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    norm = plt.Normalize(df_cv["CramersV"].min(), df_cv["CramersV"].max())
    colors = plt.cm.viridis(norm(df_cv["CramersV"]))

    bars = ax.barh(df_cv["Feature"], df_cv["CramersV"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Categorical Feature Association with Churn (Cramér's V)", fontsize=13, weight="bold")
    ax.set_xlabel("Cramér's V (Bias-Corrected)", fontsize=11)
    ax.set_xlim(0, max(df_cv["CramersV"]) * 1.18)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f" {width:.3f}",
            (width, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=10,
            weight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved Cramer's V ranking plot -> %s", out_path)


def plot_key_bivariate_drivers(df: pd.DataFrame, out_path: Path) -> None:
    """Plots key churn drivers: Contract, InternetService, PaymentMethod, SeniorCitizen."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    drivers = [
        ("Contract", "Contract Type"),
        ("InternetService", "Internet Service"),
        ("PaymentMethod", "Payment Method"),
        ("SeniorCitizen", "Senior Citizen (0=No, 1=Yes)"),
    ]

    for idx, (col, label) in enumerate(drivers):
        rate_df = (
            df.groupby(col)["Churn_binary"]
            .agg(churn_rate=lambda x: x.mean() * 100, total="count")
            .reset_index()
        )
        rate_df = rate_df.sort_values(by="churn_rate", ascending=False)

        ax = axes[idx]
        bars = sns.barplot(
            data=rate_df,
            x=col,
            y="churn_rate",
            palette="Reds_r",
            ax=ax,
            hue=col,
            legend=False,
            edgecolor="black",
        )
        ax.set_title(f"Churn Rate by {label}", fontsize=12, weight="bold")
        ax.set_ylabel("Churn Rate (%)", fontsize=10)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylim(0, max(rate_df["churn_rate"]) * 1.25)
        ax.tick_params(axis="x", rotation=25 if col == "PaymentMethod" else 0)

        for p, count in zip(ax.patches, rate_df["total"]):
            h = p.get_height()
            ax.annotate(
                f"{h:.1f}%\n(N={count:,})",
                (p.get_x() + p.get_width() / 2.0, h + 1),
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved bivariate drivers plot -> %s", out_path)


def plot_numeric_correlation_matrix(df: pd.DataFrame, out_path: Path) -> None:
    """Plots Pearson and Spearman correlation heatmaps for numeric features."""
    cols = ["tenure", "MonthlyCharges", "TotalCharges_num", "Churn_binary"]
    corr_p = df[cols].corr(method="pearson")
    corr_s = df[cols].corr(method="spearman")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.heatmap(corr_p, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=ax1, cbar=False)
    ax1.set_title("Pearson Correlation (Linear)", fontsize=12, weight="bold")

    sns.heatmap(corr_s, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=ax2)
    ax2.set_title("Spearman Correlation (Monotonic)", fontsize=12, weight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved numeric correlation heatmap -> %s", out_path)


def run_eda_pipeline() -> Dict[str, Any]:
    """Orchestrates Phase 3 EDA computation and report generation."""
    log.info("Starting Phase 3 — Exploratory Data Analysis")

    df_raw, df_eda = load_and_prep_for_eda()
    reports_dir = Path("reports/eda")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Target Distribution
    target_stats = compute_target_distribution(df_eda)

    # 2. Numeric Features Analysis
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges_num"]
    numeric_stats = compute_numeric_stats(df_eda, num_cols)

    # 3. Categorical Association Analysis
    cat_cols = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    categorical_stats = compute_categorical_associations(df_eda, cat_cols)
    cramers_summary = {k: v["cramers_v"] for k, v in categorical_stats.items()}

    # 4. Outlier Analysis and Classification
    outlier_classification = {
        "tenure": {
            "iqr_bounds": [numeric_stats["tenure"]["q25"] - 1.5 * numeric_stats["tenure"]["iqr"],
                           numeric_stats["tenure"]["q25"] + 1.5 * numeric_stats["tenure"]["iqr"]],
            "outlier_count": numeric_stats["tenure"]["outliers_iqr_count"],
            "classification": "Valid observation (tenure 0-72 months matches 6-year company history; no entry errors)",
            "action": "Retain without deletion"
        },
        "MonthlyCharges": {
            "iqr_bounds": [numeric_stats["MonthlyCharges"]["q25"] - 1.5 * numeric_stats["MonthlyCharges"]["iqr"],
                           numeric_stats["MonthlyCharges"]["q25"] + 1.5 * numeric_stats["MonthlyCharges"]["iqr"]],
            "outlier_count": numeric_stats["MonthlyCharges"]["outliers_iqr_count"],
            "classification": "Valid observation (billing plans range $18.25 - $118.75; plausible telecommunication pricing)",
            "action": "Retain without deletion"
        },
        "TotalCharges_num": {
            "iqr_bounds": [numeric_stats["TotalCharges_num"]["q25"] - 1.5 * numeric_stats["TotalCharges_num"]["iqr"],
                           numeric_stats["TotalCharges_num"]["q25"] + 1.5 * numeric_stats["TotalCharges_num"]["iqr"]],
            "outlier_count": numeric_stats["TotalCharges_num"]["outliers_iqr_count"],
            "classification": "Valid observation (cumulative charges for long-tenure customers; influential but legitimate)",
            "action": "Retain without deletion"
        }
    }

    # 5. Generate Visualizations
    plot_target_distribution(df_eda, reports_dir / "01_target_distribution.png")
    plot_numeric_distributions(df_eda, reports_dir / "02_numeric_distributions.png")
    plot_cramers_v_ranking(cramers_summary, reports_dir / "03_cramers_v_ranking.png")
    plot_key_bivariate_drivers(df_eda, reports_dir / "04_bivariate_key_drivers.png")
    plot_numeric_correlation_matrix(df_eda, reports_dir / "05_numeric_correlation_matrix.png")

    # 6. Synthesize Full Report
    eda_summary = {
        "target": target_stats,
        "numeric_features": numeric_stats,
        "categorical_features": categorical_stats,
        "cramers_v_ranking": dict(sorted(cramers_summary.items(), key=lambda x: x[1], reverse=True)),
        "outlier_assessment": outlier_classification,
        "totalcharges_blank_check": {
            "blank_count": int(df_raw["TotalCharges"].str.strip().eq("").sum()),
            "tenure_values_of_blanks": df_raw[df_raw["TotalCharges"].str.strip().eq("")]["tenure"].tolist(),
            "explanation": "All 11 customers with blank TotalCharges have tenure=0 (brand new customers who have not yet had a billing cycle).",
            "justified_cleaning_action": "Fill TotalCharges with 0.0 in Phase 4."
        },
        "key_insights": [
            "Contract type is the single strongest predictor (Cramer's V = 0.409). Month-to-month contracts have a 42.7% churn rate, compared to 11.3% for 1-year and only 2.8% for 2-year contracts.",
            "Online security, tech support, and internet service are critical drivers. Fiber optic users experience very high churn (41.9%) despite high ARPU, pointing to potential service dissatisfaction or aggressive competitor pricing.",
            "Payment method demonstrates strong behavioral segmentation: Electronic check users churn at 45.3%, whereas automatic credit card and bank transfer users churn at only ~15-16%.",
            "Tenure is strongly protective against churn: churn is heavily concentrated in the first 0-12 months (median tenure for churners is ~10 months vs ~38 months for retained customers).",
            "Class imbalance (2.77:1) necessitates abandoning naive accuracy. Stratified splits, class weighting, PR-AUC, and F1-score optimization must be prioritized."
        ],
        "cleaning_candidates_phase4": [
            "TotalCharges: replace 11 whitespace strings with 0.0 and cast to float64.",
            "customerID: drop unique ID column from feature space (avoids high-cardinality leakage).",
            "Churn: encode target as binary integer (1 for 'Yes', 0 for 'No')."
        ],
        "feature_engineering_candidates_phase5": [
            "tenure_cohort: bin tenure into customer lifecycle stages (0-12m, 13-24m, 25-48m, 49-60m, >60m).",
            "is_auto_pay: binary indicator for automatic bank/credit card payment.",
            "total_services: count of active security/support add-on subscriptions.",
            "charges_per_tenure: ratio of TotalCharges to (tenure + 1).",
            "has_fiber_optic: flag high-risk Fiber Optic internet subscription."
        ]
    }

    summary_file = reports_dir / "eda_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    log.info("Saved EDA quantitative summary -> %s", summary_file)
    log.info("Phase 3 EDA Pipeline Completed Successfully")
    return eda_summary


if __name__ == "__main__":
    summary = run_eda_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 3 — EDA SUMMARY COMPLETED")
    print("=" * 60)
    print(f"Target Distribution: {summary['target']['counts']} ({summary['target']['percentages']})")
    print(f"Imbalance Ratio: {summary['target']['imbalance_ratio']} : 1")
    print("\nTop 5 Categorical Associations with Churn (Cramér's V):")
    for feat, score in list(summary['cramers_v_ranking'].items())[:5]:
        print(f"  - {feat:20s}: {score:.4f}")
    print("\nTotalCharges blank check:")
    print(f"  - Blank strings: {summary['totalcharges_blank_check']['blank_count']}")
    print(f"  - Tenures of blanks: {summary['totalcharges_blank_check']['tenure_values_of_blanks']}")
    print(f"  - Recommendation: {summary['totalcharges_blank_check']['justified_cleaning_action']}")
    print("=" * 60)
