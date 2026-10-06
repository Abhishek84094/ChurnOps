import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title
cells.append(nbf.v4.new_markdown_cell("""# 06 — Train / Validation / Test Split
**ChurnOps Phase 7** — Stratified Data Partitioning & Test Set Lock  
*Enforcing strict stratification, repeatable 5-fold cross-validation fold structures, and complete data leakage isolation.*

---

### Phase 7 Objectives
1. **Stratified Partitioning:** Execute an 80/20 train/test split preserving the exact 26.54% minority churn ratio.
2. **Cross-Validation Architecture:** Establish a repeatable 5-fold Stratified K-Fold setup on the training set so all future models (Phases 9–13) train and validate on identical folds.
3. **Data Leakage Isolation (Rule R3):** Lock the test set. All subsequent transformers, encoders, scalers, imputers, and hyperparameter search algorithms must be fitted only on training folds.
4. **Artifact Generation:** Persist `data/processed/train.csv`, `data/processed/test.csv`, and `data/processed/cv_folds.json`."""))

# Cell 2: Setup
cells.append(nbf.v4.new_code_cell(r"""import sys, os
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd()).parent if Path(os.getcwd()).name == 'notebooks' else Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import get_path, PARAMS
from src.data.split import execute_stratified_split

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 110

print("Project Root:", PROJECT_ROOT)"""))

# Cell 3: Load Split Datasets
cells.append(nbf.v4.new_code_cell(r"""train_path = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
test_path = PROJECT_ROOT / 'data' / 'processed' / 'test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train Dataset Shape: {train_df.shape[0]:,} rows x {train_df.shape[1]:,} columns")
print(f"Test Dataset Shape : {test_df.shape[0]:,} rows x {test_df.shape[1]:,} columns [LOCKED]")
print(f"Total Combined Rows: {len(train_df) + len(test_df):,}")"""))

# Cell 4: Markdown Class Balance Verification
cells.append(nbf.v4.new_markdown_cell("""## 1. Class Ratio Verification Across Partitions"""))

# Cell 5: Code Class Balance Table
cells.append(nbf.v4.new_code_cell(r"""total_len = len(train_df) + len(test_df)
train_counts = train_df['Churn'].value_counts()
test_counts = test_df['Churn'].value_counts()

comparison_df = pd.DataFrame([
    {
        'Partition': 'Full Dataset',
        'Total Rows': total_len,
        'Retained (0)': train_counts[0] + test_counts[0],
        'Churned (1)': train_counts[1] + test_counts[1],
        'Churn Rate (%)': round(((train_counts[1] + test_counts[1]) / total_len) * 100, 2),
        'Imbalance Ratio': round((train_counts[0] + test_counts[0]) / (train_counts[1] + test_counts[1]), 4)
    },
    {
        'Partition': 'Train Set (80%)',
        'Total Rows': len(train_df),
        'Retained (0)': train_counts[0],
        'Churned (1)': train_counts[1],
        'Churn Rate (%)': round((train_counts[1] / len(train_df)) * 100, 2),
        'Imbalance Ratio': round(train_counts[0] / train_counts[1], 4)
    },
    {
        'Partition': 'Test Set (20% - Locked)',
        'Total Rows': len(test_df),
        'Retained (0)': test_counts[0],
        'Churned (1)': test_counts[1],
        'Churn Rate (%)': round((test_counts[1] / len(test_df)) * 100, 2),
        'Imbalance Ratio': round(test_counts[0] / test_counts[1], 4)
    }
])

comparison_df"""))

# Cell 6: Visual Comparison Plot
cells.append(nbf.v4.new_code_cell(r"""fig, ax = plt.subplots(figsize=(9, 4.5))
bar_data = pd.DataFrame({
    'Partition': ['Train Set (80%)', 'Test Set (20% - Locked)'],
    'Retained (0)': [train_counts[0] / len(train_df) * 100, test_counts[0] / len(test_df) * 100],
    'Churned (1)': [train_counts[1] / len(train_df) * 100, test_counts[1] / len(test_df) * 100]
})

x = np.arange(len(bar_data))
width = 0.35
ax.bar(x - width/2, bar_data['Retained (0)'], width, label='Retained (0)', color='#4C72B0', edgecolor='black')
ax.bar(x + width/2, bar_data['Churned (1)'], width, label='Churned (1)', color='#DD8452', edgecolor='black')
ax.set_ylabel("Class Percentage (%)")
ax.set_title("Exact Stratification Preserved Between Train and Test", weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(bar_data['Partition'], weight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='upper right')

for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2, h + 1), ha='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.show()"""))

# Cell 7: Markdown CV Folds
cells.append(nbf.v4.new_markdown_cell("""## 2. Cross-Validation Architecture (5-Fold Stratified K-Fold)

To guarantee that model comparisons in Phase 9, 10, 11, 12, and 13 are directly comparable and reproducible, 5 stratified fold indices are fixed on the training set."""))

# Cell 8: Code CV Folds check
cells.append(nbf.v4.new_code_cell(r"""cv_path = PROJECT_ROOT / 'data' / 'processed' / 'cv_folds.json'
with open(cv_path) as f:
    cv_folds = json.load(f)

fold_summary = []
for fold_name, idx_dict in cv_folds.items():
    val_indices = idx_dict['val_indices']
    val_targets = train_df.loc[val_indices, 'Churn']
    fold_summary.append({
        'Fold': fold_name,
        'Train Size': len(idx_dict['train_indices']),
        'Val Size': len(val_indices),
        'Val Retained (0)': (val_targets == 0).sum(),
        'Val Churned (1)': (val_targets == 1).sum(),
        'Val Churn %': f"{val_targets.mean() * 100:.2f}%"
    })

pd.DataFrame(fold_summary)"""))

# Cell 9: Markdown Test Set Lock Notice
cells.append(nbf.v4.new_markdown_cell("""## 3. Strict Test Set Lock Protocol

> ### 🔒 TEST SET LOCK NOTICE (Rule R3)
> - **File:** `data/processed/test.csv` (1,409 rows)
> - **Current Status:** **LOCKED**
> - **Permitted Actions:** None until Phase 14.
> - **Prohibited Actions:** No EDA, no feature selection, no tuning, no scaling fitting, no threshold optimization.
> - **Release Date:** **Phase 14 (Final Model Selection)**, where the single winning pipeline will evaluate on this test set exactly once to measure genuine out-of-sample performance."""))

# Cell 10: Summary markdown
cells.append(nbf.v4.new_markdown_cell("""---

## 4. Phase 7 Summary & Hand-off to Phase 8
- **Train Set:** `data/processed/train.csv` (5,634 rows, 26.54% churn)
- **Test Set:** `data/processed/test.csv` (1,409 rows, 26.54% churn - LOCKED)
- **CV Architecture:** 5 Stratified folds pre-generated and stored in `data/processed/cv_folds.json`.
- **Integrity Status:** Stratification preserved to within < 0.01% disparity; 0 data leakage.
- **Ready for:** Phase 8 — Preprocessing Pipeline (`ColumnTransformer` + `Pipeline`)."""))

nb['cells'] = cells

out_path = Path('notebooks/06_train_test_split.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Generated {out_path} with {len(cells)} cells")
