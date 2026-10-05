import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title
cells.append(nbf.v4.new_markdown_cell("""# 03 — Data Cleaning
**ChurnOps Phase 4** — Deterministic, Domain-Verified Pre-Split Cleaning  
*All cleaning operations are driven directly by EDA/Validation findings without statistical data leakage.*

---

### Phase 4 Objectives
1. **TotalCharges Resolution:** Impute `0.0` for the 11 records with whitespace strings (verified in Phase 3 to have `tenure == 0`) and cast to `float64`.
2. **Whitespace Normalization:** Strip extraneous whitespace from all text/categorical features.
3. **Target Encoding:** Map categorical `Churn` (`{'No', 'Yes'}`) to integer binary (`{0, 1}`).
4. **Identifier Elimination:** Drop `customerID` (unique identifier with 100% cardinality, zero generalizable signal).
5. **Leakage Boundary Check:** Ensure no transformations *learn* statistics from the data (e.g. median/mean imputation or scaling, which must occur within CV folds in Phase 8).
6. **Artifact Output:** Persist clean dataset to `data/interim/telco_clean.csv` and emit `reports/cleaning_log.json`."""))

# Cell 2: Setup
cells.append(nbf.v4.new_code_cell(r"""import sys, os
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd()).parent if Path(os.getcwd()).name == 'notebooks' else Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.utils.config import get_path, PARAMS
from src.data.cleaning import clean_dataset, run_cleaning_pipeline

print("Project Root:", PROJECT_ROOT)"""))

# Cell 3: Load Raw Data
cells.append(nbf.v4.new_code_cell(r"""raw_df = pd.read_csv(get_path('raw_data'))
print(f"Raw Data Shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]:,} columns")
raw_df.head(3)"""))

# Cell 4: Execute Cleaning
cells.append(nbf.v4.new_markdown_cell("""## 1. Execute Cleaning Transformations & Audit Log"""))

# Cell 5: Clean Code
cells.append(nbf.v4.new_code_cell(r"""clean_df, audit_log = clean_dataset(raw_df)
print(f"Cleaned Data Shape: {clean_df.shape[0]:,} rows x {clean_df.shape[1]:,} columns")
print(f"Total Transformations Applied: {len(audit_log['transformations'])}")"""))

# Cell 6: Audit log display
cells.append(nbf.v4.new_code_cell(r"""pd.DataFrame([
    {
        'Step': t['step'],
        'Name': t['name'],
        'Affected Rows': t['affected_rows'],
        'Rationale': t['rationale']
    }
    for t in audit_log['transformations']
])"""))

# Cell 7: Markdown TotalCharges
cells.append(nbf.v4.new_markdown_cell("""## 2. Verification of TotalCharges Imputation & Type Fix

Let us verify the exact 11 records that previously contained whitespace strings:"""))

# Cell 8: Code TotalCharges check
cells.append(nbf.v4.new_code_cell(r"""blank_indices = raw_df[raw_df['TotalCharges'].str.strip() == ''].index

comparison = pd.DataFrame({
    'CustomerID (Raw)': raw_df.loc[blank_indices, 'customerID'],
    'Tenure': raw_df.loc[blank_indices, 'tenure'],
    'MonthlyCharges': raw_df.loc[blank_indices, 'MonthlyCharges'],
    'Raw TotalCharges': raw_df.loc[blank_indices, 'TotalCharges'],
    'Cleaned TotalCharges': clean_df.loc[blank_indices, 'TotalCharges']
})
comparison"""))

# Cell 9: Markdown Target
cells.append(nbf.v4.new_markdown_cell("""## 3. Verification of Target Variable Encoding"""))

# Cell 10: Target code
cells.append(nbf.v4.new_code_cell(r"""print("Raw Churn Value Counts:")
print(raw_df['Churn'].value_counts())
print("\nCleaned Churn Value Counts:")
print(clean_df['Churn'].value_counts())

assert clean_df['Churn'].dtype == 'int64', "Churn dtype must be int64!"
assert clean_df['Churn'].sum() == (raw_df['Churn'] == 'Yes').sum(), "Churn counts do not match!"
print("\nTarget encoding successfully verified!")"""))

# Cell 11: Final integrity checks
cells.append(nbf.v4.new_markdown_cell("""## 4. Final Integrity Checks on Interim Clean Dataset"""))

# Cell 12: Code integrity
cells.append(nbf.v4.new_code_cell(r"""print(f"Total Null Values in Cleaned Dataset: {clean_df.isnull().sum().sum()}")
print(f"Duplicate Rows in Cleaned Dataset     : {clean_df.duplicated().sum()}")
print(f"Is customerID column present?          : {'customerID' in clean_df.columns}")
print("\nCleaned Data Types:")
print(clean_df.dtypes)"""))

# Cell 13: Summary markdown
cells.append(nbf.v4.new_markdown_cell("""---

## 5. Phase 4 Summary & Hand-off to Phase 5
- **Input:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows x 21 cols, unchanged)
- **Output:** `data/interim/telco_clean.csv` (7,043 rows x 20 cols)
- **Missing Values:** 0
- **Leakage Status:** Fully compliant. All modifications are deterministic domain adjustments. No statistical parameters (means, medians, scalers) were fitted on the data.
- **Ready for:** Phase 5 — Feature Engineering."""))

nb['cells'] = cells

out_path = Path('notebooks/03_data_cleaning.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Generated {out_path} with {len(cells)} cells")
