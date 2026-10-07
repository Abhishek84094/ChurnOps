import json, pathlib, textwrap

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src, "id": "md" + str(abs(hash(src)))[:6]}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src, "id": "co" + str(abs(hash(src)))[:6]}

cells = []

cells.append(md(
    "# Phase 9 — Baseline Models\n\n"
    "**Objective:** Train and evaluate four baseline classifiers using full end-to-end Scikit-Learn Pipelines\n"
    "(DomainFeatureEngineer \u2192 ColumnTransformer \u2192 Estimator) with 5-fold cross-validation.\n\n"
    "| Model | Imbalance Strategy | Notes |\n"
    "|---|---|---|\n"
    "| `DummyClassifier` | N/A | Stratified random \u2014 performance floor |\n"
    "| `LogisticRegression` | `class_weight='balanced'` | Linear baseline, needs scaling |\n"
    "| `RandomForestClassifier` | `class_weight='balanced'` | Tree ensemble, no scaling needed |\n"
    "| `XGBClassifier` | `scale_pos_weight=neg/pos` | Gradient boost, recomputed per fold |\n\n"
    "**Rules enforced:**\n"
    "- \u2705 No leakage: pipeline fitted per fold on training split only\n"
    "- \u2705 `test.csv` untouched\n"
    "- \u2705 `random_state=42` everywhere\n"
    "- \u2705 Imbalance handled via `class_weight` / `scale_pos_weight`"
))

cells.append(md("## 9.0  Setup & Imports"))

cells.append(code(
    "import sys, json, time, warnings, pathlib\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "warnings.filterwarnings('ignore')\n\n"
    "# Find project root (works when kernel cwd is notebooks/ or project root)\n"
    "def _find_root(start):\n"
    "    for p in [start] + list(start.parents):\n"
    "        if (p / 'params.yaml').exists():\n"
    "            return p\n"
    "    return start\n\n"
"import os\n"
    "PROJECT_ROOT = _find_root(pathlib.Path.cwd())\n"
    "os.chdir(PROJECT_ROOT)\n"
    "if str(PROJECT_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
    "from sklearn.dummy import DummyClassifier\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.metrics import (\n"
    "    accuracy_score, f1_score, precision_score, recall_score,\n"
    "    roc_auc_score, roc_curve, precision_recall_curve,\n"
    "    average_precision_score, confusion_matrix, ConfusionMatrixDisplay\n"
    ")\n"
    "from xgboost import XGBClassifier\n\n"
    "from src.data.preprocessing import build_full_pipeline\n"
    "from src.utils.config import PARAMS\n\n"
    "RANDOM_STATE = PARAMS['random_state']\n"
    "TARGET       = PARAMS['dataset']['target_column']\n"
    "print(f'Project root : {PROJECT_ROOT}')\n"
    "print(f'Random state : {RANDOM_STATE}')\n"
    "print(f'Target column: {TARGET}')"
))

cells.append(md("## 9.1  Load Training Data & CV Folds"))

cells.append(code(
    "df_train = pd.read_csv('data/processed/train.csv')\n"
    "with open('data/processed/cv_folds.json') as f:\n"
    "    cv_folds = json.load(f)\n\n"
    "print(f'Training set : {df_train.shape[0]:,} rows x {df_train.shape[1]} columns')\n"
    "print(f'CV folds     : {len(cv_folds)}')\n\n"
    "X = df_train.drop(columns=[TARGET])\n"
    "y = df_train[TARGET]\n\n"
    "pos_count   = int((y == 1).sum())\n"
    "neg_count   = int((y == 0).sum())\n"
    "class_ratio = neg_count / pos_count\n"
    "print(f'Positives (Churn=1) : {pos_count:,}  ({pos_count/len(y)*100:.1f}%)')\n"
    "print(f'Negatives (Churn=0) : {neg_count:,}  ({neg_count/len(y)*100:.1f}%)')\n"
    "print(f'neg/pos ratio       : {class_ratio:.4f}  -> XGBoost scale_pos_weight')"
))

cells.append(md("## 9.2  Verify CV Fold Class Balance"))

cells.append(code(
    "fold_stats = []\n"
    "for i, fold_key in enumerate(sorted(cv_folds.keys())):
    fold = cv_folds[fold_key]\n"
    "    y_tr  = y.iloc[fold['train_indices']]\n"
    "    y_val = y.iloc[fold['val_indices']]\n"
    "    fold_stats.append({\n"
    "        'Fold'        : i + 1,\n"
    "        'Train n'     : len(y_tr),\n"
    "        'Val n'       : len(y_val),\n"
    "        'Train churn%': round(y_tr.mean() * 100, 2),\n"
    "        'Val churn%'  : round(y_val.mean() * 100, 2),\n"
    "    })\n\n"
    "pd.DataFrame(fold_stats)"
))

cells.append(md("## 9.3  Build Baseline Pipelines\n\n> Each model is a `Pipeline(DomainFeatureEngineer -> ColumnTransformer -> Estimator)`.  \n> Fresh instances are built **per fold** so no fitted state leaks between folds."))

cells.append(code(
    "def _compute_spw(y_series):\n"
    "    neg = (y_series == 0).sum()\n"
    "    pos = (y_series == 1).sum()\n"
    "    return float(neg / pos) if pos > 0 else 1.0\n\n"
    "def build_pipelines(y_ref):\n"
    "    spw = _compute_spw(y_ref)\n"
    "    return {\n"
    "        'DummyClassifier': build_full_pipeline(\n"
    "            estimator=DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),\n"
    "            scale_numeric=True,\n"
    "        ),\n"
    "        'LogisticRegression': build_full_pipeline(\n"
    "            estimator=LogisticRegression(\n"
    "                class_weight='balanced', max_iter=1000,\n"
    "                solver='lbfgs', random_state=RANDOM_STATE,\n"
    "            ),\n"
    "            scale_numeric=True,\n"
    "        ),\n"
    "        'RandomForest': build_full_pipeline(\n"
    "            estimator=RandomForestClassifier(\n"
    "                n_estimators=200, class_weight='balanced',\n"
    "                random_state=RANDOM_STATE, n_jobs=-1,\n"
    "            ),\n"
    "            scale_numeric=False,\n"
    "        ),\n"
    "        'XGBoost': build_full_pipeline(\n"
    "            estimator=XGBClassifier(\n"
    "                n_estimators=200, scale_pos_weight=spw,\n"
    "                eval_metric='logloss', random_state=RANDOM_STATE,\n"
    "                n_jobs=-1, verbosity=0,\n"
    "            ),\n"
    "            scale_numeric=False,\n"
    "        ),\n"
    "    }\n\n"
    "# Preview pipeline steps\n"
    "sample_pipes = build_pipelines(y)\n"
    "print('Pipelines created:')\n"
    "for name, pipe in sample_pipes.items():\n"
    "    print(f'  {name:<24} steps={list(pipe.named_steps.keys())}')"
))

cells.append(md("## 9.4  Run 5-Fold Cross-Validation\n\n> Rule R3: Each fold builds **fresh** unfitted pipelines and fits them on the training split of that fold only. No data from the validation split is ever used for fitting."))

cells.append(code(
    "MODEL_NAMES = ['DummyClassifier', 'LogisticRegression', 'RandomForest', 'XGBoost']\n\n"
    "def evaluate_fold(pipe, X_tr, y_tr, X_val, y_val):\n"
    "    pipe.fit(X_tr, y_tr)\n"
    "    y_pred = pipe.predict(X_val)\n"
    "    try:\n"
    "        y_prob = pipe.predict_proba(X_val)[:, 1]\n"
    "    except AttributeError:\n"
    "        y_prob = pipe.decision_function(X_val)\n"
    "    return {\n"
    "        'accuracy' : round(accuracy_score(y_val, y_pred), 4),\n"
    "        'precision': round(precision_score(y_val, y_pred, zero_division=0), 4),\n"
    "        'recall'   : round(recall_score(y_val, y_pred, zero_division=0), 4),\n"
    "        'f1'       : round(f1_score(y_val, y_pred, zero_division=0), 4),\n"
    "        'roc_auc'  : round(roc_auc_score(y_val, y_prob), 4),\n"
    "        'y_prob'   : y_prob,\n"
    "        'y_pred'   : y_pred,\n"
    "        'y_true'   : y_val.values,\n"
    "    }\n\n"
    "fold_results = {name: [] for name in MODEL_NAMES}\n"
    "t_total = time.perf_counter()\n\n"
    "for fold_idx, fold in enumerate(cv_folds):\n"
    "    X_tr  = X.iloc[fold['train_indices']].reset_index(drop=True)\n"
    "    y_tr  = y.iloc[fold['train_indices']].reset_index(drop=True)\n"
    "    X_val = X.iloc[fold['val_indices']].reset_index(drop=True)\n"
    "    y_val = y.iloc[fold['val_indices']].reset_index(drop=True)\n\n"
    "    pipes = build_pipelines(y_tr)  # fresh instances per fold\n\n"
    "    print(f\"\\n{'='*65}\")\n"
    "    print(f\" Fold {fold_idx+1}/{len(cv_folds)} | train={len(y_tr)}  val={len(y_val)}  \"\n"
    "          f\"pos_rate_train={y_tr.mean():.3f}  pos_rate_val={y_val.mean():.3f}\")\n"
    "    print(f\"{'='*65}\")\n\n"
    "    for name in MODEL_NAMES:\n"
    "        pipe = pipes[name]\n"
    "        if name == 'XGBoost':\n"
    "            pipe.named_steps['model'].set_params(scale_pos_weight=_compute_spw(y_tr))\n"
    "        t0 = time.perf_counter()\n"
    "        metrics = evaluate_fold(pipe, X_tr, y_tr, X_val, y_val)\n"
    "        elapsed = time.perf_counter() - t0\n"
    "        metrics['train_time_s'] = round(elapsed, 2)\n"
    "        fold_results[name].append(metrics)\n"
    "        print(f\"  {name:<24} F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}  \"\n"
    "              f\"Recall={metrics['recall']:.4f}  [{elapsed:.1f}s]\")\n\n"
    "print(f\"\\nTotal CV time: {time.perf_counter()-t_total:.1f}s\")"
))

cells.append(md("## 9.5  Summary Table — Mean \u00b1 Std Across Folds"))

cells.append(code(
    "METRIC_COLS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n\n"
    "rows = []\n"
    "for name in MODEL_NAMES:\n"
    "    folds_data = fold_results[name]\n"
    "    row = {'Model': name}\n"
    "    for m in METRIC_COLS:\n"
    "        vals = [fd[m] for fd in folds_data]\n"
    "        row[m] = f\"{np.mean(vals):.4f} \u00b1 {np.std(vals):.4f}\"\n"
    "    row['mean_train_s'] = f\"{np.mean([fd['train_time_s'] for fd in folds_data]):.1f}\"\n"
    "    rows.append(row)\n\n"
    "df_summary = pd.DataFrame(rows).set_index('Model')\n"
    "display(df_summary)\n\n"
    "min_f1  = PARAMS['registry']['min_f1']\n"
    "min_auc = PARAMS['registry']['min_roc_auc']\n"
    "min_rec = PARAMS['registry']['min_recall']\n"
    "print(f'\\n[!] Registry gates: F1>={min_f1}  ROC-AUC>={min_auc}  Recall>={min_rec}')"
))

cells.append(md("## 9.6  Visualisations\n### 9.6.1  F1 / ROC-AUC Boxplots Across Folds"))

cells.append(code(
    "COLORS = ['#6B7280', '#3B82F6', '#10B981', '#F59E0B']\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "for ax, metric, title in zip(\n"
    "    axes, ['f1', 'roc_auc'],\n"
    "    ['F1 Score (5-fold CV)', 'ROC-AUC (5-fold CV)'],\n"
    "):\n"
    "    data_plot = [[fd[metric] for fd in fold_results[n]] for n in MODEL_NAMES]\n"
    "    bp = ax.boxplot(data_plot, patch_artist=True, widths=0.55)\n"
    "    for patch, c in zip(bp['boxes'], COLORS):\n"
    "        patch.set_facecolor(c); patch.set_alpha(0.75)\n"
    "    ax.set_xticks(range(1, len(MODEL_NAMES)+1))\n"
    "    ax.set_xticklabels(MODEL_NAMES, rotation=20, ha='right', fontsize=10)\n"
    "    ax.set_title(title, fontsize=13, fontweight='bold')\n"
    "    ax.set_ylabel(metric.upper(), fontsize=11)\n"
    "    ax.grid(axis='y', alpha=0.4)\n"
    "    thresh_val = PARAMS['registry']['min_f1'] if metric=='f1' else PARAMS['registry']['min_roc_auc']\n"
    "    ax.axhline(thresh_val, color='red', ls='--', alpha=0.7,\n"
    "               label=f'min_{metric}={thresh_val}')\n"
    "    ax.legend(fontsize=9)\n\n"
    "plt.suptitle('Phase 9 \u2014 Baseline Model Comparison (5-Fold CV)',\n"
    "             fontsize=14, fontweight='bold', y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.savefig('reports/baseline_fold_comparison.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: reports/baseline_fold_comparison.png')"
))

cells.append(md("### 9.6.2  ROC Curves (Last Fold)"))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "for name, color in zip(MODEL_NAMES, COLORS):\n"
    "    last = fold_results[name][-1]\n"
    "    fpr, tpr, _ = roc_curve(last['y_true'], last['y_prob'])\n"
    "    auc = roc_auc_score(last['y_true'], last['y_prob'])\n"
    "    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.4f})')\n"
    "ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random baseline')\n"
    "ax.set_xlabel('False Positive Rate', fontsize=12)\n"
    "ax.set_ylabel('True Positive Rate', fontsize=12)\n"
    "ax.set_title('ROC Curves \u2014 Fold 5 (Validation Set)', fontsize=13, fontweight='bold')\n"
    "ax.legend(loc='lower right', fontsize=10)\n"
    "ax.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('reports/baseline_roc_curves.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: reports/baseline_roc_curves.png')"
))

cells.append(md("### 9.6.3  Precision-Recall Curves (Last Fold)"))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "for name, color in zip(MODEL_NAMES, COLORS):\n"
    "    last = fold_results[name][-1]\n"
    "    prec, rec, _ = precision_recall_curve(last['y_true'], last['y_prob'])\n"
    "    ap = average_precision_score(last['y_true'], last['y_prob'])\n"
    "    ax.plot(rec, prec, color=color, lw=2, label=f'{name} (AP={ap:.4f})')\n"
    "ax.axhline(y.mean(), color='gray', ls='--', alpha=0.7,\n"
    "           label=f'No-skill ({y.mean():.3f})')\n"
    "ax.set_xlabel('Recall', fontsize=12)\n"
    "ax.set_ylabel('Precision', fontsize=12)\n"
    "ax.set_title('Precision-Recall Curves \u2014 Fold 5', fontsize=13, fontweight='bold')\n"
    "ax.legend(loc='upper right', fontsize=10)\n"
    "ax.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('reports/baseline_pr_curves.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: reports/baseline_pr_curves.png')"
))

cells.append(md("### 9.6.4  Confusion Matrices (Last Fold, All Models)"))

cells.append(code(
    "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\n"
    "for ax, name in zip(axes, MODEL_NAMES):\n"
    "    last = fold_results[name][-1]\n"
    "    cm = confusion_matrix(last['y_true'], last['y_pred'])\n"
    "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn','Churn'])\n"
    "    disp.plot(ax=ax, colorbar=False, cmap='Blues')\n"
    "    ax.set_title(name, fontsize=11, fontweight='bold')\n"
    "    ax.tick_params(labelsize=9)\n"
    "plt.suptitle('Confusion Matrices \u2014 Fold 5 (Validation Set)',\n"
    "             fontsize=13, fontweight='bold', y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.savefig('reports/baseline_confusion_matrices.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: reports/baseline_confusion_matrices.png')"
))

cells.append(md("## 9.7  Save Results to reports/"))

cells.append(code(
    "import datetime\n"
    "SAVE_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n\n"
    "# Strip numpy arrays before serializing\n"
    "fold_results_serializable = {}\n"
    "for name in MODEL_NAMES:\n"
    "    fold_results_serializable[name] = [\n"
    "        {k: v for k, v in fd.items() if k not in ('y_prob','y_pred','y_true')}\n"
    "        for fd in fold_results[name]\n"
    "    ]\n\n"
    "cv_report = {\n"
    "    'timestamp'          : datetime.datetime.utcnow().isoformat(),\n"
    "    'phase'              : 9,\n"
    "    'n_folds'            : len(cv_folds),\n"
    "    'train_rows'         : len(df_train),\n"
    "    'class_balance'      : {\n"
    "        'positive'      : pos_count,\n"
    "        'negative'      : neg_count,\n"
    "        'neg_pos_ratio' : round(class_ratio, 4),\n"
    "    },\n"
    "    'imbalance_strategy' : {\n"
    "        'LR_RF'  : 'class_weight=balanced',\n"
    "        'XGBoost': f'scale_pos_weight={class_ratio:.4f}',\n"
    "        'Dummy'  : 'N/A',\n"
    "    },\n"
    "    'fold_results'       : fold_results_serializable,\n"
    "}\n"
    "with open('reports/baseline_cv_results.json', 'w') as f:\n"
    "    json.dump(cv_report, f, indent=2)\n\n"
    "summary_dict = {}\n"
    "for name in MODEL_NAMES:\n"
    "    folds_data = fold_results[name]\n"
    "    s = {}\n"
    "    for m in SAVE_METRICS:\n"
    "        vals = [fd[m] for fd in folds_data]\n"
    "        s[m] = f\"{np.mean(vals):.4f} +/- {np.std(vals):.4f}\"\n"
    "    s['mean_train_time_s'] = f\"{np.mean([fd['train_time_s'] for fd in folds_data]):.2f}\"\n"
    "    summary_dict[name] = s\n\n"
    "with open('reports/baseline_summary.json', 'w') as f:\n"
    "    json.dump({'phase': 9, 'summary': summary_dict}, f, indent=2)\n\n"
    "print('Saved:')\n"
    "print('  reports/baseline_cv_results.json')\n"
    "print('  reports/baseline_summary.json')"
))

cells.append(md("## 9.8  Model Ranking & Registry Gate Check"))

cells.append(code(
    "print('\\n' + '='*70)\n"
    "print('PHASE 9 - BASELINE MODELS  |  5-Fold CV Summary')\n"
    "print('='*70)\n"
    "print(f\"{'Model':<24} {'F1 mean+/-std':>18} {'ROC-AUC':>10} {'Recall':>10} {'Precision':>11}\")\n"
    "print('-'*70)\n\n"
    "ranking = []\n"
    "for name in MODEL_NAMES:\n"
    "    folds_data = fold_results[name]\n"
    "    mf1  = np.mean([fd['f1']        for fd in folds_data])\n"
    "    mauc = np.mean([fd['roc_auc']   for fd in folds_data])\n"
    "    mrec = np.mean([fd['recall']    for fd in folds_data])\n"
    "    mpre = np.mean([fd['precision'] for fd in folds_data])\n"
    "    sf1  = np.std( [fd['f1']        for fd in folds_data])\n"
    "    ranking.append((name, mf1, mauc, mrec, mpre, sf1))\n"
    "    print(f\"{name:<24} {mf1:.4f} +/- {sf1:.4f}  {mauc:.4f}  {mrec:.4f}  {mpre:.4f}\")\n\n"
    "print('='*70)\n\n"
    "min_f1  = PARAMS['registry']['min_f1']\n"
    "min_auc = PARAMS['registry']['min_roc_auc']\n"
    "min_rec = PARAMS['registry']['min_recall']\n"
    "print(f'\\nRegistry Gate Check (F1>={min_f1}  AUC>={min_auc}  Recall>={min_rec}):')\n"
    "for name, mf1, mauc, mrec, mpre, sf1 in ranking:\n"
    "    gate = 'PASS' if mf1>=min_f1 and mauc>=min_auc and mrec>=min_rec else 'FAIL'\n"
    "    print(f'  {name:<24} -> {gate}   (F1={mf1:.4f} AUC={mauc:.4f} Recall={mrec:.4f})')\n\n"
    "non_dummy = [(n,f,a,r,p,s) for n,f,a,r,p,s in ranking if n!='DummyClassifier']\n"
    "best = max(non_dummy, key=lambda x: x[1])\n"
    "print(f'\\nBest non-dummy baseline by F1: {best[0]}  (mean F1={best[1]:.4f})')\n"
    "print('\\nNext: Phase 10 - Cross-Validation deep-dive.')"
))

cells.append(md("## 9.9  Update PROGRESS.md"))

cells.append(code(
    "from datetime import date\n"
    "prog_path = Path('PROGRESS.md')\n"
    "content = prog_path.read_text(encoding='utf-8')\n"
    "old = '| 9     | Baseline Models               | \\u23f3 Pending  |            |'\n"
    "new = f'| 9     | Baseline Models               | \\u2705 Complete | {date.today()} |'\n"
    "if old in content:\n"
    "    prog_path.write_text(content.replace(old, new), encoding='utf-8')\n"
    "    print(f'PROGRESS.md updated: Phase 9 -> Complete ({date.today()})')\n"
    "else:\n"
    "    print('Pattern not found - check PROGRESS.md manually.')"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.8"}
    },
    "cells": cells
}

out = pathlib.Path("notebooks/08_baseline_models.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written: {out}  ({out.stat().st_size:,} bytes)")
