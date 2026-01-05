# %%
# # 01_Analysis

# %%
# In summary, 01_Analysis.py contains code for predictive modeling and statistical analyses comparing cognitive decline prediction across OASIS-3 and ADNI datasets. The code is structured as follows:
# 
#     0. Setup and Helper Functions
#         0.1 Package imports
#         0.2 Helper functions for data processing, model fitting, and evaluation
# 
#     1. Data Loading and Within-Dataset Predictions
#         1.1 Data loading and preprocessing
#         1.2 Within-dataset predictions (OASIS-3 and ADNI)
#         1.3 Results aggregation and summary
# 
#     2. Permutation Importance Analysis
#         2.1 Permutation importance for individual features
#         2.2 Feature importance visualization and analysis
#         2.3 Coalition-based feature importance
# 
#     3. Across-Dataset Predictions
#         3.1 Cross-dataset prediction (clinical, structural, combined, top-15)
#         3.2 Cross-dataset results aggregation
# 
#     4. Statistical Model Comparisons
#         4.1 Statistical Testing Framework and Model Comparisons
#         4.2 Absolute Error Comparisons Between Models
# 
#     5. Subgroup Analysis
#         5.1 Absolute error analysis preparation
#         5.2 Subgroup comparisons by diagnosis, sex, APOE E4, and age
#         5.3 Results formatting and export
# 
#     6. Post-Hoc Matched Samples Analysis
#         6.1 Quantile-based sample matching
#         6.2 Matched sample predictions and evaluation
#         6.3 Results for both combined and top-15 models
# 
# Note: Running the predictive models is computationally expensive with n_splits=1000. Consider reducing splits for testing. 
# Full results tables are available in the 02_Supplementary_Material folder at https://osf.io/up65f/files/osfstorage.


# %%
# ## 0. Setup and Helper Functions
# ### 0.1 Package Imports

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so

import sklearn
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Arial']

from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu

from tqdm.notebook import tqdm

from pathlib import Path
from joblib import Parallel, delayed

np.random.seed(21) # by setting a seed, you can ensure that every time you run the code, the sequence of random numbers generated will be the same

# %%
# ### 0.2 Helper Functions for Data Processing, Model Fitting, and Evaluation

def get_data(clinical_features_path, data_slopes_path, structural_data_path, missing_cols=[]):
    clinical_features = pd.read_pickle(clinical_features_path)
    data_slopes = pd.read_pickle(data_slopes_path)
    structural_data = pd.read_pickle(structural_data_path)

    # clinical_features.columns.str.strip() str --> columns as strings not as indexes
    clinical_features.columns = clinical_features.columns.str.removeprefix("clin__npsy__").str.removeprefix("clin__risk__").str.removeprefix("clin__assess__")

    clinical_features["demo_sex"] = clinical_features["demo_sex"].map({'F': 0, 'M': 1, 0: 0, 1: 1})
    clinical_features["diag"] = clinical_features["diag"].map({"hc": 0, "mci": 1, "dem": 2.0, "NL": 0.0, "MCI": 1.0, "Dementia": 2.0})

    # defining the dependent and independent variables and removing the variable "subject" from the dataframes
    y = data_slopes.copy().set_index("subject")
    X_clin = clinical_features.copy().set_index("subject")
    X_fs = structural_data.copy().set_index("subject")
    X_clin_fs = pd.concat([X_clin, X_fs], axis=1)

    # remove missing columns if specified
    if missing_cols:
        X_clin = X_clin.drop(columns=missing_cols, errors='ignore')
        X_fs = X_fs.drop(columns=missing_cols, errors='ignore')
        X_clin_fs = X_clin_fs.drop(columns=missing_cols, errors='ignore')

    return y, X_clin, X_fs, X_clin_fs


def get_iterative_imputer(X_train, X_test, random_state):
    imputer = IterativeImputer(add_indicator=True, random_state=random_state, n_nearest_features=50)
    missing_cols_train = [c + "_missing" for c in X_train.columns[X_train.isna().any()]]
    missing_cols_test = [c + "_missing" for c in X_test.columns[X_test.isna().any()]]    
    feature_names_train = X_train.columns.to_list() + missing_cols_train
    feature_names_test = X_test.columns.to_list() + missing_cols_test
    return imputer, feature_names_train, feature_names_test

# %%


def fit_pipeline(X, y, train_idx, test_idx, root_dir):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    # iterative imputation

    # define imputer
    imputer, feature_names_train, feature_names_test = get_iterative_imputer(X_train, X_test, random_state=0)

    # define pipeline
    pipeline = make_pipeline(imputer, RandomForestRegressor(random_state=0))
    
    # fit pipeline
    pipeline.fit(X_train, y_train)

    # predict
    y_pred = pd.DataFrame(pipeline.predict(X_test)).rename(columns={0: 'mmse_pred', 1: 'sob_pred'})
    y_pred.index = y_test.index.values

    # concatenate
    df = pd.concat([y_pred, y_test], axis = 1)

    # get MAE, MSE, R2 for the predictions with scikit-learn
    
    metrics = {
        'r2': r2_score,
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
    }
    results_df = pd.DataFrame({
        var: {name: func(df[f'{var}_slope'], df[f'{var}_pred']) for name, func in metrics.items()}
        for var in ['sob', 'mmse']
    }).T



    # save pipeline and dataframe
    if root_dir:
        save_pipeline_and_df(pipeline, df, results_df, root_dir)

    return df, pipeline, results_df

def save_pipeline_and_df(pipeline, df, results_df, root_dir):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, f"{root_dir}/pipeline.joblib", compress=('xz', 3))
    df.to_csv(f"{root_dir}/predictions.csv", index=True, header=True)
    results_df.to_csv(f"{root_dir}/results.csv", index=True, header=True)

def compress_pipeline_file(root_dir):
    pipeline_path = Path(root_dir) / "pipeline.joblib"
    if pipeline_path.exists():
        with open(pipeline_path, 'rb') as f:
            pipeline = joblib.load(f)
        with open(pipeline_path, 'wb') as f:
            joblib.dump(pipeline, f, compress=('xz', 3))


def split_and_fit(X, y, root_dir, n_splits=1000, test_size=0.2):
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

    # split data
    for iter, (train_idx, test_idx) in tqdm(enumerate(ss.split(X, y)), total=n_splits):
        path = Path(f"{root_dir}/iteration_{iter+1}")
        path.mkdir(parents=True, exist_ok=True)
        # check if both pipeline and df already exist
        if (path / "pipeline.joblib").exists() and (path / "predictions.csv").exists():
            continue

        fit_pipeline(X, y, train_idx, test_idx, path)

def get_results(root, iters=range(1000)):
    dfs = [pd.read_csv(f"{root}/iteration_{i+1}/results.csv", index_col=0) for i in iters]
    return pd.concat(dfs, keys=iters, names=["iteration", "target"])

def get_results_summary(root, iters=range(1000)):
    results = get_results(root, iters)
    summary = results.groupby(level=0).agg(['mean', 'median'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary

# %%
# ## 1. Data Loading and Within-Dataset Predictions
# ### 1.1 Data Loading and Preprocessing

# Load data
oasis_y, oasis_X_clin, oasis_X_fs, oasis_X_clin_fs = get_data("OASIS_clinical_features_X.pkl", "OASIS_data_slopes_y.pkl", "OASIS_structGlobScort.pkl")
adni_y, adni_X_clin, adni_X_fs, adni_X_clin_fs = get_data("ADNI_1237_data_clinical_features_X_clinsessionn_2025.pkl", "ADNI_1237_data_slopes_y_2025.pkl", "ADNI_1237_data_structGlobScort_2025.pkl")

missing_adni = ["smoke_PACKSPER", "TMTA_TRAILALI", "smoke_TOBAC100", "smoke_SMOKYRS", "TMTB_TRAILBLI", "smoke_TOBAC30", "WMSds_DIGIF", "WMSds_DIGIFLEN", "WF_VEG", "WMSds_DIGIB", "WMSds_DIGIBLEN", "WAIS_WAIS", "familyHist_DADDEM", "familyHist_MOMDEM"]
missing_oasis = ["familyHist_sumSibDem","familyHist_ratioSibDem","TMTB_TRAILBRR","TMTB_TRAILBLI","TMTA_TRAILALI","TMTA_TRAILARR"]

missing_cols = missing_adni + missing_oasis

oasis_y, oasis_X_clin_filtered, oasis_X_fs_filtered, oasis_X_clin_fs_filtered = get_data("OASIS_clinical_features_X.pkl", "OASIS_data_slopes_y.pkl", "OASIS_structGlobScort.pkl", missing_cols=missing_cols)
adni_y, adni_X_clin_filtered, adni_X_fs_filtered, adni_X_clin_fs_filtered = get_data("ADNI_1237_data_clinical_features_X_clinsessionn_2025.pkl", "ADNI_1237_data_slopes_y_2025.pkl", "ADNI_1237_data_structGlobScort_2025.pkl", missing_cols=missing_cols)

data_combinations = [
    (oasis_y, oasis_X_clin, "OASIS_clin_predictions"),
    (oasis_y, oasis_X_fs, "OASIS_struct_predictions"),
    (oasis_y, oasis_X_clin_fs, "OASIS_clin_struct_predictions"),
    (adni_y, adni_X_clin, "ADNI_clin_predictions"),
    (adni_y, adni_X_fs, "ADNI_struct_predictions"),
    (adni_y, adni_X_clin_fs, "ADNI_clin_struct_predictions"),
    (oasis_y, oasis_X_clin_filtered, "OASIS_clin_predictions_filtered"),
    (oasis_y, oasis_X_fs_filtered, "OASIS_struct_predictions_filtered"),
    (oasis_y, oasis_X_clin_fs_filtered, "OASIS_clin_struct_predictions_filtered"),
    (adni_y, adni_X_clin_filtered, "ADNI_clin_predictions_filtered"),
    (adni_y, adni_X_fs_filtered, "ADNI_struct_predictions_filtered"),
    (adni_y, adni_X_clin_fs_filtered, "ADNI_clin_struct_predictions_filtered"),
]

oasis_X_clin_filtered.to_csv("OASIS_clin_filtered.csv", index=True, header=True)
adni_X_clin_filtered.to_csv("ADNI_clin_filtered.csv", index=True, header=True)

# ### 1.2 Within-Dataset Predictions (OASIS-3 and ADNI)

for y, X, root_dir in data_combinations:
    print(f"Processing {root_dir}...")
    split_and_fit(X, y, root_dir, n_splits=1000)

# %%
# ### 1.3 Results Aggregation and Summary

def get_dataset_results(clin_file, struct_file, clin_struct_file, iters=range(1000)):
    clin_results = get_results(clin_file, iters)
    struct_results = get_results(struct_file, iters)
    clin_struct_results = get_results(clin_struct_file, iters)

    results = pd.concat([clin_results, struct_results, clin_struct_results], keys=["clin", "struct", "clin_struct"], names=["modality", "iteration", "target"])

    return results

oasis_results = get_dataset_results("OASIS_clin_predictions", "OASIS_struct_predictions", "OASIS_clin_struct_predictions")
adni_results = get_dataset_results("ADNI_clin_predictions", "ADNI_struct_predictions", "ADNI_clin_struct_predictions")
oasis_filtered_results = get_dataset_results("OASIS_clin_predictions_filtered", "OASIS_struct_predictions_filtered", "OASIS_clin_struct_predictions_filtered")
adni_filtered_results = get_dataset_results("ADNI_clin_predictions_filtered", "ADNI_struct_predictions_filtered", "ADNI_clin_struct_predictions_filtered")

results = pd.concat([oasis_results, adni_results, oasis_filtered_results, adni_filtered_results], keys=["OASIS", "ADNI", "OASIS_filtered", "ADNI_filtered"], names=["dataset", "modality", "iteration", "target"])
results.to_csv("results_within.csv")

results.groupby(level=["dataset", "modality", "target"]).agg(['mean', 'median'])


# %%
# ## 2. Permutation Importance Analysis
# ### 2.1 Permutation Importance for Individual Features

def load_pipeline(iter, root_dir):
    path = Path(f"{root_dir}/iteration_{iter+1}/pipeline.joblib")
    if path.exists():
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"Pipeline for iteration {iter+1} not found in {root_dir}.")


def get_permutation_importance_iter(pipeline, X_test, y_test, n_repeats=100, random_state=0):
    pi = permutation_importance(pipeline, X_test, y_test, n_repeats=n_repeats, scoring='r2', random_state=random_state)
    return pd.DataFrame({"feature": X_test.columns, "permutation_importance": pi.importances_mean})

def get_permutation_importance(X, y, root_dir, n_splits=1000, n_repeats=100):
    
    # quickly check that every path exists
    root_dir = Path(root_dir)
    for iter in range(n_splits):
        path = root_dir / f"iteration_{iter+1}"
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist. Make sure to run split_and_fit first.")
    
    def process_iteration(iter):
        path = root_dir / f"iteration_{iter+1}"
        pi_path = path / f"permutation_importance_repeats-{n_repeats}.csv"
        if pi_path.exists():
            print(f"Skipping iteration {iter+1}, already processed.")
            return pd.read_csv(pi_path, index_col=0)
        else:
            ss = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
            train_indices = []
            test_indices = []
            for i, (train_idx, test_idx) in enumerate(ss.split(X, y)):
                if i == iter:
                    train_indices = train_idx
                    test_indices = test_idx
                    break
            
            pipeline = load_pipeline(iter, root_dir)
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]
            
            pi_df = get_permutation_importance_iter(pipeline, X_test, y_test, n_repeats=n_repeats)
            pi_df.to_csv(pi_path)
            return pi_df
    
    # Run the processing in parallel with 4 jobs
    results = Parallel(n_jobs=4)(
        delayed(process_iteration)(iter) for iter in tqdm(range(n_splits), desc="Processing iterations")
    )
    
    results = pd.concat(results, keys=range(n_splits), names=["iteration", "feature"])
    results = results.droplevel("feature")

    return results

oasis_pi = get_permutation_importance(oasis_X_clin_fs, oasis_y,"OASIS_clin_struct_predictions", n_splits=1000, n_repeats=5)
adni_pi = get_permutation_importance(adni_X_clin_fs, adni_y, "ADNI_clin_struct_predictions", n_splits=1000, n_repeats=5)
oasis_filtered_pi = get_permutation_importance(oasis_X_clin_fs_filtered, oasis_y, "OASIS_clin_struct_predictions_filtered", n_splits=1000, n_repeats=5)
adni_filtered_pi = get_permutation_importance(adni_X_clin_fs_filtered, adni_y, "ADNI_clin_struct_predictions_filtered", n_splits=1000, n_repeats=5)

oasis_filtered_pi.set_index("feature", append=True).unstack("feature").droplevel(0, axis=1).to_csv("OASIS_clin_struct_permutation_importance.csv")
adni_filtered_pi.set_index("feature", append=True).unstack("feature").droplevel(0, axis=1).to_csv("ADNI_clin_struct_permutation_importance.csv")

# ### 2.2 Feature Importance Visualization and Analysis

# %%
def get_across_permutation_importance(X, y, root_dir, n_repeats=100, label="permutation_importance"):
    # quickly check that every path exists
    path = Path(root_dir)
    
    pi_path = path / f"{label}_repeats-{n_repeats}.csv"
    if pi_path.exists():
        print(f"Skipping {label}, already processed.")
        pi_df = pd.read_csv(pi_path, index_col=0)
    else:
        pipeline = joblib.load(f"{root_dir}/pipeline.joblib")
        pi = permutation_importance(pipeline, X, y, n_repeats=n_repeats, scoring='r2', random_state=0)
        pi_df = pd.DataFrame({"feature": X.columns, "permutation_importance": pi.importances_mean, "permutation_importance_std": pi.importances_std})
        pi_df.to_csv(pi_path)
    return pi_df

oasis2adni_pi = get_across_permutation_importance(adni_X_clin_fs_filtered, adni_y, "OASIS_to_ADNI_clin_struct", n_repeats=100)
adni2oasis_pi = get_across_permutation_importance(oasis_X_clin_fs_filtered, oasis_y, "ADNI_to_OASIS_clin_struct", n_repeats=100)


# ### 2.3 Coalition-Based Feature Importance

# %%
feature_coalitions = pd.read_csv("features.csv", header=0)

# group by coalition and feature type
coalition_dict = feature_coalitions.groupby("coalition")["feature"].apply(list).to_dict()
feature_type_dict = feature_coalitions.groupby("feature-type")["feature"].apply(list).to_dict()


# get importance by feature coalition
def importance_by_coalition(pipeline, X_test, y_test, coalition_dict, random_state=0):
    # manual reimplementation of permutation_importance
    result = {}

    for coalition, features_in in coalition_dict.items():
        # permute the rows within the coalition
        X_test_coalition = X_test.copy()
        X_test_coalition[features_in] = X_test_coalition[features_in].sample(frac=1, random_state=random_state).values
        pred = pipeline.predict(X_test_coalition)
        result[coalition] = metrics.r2_score(y_test, pred)

    pred = pipeline.predict(X_test)

    all_score = metrics.r2_score(y_test, pred)

    result = pd.DataFrame(result, index=['permutation_importance'])
    result = all_score - result.T

    return result


def get_importance_by_coalition(X, y, root_dir, coalition_dict, n_splits=1000, n_repeats=5, label="coalition_importance"):
    # quickly check that every path exists
    root_dir = Path(root_dir)
    for i in range(n_splits):
        path = root_dir / f"iteration_{i+1}"
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist. Make sure to run split_and_fit first.")

    def process_iteration(iter):
        path = root_dir / f"iteration_{iter+1}"
        pi_path = path / f"{label}_repeats-{n_repeats}.csv"
        if pi_path.exists():
            return pd.read_csv(pi_path, index_col=0)
        else:
            ss = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
            test_indices = []
            for i, (train_idx, test_idx) in enumerate(ss.split(X, y)):
                if i == iter:
                    test_indices = test_idx
                    break
            
            pipeline = load_pipeline(iter, root_dir)
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            pi_df = [importance_by_coalition(pipeline, X_test, y_test, coalition_dict, random_state=i) for i in range(n_repeats)]
            pi_df = pd.concat(pi_df).groupby(level=0).mean()
            pi_df.to_csv(pi_path)
            return pi_df

    # Run the processing in parallel with 4 jobs
    results = Parallel(n_jobs=4)(
        delayed(process_iteration)(iter) for iter in tqdm(range(n_splits), desc="Processing iterations")
    )
    results = pd.concat(results, keys=range(n_splits), names=["iteration", "coalition"])

    return results

oasis_filtered_ci = get_importance_by_coalition(oasis_X_clin_fs_filtered, oasis_y, "OASIS_clin_struct_predictions_filtered", coalition_dict, n_splits=1000, n_repeats=5)
adni_filtered_ci = get_importance_by_coalition(adni_X_clin_fs_filtered, adni_y, "ADNI_clin_struct_predictions_filtered", coalition_dict, n_splits=1000, n_repeats=5)
oasis_filtered_type_ci = get_importance_by_coalition(oasis_X_clin_fs_filtered, oasis_y, "OASIS_clin_struct_predictions_filtered", feature_type_dict, n_splits=1000, n_repeats=5, label="feature_type_importance")
adni_filtered_type_ci = get_importance_by_coalition(adni_X_clin_fs_filtered, adni_y, "ADNI_clin_struct_predictions_filtered", feature_type_dict, n_splits=1000, n_repeats=5, label="feature_type_importance")

# %%
def get_mean_ci(df, alpha=0.05, df_dof=4):
    mean = df.mean()
    ci = stats.t.interval(alpha, df_dof, loc=mean, scale=stats.sem(df))
    ci_lower, ci_upper = ci
    return pd.Series({"mean": mean, "ci_lower": ci_lower, "ci_upper": ci_upper})

def get_mean_ci_nb(df, alpha=0.05, VIF=(1 + 0.2/0.8)):
    R = len(df)
    mean = df.mean()
    var = df.var(ddof=1)  # unbiased variance
    inflation = 1 + VIF
    se = np.sqrt(var * inflation / R)
    dof = R - 1
    t_val = stats.t.ppf(1 - alpha/2, dof)
    ci_lower = mean - t_val * se
    ci_upper = mean + t_val * se

    return pd.Series({"mean": mean, "ci_lower": ci_lower, "ci_upper": ci_upper})

def get_ci_nb(df, alpha=0.05, VIF=(1 + 0.2/0.8)):
    res = get_mean_ci_nb(df, alpha, VIF)
    ci_lower = res['ci_lower']
    ci_upper = res['ci_upper']
    return (ci_lower, ci_upper)

# %%

# plot these as point plots with error bars
def plot_permutation_importance(pi_df, title):
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=pi_df.reset_index(),
        x='feature',
        y='permutation_importance',
        capsize=0.1,
        errorbar=get_ci_nb,
        markers='o',
        color='blue',
        markersize=2,
        linestyle='none',
        linewidth=1
    )
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Permutation Importance (Mean ± CI)')
    plt.tight_layout()
    plt.show()

top_15_features_oasis = oasis_pi.groupby("feature").mean()["permutation_importance"].sort_values(ascending=False).head(15).index.tolist()
top_15_features_adni = adni_pi.groupby("feature").mean()["permutation_importance"].sort_values(ascending=False).head(15).index.tolist()
top_15_features = list(set(top_15_features_oasis + top_15_features_adni))

plot_permutation_importance(oasis_pi.loc[oasis_pi["feature"].isin(top_15_features)], "OASIS-3: Top 15 Features Permutation Importance")
plot_permutation_importance(adni_pi.loc[adni_pi["feature"].isin(top_15_features)], "ADNI: Top 15 Features Permutation Importance")

# for across datasets:

# %%

def plot_across_permutation_importance(pi_df, title, ax=None):
    pi_df = pi_df.sort_values("permutation_importance", ascending=False).head(15).copy()
    pi_df['feature'] = pd.Categorical(pi_df['feature'], categories=pi_df['feature'], ordered=True)
    pi_df["ymin"] = pi_df["permutation_importance"] - pi_df["permutation_importance_std"]
    pi_df["ymax"] = pi_df["permutation_importance"] + pi_df["permutation_importance_std"]

    p = so.Plot(data=pi_df, y='feature', x='permutation_importance', xmin='ymin', xmax='ymax')
    p = p.add(so.Dot())
    p = p.add(so.Range())
    p = p.label(title=title, y="Feature", x="Permutation Importance ($R^2$ decrease)")
    
    # Set font to Arial for all text elements
    p = p.theme({
        "axes.labelweight": "normal", 
        "axes.labelsize": 10,
        "axes.titlesize": 12,
    })
    
    p.on(ax).plot()

# Create a figure with two subplots arranged vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 12), constrained_layout=True, sharex=True)


feature_dict = {
    'session_n': "Number of previous visits",
    'diag': "Diagnosis",
    'demo_sex': "Sex",
    'demo_age': "Age",
    'demo_education': "Education",
    'diabetes_DIABETES': "Diabetes",
    'hypercho_HYPERCHO': "Hypercholesterolemia",
    'cvasc_HYPERTEN': "Hypertension",
    'cvasc_CBSTROKE': "Stroke",
    'cvasc_CBTIA': "Transient Ischemic Attack",
    'cvasc_CVHATT': "Heart Attack",
    'cvasc_CVAFIB': "Atrial Fibrillation",
    'cvasc_CVANGIO': "Angiography",
    'cvasc_CVOTHR': "Other Cardiovascular",
    'cdr_commun': "CDR (Communication)",
    'cdr_homehobb': "CDR (Home and Hobbies)",
    'cdr_judgment': "CDR (Judgment)",
    'cdr_memory': "CDR (Memory)",
    'cdr_orient': "CDR (Orientation)",
    'cdr_perscare': "CDR (Personal Care)",
    'cdr_sob': "CDR-SOB",
    'cdr_cdrGlobal': "CDR (Global)",
    'mmse_mmse': "MMSE (Total)",
    'gds_gdsSum': "GDS (Sum)",
    'faq_BILLS': "FAQ (Bills)",
    'faq_TAXES': "FAQ (Taxes)",
    'faq_SHOPPING': "FAQ (Shopping)",
    'faq_GAMES': "FAQ (Games)",
    'faq_STOVE': "FAQ (Stove)",
    'faq_MEALPREP': "FAQ (Meal Preparation)",
    'faq_EVENTS': "FAQ (Events)",
    'faq_PAYATTN': "FAQ (Pay Attention)",
    'faq_REMDATES': "FAQ (Remember Dates)",
    'faq_TRAVEL': "FAQ (Travel)",
    'faq_faqSum': "FAQ (Total)",
    'npiq_npiqPresSum': "NPI-Q (Presence Sum)",
    'npiq_npiqSevSum': "NPI-Q (Severity Sum)",
    'apoe_e2count': "APOE ε2 Count",
    'apoe_e3count': "APOE ε3 Count",
    'apoe_e4count': "APOE ε4 Count",
    'WMSlm_LOGIMEM': "WMS (Logical Memory)",
    'WMSlm_MEMUNITS': "WMS (Memory Units)",
    'WMSlm_MEMTIME': "WMS (Memory Time)",
    'WF_ANIMALS': "Word Fluency (Animals)",
    'TMTA_TRAILA': "Trail Making Test A",
    'TMTB_TRAILB': "Trail Making Test B",
    'TMTB_TRAILBnorm': "Trail Making Test B (normalized)",
    'BOSTON_BOSTON': "Boston Naming Test (Total)",
    'fs__globalVolume__3rd-Ventricle': "Volume of 3rd Ventricle",
    'fs__globalVolume__4th-Ventricle': "Volume of 4th Ventricle",
    'fs__globalVolume__CC_Anterior': "Volume of Corpus Callosum (Anterior)",
    'fs__globalVolume__CC_Central': "Volume of Corpus Callosum (Central)",
    'fs__globalVolume__CC_Mid_Anterior': "Volume of Corpus Callosum (Mid Anterior)",
    'fs__globalVolume__CC_Mid_Posterior': "Volume of Corpus Callosum (Mid Posterior)",
    'fs__globalVolume__CC_Posterior': "Volume of Corpus Callosum (Posterior)",
    'fs__globalVolume__Left-Cerebellum-Cortex': "Volume of Cerebellum Cortex (Left)",
    'fs__globalVolume__Left-Cerebellum-White-Matter': "Volume of Cerebellum White Matter (Left)",
    'fs__globalVolume__Left-Lateral-Ventricle': "Volume of Lateral Ventricle (Left)",
    'fs__globalVolume__Right-Cerebellum-Cortex': "Volume of Cerebellum Cortex (Right)",
    'fs__globalVolume__Right-Cerebellum-White-Matter': "Volume of Cerebellum White Matter (Right)",
    'fs__globalVolume__Right-Lateral-Ventricle': "Volume of Lateral Ventricle (Right)",
    'fs__globalVolume__SubCortGrayVol': "Volume of Subcortical Gray Matter",
    'fs__globalVolume__TotalGrayVol': "Volume of Total Gray Matter",
    'fs__globalVolume__lhCerebralWhiteMatterVol': "Volume of Cerebral White Matter (Left)",
    'fs__globalVolume__lhCortexVol': "Volume of Cortex (Left)",
    'fs__globalVolume__lh_MeanThickness_thickness': "Mean Cortical Thickness (Left)",
    'fs__globalVolume__rhCerebralWhiteMatterVol': "Volume of Cerebral White Matter (Right)",
    'fs__globalVolume__rhCortexVol': "Volume of Cortex (Right)",
    'fs__globalVolume__rh_MeanThickness_thickness': "Mean Cortical Thickness (Right)",
    'fs__subcortVolume__Left-Accumbens-area': "Volume of Accumbens (Left)",
    'fs__subcortVolume__Left-Amygdala': "Volume of Amygdala (Left)",
    'fs__subcortVolume__Left-Caudate': "Volume of Caudate (Left)",
    'fs__subcortVolume__Left-Hippocampus': "Volume of Hippocampus (Left)",
    'fs__subcortVolume__Left-Pallidum': "Volume of Pallidum (Left)",
    'fs__subcortVolume__Left-Putamen': "Volume of Putamen (Left)",
    'fs__subcortVolume__Left-Thalamus-Proper': "Volume of Thalamus (Left)",
    'fs__subcortVolume__Right-Accumbens-area': "Volume of Accumbens (Right)",
    'fs__subcortVolume__Right-Amygdala': "Volume of Amygdala (Right)",
    'fs__subcortVolume__Right-Caudate': "Volume of Caudate (Right)",
    'fs__subcortVolume__Right-Hippocampus': "Volume of Hippocampus (Right)",
    'fs__subcortVolume__Right-Pallidum': "Volume of Pallidum (Right)",
    'fs__subcortVolume__Right-Putamen': "Volume of Putamen (Right)",
    'fs__subcortVolume__Right-Thalamus-Proper': "Volume of Thalamus (Right)",
}

oasis2adni_pi['feature_names'] = oasis2adni_pi['feature'].map(feature_dict)
adni2oasis_pi['feature_names'] = adni2oasis_pi['feature'].map(feature_dict)

# Plot for OASIS to ADNI
plot_across_permutation_importance(oasis2adni_pi.rename(columns={"feature_names": "feature", "feature": "feature_i"}), "A) OASIS-3 → ADNI: Feature Importance", ax=axes[0])

# Plot for ADNI to OASIS
plot_across_permutation_importance(adni2oasis_pi.rename(columns={"feature_names": "feature", "feature": "feature_i"}), "B) ADNI → OASIS-3: Feature Importance", ax=axes[1])


plt.savefig("cross_dataset_feature_importance.png", dpi=300, bbox_inches="tight")

top15_features_oasis2adni = oasis2adni_pi.sort_values("permutation_importance", ascending=False).head(15)["feature"].to_list()
top15_features_adni2oasis = adni2oasis_pi.sort_values("permutation_importance", ascending=False).head(15)["feature"].to_list()

# %%

top_15_features_oasis_filtered = oasis_filtered_pi.groupby("feature").mean()["permutation_importance"].sort_values(ascending=False).head(15).index.tolist()
top_15_features_adni_filtered = adni_filtered_pi.groupby("feature").mean()["permutation_importance"].sort_values(ascending=False).head(15).index.tolist()
top_15_features_filtered = list(set(top_15_features_oasis_filtered + top_15_features_adni_filtered))
top_15_features_intersection = list(set(top_15_features_oasis_filtered) & set(top_15_features_adni_filtered))

adni_overlap = set(top15_features_adni2oasis) & set(top_15_features_adni_filtered)
adni_difference1 = set(top_15_features_adni_filtered) - set(top15_features_adni2oasis)
adni_difference2 = set(top15_features_adni2oasis) - set(top_15_features_adni_filtered)

oasis_overlap = set(top15_features_oasis2adni) & set(top_15_features_oasis_filtered)
oasis_difference1 = set(top_15_features_oasis_filtered) - set(top15_features_oasis2adni)
oasis_difference2 = set(top15_features_oasis2adni) - set(top_15_features_oasis_filtered)

plot_permutation_importance(oasis_filtered_pi.loc[oasis_filtered_pi["feature"].isin(top_15_features)], "OASIS-3 (Filtered): Top 15 Features Permutation Importance")
plot_permutation_importance(adni_filtered_pi.loc[adni_filtered_pi["feature"].isin(top_15_features)], "ADNI (Filtered): Top 15 Features Permutation Importance")

plot_permutation_importance(oasis_filtered_ci.reset_index("coalition").rename(columns={"coalition": "feature"}), "OASIS-3 (Filtered): Permutation Importance by Coalition")
plot_permutation_importance(adni_filtered_ci.reset_index("coalition").rename(columns={"coalition": "feature"}), "ADNI (Filtered): Permutation Importance by Coalition")

# %%
def round(x, n_figs):
    x = float(x)
    power = 10 ** np.floor(np.log10(abs(x)))
    rounded = np.round(x / power, n_figs - 1) * power
    rounded = float(rounded)

    # Use 'f' format to preserve trailing zeros if needed
    digits_after_decimal = max(n_figs - int(np.floor(np.log10(abs(rounded)))) - 1, 0)
    format_str = f"{{:.{digits_after_decimal}f}}"
    result = format_str.format(rounded)
    return result

# oasis_filtered_pi_top15 = oasis_filtered_pi.loc[oasis_filtered_pi["feature"].isin(top_15_features_oasis_filtered)].copy()
oasis_filtered_pi_top15 = oasis_filtered_pi.copy()
oasis_filtered_pi_top15 = oasis_filtered_pi_top15.groupby("feature")["permutation_importance"].apply(get_mean_ci_nb).unstack()
oasis_filtered_pi_top15 = oasis_filtered_pi_top15.sort_values("mean", ascending=False)

oasis_filtered_pi_top15.loc[top_15_features_oasis_filtered].applymap(round, n_figs=2).apply(lambda x: f"{x['mean']} ({x['ci_lower']}, {x['ci_upper']})", axis=1)
oasis_filtered_pi_top15.applymap(round, n_figs=2).apply(lambda x: f"{x['mean']} ({x['ci_lower']}, {x['ci_upper']})", axis=1).to_csv("OASIS_importance.csv")


adni_filtered_pi_top15 = adni_filtered_pi.copy()
adni_filtered_pi_top15 = adni_filtered_pi_top15.groupby("feature")["permutation_importance"].apply(get_mean_ci_nb).unstack()
adni_filtered_pi_top15 = adni_filtered_pi_top15.sort_values("mean", ascending=False)


adni_filtered_pi_top15.loc[top_15_features_adni_filtered].applymap(round, n_figs=2).apply(lambda x: f"{x['mean']} ({x['ci_lower']}, {x['ci_upper']})", axis=1)
adni_filtered_pi_top15.applymap(round, n_figs=2).apply(lambda x: f"{x['mean']} ({x['ci_lower']}, {x['ci_upper']})", axis=1).to_csv("ADNI_importance.csv")


# what is unique to top_15_features_oasis_filtered?
set(top_15_features_oasis_filtered) - set(top_15_features_adni_filtered)
# what is unique to top_15_features_adni_filtered?
set(top_15_features_adni_filtered) - set(top_15_features_oasis_filtered)

oasis_filtered_ci_table = oasis_filtered_type_ci.groupby("coalition")["permutation_importance"].apply(get_mean_ci_nb).unstack()
oasis_filtered_ci_table = oasis_filtered_ci_table.sort_values("mean", ascending=False)
oasis_filtered_ci_table.applymap(round, n_figs=2).apply(lambda x: f"{x['mean']} ({x['ci_lower']}, {x['ci_upper']})", axis=1)

adni_filtered_ci_table = adni_filtered_type_ci.groupby("coalition")["permutation_importance"].apply(get_mean_ci_nb).unstack()
adni_filtered_ci_table = adni_filtered_ci_table.sort_values("mean", ascending=False)
adni_filtered_ci_table.applymap(round, n_figs=2).apply(lambda x: f"{x['mean']} ({x['ci_lower']}, {x['ci_upper']})", axis=1)


# %%
# ## 3. Across-Dataset Predictions
# ### 3.1 Cross-Dataset Prediction Functions and Execution

def predict_across_datasets(X_train, y_train, X_test, y_test, root_dir):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if root_dir / "pipeline.joblib" in root_dir.iterdir() and root_dir / "predictions.csv" in root_dir.iterdir():
        print(f"Skipping {root_dir}, already processed.")
        return
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    train_idx = np.arange(len(X_train))
    test_idx = np.arange(len(X_train), len(X))
    # Train a model on the combined data
    fit_pipeline(X, y, train_idx, test_idx, root_dir)

# train on oasis, test on adni
across_data_combinations = [
    (oasis_y, oasis_X_clin_filtered, adni_y, adni_X_clin_filtered, "OASIS_to_ADNI_clin"),
    (oasis_y, oasis_X_fs_filtered, adni_y, adni_X_fs_filtered, "OASIS_to_ADNI_struct"),
    (oasis_y, oasis_X_clin_fs_filtered, adni_y, adni_X_clin_fs_filtered, "OASIS_to_ADNI_clin_struct"),
    (adni_y, adni_X_clin_filtered, oasis_y, oasis_X_clin_filtered, "ADNI_to_OASIS_clin"),
    (adni_y, adni_X_fs_filtered, oasis_y, oasis_X_fs_filtered, "ADNI_to_OASIS_struct"),
    (adni_y, adni_X_clin_fs_filtered, oasis_y, oasis_X_clin_fs_filtered, "ADNI_to_OASIS_clin_struct"),
    (oasis_y, oasis_X_clin_fs_filtered[top_15_features_oasis_filtered], adni_y, adni_X_clin_fs_filtered[top_15_features_oasis_filtered], "OASIS_to_ADNI_clin_struct_top15"),
    (adni_y, adni_X_clin_fs_filtered[top_15_features_adni_filtered], oasis_y, oasis_X_clin_fs_filtered[top_15_features_adni_filtered], "ADNI_to_OASIS_clin_struct_top15"),
]

for y_train, X_train, y_test, X_test, root_dir in across_data_combinations:
    print(f"Processing {root_dir}...")
    predict_across_datasets(X_train, y_train, X_test, y_test, root_dir)

# ### 3.2 Cross-Dataset Results Aggregation

def get_across_data_results(root):
    results = pd.read_csv(f"{root}/results.csv", index_col=0)
    summary = results.groupby(level=0).agg(['mean', 'median'])
    return summary


oasis2adni_results = [get_across_data_results(f"OASIS_to_ADNI_{c}") for c in ["clin", "struct", "clin_struct", "clin_struct_top15"]]
oasis2adni_results = pd.concat(oasis2adni_results, keys=["clin", "struct", "clin_struct", "clin_struct_top15"], names=["modality", "target"])
adni2oasis_results = [get_across_data_results(f"ADNI_to_OASIS_{c}") for c in ["clin", "struct", "clin_struct", "clin_struct_top15"]]
adni2oasis_results = pd.concat(adni2oasis_results, keys=["clin", "struct", "clin_struct", "clin_struct_top15"], names=["modality", "target"])

across_results = pd.concat([oasis2adni_results, adni2oasis_results], axis=0, 
                           keys=["OASIS_to_ADNI", "ADNI_to_OASIS"], names=["dataset", "modality", "target"])

# %%
# ## 4. Statistical Model Comparisons
# ### 4.1 Statistical Testing Framework and Model Comparisons

def r2_test(df, subset1, subset2=None, metric='r2', null_value=0):
    dataset1, modality1, target1 = subset1

    test_func = wilcoxon
    df1 = df.loc[(dataset1, modality1, slice(None), target1)]
    if subset2 is not None:
        dataset2, modality2, target2 = subset2
        df2 = df.loc[(dataset2, modality2, slice(None), target2)]
        res = test_func(df1[metric], df2[metric], alternative='two-sided')
        proportion = (df1[metric] > df2[metric]).mean()
    else:
        res = test_func(df1[metric] - null_value, alternative='two-sided')
        proportion = (df1[metric] > null_value).mean()
    statistic, p_value = res.statistic, res.pvalue
    return [statistic, p_value, proportion, metric]

model_comparisons = []

# Within dataset comparisons
for metric in ["r2", "mse", "mae"]:
    for target in ["sob", "mmse"]:
        for modality1, modality2 in [("clin", "struct"), ("struct", "clin_struct"), ("clin_struct", "clin")]:
            for dataset in ["OASIS_filtered", "ADNI_filtered"]:
                comparison_label = f"{dataset}_{modality1}_vs_{modality2}_{target}"
                comparison = r2_test(results, (dataset, modality1, target), (dataset, modality2, target), metric=metric)
                comparison = pd.Series(comparison, index=["statistic", "p_value", "proportion", "metric"], name=comparison_label)
                model_comparisons.append(comparison)

    # Between dataset comparisons
    for target in ["sob", "mmse"]:
        for modality in ["clin", "struct", "clin_struct"]:
            comparison_label = f"OASIS_filtered_{modality}_vs_ADNI_filtered_{modality}_{target}"
            comparison = r2_test(results, ("OASIS_filtered", modality, target), ("ADNI_filtered", modality, target), metric=metric)
            comparison = pd.Series(comparison, index=["statistic", "p_value", "proportion", "metric"], name=comparison_label)
            model_comparisons.append(comparison)

    # Within vs across comparisons
    for target in ["sob", "mmse"]:
        for modality in ["clin", "struct", "clin_struct"]:
            for dataset_across in ["OASIS_to_ADNI", "ADNI_to_OASIS"]:
                for dataset_within in ["OASIS_filtered", "ADNI_filtered"]:
                    comparison_label = f"{dataset_across}_{modality}_vs_{dataset_within}_{modality}_{target}"
                    null = across_results.loc[dataset_across, modality, target].loc["r2", "mean"]
                    comparison = r2_test(results, (dataset_within, modality, target), None, null_value=null, metric=metric)
                    comparison = pd.Series(comparison, index=["statistic", "p_value", "proportion", "metric"], name=comparison_label)
                    model_comparisons.append(comparison)

model_comparisons = pd.DataFrame(model_comparisons).rename_axis("comparison")

model_comparisons.round(3).loc["OASIS_filtered_clin_vs_struct_sob"]
model_comparisons.round(3).loc["OASIS_filtered_clin_vs_struct_mmse"]
model_comparisons.loc["OASIS_filtered_clin_struct_vs_clin_sob"]
model_comparisons.loc["OASIS_filtered_clin_struct_vs_clin_mmse"]
model_comparisons.loc["OASIS_filtered_struct_vs_clin_struct_sob"]
model_comparisons.loc["OASIS_filtered_struct_vs_clin_struct_mmse"]
model_comparisons.loc["OASIS_filtered_clin_vs_struct_sob"]
model_comparisons.loc["OASIS_filtered_clin_vs_struct_mmse"]

model_comparisons.loc["ADNI_filtered_clin_struct_vs_clin_sob"]
model_comparisons.loc["ADNI_filtered_clin_struct_vs_clin_mmse"]
model_comparisons.loc["ADNI_filtered_struct_vs_clin_struct_sob"]
model_comparisons.loc["ADNI_filtered_struct_vs_clin_struct_mmse"]
model_comparisons.loc["ADNI_filtered_clin_vs_struct_sob"]
model_comparisons.loc["ADNI_filtered_clin_vs_struct_mmse"]

model_comparisons.loc["OASIS_filtered_clin_struct_vs_ADNI_filtered_clin_struct_sob"]
model_comparisons.loc["OASIS_filtered_clin_struct_vs_ADNI_filtered_clin_struct_mmse"]

model_comparisons.loc["OASIS_filtered_clin_vs_ADNI_filtered_clin_sob"]
model_comparisons.loc["OASIS_filtered_clin_vs_ADNI_filtered_clin_mmse"]

model_comparisons.loc["OASIS_filtered_struct_vs_ADNI_filtered_struct_sob"]
model_comparisons.loc["OASIS_filtered_struct_vs_ADNI_filtered_struct_mmse"]

model_comparisons.loc[
    ["OASIS_to_ADNI_clin_struct_vs_OASIS_filtered_clin_struct_sob",
    "OASIS_to_ADNI_clin_struct_vs_OASIS_filtered_clin_struct_mmse",
    "OASIS_to_ADNI_clin_vs_OASIS_filtered_clin_sob",
    "OASIS_to_ADNI_clin_vs_OASIS_filtered_clin_mmse",
    "OASIS_to_ADNI_struct_vs_OASIS_filtered_struct_sob",
    "OASIS_to_ADNI_struct_vs_OASIS_filtered_struct_mmse",
    "ADNI_to_OASIS_clin_struct_vs_OASIS_filtered_clin_struct_sob",
    "ADNI_to_OASIS_clin_struct_vs_OASIS_filtered_clin_struct_mmse",
    "ADNI_to_OASIS_clin_vs_OASIS_filtered_clin_sob",
    "ADNI_to_OASIS_clin_vs_OASIS_filtered_clin_mmse",
    "ADNI_to_OASIS_struct_vs_OASIS_filtered_struct_sob",
    "ADNI_to_OASIS_struct_vs_OASIS_filtered_struct_mmse"]
    ].query("metric == 'r2'")

model_comparisons.loc[
    ["OASIS_to_ADNI_clin_struct_vs_ADNI_filtered_clin_struct_sob",
    "OASIS_to_ADNI_clin_struct_vs_ADNI_filtered_clin_struct_mmse",
    "OASIS_to_ADNI_clin_vs_ADNI_filtered_clin_sob",
    "OASIS_to_ADNI_clin_vs_ADNI_filtered_clin_mmse",
    "OASIS_to_ADNI_struct_vs_ADNI_filtered_struct_sob",
    "OASIS_to_ADNI_struct_vs_ADNI_filtered_struct_mmse",
    "ADNI_to_OASIS_clin_struct_vs_ADNI_filtered_clin_struct_sob",
    "ADNI_to_OASIS_clin_struct_vs_ADNI_filtered_clin_struct_mmse",
    "ADNI_to_OASIS_clin_vs_ADNI_filtered_clin_sob",
    "ADNI_to_OASIS_clin_vs_ADNI_filtered_clin_mmse",
    "ADNI_to_OASIS_struct_vs_ADNI_filtered_struct_sob",
    "ADNI_to_OASIS_struct_vs_ADNI_filtered_struct_mmse"]
    ].query("metric == 'r2'")

target="mmse"
(results
    .groupby(level=["dataset", "modality", "target"])
    .agg(['mean', 'median'])
    .xs("median", level=1, axis=1)
    .xs(target, level="target", axis=0)
    .loc[["OASIS_filtered", "ADNI_filtered"], ["clin", "struct", "clin_struct"], :]
    .round(2)
    [["r2", "mse", "mae"]])

across_results.round(2).xs(target, level="target", axis=0).xs("median", level=1, axis=1)[["r2", "mse", "mae"]]

r2_comparisons = model_comparisons.set_index("metric", append=True)
r2_comparisons = r2_comparisons.xs("r2", level="metric", axis=0)

##
r2_comparisons.loc[f"OASIS_filtered_clin_vs_struct_{target}"]
r2_comparisons.loc[f"OASIS_filtered_clin_struct_vs_clin_{target}"]
r2_comparisons.loc[f"OASIS_filtered_struct_vs_clin_struct_{target}"]
##
r2_comparisons.loc[f"ADNI_filtered_clin_vs_struct_{target}"]
r2_comparisons.loc[f"ADNI_filtered_clin_struct_vs_clin_{target}"]
r2_comparisons.loc[f"ADNI_filtered_struct_vs_clin_struct_{target}"]
##
r2_comparisons.loc[f"OASIS_filtered_clin_vs_ADNI_filtered_clin_{target}"]
r2_comparisons.loc[f"OASIS_filtered_struct_vs_ADNI_filtered_struct_{target}"]
r2_comparisons.loc[f"OASIS_filtered_clin_struct_vs_ADNI_filtered_clin_struct_{target}"]
##
r2_comparisons.loc[f"OASIS_to_ADNI_clin_vs_OASIS_filtered_clin_{target}"]
r2_comparisons.loc[f"OASIS_to_ADNI_struct_vs_OASIS_filtered_struct_{target}"]
r2_comparisons.loc[f"OASIS_to_ADNI_clin_struct_vs_OASIS_filtered_clin_struct_{target}"]
##
r2_comparisons.loc[f"OASIS_to_ADNI_clin_vs_ADNI_filtered_clin_{target}"]
r2_comparisons.loc[f"OASIS_to_ADNI_struct_vs_ADNI_filtered_struct_{target}"]
r2_comparisons.loc[f"OASIS_to_ADNI_clin_struct_vs_ADNI_filtered_clin_struct_{target}"]
##
r2_comparisons.loc[f"ADNI_to_OASIS_clin_vs_OASIS_filtered_clin_{target}"]
r2_comparisons.loc[f"ADNI_to_OASIS_struct_vs_OASIS_filtered_struct_{target}"]
r2_comparisons.loc[f"ADNI_to_OASIS_clin_struct_vs_OASIS_filtered_clin_struct_{target}"]
##
r2_comparisons.loc[f"ADNI_to_OASIS_clin_vs_ADNI_filtered_clin_{target}"]
r2_comparisons.loc[f"ADNI_to_OASIS_struct_vs_ADNI_filtered_struct_{target}"]
r2_comparisons.loc[f"ADNI_to_OASIS_clin_struct_vs_ADNI_filtered_clin_struct_{target}"]
##


# %%
# ### 4.2 Absolute Error Comparisons Between Models

def get_abserr(root_dir):
    abserr = pd.read_csv(f"{root_dir}/predictions.csv", index_col=0)
    abserr['sob_abs_error'] = (abserr['sob_slope'] - abserr['sob_pred']).abs()
    abserr['mmse_abs_error'] = (abserr['mmse_slope'] - abserr['mmse_pred']).abs()
    return abserr

def get_abserr_comparison(root_dir1, root_dir2, paired=False):
    abserr1 = get_abserr(root_dir1)
    abserr2 = get_abserr(root_dir2)
    
    if paired:
        test_fun = wilcoxon
        # check that abserr1 and abserr2 have the same index
        if not abserr1.index.equals(abserr2.index):
            raise ValueError("The indices of abserr1 and abserr2 must match for paired tests.")
        mmse_proportion = (abserr1['mmse_abs_error'] > abserr2['mmse_abs_error']).mean()
        sob_proportion = (abserr1['sob_abs_error'] > abserr2['sob_abs_error']).mean()
    else:
        test_fun = mannwhitneyu
        mmse_proportion = np.nan
        sob_proportion = np.nan


    mmse_test = test_fun(abserr1['mmse_abs_error'], abserr2['mmse_abs_error'])
    sob_test = test_fun(abserr1['sob_abs_error'], abserr2['sob_abs_error'])


    res = {
        'mmse_statistic': mmse_test.statistic,
        'mmse_pvalue': mmse_test.pvalue,
        'mmse_proportion': mmse_proportion,
        'sob_statistic': sob_test.statistic,
        'sob_pvalue': sob_test.pvalue,
        'sob_proportion': sob_proportion,
        'paired': paired,
        f"{root_dir1}_mmse_mean": abserr1['mmse_abs_error'].mean(),
        f"{root_dir2}_mmse_mean": abserr2['mmse_abs_error'].mean(),
        f"{root_dir1}_sob_mean": abserr1['sob_abs_error'].mean(),
        f"{root_dir2}_sob_mean": abserr2['sob_abs_error'].mean()
    }

    res = pd.Series(res)

    return res

# ### 6.6 OASIS-3 → ADNI: Combined vs. Top 15 (Absolute errors)
get_abserr_comparison("OASIS_to_ADNI_clin_struct", "OASIS_to_ADNI_clin_struct_top15", paired=True)
# ### 6.7 ADNI → OASIS-3: Combined vs. Top 15
get_abserr_comparison("ADNI_to_OASIS_clin_struct", "ADNI_to_OASIS_clin_struct_top15", paired=True)
# ### 6.8 OASIS-3 → ADNI vs. ADNI → OASIS-3: Combined
get_abserr_comparison("OASIS_to_ADNI_clin_struct", "ADNI_to_OASIS_clin_struct", paired=False)
# ### 6.9 OASIS-3 → ADNI vs. ADNI → OASIS-3: Top 15
get_abserr_comparison("OASIS_to_ADNI_clin_struct_top15", "ADNI_to_OASIS_clin_struct_top15", paired=False)

across_results.loc[:, "clin_struct", :]


# %%
# ## 5. Subgroup Analysis
# ### 5.1 Preparation of Data for Subgroup Comparisons

subset_adni = pd.concat([adni_X_clin_filtered, get_abserr("OASIS_to_ADNI_clin_struct")], axis=1)
subset_oasis = pd.concat([oasis_X_clin_fs_filtered, get_abserr("ADNI_to_OASIS_clin_struct")], axis=1)
subset_adni_top15 = pd.concat([adni_X_clin_filtered, get_abserr("OASIS_to_ADNI_clin_struct_top15")], axis=1)
subset_oasis_top15 = pd.concat([oasis_X_clin_fs_filtered, get_abserr("ADNI_to_OASIS_clin_struct_top15")], axis=1)

# ### 5.2 Subgroup Comparison Functions and Analysis

# accessor function to make subset comparisons
def get_subset(variable, value, df, comparison_type='equal'):
    if comparison_type == 'equal':
        return df[df[variable] == value]
    elif comparison_type == 'greater':
        return df[df[variable] > value]
    elif comparison_type == 'less':
        return df[df[variable] < value]
    elif comparison_type == 'different':
        return df[df[variable] != value]
    elif comparison_type == 'greater_equal':
        return df[df[variable] >= value]
    else:
        raise ValueError("Invalid comparison type.")

def get_subset_comparison(df, variable, comparison_1, comparison_2, label=None):
    (value1, comparison_type1) = comparison_1
    (value2, comparison_type2) = comparison_2
    subset1 = get_subset(variable, value1, df, comparison_type1)
    subset2 = get_subset(variable, value2, df, comparison_type2)

    sob_test = mannwhitneyu(subset1['sob_abs_error'], subset2['sob_abs_error'])
    mmse_test = mannwhitneyu(subset1['mmse_abs_error'], subset2['mmse_abs_error'])

    sob_mae1 = subset1['sob_abs_error'].mean()
    mmse_mae1 = subset1['mmse_abs_error'].mean()
    sob_mae2 = subset2['sob_abs_error'].mean()
    mmse_mae2 = subset2['mmse_abs_error'].mean()

    return pd.Series({
        'sob_statistic': sob_test.statistic,
        'sob_pvalue': sob_test.pvalue,
        'mmse_statistic': mmse_test.statistic,
        'mmse_pvalue': mmse_test.pvalue,
        'sob_mae1': sob_mae1,
        'mmse_mae1': mmse_mae1,
        'sob_mae2': sob_mae2,
        'mmse_mae2': mmse_mae2
    }, name=label)

def subgroup_analysis(subset):
    res = []
    # ### 8.1 Subgroups Segregated by Diagnosis
    # ##### HC (0) vs. MCI (1) and AD (2)
    res.append(get_subset_comparison(subset, 'diag', (0, 'equal'), (0, 'different'), label='HC vs. MCI and AD'))
    # #### MCI vs. HC and AD
    res.append(get_subset_comparison(subset, 'diag', (1, 'equal'), (1, 'different'), label='MCI vs. HC and AD'))
    # #### AD vs. HC and MCI
    res.append(get_subset_comparison(subset, 'diag', (2, 'equal'), (2, 'different'), label='AD vs. HC and MCI'))
    # #### HC vs. MCI
    res.append(get_subset_comparison(subset, 'diag', (0, 'equal'), (1, 'equal'), label='HC vs. MCI'))
    # #### HC vs. AD
    res.append(get_subset_comparison(subset, 'diag', (0, 'equal'), (2, 'equal'), label='HC vs. AD'))
    # #### MCI vs. AD
    res.append(get_subset_comparison(subset, 'diag', (1, 'equal'), (2, 'equal'), label='MCI vs. AD'))
    # ### 8.2 Subgroups Segregated by Sex
    # #### Female (0) vs. Male (1)
    res.append(get_subset_comparison(subset, 'demo_sex', (0, 'equal'), (1, 'equal'), label='Female vs. Male'))
    # ### 8.3 Subgroups Segregated by APOE E4
    # #### APOE E4: 0 vs. 1 and 2
    res.append(get_subset_comparison(subset, 'apoe_e4count', (0, 'equal'), (0, 'greater'), label='APOE E4: 0 vs. 1 and 2'))
    # #### APOE E4: 1 vs. 0 and 2
    res.append(get_subset_comparison(subset, 'apoe_e4count', (1, 'equal'), (1, 'different'), label='APOE E4: 1 vs. 0 and 2'))
    # #### APOE E4: 2 vs. 0 and 1
    res.append(get_subset_comparison(subset, 'apoe_e4count', (2, 'equal'), (2, 'less'), label='APOE E4: 2 vs. 0 and 1'))
    # #### APOE E4: 0 vs. 1
    res.append(get_subset_comparison(subset, 'apoe_e4count', (0, 'equal'), (1, 'equal'), label='APOE E4: 0 vs. 1'))
    # #### APOE E4: 0 vs. 2
    res.append(get_subset_comparison(subset, 'apoe_e4count', (0, 'equal'), (2, 'equal'), label='APOE E4: 0 vs. 2'))
    # #### APOE E4: 1 vs. 2
    res.append(get_subset_comparison(subset, 'apoe_e4count', (1, 'equal'), (2, 'equal'), label='APOE E4: 1 vs. 2'))
    # ### 8.4 Subgroups Segregated by Age
    subset['demo_age_group'] = pd.cut(subset['demo_age'], bins=[-np.inf, 65, 70, 75, 80, np.inf], labels=['<65', '65-70', '70-75', '75-80', '>80'])
    # #### AGE: <65 vs. >=65 (65-70 and 70-75 and 75-80 and and >80)
    res.append(get_subset_comparison(subset, 'demo_age_group', ('65-70', 'less'), ('65-70', 'greater_equal'), label='Age: <65 vs. >=65'))
    # #### AGE: 70-75 vs. <70 and >= 75
    res.append(get_subset_comparison(subset, 'demo_age_group', ('70-75', 'less'), ('70-75', 'greater_equal'), label='Age: <70 vs. >= 70'))
    # #### AGE: 75-80 vs. <75 and >= 80
    res.append(get_subset_comparison(subset, 'demo_age_group', ('75-80', 'less'), ('75-80', 'greater_equal'), label='Age: <75 vs. >= 80'))
    # #### AGE: >= 80 vs. <80
    res.append(get_subset_comparison(subset, 'demo_age_group', ('>80', 'less'), ('>80', 'greater_equal'), label='Age: <80 vs. >= 80'))

    res = pd.DataFrame(res)
    
    return res

res_subset_adni = subgroup_analysis(subset_adni)[["sob_pvalue", "sob_mae1", "sob_mae2", "mmse_pvalue", "mmse_mae1", "mmse_mae2"]]
res_subset_oasis = subgroup_analysis(subset_oasis)[["sob_pvalue", "sob_mae1", "sob_mae2", "mmse_pvalue", "mmse_mae1", "mmse_mae2"]]

# ### 5.3 Results Formatting and Export

def apa_format(column):
    """
    Format a dataframe column in APA style.
    """
    # first check if the column name contains 'pvalue'
    if 'pvalue' in column.name:
        # format p-values
        formatted = column.apply(lambda x: f"{x:.3f}" if x > 0.001 else "< .001")
    else:
        formatted = column.apply(lambda x: round(x, 2) if isinstance(x, float) else str(x))
    return formatted

# Format the results in APA style
res_subset_adni.apply(apa_format).to_csv("ADNI_subgroup_analysis.csv")
res_subset_oasis.apply(apa_format).to_csv("OASIS_subgroup_analysis.csv")

# %%
# ## 6. Post-Hoc Matched Samples Analysis
# ### 6.1 Quantile-Based Sample Matching

def get_matched_samples(source_y, target_y, quantiles=np.arange(0, 1.1, 0.1)):
    """
    Get matched samples based on quantiles of the source dataset.
    """
    quant_source_sob = source_y["sob_slope"].quantile(quantiles)
    quant_source_mmse = source_y["mmse_slope"].quantile(quantiles)
    
    target_sob_cut = pd.cut(target_y["sob_slope"], quant_source_sob, include_lowest=True, duplicates="drop")
    target_mmse_cut = pd.cut(target_y["mmse_slope"], quant_source_mmse, include_lowest=True, duplicates="drop")
    
    # number of people in target with each of the categories so that it matches the proportion in source
    source_sob_n = pd.cut(source_y["sob_slope"], quant_source_sob, include_lowest=True, duplicates="drop").value_counts()
    source_mmse_n = pd.cut(source_y["mmse_slope"], quant_source_mmse, include_lowest=True, duplicates="drop").value_counts()
    
    return target_sob_cut, target_mmse_cut, source_sob_n, source_mmse_n


# trained on OASIS-3, tested on ADNI
ADNI_sob_cut, ADNI_mmse_cut, OASIS_sob_n, OASIS_mmse_n = get_matched_samples(oasis_y, adni_y)
# trained on ADNI, tested on OASIS-3
OASIS_sob_cut, OASIS_mmse_cut, ADNI_sob_n, ADNI_mmse_n = get_matched_samples(adni_y, oasis_y)

subset_adni = pd.concat([subset_adni, ADNI_sob_cut.rename("sob_qcat"), ADNI_mmse_cut.rename("mmse_qcat")], axis=1)
subset_oasis = pd.concat([subset_oasis, OASIS_sob_cut.rename("sob_qcat"), OASIS_mmse_cut.rename("mmse_qcat")], axis=1)
subset_adni_top15 = pd.concat([subset_adni_top15, ADNI_sob_cut.rename("sob_qcat"), ADNI_mmse_cut.rename("mmse_qcat")], axis=1)
subset_oasis_top15 = pd.concat([subset_oasis_top15, OASIS_sob_cut.rename("sob_qcat"), OASIS_mmse_cut.rename("mmse_qcat")], axis=1)


# ### 6.2 Matched Sample Predictions and Evaluation

# ### 10.3 OASIS-3 → ADNI (Combined)

# %%
feature_names_train = oasis_X_clin_fs_filtered.columns.tolist()


# %%

# %%


def subset_results(subset, SOB_n, MMSE_n):
    qcat_results = []
    for _ in tqdm(range(0, 1000)):
        df_sob = subset.groupby("sob_qcat", as_index=False, group_keys=False, observed=True).apply(lambda x: x.sample(n = SOB_n[x.name], replace=True), include_groups=False)
        df_mmse = subset.groupby("mmse_qcat", as_index=False, group_keys=False, observed=True).apply(lambda x: x.sample(n = MMSE_n[x.name], replace=True), include_groups=False)
        sob_test = df_sob['sob_slope']
        mmse_test = df_mmse['mmse_slope']
        sob_pred = df_sob['sob_pred']
        mmse_pred = df_mmse['mmse_pred']

        qcat_results.append({
        'sob_r2': sklearn.metrics.r2_score(sob_test, sob_pred),
        'sob_mse': sklearn.metrics.mean_squared_error(sob_test, sob_pred),
        'sob_mae': sklearn.metrics.mean_absolute_error(sob_test, sob_pred),
        'mmse_r2': sklearn.metrics.r2_score(mmse_test, mmse_pred),
        'mmse_mse': sklearn.metrics.mean_squared_error(mmse_test, mmse_pred),
        'mmse_mae': sklearn.metrics.mean_absolute_error(mmse_test, mmse_pred),
        })

    qcat_results = pd.DataFrame(qcat_results)

    return qcat_results

# ### 6.3 Results for Both Combined and Top-15 Models

subset_adni_results = subset_results(subset_adni, OASIS_sob_n, OASIS_mmse_n)
subset_adni_top15_results = subset_results(subset_adni_top15, OASIS_sob_n, OASIS_mmse_n)
subset_oasis_results = subset_results(subset_oasis, ADNI_sob_n, ADNI_mmse_n)
subset_oasis_top15_results = subset_results(subset_oasis_top15, ADNI_sob_n, ADNI_mmse_n)

subset_adni_results.agg(["median"]).map(round, n_figs=2)
subset_adni_top15_results.agg(["median"]).map(round, n_figs=2)

subset_oasis_results.agg(["median"]).map(round, n_figs=2)
subset_oasis_top15_results.agg(["median"]).map(round, n_figs=2)

# %%