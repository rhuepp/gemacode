# %%
# # 02_Figures

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.categorical import LetterValues

from matplotlib.font_manager import FontProperties
from sklearn.metrics import r2_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import seaborn as sns  # Optional, but it provides additional styling


np.random.seed(21) # by setting a seed, you can ensure that every time you run the code, the sequence of random numbers generated will be the same

# %%
# ## Data for Figures

# %%
# CSV files with model performance of different models


# %%
# ADNI data
clinical_features_adni = pd.read_pickle("ADNI_1237_data_clinical_features_X_clinsessionn_2025.pkl")
data_slopes_adni = pd.read_pickle("ADNI_1237_data_slopes_y_2025.pkl")
structural_data_adni = pd.read_pickle("ADNI_1237_data_structGlobScort_2025.pkl")

# OASIS-3 data
clinical_features_oasis = pd.read_pickle("OASIS_clinical_features_X.pkl")
data_slopes_oasis = pd.read_pickle("OASIS_data_slopes_y.pkl")
structural_data_oasis = pd.read_pickle("OASIS_structGlobScort.pkl")

# %%
# clinical_features.columns.str.strip() str → columns as strings not as indexes
clinical_features_adni.columns = clinical_features_adni.columns.str.removeprefix("clin__npsy__").str.removeprefix("clin__risk__").str.removeprefix("clin__assess__")
clinical_features_oasis.columns = clinical_features_oasis.columns.str.removeprefix("clin__npsy__").str.removeprefix("clin__risk__").str.removeprefix("clin__assess__")

# %%
# structural_data.columns.str.strip() str → columns as strings not as indexes
structural_data_oasis.columns = structural_data_oasis.columns.str.removeprefix("fs__globalVolume__").str.removeprefix("fs__subcortVolume__")
structural_data_adni.columns = structural_data_adni.columns.str.removeprefix("fs__globalVolume__").str.removeprefix("fs__subcortVolume__")

# %%

df_adni2oasis_full = pd.read_csv("ADNI_to_OASIS_clin_struct/predictions.csv", index_col=0)
df_adni2oasis_struct = pd.read_csv("ADNI_to_OASIS_struct/predictions.csv", index_col=0)
df_adni2oasis_clin = pd.read_csv("ADNI_to_OASIS_clin/predictions.csv", index_col=0)
df_adni2oasis_top15 = pd.read_csv("ADNI_to_OASIS_clin_struct_top15/predictions.csv", index_col=0)
df_oasis2adni_full = pd.read_csv("OASIS_to_ADNI_clin_struct/predictions.csv", index_col=0)
df_oasis2adni_struct = pd.read_csv("OASIS_to_ADNI_struct/predictions.csv", index_col=0)
df_oasis2adni_clin = pd.read_csv("OASIS_to_ADNI_clin/predictions.csv", index_col=0)
df_oasis2adni_top15 = pd.read_csv("OASIS_to_ADNI_clin_struct_top15/predictions.csv", index_col=0)

# %%
# ## Figure 1: Model Performance in OASIS-3 and ADNI.

# %%
# create a figure and axes using subplots


def boxplot_model_rsq(data_df, plot_var, ax=None, color=None):
    # non-brain
    sns.boxplot(data=data_df[1], x=plot_var, ax=ax, color=color)
    ax.set_xlabel('')
    ax.set_ylabel(data_df[0], rotation=0, fontdict={'fontsize': 16, 'fontweight':'bold', 'fontname': 'Arial'}, labelpad=10, ha='right')
    for pos in ['top', 'bottom']:
        ax.spines[pos].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    # add median labels
    medians_0 = data_df[1][plot_var].median()
    # median + IQR
    text_hpos = data_df[1][plot_var].quantile(0.80)
    text_vpos = 0
    ax.text(text_hpos, text_vpos, f"{medians_0:#.2g}",
                ha='left', va='bottom', fontdict={'fontsize': 14, 'fontweight':'bold', 'fontname': 'Arial'})
    
def plot_boxplot_model_rsqs(dfs, colors, plot_var, xlim=None, save_path=None, save_fig=False, overwrite=False, plot_median=None):
    if len(dfs) != len(colors):
        raise ValueError("dfs and colors must have the same length")
    # create a figure and axes using subplots
    fig, axes = plt.subplots(nrows=len(dfs), ncols=1, figsize=(9, 3), sharex=True, gridspec_kw={'hspace': 0})

    for (df, color, ax) in zip(dfs.items(), colors, axes):
        boxplot_model_rsq(df, plot_var, ax=ax, color=color)
        
    for ax in axes:
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0.2, color='grey', linestyle='-', linewidth=0.3)
        ax.axvline(x=0.4, color='grey', linestyle='-', linewidth=0.3)
        ax.axvline(x=0.6, color='grey', linestyle='-', linewidth=0.3)
        ax.axvline(x=-0.2, color='grey', linestyle='-', linewidth=0.3)
        ax.axvline(x=-0.4, color='grey', linestyle='-', linewidth=0.3)
        ax.axvline(x=-0.6, color='grey', linestyle='-', linewidth=0.3)
        if plot_median is not None:
            median_plot = dfs[plot_median][plot_var].median()
            ax.axvline(median_plot, color='black', linestyle='--', linewidth=1.2)

    axes[0].spines['top'].set_visible(True)
    axes[-1].spines['bottom'].set_visible(True)
                                
    plt.xlabel(f'R\u00B2', fontdict={'fontsize': 16, 'fontstyle':'italic', 'fontweight':'bold', 'fontname': 'Arial'})

    font = FontProperties()
    font.set_family('Arial')
    font.set_size(14)
    plt.xticks(fontproperties=font)
    plt.xlim(xlim)


    # adjust the layout of the subplots
    plt.tight_layout()

    # save figure
    if save_fig and overwrite or save_fig and not os.path.exists(save_path):
            plt.savefig(save_path, dpi=3000)
    elif save_fig and os.path.exists(save_path):
        raise ValueError(f"File already exists at {save_path}")

    # show the plots
    plt.show()


# %%
def bootstrapped_r2_score(df, datasets, feature_set, n_iterations=1, random_state=0):
    r2_scores = []
    for i in tqdm(range(n_iterations)):
        df_sample = df.sample(n=df.shape[0], replace=True, random_state=random_state+i)
        r2_scores.append(r2_score(df_sample[["mmse_slope", "sob_slope"]], df_sample[["mmse_pred", "sob_pred"]], multioutput='raw_values'))
    r2_scores = np.array(r2_scores)
    df_boot = pd.DataFrame(r2_scores, columns=["mmse", "sob"])
    df_boot["data"] = datasets
    df_boot["feature_set"] = feature_set
    return df_boot

boot_adni2oasis_clin = bootstrapped_r2_score(df_adni2oasis_clin, "adni2oasis", "all_features", n_iterations=1000)
boot_adni2oasis_struct = bootstrapped_r2_score(df_adni2oasis_struct, "adni2oasis", "all_features", n_iterations=1000)
boot_adni2oasis_full = bootstrapped_r2_score(df_adni2oasis_full, "adni2oasis", "all_features", n_iterations=1000)
boot_adni2oasis_top15 = bootstrapped_r2_score(df_adni2oasis_top15, "adni2oasis", "15_features", n_iterations=1000)
boot_oasis2adni_clin = bootstrapped_r2_score(df_oasis2adni_clin, "oasis2adni", "all_features", n_iterations=1000)
boot_oasis2adni_struct = bootstrapped_r2_score(df_oasis2adni_struct, "oasis2adni", "all_features", n_iterations=1000)
boot_oasis2adni_full = bootstrapped_r2_score(df_oasis2adni_full, "oasis2adni", "all_features", n_iterations=1000)
boot_oasis2adni_top15 = bootstrapped_r2_score(df_oasis2adni_top15, "oasis2adni", "15_features", n_iterations=1000)

boot_res = pd.concat([boot_adni2oasis_clin, boot_adni2oasis_struct, boot_adni2oasis_full, boot_adni2oasis_top15, boot_oasis2adni_clin, boot_oasis2adni_struct, boot_oasis2adni_full, boot_oasis2adni_top15])
boot_res = boot_res.melt(id_vars=["data", "feature_set"], value_vars=["mmse", "sob"], var_name="metric", value_name="r2")

# %%


def get_fliers(data, k_depth="tukey"):
    estimator = LetterValues(k_depth, None, None)
    lv_data = estimator(data)
    return pd.Series(lv_data["fliers"])


# oasis_df = pd.concat({"non-brain": OASIS_clin, "structural MRI": OASIS_struct, "combined": OASIS_clin_struct}, names=["feature_set"]).reset_index(level=0)
# adni_df = pd.concat({"non-brain": ADNI_clin, "structural MRI": ADNI_struct, "combined": ADNI_clin_struct}, names=["feature_set"]).reset_index(level=0)
boot_adni2oasis_df = pd.concat({"non-brain": boot_adni2oasis_clin, "structural MRI": boot_adni2oasis_struct, "combined": boot_adni2oasis_full, "top 15": boot_adni2oasis_top15}, names=["feature_set2"]).reset_index(level=0)
boot_oasis2adni_df = pd.concat({"non-brain": boot_oasis2adni_clin, "structural MRI": boot_oasis2adni_struct, "combined": boot_oasis2adni_full, "top 15": boot_oasis2adni_top15}, names=["feature_set2"]).reset_index(level=0)
boot_oasis2adni_df = boot_oasis2adni_df.drop(columns="feature_set").rename(columns={"feature_set2": "feature_set"})
boot_adni2oasis_df = boot_adni2oasis_df.drop(columns="feature_set").rename(columns={"feature_set2": "feature_set"})

def get_rsq_plot(data, y, x, colors, title = "", xlabel = "R^2", ax=None, showfliers=False, ref_cat="non-brain +\nstructural MRI", boxenplot=False, digits=2):
    if ax is None:
        ax = plt.gca()
    ax.axvline(0, color='black', linestyle='--', linewidth=1.2)
    if ref_cat is not None:
        ref_val = data[data[y] == ref_cat][x].median()
        ax.axvline(ref_val, color='black', linestyle='-', linewidth=1.2)
    # add v line every 0.25
    [ax.axvline(p, color='gray', linestyle='--', linewidth=0.3) for p in np.arange(-0.6, 0.8, 0.2)]


    if boxenplot == True:
            
        if showfliers == True:
            fliers = data.groupby("feature_set", sort=False)[x].apply(get_fliers, k_depth=7).reset_index(level=0)
            sns.stripplot(data=fliers, y=y, x=x, hue=y, palette=colors, dodge=False, ax=ax, size=2, edgecolor='black', linewidth=0.5, alpha=0.5)

        sns.boxenplot(data=data, y=y, x=x, hue=y, palette=colors, dodge=False, saturation=1, ax=ax, linewidth=0.5, width_method="area", showfliers=False, k_depth=7)

    else:
        if showfliers == True:
            fliers = data.groupby("feature_set", sort=False)[x].apply(lambda x: x[(x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) | (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))]).reset_index(level=0)
            sns.stripplot(data=fliers, y=y, x=x, hue=y, palette=colors, dodge=False, ax=ax, size=2, edgecolor='black', linewidth=0.5, alpha=0.5)

        sns.boxplot(data=data, y=y, x=x, hue=y, palette=colors, dodge=False, ax=ax, showfliers=False, linewidth=0.5)

    medians = data.groupby(y, sort=False)[x].median()
    for i, median in enumerate(medians):
        # ax.text(median, i, f"{median:#.2g}", ha='center', va='center', fontdict={'fontsize': 12, 'fontweight':'bold', 'fontname': 'Arial'})
        # instead of significant digits, use the number of digits specified in the function
        ax.text(median, i, f"{median:.{digits}f}", ha='center', va='center', fontdict={'fontsize': 12, 'fontweight':'bold', 'fontname': 'Arial'})

    ax.tick_params(axis='both', labelfontfamily='Arial', labelsize=12)

    ax.set_xlabel(xlabel, fontdict={'fontsize': 12, 'fontstyle':'italic', 'fontweight':'bold', 'fontname': 'Arial'})
    ax.set_ylabel('')
    ax.set_title(title, fontdict={'fontsize': 16, 'fontweight':'bold', 'fontname': 'Arial'})

results_within = pd.read_csv("results_within.csv", index_col=[0, 1, 2, 3])
results_within = results_within.reset_index("modality")
results_within["feature_set"] = results_within["modality"].replace({
    "clin": "non-brain",
    "struct": "structural MRI",
    "clin_struct": "combined",
    "clin_struct_top15": "top 15",
})

oasis_df = results_within.loc["OASIS_filtered"]
adni_df = results_within.loc["ADNI_filtered"]

results_within = results_within.set_index("modality", append=True)

fig, axs = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=False, gridspec_kw={'height_ratios': [3,3,4,4], 'hspace': 0.45, 'wspace': 0.5})
get_rsq_plot(oasis_df.loc[:, "sob", :], "feature_set", "r2", ["#ACD5EF", "#CFE6A3", "#E89D9D"], "OASIS-3: CDR-SOB", "R\u00B2", axs[0, 0], showfliers=True)
get_rsq_plot(oasis_df.loc[:, "mmse", :], "feature_set", "r2", ["#ACD5EF", "#CFE6A3", "#E89D9D"], "OASIS-3: MMSE", "R\u00B2", axs[0, 1], showfliers=True)
get_rsq_plot(adni_df.loc[:, "sob", :], "feature_set", "r2", ["#ACD5EF", "#CFE6A3", "#E89D9D"], "ADNI: CDR-SOB", "R\u00B2", axs[1, 0], showfliers=True)
get_rsq_plot(adni_df.loc[:, "mmse", :], "feature_set", "r2", ["#ACD5EF", "#CFE6A3", "#E89D9D"], "ADNI: MMSE", "R\u00B2", axs[1, 1], showfliers=True)
get_rsq_plot(boot_oasis2adni_df, "feature_set", "sob", ["#ACD5EF", "#CFE6A3", "#E89D9D", "#FFDB93"], "OASIS-3 → ADNI: CDR-SOB", "R\u00B2", axs[2, 0], showfliers=True)
get_rsq_plot(boot_oasis2adni_df, "feature_set", "mmse", ["#ACD5EF", "#CFE6A3", "#E89D9D", "#FFDB93"], "OASIS-3 → ADNI: MMSE", "R\u00B2", axs[2, 1], showfliers=True)
get_rsq_plot(boot_adni2oasis_df, "feature_set", "sob", ["#ACD5EF", "#CFE6A3", "#E89D9D", "#FFDB93"], "ADNI → OASIS-3: CDR-SOB", "R\u00B2", axs[3, 0], showfliers=True)
get_rsq_plot(boot_adni2oasis_df, "feature_set", "mmse", ["#ACD5EF", "#CFE6A3", "#E89D9D", "#FFDB93"], "ADNI → OASIS-3: MMSE", "R\u00B2", axs[3, 1], showfliers=True)

font = FontProperties()
# set family arial
font.set_family('Arial')
font.set_size(14)

# add letter labels to subplots
for i, ax in enumerate(axs.ravel()):
    ax.text(-0.38, 1.1, chr(65+i), transform=ax.transAxes, fontsize=16, font_properties = font, fontweight='bold')

plt.tight_layout()

# save 
plt.savefig('r2_perf.png', dpi=300)

# %%
# ## Figure 2: Distribution of Absolute Error According to Subgroups.

# %%
# absolute error for each subject of the test set

OASIS_ADNI_clin_struct_abs = pd.DataFrame({
    'sob_abs_error': abs(df_oasis2adni_full['sob_slope'] - df_oasis2adni_full['sob_pred']),
    'mmse_abs_error': abs(df_oasis2adni_full['mmse_slope'] - df_oasis2adni_full['mmse_pred']),
})

ADNI_OASIS_clin_struct_abs = pd.DataFrame({
    'sob_abs_error': abs(df_adni2oasis_full['sob_slope'] - df_adni2oasis_full['sob_pred']),
    'mmse_abs_error': abs(df_adni2oasis_full['mmse_slope'] - df_adni2oasis_full['mmse_pred']),
})

# %%
oasis_X_clin_filtered = pd.read_csv("OASIS_clin_filtered.csv", index_col=0)
adni_X_clin_filtered = pd.read_csv("ADNI_clin_filtered.csv", index_col=0)
# ### OASIS-3 → ADNI: Prediction With Non-Brain Data and Structural MRI Data
OASIS_ADNI_fine = pd.concat([adni_X_clin_filtered, OASIS_ADNI_clin_struct_abs], axis =1)
# ### ADNI → OASIS-3: Prediction With Non-Brain Data and Structural MRI Data
ADNI_OASIS_fine = pd.concat([oasis_X_clin_filtered, ADNI_OASIS_clin_struct_abs], axis =1)

# %%

# Assuming ADNI_OASIS_fine is your DataFrame
def violinplot_fine_groups(fine_df, plot_var, title, ylim, width=0.8, boxenplot=False, remove_outliers=False, ax=None, font=None, xlabels=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    HC = fine_df[plot_var][fine_df["diag"] == 0]
    MCI = fine_df[plot_var][fine_df["diag"] == 1]
    AD = fine_df[plot_var][fine_df["diag"] == 2]
    Female = fine_df[plot_var][fine_df["demo_sex"] == 0]
    Male = fine_df[plot_var][fine_df["demo_sex"] == 1]
    APOE_E4_0 = fine_df[plot_var][fine_df["apoe_e4count"] == 0]
    APOE_E4_1 = fine_df[plot_var][fine_df["apoe_e4count"] == 1]
    APOE_E4_2 = fine_df[plot_var][fine_df["apoe_e4count"] == 2]
    age_U65 = fine_df[plot_var][fine_df["demo_age"] < 65]
    age_65_70 = fine_df[plot_var][(fine_df["demo_age"] >= 65) & (fine_df["demo_age"] < 70)]
    age_70_75 = fine_df[plot_var][(fine_df["demo_age"] >= 70) & (fine_df["demo_age"] < 75)]
    age_75_80 = fine_df[plot_var][(fine_df["demo_age"] >= 75) & (fine_df["demo_age"] < 80)]
    age_UE80 = fine_df[plot_var][fine_df["demo_age"] >= 80]

    # Combine the data into a list
    data = [
        HC,
        MCI,
        AD,
        Female,
        Male,
        APOE_E4_0,
        APOE_E4_1,
        APOE_E4_2,
        age_U65,
        age_65_70,
        age_70_75,
        age_75_80,
        age_UE80,
    ]

    data_df = pd.DataFrame(
        data,
        index=[
            "HC",
            "MCI",
            "AD",
            "Female",
            "Male",
            "APOE_E4_0",
            "APOE_E4_1",
            "APOE_E4_2",
            "age_U65",
            "age_65_70",
            "age_70_75",
            "age_75_80",
            "age_UE80",
        ],
    )
    data_df = data_df.T
    data_df = data_df.melt(var_name="groups", value_name="values")
    data_df = data_df.dropna()

    if remove_outliers:
        def is_outlier(data):
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            return ~data.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        outlier_idx = data_df.groupby("groups", group_keys=False)["values"].apply(is_outlier)
        outlier_df = data_df[outlier_idx]
        # data_df = data_df[~outlier_idx]

    # Define color groups for ticks
    color_groups = [(1, 2, 3), (4, 5), (6, 7, 8), (9, 10, 11, 12, 13)]

    # Define colors for each subgroup
    group_colors = ["#33a02c", "#0343DF", "red", "orange"]

    # colors, based on the color groups
    colors = [group_colors[next(i for i, group in enumerate(color_groups) if pos in group)] for pos in range(1, 14)]

    # Define specific spacing for certain subgroups
    """custom_spacing = {3: 1.5, 5: 5.5, 9: 9.5}"""

    """positions = [custom_spacing.get(pos, pos) for pos in range(1, 14)]  # Use custom spacing if specified
    """
    # use seaborn to create the violin plot
    # sns.violinplot(
    #     data=data_df,
    #     ax=ax,
    #     linewidth=None,
    #     x="groups",
    #     y="values",
    #     alpha=0.35,
    #     inner="quart",
    #     cut=0,
    #     density_norm="area",
    #     palette=colors,
    #     hue="groups",
    #     # common_norm=True,
    #     saturation=1,
    #     width=width,
    # )

    sns.boxplot(
        data=data_df,
        ax=ax,
        linewidth=None,
        x="groups",
        y="values",
        color="white",
        width=width,
        showfliers=False,
        # boxprops=dict(linewidth=1.5),
        # whiskerprops=dict(linewidth=1.5),
        # capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5),
        palette=colors,
        hue="groups",
        saturation=1,
        boxprops=dict(alpha=.35)
    )


    if remove_outliers:
        # Add the outliers to the plot with a beeswarm plot
        sns.stripplot(
            data=outlier_df,
            ax=ax,
            x="groups",
            y="values",
            color="black",
            size=2,
            alpha=0.5,
            marker="o",
            edgecolor="black",
        )

    # Customize the plot
    ax.set_title(title, font = font, fontsize=14, fontweight='bold')
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), font = font)
    ax.set_xticks(range(0, 13))
    if xlabels:
        ax.set_xticklabels(
            [
                "HC",
                "MCI",
                "AD",
                "Female",
                "Male",
                "APOE_E4_0",
                "APOE_E4_1",
                "APOE_E4_2",
                "< 65",
                "65-70",
                "70-75",
                "75-80",
                ">= 80",
            ],
            rotation=45,
            ha="right",
            font = font,
        )
        ax.set_xlabel("Subgroup", font = font)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    ax.set_ylabel("Absolute error", font = font)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Calculate means for each subgroup
    means = [group.mean() for group in data]

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Add overlapping line plots connecting the means for specified ranges
    ax.plot(range(0, 3), means[:3], color="black", linestyle="-", marker="o", markersize=4, label="Mean")
    ax.plot(range(3, 5), means[3:5], color="black", linestyle="-", marker="o", markersize=4)
    ax.plot(range(5, 8), means[5:8], color="black", linestyle="-", marker="o", markersize=4)
    ax.plot(range(8, 13), means[8:], color="black", linestyle="-", marker="o", markersize=4)


def barplot_fine_groups(fine_df, ax=None, font=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 1.5))

    HC = (fine_df['diag'] == 0).sum() / len(fine_df)
    MCI = (fine_df['diag'] == 1).sum() / len(fine_df)
    AD = (fine_df['diag'] == 2).sum() / len(fine_df)
    Female = (fine_df['demo_sex'] == 0).sum() / len(fine_df)
    Male = (fine_df['demo_sex'] == 1).sum() / len(fine_df)
    APOE_E4_0 = (fine_df['apoe_e4count'] == 0).sum() / len(fine_df)
    APOE_E4_1 = (fine_df['apoe_e4count'] == 1).sum() / len(fine_df)
    APOE_E4_2 = (fine_df['apoe_e4count'] == 2).sum() / len(fine_df)
    age_U65 = (fine_df['demo_age'] < 65).sum() / len(fine_df['demo_age'])
    age_65_70 = ((fine_df['demo_age'] >= 65) & (fine_df['demo_age'] < 70)).sum() / len(fine_df['demo_age'])
    age_70_75 = ((fine_df['demo_age'] >= 70) & (fine_df['demo_age'] < 75)).sum() / len(fine_df['demo_age'])
    age_75_80 = ((fine_df['demo_age'] >= 75) & (fine_df['demo_age'] < 80)).sum() / len(fine_df['demo_age'])
    age_UE80 = (fine_df['demo_age'] >= 80).sum() / len(fine_df['demo_age'])


    var = ['HC', 'MCI', 'AD', 'Female', 'Male', 'APOE_E4_0', 'APOE_E4_1', 'APOE_E4_2', 'age_U65', 'age_65_70', 'age_70_75', 'age_75_80', 'age_UE80']
    counts = [HC, MCI, AD, Female, Male, APOE_E4_0, APOE_E4_1, APOE_E4_2, age_U65, age_65_70, age_70_75, age_75_80, age_UE80]
    bar_labels = ['HC', 'MCI', 'AD', 'Female', 'Male', 'APOE_E4_0', 'APOE_E4_1', 'APOE_E4_2', 'age_U65', 'age_65_70', 'age_70_75', 'age_75_80', 'age_UE80']
    bar_colors = ['#33a02c', '#33a02c','#33a02c','#0343DF', '#0343DF', 'red', 'red', 'red', 'orange', 'orange', 'orange', 'orange', 'orange']

    ax.bar(var, counts, label=bar_labels, color=bar_colors, alpha=0.35, edgecolor='black', linewidth=0.8)
    ax.set_xticks(range(0,13))
    ax.set_xticklabels(['HC', 'MCI', 'AD', 'Female', 'Male', '$APOE\, \epsilon4 = 0$', '$APOE\, \epsilon4 = 1$', '$APOE\, \epsilon4 = 2$', 'Age $<$ 65',
                    '$65 \leq$ Age $<$ 70', '$70 \leq$ Age $<$ 75', '$75 \leq$ Age $<$ 80', 'Age $\geq$ 80'], rotation=45, ha='right', font = font)
    ax.set_ylabel('Subjects (%)', font = font)
    ax.set_ylim(0, 1)
    ax.set_title(title, font = font, fontsize=14, fontweight='bold')

    # format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

# %%
ADNI_OASIS_fine.groupby('diag')["sob_abs_error"].mean()

# %%
# 3 x 2 grid, 3 row should have smaller height, increase separation between rows, diminish between columns
fig, axs = plt.subplots(3, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1, 0.5], 'hspace': 0.3, 'wspace': 0.25})
violinplot_fine_groups(OASIS_ADNI_fine, "sob_abs_error", "OASIS-3 → ADNI: CDR-SOB", ylim = (0,3), remove_outliers=True, ax=axs[0, 0], xlabels=False, font=font)
violinplot_fine_groups(OASIS_ADNI_fine, "mmse_abs_error", "OASIS-3 → ADNI: MMSE", ylim = (0,7), remove_outliers=True, ax=axs[1, 0], xlabels=False, font=font)
barplot_fine_groups(OASIS_ADNI_fine, ax=axs[2, 0], font=font, title='Subgroup distribution (ADNI)')

violinplot_fine_groups(ADNI_OASIS_fine, "sob_abs_error", "ADNI → OASIS-3: CDR-SOB", ylim = (0,3), remove_outliers=True, ax=axs[0, 1], xlabels=False, font=font)
violinplot_fine_groups(ADNI_OASIS_fine, "mmse_abs_error", "ADNI → OASIS-3: MMSE", ylim = (0,7), remove_outliers=True, ax=axs[1, 1], xlabels=False, font=font)
barplot_fine_groups(ADNI_OASIS_fine, ax=axs[2, 1], font=font, title='Subgroup distribution (OASIS-3)')

# enumerate subplots and add letters
for i, ax in enumerate(axs.ravel()):
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font)
    ax.text(-0.175, 1.05, chr(65+i), transform=ax.transAxes, fontsize=16, font_properties = font, fontweight='bold')

# save figure
plt.savefig('subgroup_violinplot.png', dpi=300, bbox_inches='tight')

# %%
# # Supplementary Figures

# %%
# ## Figure B: Distribution of CDR-SOB and MMSE Change in OASIS-3 and ADNI.

# %%
def plot_histogram(data, bins, color, title, xlabel, ylabel, xlim, ylim, ref_vline=None, ha_summary="left", summary_fun = np.mean, save_path=None, save_fig=False, overwrite=False, ax=None, font=None):
    if ax is None:
        ax = plt.gca()
    ax.hist(x= data , bins= bins, color=color,
                            alpha=1, rwidth=0.8)
    ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel(xlabel, font_properties = font, fontweight='bold')
    ax.set_ylabel(ylabel, font_properties = font, fontweight='bold')
    ax.set_title(title, font_properties = font, fontweight='bold')


    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.set_xticks(ax.get_xticks())
    # ax.set_yticks(ax.get_yticks())

    # ax.set_xticklabels(ax.get_xticks(), fontproperties=font)
    # ax.set_yticklabels(ax.get_yticks(), fontproperties=font)

    for label in ax.get_xticklabels() :
        label.set_fontproperties(font)
    for label in ax.get_yticklabels() :
        label.set_fontproperties(font)
        
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    # add a vertical line at the mean
    summary_slope = summary_fun(data)
    ax.axvline(summary_slope, color='black', linestyle='--', linewidth=1.5)

    if ref_vline is not None:
        ax.axvline(x=ref_vline, color='black', linestyle='-', linewidth=0.5)
    # save figure
    #plt.savefig('Descriptives_Scatterplot/OASIS-3_CDR-SOB_slope.png', dpi=3000)

    if ha_summary == "left":
        summary_xpos = summary_slope + 0.01*(xlim[1]-xlim[0])
    else:
        summary_xpos = summary_slope - 0.01*(xlim[1]-xlim[0])
    summary_ypos = 0.9*ylim[1]
    # add summary statistics near the top to the right of summary line
    ax.text(summary_xpos, summary_ypos, f"{summary_slope:#.2g}",
            ha=ha_summary, va='center', font_properties = font, fontweight='bold')

# %%
font = FontProperties()
font.set_family('Arial')
font.set_size(14)

fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False, gridspec_kw={'hspace': 0.45, 'wspace': 0.3})
plot_histogram(data_slopes_oasis['sob_slope'], 27, '#ACD5EF', 'OASIS-3: CDR-SOB', 'CDR-SOB change', 'Frequency', (-2.75, 8.75), (0, 450), ax=axs[0,0], font=font)
plot_histogram(data_slopes_oasis['mmse_slope'], 27, '#ACD5EF', 'OASIS-3: MMSE', 'MMSE change', 'Frequency', (-14.75, 6), (0, 400), ha_summary="right", ax=axs[1,0], font=font)
plot_histogram(data_slopes_adni['sob_slope'], 50, '#CFE6A3', 'ADNI: CDR-SOB', 'CDR-SOB change', 'Frequency', (-2.75, 8.75), (0, 450), ax=axs[0,1], font=font)
plot_histogram(data_slopes_adni['mmse_slope'], 50, '#CFE6A3', 'ADNI: MMSE', 'MMSE change', 'Frequency', (-14.75, 6), (0, 400), ha_summary="right", ax=axs[1,1], font=font)

# add letter labels to subplots
for i, ax in enumerate(axs.ravel()):
    ax.text(-0.205, 1.1, chr(65+i), transform=ax.transAxes, fontsize=16, font_properties = font, fontweight='bold')

plt.tight_layout()

plt.savefig("change_distribution.png", dpi=300)

# %%
min_slope = np.min(data_slopes_oasis['sob_slope'])
print(min_slope)

max_slope = np.max(data_slopes_oasis['sob_slope'])
print(max_slope)

min_slope = np.min(data_slopes_adni['sob_slope'])
print(min_slope)

max_slope = np.max(data_slopes_adni['sob_slope'])
print(max_slope)


# %%
# ## Figure C: Distribution of R<sup>2</sup> Across 1000 Splits in OASIS-3 Models.

# %%
font = {'family': 'Arial', 'weight': 'bold', 'size': 14}
italic_font = {'family': 'Arial', 'weight': 'bold', 'size': 14, 'style': 'italic'}
tick_font = {'family': 'Arial', 'weight': 'normal', 'size': 14}

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=False, sharey=False, gridspec_kw={'hspace': 0.45, 'wspace': 0.3})

# Plot histograms
plot_histogram(results_within.loc["OASIS_filtered", :, "sob", "clin"]["r2"], 'auto', '#ACD5EF', 'OASIS-3 (non-brain): CDR-SOB', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[0,0])
plot_histogram(results_within.loc["OASIS_filtered", :, "mmse", "clin"]["r2"], 'auto', '#ACD5EF', 'OASIS-3 (non-brain): MMSE', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[0,1])
plot_histogram(results_within.loc["OASIS_filtered", :, "sob", "struct"]["r2"], 'auto', '#CFE6A3', 'OASIS-3 (structural MRI): CDR-SOB', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[1,0])
plot_histogram(results_within.loc["OASIS_filtered", :, "mmse", "struct"]["r2"], 'auto', '#CFE6A3', 'OASIS-3 (structural MRI): MMSE', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[1,1])
plot_histogram(results_within.loc["OASIS_filtered", :, "sob", "clin_struct"]["r2"], 'auto', '#E89D9D', 'OASIS-3 (combined): CDR-SOB', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[2,0])
plot_histogram(results_within.loc["OASIS_filtered", :, "mmse", "clin_struct"]["r2"], 'auto', '#E89D9D', 'OASIS-3 (combined): MMSE', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[2,1])

for ax in axs.ravel():
    ax.set_title(ax.get_title(), fontdict=font)
    ax.set_ylabel(ax.get_ylabel(), fontdict=font)

for ax in axs.ravel():
    ax.set_xlabel(f'R\u00B2', fontdict=italic_font)

for ax in axs.ravel():
    ax.tick_params(axis='both', labelsize=14, labelcolor='black', width=1)
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=tick_font)  # Ensure tick labels are regular font
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=tick_font)  # Ensure tick labels are regular font

# Add letter labels to subplots
for i, ax in enumerate(axs.ravel()):
    ax.text(-0.205, 1.1, chr(65+i), transform=ax.transAxes, fontsize=16, fontweight='bold')

plt.tight_layout()

plt.savefig("OASIS_R2_distribution.png", dpi=300)


# %%
# ## Figure D: Distribution of R<sup>2</sup> Across 1000 Splits in ADNI Models.

# %%
font = {'family': 'Arial', 'weight': 'bold', 'size': 14}
italic_font = {'family': 'Arial', 'weight': 'bold', 'size': 14, 'style': 'italic'}
tick_font = {'family': 'Arial', 'weight': 'normal', 'size': 14}

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=False, sharey=False, gridspec_kw={'hspace': 0.45, 'wspace': 0.3})

# Plot histograms
plot_histogram(results_within.loc["ADNI_filtered", :, "sob", "clin"]["r2"], 'auto', '#ACD5EF', 'ADNI (non-brain): CDR-SOB', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[0,0])
plot_histogram(results_within.loc["ADNI_filtered", :, "mmse", "clin"]["r2"], 'auto', '#ACD5EF', 'ADNI (non-brain): MMSE', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[0,1])
plot_histogram(results_within.loc["ADNI_filtered", :, "sob", "struct"]["r2"], 'auto', '#CFE6A3', 'ADNI (structural MRI): CDR-SOB', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[1,0])
plot_histogram(results_within.loc["ADNI_filtered", :, "mmse", "struct"]["r2"], 'auto', '#CFE6A3', 'ADNI (structural MRI): MMSE', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[1,1])
plot_histogram(results_within.loc["ADNI_filtered", :, "sob", "clin_struct"]["r2"], 'auto', '#E89D9D', 'ADNI (combined): CDR-SOB', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[2,0])
plot_histogram(results_within.loc["ADNI_filtered", :, "mmse", "clin_struct"]["r2"], 'auto', '#E89D9D', 'ADNI (combined): MMSE', f'R\u00B2', 'Frequency', (-0.75, 0.7), (0, 140), ref_vline=0, summary_fun=np.median, ax=axs[2,1])

for ax in axs.ravel():
    ax.set_title(ax.get_title(), fontdict=font)
    ax.set_ylabel(ax.get_ylabel(), fontdict=font)

for ax in axs.ravel():
    ax.set_xlabel(f'R\u00B2', fontdict=italic_font)

for ax in axs.ravel():
    ax.tick_params(axis='both', labelsize=14, labelcolor='black', width=1)
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=tick_font)  # Ensure tick labels are regular font
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=tick_font)  # Ensure tick labels are regular font

# Add letter labels to subplots
for i, ax in enumerate(axs.ravel()):
    ax.text(-0.205, 1.1, chr(65+i), transform=ax.transAxes, fontsize=16, fontweight='bold')

plt.tight_layout()

plt.savefig("ADNI_R2_distribution.png", dpi=300)


# %%
# ## Figure E: True Versus Predicted Values of CDR-SOB and MMSE Change in Across Datasets Predictions.

# %%


def plot_scatter(x, y, xlabel, ylabel, title, ax=None, font=None, color=None, labels=None):
    if ax is None:
        ax = plt.gca()


    # set x and y axis limits
    ax.set_aspect('equal')

    # add a diagonal line
    ax.axline(xy1=(0,0), slope=1, linestyle='--', color='black', linewidth=1, zorder=3)
    # horizontal and vertical lines
    ax.axvline(x=0, linestyle='-', color='black', linewidth=1, zorder=1)
    ax.axhline(y=0, linestyle='-', color='black', linewidth=1, zorder=1)
    # create a scatter plot and add color and label
    ax.set_axisbelow(True)
    ax.grid(axis='both', alpha=0.75, zorder=0)
    if color is None:
        ax.scatter(x, y, edgecolor='#0343DF', alpha=1, c='#ACD5EF', marker='o', s = 10, zorder=2)
    else:
        colorlist = ["#ACD5EF", "#E89D9D", "#CFE6A3"]
        edgecolorlist = ['#0343DF','red','#33a02c']
        # ensure color is integer
        color = color.astype(int)
        for i in range(len(colorlist)):
            x_i = x[color == i]
            y_i = y[color == i]
            label_i = labels[i]
            ax.scatter(x_i, y_i, alpha=1, marker='o', s = 10, c=colorlist[i], edgecolor=edgecolorlist[i], label=label_i, linewidths=0.5)
    # add labels and title and change font
    ax.set_xlabel(xlabel, fontdict={'fontsize': 14, 'fontweight': 'bold', 'fontname': 'Arial'})
    ax.set_ylabel(ylabel, fontdict={'fontsize': 14, 'fontweight': 'bold', 'fontname': 'Arial'})
    ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold', 'fontname': 'Arial'})
    ax.tick_params(axis='both', which='major', labelsize=14, labelfontfamily='Arial')

    # flip x and y
    

    # plt.xticks(fontproperties=font)
    # plt.yticks(fontproperties=font)


# 2x2 grid of scatter plots
fig, axs = plt.subplots(2, 2, figsize=(5, 9), sharex="col", sharey="col", gridspec_kw={'hspace': 0.3, 'wspace': 1})
plot_scatter(df_oasis2adni_full['sob_pred'], df_oasis2adni_full['sob_slope'], '', 'True slope', 'OASIS-3 → ADNI\n(CDR-SOB)', ax=axs[0, 0], font=font)
plot_scatter(df_oasis2adni_full['mmse_pred'], df_oasis2adni_full['mmse_slope'], '', '', 'OASIS-3 → ADNI\n(MMSE)', ax=axs[0, 1], font=font)
plot_scatter(df_adni2oasis_full['sob_pred'], df_adni2oasis_full['sob_slope'], 'Predicted slope', 'True slope', 'ADNI → OASIS-3\n(CDR-SOB)', ax=axs[1, 0], font=font)
plot_scatter(df_adni2oasis_full['mmse_pred'], df_adni2oasis_full['mmse_slope'], 'Predicted slope', '', 'ADNI → OASIS-3\n(MMSE)', ax=axs[1, 1], font=font)
plt.tight_layout()

# add A, B, C, D labels
for i, ax in enumerate(axs.ravel()):
    ax.text(-0.6, 1.05, chr(65+i), transform=ax.transAxes, fontsize=16, font_properties = font, fontweight='bold')

# save figure
plt.savefig("ADNI_OASIS_pred_scatterplot.png", dpi=300)

# %%
# ## Figure F: True Versus Predicted Values of CDR-SOB and MMSE Change for Subgroups Differentiated by Clinical Diagnoses in Across Datasets Predictions.

# %%
df_adni2oasis_extended = pd.merge(df_adni2oasis_full, clinical_features_oasis, left_index=True, right_on="subject")
df_oasis2adni_extended = pd.merge(df_oasis2adni_full, clinical_features_adni, left_index=True, right_on="subject")
df_adni2oasis_extended = df_adni2oasis_extended.dropna(subset="diag")
df_adni2oasis_extended["diag"] = df_adni2oasis_extended["diag"].map({"hc": 0, "mci": 1, "dem": 2})
df_oasis2adni_extended = df_oasis2adni_extended.dropna(subset="diag")
df_oasis2adni_extended["diag"] = df_oasis2adni_extended["diag"].map({"NL": 0, "MCI": 1, "Dementia": 2})

# scatter = ax.scatter(df_extended_AD['sob_pred_jitter'], df_extended_AD['sob_slope_jitter'], edgecolor='#33a02c', alpha=0.7, c='#CFE6A3', marker='s', label='AD', s = 15)
# scatter = ax.scatter(df_extended_MCI['sob_pred_jitter'], df_extended_MCI['sob_slope_jitter'], edgecolor='red', alpha=0.7, c='#E89D9D', marker='^', label='MCI', s = 15)
# scatter = ax.scatter(df_extended_HC['sob_pred_jitter'], df_extended_HC['sob_slope_jitter'], edgecolor='#0343DF', alpha=0.7, c='#ACD5EF', marker='o', label='HC', s = 15)

# 2x2 grid of scatter plots
fig, axs = plt.subplots(2, 2, figsize=(5, 9), sharex="col", sharey="col", gridspec_kw={'hspace': 0.3, 'wspace': 1})
plot_scatter(df_oasis2adni_extended['sob_pred'], df_oasis2adni_extended['sob_slope'], 'Predicted CDR-SOB', 'True CDR-SOB', 'OASIS-3 → ADNI', ax=axs[0, 0], font=font, color=df_oasis2adni_extended["diag"], labels=["HC", "MCI", "AD"])
plot_scatter(df_oasis2adni_extended['mmse_pred'], df_oasis2adni_extended['mmse_slope'], 'Predicted MMSE', 'True MMSE', 'OASIS-3 → ADNI', ax=axs[0, 1], font=font, color=df_oasis2adni_extended["diag"], labels=["HC", "MCI", "AD"])
plot_scatter(df_adni2oasis_extended['sob_pred'], df_adni2oasis_extended['sob_slope'], 'Predicted CDR-SOB', 'True CDR-SOB', 'ADNI → OASIS-3', ax=axs[1, 0], font=font, color=df_adni2oasis_extended["diag"], labels=["HC", "MCI", "AD"])
plot_scatter(df_adni2oasis_extended['mmse_pred'], df_adni2oasis_extended['mmse_slope'], 'Predicted MMSE', 'True MMSE', 'ADNI → OASIS-3', ax=axs[1, 1], font=font, color=df_adni2oasis_extended["diag"], labels=["HC", "MCI", "AD"])
plt.tight_layout()
axs[1,1].legend(fontsize=14, prop={'family': 'Arial'}, loc='center left', bbox_to_anchor=(1, 0.5), title="Diagnosis", title_fontsize="14")

# add A, B, C, D labels
for i, ax in enumerate(axs.ravel()):
    ax.text(-0.6, 1.05, chr(65+i), transform=ax.transAxes, fontsize=16, font_properties = font, fontweight='bold')

# save figure
plt.savefig("ADNI_OASIS_pred_scatterplot_diag.png", dpi=300, bbox_inches='tight')



# %%
