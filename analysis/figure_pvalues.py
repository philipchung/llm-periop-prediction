# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps

from llm.prompt_names import prompt_name_df
from llm_utils import read_pandas

pd.options.display.max_columns = None
pd.options.display.max_rows = 100


# %%
# Functions for Creating Figures for P-value comparison across prompts for each task


def make_pvalue_matrices(pvalues_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_strategies = prompt_name_df.Formatted.tolist()
    num_prompt_strategies = len(prompt_strategies)
    corrected_pvalue_df = pd.DataFrame(
        data=np.zeros(shape=(num_prompt_strategies, num_prompt_strategies), dtype=float),
        index=prompt_strategies,
        columns=prompt_strategies,
    )
    reject_H0_df = pd.DataFrame(
        data=np.zeros(shape=(num_prompt_strategies, num_prompt_strategies), dtype=bool),
        index=prompt_strategies,
        columns=prompt_strategies,
    )

    for row in pvalues_table.itertuples():
        j = row.Prompt1
        i = row.Prompt2
        corrected_pvalue_df.at[i, j] = row.corrected_pvalue
        reject_H0_df.at[i, j] = row.reject_H0
    return corrected_pvalue_df, reject_H0_df


def make_pvalue_figure(
    reject_H0_df: pd.DataFrame, pvalue_df: pd.DataFrame, title: str = "P-Values"
) -> tuple[plt.Figure, plt.Axes]:
    """Create triangular heatmap with corrected p-values for each pairwise comparison.
    The colors in the heatmap grid represent whether the finding is statistically significant.
    Dark blue means statistically significant (reject H0 = True).
    Light blue means not statistically significant (reject H0 = False).

    Args:
        reject_H0_df (pd.DataFrame): Dataframe of boolean values.
        pvalue_df (pd.DataFrame): Dataframe of p-values.
        title (str, optional): _description_. Defaults to "Corrected P-Values".

    Returns:
        tuple[plt.Figure, plt.Axes]: _description_
    """
    # Create Upper Triangular Mask
    mask = np.triu(np.ones_like(reject_H0_df))
    # Colormap (Reject H0 = Dark Blue, No Reject Ho = Light Blue)
    cmap = colormaps["Blues"]
    # Create Figure
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    p1 = sns.heatmap(
        data=reject_H0_df,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,
        annot=pvalue_df,
        fmt=".2G",
        linewidth=0.5,
        ax=ax,
    )
    p1.set_title(title)
    return fig, ax


# %%
# ## Load p-values data
# Initialize Dicts to hold result dataframes
mcc_pvalues = {}
f1_pvalues = {}
mae_pvalues = {}
maxerror_pvalues = {}

# Task Names
classification_tasks = [
    "asa",
    "hospital_admission",
    "icu_admission",
    "unplanned_admit",
    "hospital_mortality",
]
regression_tasks = ["phase1_duration", "hospital_duration", "icu_duration"]
tasks = classification_tasks + regression_tasks

for task in classification_tasks:
    save_dir = Path.cwd() / "results" / task
    f1_pvalues |= {task: read_pandas(save_dir / "f1_prompt_pvalues.csv")}
    mcc_pvalues |= {task: read_pandas(save_dir / "mcc_prompt_pvalues.csv")}

for task in regression_tasks:
    save_dir = Path.cwd() / "results" / task
    mae_pvalues |= {task: read_pandas(save_dir / "mae_prompt_pvalues.csv")}
    maxerror_pvalues |= {task: read_pandas(save_dir / "maxerror_prompt_pvalues.csv")}

# %%
# Create Plots
plot_save_dir = Path.cwd() / "results" / "pvalues"
plot_save_dir.mkdir(parents=True, exist_ok=True)

## Classification Tasks
# ASA Physical Status Classification
task = "asa"
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=f1_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="ASA Physical Status Classification: P-Values Comparing Prompt Strategies with F1 Score ",
)
fig.savefig(plot_save_dir / f"{task}_f1_pvalues.png")
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=mcc_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="ASA Physical Status Classification: P-Values Comparing Prompt Strategies with MCC",
)
fig.savefig(plot_save_dir / f"{task}_mcc_pvalues.png")

# Hospital Admission
task = "hospital_admission"
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=f1_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="Hospital Admission: P-Values Comparing Prompt Strategies with F1 Score ",
)
fig.savefig(plot_save_dir / f"{task}_f1_pvalues.png")
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=mcc_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="Hospital Admission: P-Values Comparing Prompt Strategies with MCC",
)
fig.savefig(plot_save_dir / f"{task}_mcc_pvalues.png")

# ICU Admission
task = "icu_admission"
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=f1_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="ICU Admission: P-Values Comparing Prompt Strategies with F1 Score ",
)
fig.savefig(plot_save_dir / f"{task}_f1_pvalues.png")
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=mcc_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="ICU Admission: P-Values Comparing Prompt Strategies with MCC",
)
fig.savefig(plot_save_dir / f"{task}_mcc_pvalues.png")

# Unplanned Admission
task = "unplanned_admit"
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=f1_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="Unplanned Admission: P-Values Comparing Prompt Strategies with F1 Score ",
)
fig.savefig(plot_save_dir / f"{task}_f1_pvalues.png")
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=mcc_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="Unplanned Admission: P-Values Comparing Prompt Strategies with MCC",
)
fig.savefig(plot_save_dir / f"{task}_mcc_pvalues.png")

# Hospital Mortality
task = "hospital_mortality"
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=f1_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="Hospital Mortality: P-Values Comparing Prompt Strategies with F1 Score ",
)
fig.savefig(plot_save_dir / f"{task}_f1_pvalues.png")
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=mcc_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="Hospital Mortality: P-Values Comparing Prompt Strategies with MCC",
)
fig.savefig(plot_save_dir / f"{task}_mcc_pvalues.png")

## Regression Tasks
# PACU Phase 1 Duration
task = "phase1_duration"
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=mae_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="PACU Phase 1 Duration: P-Values Comparing Prompt Strategies with MAE ",
)
fig.savefig(plot_save_dir / f"{task}_mae_pvalues.png")
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=maxerror_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="PACU Phase 1 Duration: P-Values Comparing Prompt Strategies with MaxError",
)
fig.savefig(plot_save_dir / f"{task}_maxerror_pvalues.png")

# Hospital Duration
task = "hospital_duration"
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=mae_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="Hospital Duration: P-Values Comparing Prompt Strategies with MAE ",
)
fig.savefig(plot_save_dir / f"{task}_mae_pvalues.png")
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=maxerror_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="Hospital Duration: P-Values Comparing Prompt Strategies with MaxError",
)
fig.savefig(plot_save_dir / f"{task}_maxerror_pvalues.png")

# ICU Duration
task = "icu_duration"
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=mae_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="ICU Duration: P-Values Comparing Prompt Strategies with MAE ",
)
fig.savefig(plot_save_dir / f"{task}_mae_pvalues.png")
pvalues, reject_H0 = make_pvalue_matrices(pvalues_table=maxerror_pvalues[task])
fig, ax = make_pvalue_figure(
    reject_H0_df=reject_H0,
    pvalue_df=pvalues,
    title="ICU Duration: P-Values Comparing Prompt Strategies with MaxError",
)
fig.savefig(plot_save_dir / f"{task}_maxerror_pvalues.png")
# %%
