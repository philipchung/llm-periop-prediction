# %%
# ## Bootstrap Confusion Matrix Metrics
#
# Requires that we run generation for each task
# and save the "predictions.feather" file for each
# categorical prediciton task:
# * `analysis_asa.py`
# * `analysis_hospital_admission.py`
# * `analysis_icu_admission.py`
# * `analysis_unplanned_admit.py`
# * `analysis_hospital_mortality.py`
# %%
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from llm.metrics import ConfusionMatrixMetrics, dummy_classifier_confusion_matrix_metrics
from llm.prompt_names import format_prompt_name, prompt_name_df
from llm_utils import ProgressBar, read_pandas, save_pandas

pd.options.display.max_columns = None
pd.options.display.max_rows = 100

save_dir = Path.cwd() / "results" / "confmat_metrics"

# Prediction Task
predictions_dfs = {}
tasks = ["asa", "hospital_admission", "icu_admission", "unplanned_admit", "hospital_mortality"]
with ProgressBar() as p:
    t1 = p.add_task("t1")
    for task in p.track(tasks, task_id=t1):
        p.update(t1, description=f"Loading Predictions for Task: {task}")
        phi_save_dir = Path.cwd() / "phi_results" / task
        predictions_df = read_pandas(path=phi_save_dir / "predictions.feather")
        # Remove Null Answers (if any, across all prompt strategy)
        answers_df = predictions_df.loc[:, prompt_name_df.Answer]
        null_answers = answers_df.isna().any(axis=1)
        null_df = predictions_df.loc[null_answers]
        nonnull_df = predictions_df.loc[~null_answers]
        # Persist Dataframe
        predictions_dfs |= {task: nonnull_df}


# %% [markdown]
# ## Compute Confusion Matrix & Metrics Derived from Confusion Matrix
#
# Includes:
# * True Positive Rate (TPR, Sensitivity, Recall)
# * True Negative Rate (TNR, Specificity)
# * Positive Predictive Value (PPV, Precision)
# * Negative Predictive Value (NPV)
#
# Confusion matrices are computed using original predictions & labels
# without bootstrap.
# Confusion matrix metrics are computed using bootstrap to estimate the
# mean and confidence interval.
# %%
def format_mean_ci(row: pd.Series) -> str:
    return f"{row.mean_value:.2f} ({row.lower_ci:.2f}, {row.upper_ci:.2f})"


def mean_ci_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(format_mean_ci, axis="columns").to_dict()


# Compute confusion matricies for each task x prompt strategy
tasks = ["asa", "hospital_admission", "icu_admission", "unplanned_admit", "hospital_mortality"]
answer_col_names = prompt_name_df.Answer
num_bootstrap_iterations = 2500
conf_mat_dict = {
    "asa": {},
    "hospital_admission": {},
    "icu_admission": {},
    "unplanned_admit": {},
    "hospital_mortality": {},
}
boot_metrics_dict = copy.deepcopy(conf_mat_dict)
mean_ci_dict = copy.deepcopy(conf_mat_dict)
with ProgressBar() as p:
    t1 = p.add_task("t1")
    t2 = p.add_task("t2")
    for task in p.track(tasks, task_id=t1):
        p.update(t1, description=f"Task: {task}")
        # Load Data
        df = predictions_dfs[task]
        # Class Labels
        if task == "asa":
            class_labels = [1, 2, 3, 4, 5, 6]
        else:
            class_labels = [True, False]

        # Compute Dummy Classifier Baseline Metrics
        p.update(t2, description="Metrics: Baseline")
        targets = df.Label.astype(int)
        baseline = dummy_classifier_confusion_matrix_metrics(
            targets=targets,
            class_labels=class_labels,
            strategy="uniform",
            num_bootstrap_iterations=num_bootstrap_iterations,
        )
        baseline_mean_ci = pd.Series(mean_ci_all_metrics(baseline), name="Baseline")
        mean_ci_dict[task] |= {"Baseline": baseline_mean_ci}

        # Compute Confusion Matrix Metrics for all Prompt Strategies
        for col in p.track(answer_col_names, task_id=t2):
            p.update(t2, description=f"Metrics: {col}")
            formatted_prompt_name = format_prompt_name(col)
            # Coerce preds & targets data type
            preds = df[col].astype(int)
            targets = df.Label.astype(int)
            # Compute Confusion Matrix
            cm = ConfusionMatrixMetrics(
                preds=preds,
                targets=targets,
                class_labels=class_labels,
                name=formatted_prompt_name,
                num_bootstrap_iterations=num_bootstrap_iterations,
            )
            boot_metrics_df = cm.compute()
            conf_mat = cm.confusion_matrix
            # Store Confusion Matrix
            conf_mat_dict[task] |= {formatted_prompt_name: conf_mat}
            # Store Bootstrap Metrics DataFrame
            boot_metrics_dict[task] |= {formatted_prompt_name: boot_metrics_df}
            # Store Formatted Metrics (Mean + 95% Confidence Interval)
            mean_ci_d = mean_ci_all_metrics(boot_metrics_df)
            mean_ci_dict[task] |= {formatted_prompt_name: mean_ci_d}
        p.reset(t2)

# %%
task = "asa"
cm_metrics_table = pd.DataFrame.from_dict(mean_ci_dict[task], orient="index")
cm_metrics_table
# %%
for task in tasks:
    # Confusion Matrix Metrics
    cm_metrics_table = pd.DataFrame.from_dict(mean_ci_dict[task], orient="index")
    # TPR, TNR, PPV, NPV
    tpr_table = cm_metrics_table.filter(like="TPR")
    tnr_table = cm_metrics_table.filter(like="TNR")
    ppv_table = cm_metrics_table.filter(like="PPV")
    npv_table = cm_metrics_table.filter(like="NPV")
    # Combined Table (TPR, TNR, PPV, NPV)
    combined_table = pd.concat([tpr_table, tnr_table, ppv_table, npv_table], axis=1)
    # Save Tables
    (save_dir / f"{task}").mkdir(exist_ok=True, parents=True)
    save_pandas(df=tpr_table, path=save_dir / f"{task}" / "tpr.csv")
    save_pandas(df=tnr_table, path=save_dir / f"{task}" / "tnr.csv")
    save_pandas(df=ppv_table, path=save_dir / f"{task}" / "ppv.csv")
    save_pandas(df=npv_table, path=save_dir / f"{task}" / "npv.csv")
    save_pandas(df=combined_table, path=save_dir / f"{task}" / "combined.csv")

# %% [markdown]
# ## Plot Confusion Matrices for each Task
# %%
task = "asa"
conf_mats_for_task = conf_mat_dict[task]
# Plot Confusion Matrix for Task for All Prompts
fig, ax = plt.subplots(
    nrows=2, ncols=6, layout="constrained", figsize=(13, 6), sharex=True, sharey=True
)
for (prompt_strategy, cm), subplot_ax in zip(conf_mats_for_task.items(), ax.flat):
    sns.heatmap(
        data=cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=subplot_ax,
    )
    subplot_ax.set_title(prompt_strategy)
    subplot_ax.set_xlabel(None)
    subplot_ax.set_ylabel(None)
    fig.suptitle("ASA Physical Status Classification", fontsize=14)
    fig.supxlabel("Actual", fontsize=12)
    fig.supylabel("Predicted", fontsize=12)

fig.savefig(save_dir / f"{task}" / "confusion_matrix.png")
fig.show()

task = "hospital_admission"
conf_mats_for_task = conf_mat_dict[task]
# Plot Confusion Matrix for Task for All Prompts
fig, ax = plt.subplots(
    nrows=2, ncols=6, layout="constrained", figsize=(13, 6), sharex=True, sharey=True
)
for (prompt_strategy, cm), subplot_ax in zip(conf_mats_for_task.items(), ax.flat):
    sns.heatmap(
        data=cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=subplot_ax,
    )
    subplot_ax.set_title(prompt_strategy)
    subplot_ax.set_xlabel(None)
    subplot_ax.set_ylabel(None)
    fig.suptitle("Hospital Admission", fontsize=14)
    fig.supxlabel("Actual", fontsize=12)
    fig.supylabel("Predicted", fontsize=12)

fig.savefig(save_dir / f"{task}" / "confusion_matrix.png")
fig.show()

task = "icu_admission"
conf_mats_for_task = conf_mat_dict[task]
# Plot Confusion Matrix for Task for All Prompts
fig, ax = plt.subplots(
    nrows=2, ncols=6, layout="constrained", figsize=(13, 6), sharex=True, sharey=True
)
for (prompt_strategy, cm), subplot_ax in zip(conf_mats_for_task.items(), ax.flat):
    sns.heatmap(
        data=cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=subplot_ax,
    )
    subplot_ax.set_title(prompt_strategy)
    subplot_ax.set_xlabel(None)
    subplot_ax.set_ylabel(None)
    fig.suptitle("ICU Admission", fontsize=14)
    fig.supxlabel("Actual", fontsize=12)
    fig.supylabel("Predicted", fontsize=12)

fig.savefig(save_dir / f"{task}" / "confusion_matrix.png")
fig.show()

task = "unplanned_admit"
conf_mats_for_task = conf_mat_dict[task]
# Plot Confusion Matrix for Task for All Prompts
fig, ax = plt.subplots(
    nrows=2, ncols=6, layout="constrained", figsize=(13, 6), sharex=True, sharey=True
)
for (prompt_strategy, cm), subplot_ax in zip(conf_mats_for_task.items(), ax.flat):
    sns.heatmap(
        data=cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=subplot_ax,
    )
    subplot_ax.set_title(prompt_strategy)
    subplot_ax.set_xlabel(None)
    subplot_ax.set_ylabel(None)
    fig.suptitle("Unplanned Admission", fontsize=14)
    fig.supxlabel("Actual", fontsize=12)
    fig.supylabel("Predicted", fontsize=12)

fig.savefig(save_dir / f"{task}" / "confusion_matrix.png")
fig.show()

task = "hospital_mortality"
conf_mats_for_task = conf_mat_dict[task]
# Plot Confusion Matrix for Task for All Prompts
fig, ax = plt.subplots(
    nrows=2, ncols=6, layout="constrained", figsize=(13, 6), sharex=True, sharey=True
)
for (prompt_strategy, cm), subplot_ax in zip(conf_mats_for_task.items(), ax.flat):
    sns.heatmap(
        data=cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=subplot_ax,
    )
    subplot_ax.set_title(prompt_strategy)
    subplot_ax.set_xlabel(None)
    subplot_ax.set_ylabel(None)
    fig.suptitle("Hospital Mortality", fontsize=14)
    fig.supxlabel("Actual", fontsize=12)
    fig.supylabel("Predicted", fontsize=12)

fig.savefig(save_dir / f"{task}" / "confusion_matrix.png")
fig.show()

# %%
