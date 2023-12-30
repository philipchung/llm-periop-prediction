# %%
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from llm.plot import plot_metric_for_prompts_on_axes
from llm_utils import read_pandas

pd.options.display.max_columns = None
pd.options.display.max_rows = 100

# Initialize Dicts to hold result dataframes
f1_boot_mean_long_dfs = {}
f1_baseline_boot_metrics_dfs = {}

mcc_boot_mean_long_dfs = {}
mcc_baseline_boot_metrics_dfs = {}

mae_boot_mean_long_dfs = {}
mae_baseline_boot_metrics_dfs = {}

maxerror_boot_mean_long_dfs = {}
maxerror_baseline_boot_metrics_dfs = {}

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

# Load Classification Task Results
for task in classification_tasks:
    save_dir = Path.cwd() / "results" / task
    # Get Baseline Metrics
    baseline_boot_metrics = read_pandas(save_dir / "baseline_boot_metrics.csv", set_index="Metric")
    f1_baseline_boot_metrics_dfs |= {
        task: baseline_boot_metrics.loc["Aggregate/F1/Micro", "mean_value"]
    }
    mcc_baseline_boot_metrics_dfs |= {
        task: baseline_boot_metrics.loc["Aggregate/MCC/MCC", "mean_value"]
    }
    # Get Metrics for Each Prompt Strategy
    f1_boot_mean_long_dfs |= {
        task: read_pandas(save_dir / "f1_boot_mean_long.csv", set_index="Unnamed: 0").reset_index(
            drop=True
        )
    }
    mcc_boot_mean_long_dfs |= {
        task: read_pandas(save_dir / "mcc_boot_mean_long.csv", set_index="Unnamed: 0").reset_index(
            drop=True
        )
    }

# Load Regression Task Results
for task in regression_tasks:
    save_dir = Path.cwd() / "results" / task
    # Get Baseline Metrics
    baseline_boot_metrics = read_pandas(save_dir / "baseline_boot_metrics.csv", set_index="Metric")
    mae_baseline_boot_metrics_dfs |= {
        task: baseline_boot_metrics.loc["Aggregate/MAE/MAE", "mean_value"]
    }
    maxerror_baseline_boot_metrics_dfs |= {
        task: baseline_boot_metrics.loc["Aggregate/MAE/MaxError", "mean_value"]
    }
    # Get Metrics for Each Prompt Strategy
    mae_boot_mean_long_dfs |= {
        task: read_pandas(save_dir / "mae_boot_mean_long.csv", set_index="Unnamed: 0").reset_index(
            drop=True
        )
    }
    maxerror_boot_mean_long_dfs |= {
        task: read_pandas(
            save_dir / "maxerror_boot_mean_long.csv", set_index="Unnamed: 0"
        ).reset_index(drop=True)
    }

# %%
# Create Plot
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 16), layout="constrained")

task = "asa"
p1 = plot_metric_for_prompts_on_axes(
    data=f1_boot_mean_long_dfs[task],
    ax=ax[0, 0],
    title="ASA Physical Status Classification",
    ylabel="F1 Score (→ is better)",
    baseline_value=f1_baseline_boot_metrics_dfs[task],
    legend_on=False,
)
task = "hospital_admission"
p2 = plot_metric_for_prompts_on_axes(
    data=f1_boot_mean_long_dfs[task],
    ax=ax[0, 1],
    title="Hospital Admission",
    ylabel="F1 Score (→ is better)",
    baseline_value=f1_baseline_boot_metrics_dfs[task],
    legend_on=False,
)
task = "icu_admission"
p3 = plot_metric_for_prompts_on_axes(
    data=f1_boot_mean_long_dfs[task],
    ax=ax[0, 2],
    title="ICU Admission",
    ylabel="F1 Score (→ is better)",
    baseline_value=f1_baseline_boot_metrics_dfs[task],
    legend_on=False,
)
task = "unplanned_admit"
p4 = plot_metric_for_prompts_on_axes(
    data=f1_boot_mean_long_dfs[task],
    ax=ax[1, 0],
    title="Unplanned Admission",
    ylabel="F1 Score (→ is better)",
    baseline_value=f1_baseline_boot_metrics_dfs[task],
    legend_on=False,
)
task = "hospital_mortality"
p5 = plot_metric_for_prompts_on_axes(
    data=f1_boot_mean_long_dfs[task],
    ax=ax[1, 1],
    title="Hospital Mortality",
    ylabel="F1 Score (→ is better)",
    baseline_value=f1_baseline_boot_metrics_dfs[task],
    legend_on=False,
)
task = "phase1_duration"
p6 = plot_metric_for_prompts_on_axes(
    data=mae_boot_mean_long_dfs[task],
    ax=ax[1, 2],
    title="PACU Phase 1 Duration",
    ylabel="Mean Absolute Error Minutes (← is better)",
    baseline_value=mae_baseline_boot_metrics_dfs[task],
    legend_on=False,
)
task = "hospital_duration"
p7 = plot_metric_for_prompts_on_axes(
    data=mae_boot_mean_long_dfs[task],
    ax=ax[2, 0],
    title="Hospital Duration",
    ylabel="Mean Absolute Error Days (← is better)",
    baseline_value=mae_baseline_boot_metrics_dfs[task],
    legend_on=False,
)
task = "icu_duration"
p8 = plot_metric_for_prompts_on_axes(
    data=mae_boot_mean_long_dfs[task],
    ax=ax[2, 1],
    title="ICU Duration",
    ylabel="Mean Absolute Error Days (← is better)",
    baseline_value=mae_baseline_boot_metrics_dfs[task],
    legend_on=False,
)
# Turn off Axes in Bottom Right subplot
ax[2, 2].set_axis_off()
ax[2, 2].set_xticks([])
ax[2, 2].set_yticks([])
# Legend
gray_dash = mlines.Line2D(
    [],
    [],
    color="tab:gray",
    linestyle="--",
)
spacer = mlines.Line2D(
    [],
    [],
    color="white",
    linestyle=None,
)
handles = p1.get_legend_handles_labels()[0][:4]
labels = p1.get_legend_handles_labels()[1][:4]
handles = handles + [spacer, spacer, gray_dash]
labels = labels + ["", "", "Baseline"]
legend = ax[2, 2].legend(
    title="Notes Length Group",
    loc="center",
    handles=handles,
    labels=labels,
    frameon=False,
    title_fontsize=18,
    fontsize=16,
)

fig.show()
# %%
save_dir = Path.cwd() / "results" / "combined"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "task_prompt_performance.png"
fig.savefig(save_path)

# %%
