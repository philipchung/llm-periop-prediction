# ruff: noqa: E501
# %% [markdown]
# ### Visualization of Duration Predictions
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from llm.prompt_names import format_prompt_name, prompt_name_df
from llm_utils import read_pandas

pd.options.display.max_columns = None
pd.options.display.max_rows = 100

# Load Prediction Results
phi_save_dir = save_dir = Path.cwd() / "phi_results"
save_dir = Path.cwd() / "results" / "duration_predictions"
save_dir.mkdir(exist_ok=True, parents=True)

phase1_duration = read_pandas(phi_save_dir / "phase1_duration" / "predictions.feather")
hospital_duration = read_pandas(phi_save_dir / "hospital_duration" / "predictions.feather")
icu_duration = read_pandas(phi_save_dir / "icu_duration" / "predictions.feather")

# %%
# PACU Phase 1 Duration
targets = phase1_duration.Label
preds_df = phase1_duration.loc[:, prompt_name_df.Answer]
preds_df.columns = [format_prompt_name(x) for x in preds_df.columns]

fig, ax = plt.subplots(
    nrows=4, ncols=3, figsize=(10, 10), layout="constrained", sharex=True, sharey=True
)
for prompt_strategy, subplot_ax in zip(preds_df.columns, ax.flat):
    sns.scatterplot(x=targets, y=preds_df.loc[:, prompt_strategy], ax=subplot_ax)
    subplot_ax.set_title(prompt_strategy)
    subplot_ax.set_ylabel(None)
    subplot_ax.set_xlabel(None)
    subplot_ax.set_xlim([0, 400])
    subplot_ax.set_ylim([0, 400])
    subplot_ax.set_axisbelow(True)
    subplot_ax.grid(axis="both")
fig.suptitle("PACU Phase 1 Duration", fontsize=14)
fig.supxlabel("Actual (minutes)", fontsize=12)
fig.supylabel("Predicted (minutes)", fontsize=12)
fig.savefig(save_dir / "phase1_duration.png")
fig.show()

# %%
# Hospital Duration
targets = hospital_duration.Label
preds_df = hospital_duration.loc[:, prompt_name_df.Answer]
preds_df.columns = [format_prompt_name(x) for x in preds_df.columns]

fig, ax = plt.subplots(
    nrows=4, ncols=3, figsize=(10, 10), layout="constrained", sharex=True, sharey=True
)
for prompt_strategy, subplot_ax in zip(preds_df.columns, ax.flat):
    sns.scatterplot(x=targets, y=preds_df.loc[:, prompt_strategy], ax=subplot_ax)
    subplot_ax.set_title(prompt_strategy)
    subplot_ax.set_ylabel(None)
    subplot_ax.set_xlabel(None)
    subplot_ax.set_xlim([0, 60])
    subplot_ax.set_ylim([0, 60])
    subplot_ax.set_axisbelow(True)
    subplot_ax.grid(axis="both")
fig.suptitle("Hospital Duration", fontsize=14)
fig.supxlabel("Actual (days)", fontsize=12)
fig.supylabel("Predicted (days)", fontsize=12)
fig.savefig(save_dir / "hospital_duration.png")
fig.show()
# %%
# ICU Duration
targets = icu_duration.Label
preds_df = icu_duration.loc[:, prompt_name_df.Answer]
preds_df.columns = [format_prompt_name(x) for x in preds_df.columns]

fig, ax = plt.subplots(
    nrows=4, ncols=3, figsize=(10, 10), layout="constrained", sharex=True, sharey=True
)
for prompt_strategy, subplot_ax in zip(preds_df.columns, ax.flat):
    sns.scatterplot(x=targets, y=preds_df.loc[:, prompt_strategy], ax=subplot_ax)
    subplot_ax.set_title(prompt_strategy)
    subplot_ax.set_ylabel(None)
    subplot_ax.set_xlabel(None)
    subplot_ax.set_xlim([0, 30])
    subplot_ax.set_ylim([0, 30])
    subplot_ax.set_axisbelow(True)
    subplot_ax.grid(axis="both")
fig.suptitle("ICU Duration", fontsize=14)
fig.supxlabel("Actual (days)", fontsize=12)
fig.supylabel("Predicted (days)", fontsize=12)
fig.savefig(save_dir / "icu_duration.png")
fig.show()
# %%
