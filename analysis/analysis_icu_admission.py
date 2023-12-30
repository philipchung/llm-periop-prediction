# ruff: noqa: E501
# %% [markdown]
# ### Analysis of Experiment: ICU Admission
#
# Experiment Run:
# python scripts/generate.py --experiment-name=icu_admission --task=icu_admission --num-fewshot=5 --num-fewshot=10 --num-fewshot=20 --num-fewshot=50 --num-concurrent=7 --model=gpt-4-1106
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llm.chat_model import ChatModel
from llm.experiment import Experiment
from llm.metrics import ClassificationMetrics, dummy_classifier_metrics
from llm.plot import plot_metric_for_prompts
from llm.prompt_names import format_prompt_name, prompt_name_df
from llm.statistics import pairwise_mannwhitneyu, pairwise_wilcoxon
from llm_utils import ProgressBar, read_pandas, save_pandas

pd.options.display.max_columns = None
pd.options.display.max_rows = 100

# Prediction Task
task = "icu_admission"
save_dir = Path.cwd() / "results" / task
boot_save_dir = save_dir / "boot_metrics"
plots_save_dir = save_dir / "plots"
phi_save_dir = Path.cwd() / "phi_results" / task
phi_inputs_save_dir = phi_save_dir / "inputs"
phi_outputs_save_dir = phi_save_dir / "outputs"
boot_save_dir.mkdir(exist_ok=True, parents=True)
plots_save_dir.mkdir(exist_ok=True, parents=True)
phi_inputs_save_dir.mkdir(exist_ok=True, parents=True)
phi_outputs_save_dir.mkdir(exist_ok=True, parents=True)

exp = Experiment(
    experiment_name=task,
    task=task,
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    # Optional config
    num_concurrent=7,
)
num_fewshot = [5, 10, 20, 50]
# %%
try:
    exp_df = read_pandas(path=phi_save_dir / "predictions.feather")
    inputs_dict = {}
    outputs_dict = {}
    dict_keys = [
        "inference_summary",
        "fewshot_summary",
        "fewshot_cot_generation",
        "zeroshot_qa_from_notes",
        "zeroshot_qa_from_notes_summary",
        "fewshot_qa_from_notes_summary",
        "zeroshot_cot_qa_from_notes",
        "zeroshot_cot_qa_from_notes_summary",
        "fewshot_cot_qa_from_notes_summary",
    ]
    with ProgressBar() as p:
        # t1 = p.add_task("t1")
        # for key in p.track(dict_keys, task_id=t1):
        #     p.update(t1, description=f"Inputs: {key}")
        #     inputs_dict |= {
        #         key: read_pandas(path=phi_inputs_save_dir / f"{key}.feather", set_index="index")
        #     }
        t2 = p.add_task("t2")
        for key in p.track(dict_keys, task_id=t2):
            p.update(t2, description=f"Outputs: {key}")
            outputs_dict |= {
                key: read_pandas(path=phi_outputs_save_dir / f"{key}.feather", set_index="index")
            }

except Exception:
    # Inference Summaries
    print("Generate Inference Summaries")
    exp.generate_notes_summary_for_inference_dataset()
    # Fewshot Summaries & CoT Rationale Generation
    print("Generate Fewshot Summaries")
    exp.generate_notes_summary_for_fewshot_dataset()
    print("Generate Fewshot CoT Rationales")
    exp.generate_cot_rationale_for_fewshot_dataset()

    # Zeroshot Q&A
    print("Generate Zeroshot Q&A from Notes")
    exp.generate_zeroshot_qa_from_notes_for_inference_dataset()
    print("Generate Zeroshot Q&A from Summaries")
    exp.generate_zeroshot_qa_from_notes_summary_for_inference_dataset()

    # Zeroshot CoT Q&A
    print("Generate Zeroshot CoT Q&A from Notes")
    exp.generate_zeroshot_cot_qa_from_notes_for_inference_dataset()
    print("Generate Zeroshot CoT Q&A from Summaries")
    exp.generate_zeroshot_cot_qa_from_notes_summary_for_inference_dataset()

    # Fewshot & Fewshot CoT Q&A
    if isinstance(num_fewshot, int):
        num_fewshot = [num_fewshot]
    for n in num_fewshot:
        print(f"Generate {n}-shot Q&A from Summaries")
        exp.generate_fewshot_qa_from_notes_summary_for_inference_dataset(num_fewshot=n)
        print(f"Generate {n}-shot CoT Q&A from Summaries")
        exp.generate_fewshot_cot_qa_from_notes_summary_for_inference_dataset(num_fewshot=n)

    exp_df = exp.inference_dataset
    inputs_dict = exp.inputs
    outputs_dict = exp.outputs
    save_pandas(df=exp_df, path=phi_save_dir / "predictions.feather")
    for key, value in exp.inputs.items():
        save_pandas(df=value, path=phi_inputs_save_dir / f"{key}.feather")
    for key, value in exp.outputs.items():
        save_pandas(df=value, path=phi_outputs_save_dir / f"{key}.feather")

# %%
# Compute Cost for GPT-4-turbo based on actual prompt & completion tokens used
GPT4TURBO_PROMPT_TOKEN_COST = 0.01 / 1000
GPT4TURBO_COMPLETION_TOKEN_COST = 0.03 / 1000


def get_token_usage(response_dict: dict) -> dict:
    return response_dict["usage"]


def cost_for_prompt(df: pd.DataFrame) -> float:
    usage = df.response_dict.apply(get_token_usage)
    usage = pd.DataFrame(usage.tolist())
    prompt_tokens = usage.prompt_tokens.sum()
    completion_tokens = usage.completion_tokens.sum()
    total_cost = (
        GPT4TURBO_PROMPT_TOKEN_COST * prompt_tokens
        + GPT4TURBO_COMPLETION_TOKEN_COST * completion_tokens
    )
    return total_cost


cost_for_prompts = [cost_for_prompt(x) for x in outputs_dict.values()]
total_experiment_cost = sum(cost_for_prompts)
print(f"Total Experiment Cost: ${total_experiment_cost:.2f}")

# %%
# Stratify examples by Length of Notes (all of the last10 concatenated)
total_note_token_length = exp_df.NoteTextTokenLength.apply(sum)
groups, bins = pd.qcut(
    total_note_token_length, q=3, labels=["short", "medium", "long"], retbins=True
)
exp_df = exp_df.assign(NoteLengthGroup=groups)

# Plot Distribution
fig = plt.figure()
ax = total_note_token_length.plot(
    kind="kde", title="Note Length Distribution (all 10 notes concatenated)"
)
ax.set_xlabel("Token Length")
for x in bins:
    plt.axvline(x=x, color="gray", linestyle="--")
fig.savefig(plots_save_dir / "note_length_distribution.png")

# %% [markdown]
# ## Handle Null LLM Outputs
#
# Experiment code will automatically have LLM retry 5 times to perform task.
# However, there may be some scenarios where LLM refuses to answer (often appropriately).
# In these cases, a `None` answer is recorded.  We identify examples where this has
# occurred for any of the experimental conditions and we separate them.  These are a
# small minority of examples.
# %%
# Get examples which have a null answer in any of the answer columns
answers_df = exp_df.loc[:, prompt_name_df.Answer]
null_answers = answers_df.isna().any(axis=1)
# Split examples
exp_null_df = exp_df.loc[null_answers]
exp_nonnull_df = exp_df.loc[~null_answers]
print(f"Examples with null answer in any prompt: {len(exp_null_df)}")
print(f"Examples with non-null answer in any prompt: {len(exp_nonnull_df)}")
# NOTE: examples in `exp_null_df` are not included in metrics computation.
# %% [markdown]
# ## Compute Metrics
#
# We compute metrics for each length group for each prompt strategy and cache result
# on disk. Metrics are computed using bootstrap to estimate the mean and confidence interval.
# %%
# Compute metrics w/ bootstrap, stratified by Length Group
length_groups = ["short", "medium", "long", "all"]
metrics = {"short": {}, "medium": {}, "long": {}, "all": {}}
class_labels = [True, False]
with ProgressBar() as p:
    t1 = p.add_task("t1")
    t2 = p.add_task("t2")
    for length_group in p.track(length_groups, task_id=t1):
        p.update(t1, description=f"Length Group: {length_group}")
        for col in p.track(
            prompt_name_df.Answer,
            task_id=t2,
        ):
            p.update(t2, description=f"Metrics: {col}")

            try:
                # Try to load cached result
                boot_metrics_df = read_pandas(
                    path=boot_save_dir / f"{col}-{length_group}.feather", set_index="Metric"
                )
            except Exception:
                # Compute bootstrap metrics
                if length_group == "all":
                    df = exp_nonnull_df
                else:
                    df = exp_nonnull_df.query(f"NoteLengthGroup == '{length_group}'")
                preds = df[col]
                targets = df.Label
                boot_metrics_df = ClassificationMetrics(
                    preds=preds,
                    targets=targets,
                    class_labels=class_labels,
                    name=col,
                ).compute()
                save_pandas(boot_metrics_df, path=boot_save_dir / f"{col}-{length_group}.feather")
            metrics[length_group] |= {col: boot_metrics_df}

# Convert Dict of Dict into Dataframe where each cell is a MeanCI object
# rows = length groups, columns = experimental conditions
metrics = pd.DataFrame(metrics)
# %%
# Compute Baseline Metrics from Dummy Classifier
df = exp_nonnull_df
targets = df.Label
baseline_boot_metrics = dummy_classifier_metrics(targets=targets, strategy="uniform")
save_pandas(df=baseline_boot_metrics, path=save_dir / "baseline_boot_metrics.csv")
# %% [markdown]
# ## Test for Statistical Significances
#
# We do 2 kinds of pairwise comparisons:

# 1. Compare pairwise different length groups (short, medium, long) within each prompt
#   strategy. MannWhitneyU test w/ false discovery rate correction since the stratified
#   examples are independent from one another.
#   This answers the question: For same prompt, are there differences in performance
#   for patient cases where we have shorter/smaller number of notes vs. more/longer notes.
# 2. Compare pairwise each prompt strategy (using "all" examples). Wilcoxon rank-sum
#   test w/ false discovery rate correction since all prompt strategy experimental conditions
#   use the same examples.
#   This answers the question: For different prompts, are there differences in performance.
# %% [markdown]
# ### F1
# %%
metric_name = "Aggregate/F1/Micro"
metric_boot_values = metrics.applymap(lambda df: df.loc[metric_name, "boot_values"])

# 1. Pairwise Comparison for different length groups within each prompt
pval_dfs = []
for prompt in prompt_name_df.Answer:
    # Get Length Groups for prompt
    df = metric_boot_values.loc[prompt, ["short", "medium", "long"]].to_frame().T
    # Expand series of list into dataframe
    df2 = df.explode(column=df.columns.tolist())
    # Compute P-Values
    pval_df = pairwise_mannwhitneyu(df2, correction_method="fdr_bh")
    pval_df = pval_df.assign(Prompt=format_prompt_name(prompt))
    pval_dfs += [pval_df]
# P-Values comparing Length Groups within each Prompt condition
length_group_comparisons = pd.concat(pval_dfs, axis=0).set_index(["Prompt", "Prompt1", "Prompt2"])
length_group_comparisons.to_csv(save_dir / "f1_length_group_pvalues.csv")
length_group_comparisons
# %%
# 2. Pairwise comparison for each prompt, using "all" examples
df = metric_boot_values.loc[:, "all"].to_frame().T
df.columns = [format_prompt_name(x) for x in df.columns]
# Expand series of list into dataframe
df2 = df.explode(column=df.columns.tolist())
# Compute P-Values
wilcoxon_df = pairwise_wilcoxon(df2, correction_method="fdr_bh")
prompt_comparisons = wilcoxon_df.set_index(["Prompt1", "Prompt2"])
prompt_comparisons.to_csv(save_dir / "f1_prompt_pvalues.csv")
prompt_comparisons
# %% [markdown]
# ### Matthew's Correlation Coefficient (MCC)
# %%
metric_name = "Aggregate/MCC/MCC"
metric_boot_values = metrics.applymap(lambda df: df.loc[metric_name, "boot_values"])

# 1. Pairwise Comparison for different length groups within each prompt
pval_dfs = []
for prompt in prompt_name_df.Answer:
    # Get Length Groups for prompt
    df = metric_boot_values.loc[prompt, ["short", "medium", "long"]].to_frame().T
    # Expand series of list into dataframe
    df2 = df.explode(column=df.columns.tolist())
    # Compute P-Values
    pval_df = pairwise_mannwhitneyu(df2, correction_method="fdr_bh")
    pval_df = pval_df.assign(Prompt=format_prompt_name(prompt))
    pval_dfs += [pval_df]
# P-Values comparing Length Groups within each Prompt condition
length_group_comparisons = pd.concat(pval_dfs, axis=0).set_index(["Prompt", "Prompt1", "Prompt2"])
length_group_comparisons.to_csv(save_dir / "mcc_length_group_pvalues.csv")
length_group_comparisons
# %%
# 2. Pairwise comparison for each prompt, using "all" examples
df = metric_boot_values.loc[:, "all"].to_frame().T
df.columns = [format_prompt_name(x) for x in df.columns]
# Expand series of list into dataframe
df2 = df.explode(column=df.columns.tolist())
# Compute P-Values
wilcoxon_df = pairwise_wilcoxon(df2, correction_method="fdr_bh")
prompt_comparisons = wilcoxon_df.set_index(["Prompt1", "Prompt2"])
prompt_comparisons.to_csv(save_dir / "mcc_prompt_pvalues.csv")
prompt_comparisons
# %% [markdown]
# ## Plot Metrics for all prompts and length groups
# %% [markdown]
# ### F1
# %%
# Get specific metric and all bootstrap values for that metric
metric_name = "Aggregate/F1/Micro"
# Bootstrap metric values, the mean and lower+upper CI of those values
boot_mean = metrics.applymap(lambda df: df.loc[metric_name, "mean_value"])
boot_mean.index = [format_prompt_name(x) for x in boot_mean.index]
boot_mean_long = boot_mean.reset_index(names="prompt").melt(
    id_vars="prompt", var_name="length_group", value_name="value"
)
baseline_value = baseline_boot_metrics.loc[metric_name, "mean_value"]

fig, ax = plot_metric_for_prompts(
    data=boot_mean_long,
    qa_formatted_prompt_names=prompt_name_df.query("Type == 'Non-CoT'").Formatted.tolist(),
    cot_qa_formatted_prompt_names=prompt_name_df.query("Type == 'CoT'").Formatted.tolist(),
    title="ICU Admission",
    ylabel="F1",
    baseline_value=baseline_value,
)
save_path = plots_save_dir / "f1.png"
fig.savefig(save_path.as_posix())
fig.show()

# Save data required to generate plot
save_pandas(df=boot_mean_long, path=save_dir / "f1_boot_mean_long.csv")
# %% [markdown]
# ### Matthew's Correlation Coefficient (MCC)
# %%
# Get specific metric and all bootstrap values for that metric
metric_name = "Aggregate/MCC/MCC"
boot_mean = metrics.applymap(lambda df: df.loc[metric_name, "mean_value"])
boot_mean.index = [format_prompt_name(x) for x in boot_mean.index]
boot_mean_long = boot_mean.reset_index(names="prompt").melt(
    id_vars="prompt", var_name="length_group", value_name="value"
)
baseline_value = baseline_boot_metrics.loc[metric_name, "mean_value"]

fig, ax = plot_metric_for_prompts(
    data=boot_mean_long,
    qa_formatted_prompt_names=prompt_name_df.query("Type == 'Non-CoT'").Formatted.tolist(),
    cot_qa_formatted_prompt_names=prompt_name_df.query("Type == 'CoT'").Formatted.tolist(),
    title="ICU Admission",
    ylabel="Matthew's Correlation Coefficient (MCC)",
    baseline_value=baseline_value,
)
save_path = plots_save_dir / "mcc.png"
fig.savefig(save_path.as_posix())
fig.show()

# Save data required to generate plot
save_pandas(df=boot_mean_long, path=save_dir / "mcc_boot_mean_long.csv")
# %% [markdown]
# ## Table with Mean Metric and Confidence Intervals for each Prompt Strategy
# %%
# Mean & CI for every metric & experimental condition


def format_mean_ci(row: pd.Series) -> str:
    return f"{row.mean_value:.2f} ({row.lower_ci:.2f}, {row.upper_ci:.2f})"


def mean_ci_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(format_mean_ci, axis="columns").to_dict()


mean_ci_df = metrics.applymap(mean_ci_all_metrics)
mean_ci_df.index = [format_prompt_name(x) for x in mean_ci_df.index]
# Add baseline
baseline_mean_ci = (
    pd.Series(
        {
            "all": baseline_boot_metrics.apply(format_mean_ci, axis="columns").to_dict(),
            "long": "--",
            "medium": "--",
            "short": "--",
        },
        name="Baseline",
    )
    .to_frame()
    .T
)
mean_ci_df = pd.concat([baseline_mean_ci, mean_ci_df], axis=0)
mean_ci_df
# %%
mean_ci_for_all_prompts_and_stratifications = pd.DataFrame(mean_ci_df.stack().to_dict())
mean_ci_for_all_prompts_and_stratifications.to_csv(
    save_dir / "mean_ci_all_prompts_and_stratifications.csv"
)
mean_ci_for_all_prompts_and_stratifications
# %%
# Compare Mean (CI) for each prompt strategy ("all" only)
mean_ci_comparing_prompts = mean_ci_for_all_prompts_and_stratifications.loc[
    :, (slice(None), "all")
].droplevel(level=1, axis="columns")
mean_ci_comparing_prompts.to_csv(save_dir / "mean_ci_all_prompts.csv")
mean_ci_comparing_prompts
# %%
# Look at only MCC Metric
metric_name = "Aggregate/MCC/MCC"
mcc = mean_ci_for_all_prompts_and_stratifications.loc[metric_name].unstack(-1)
mcc = mcc.loc[["Baseline"] + prompt_name_df.Formatted.tolist()]
mcc.to_csv(save_dir / "mcc.csv")
mcc
# %%
# Look at only F1 Metric
metric_name = "Aggregate/F1/Micro"
f1 = mean_ci_for_all_prompts_and_stratifications.loc[metric_name].unstack(-1)
f1 = f1.loc[["Baseline"] + prompt_name_df.Formatted.tolist()]
f1.to_csv(save_dir / "f1.csv")
f1
# %%
