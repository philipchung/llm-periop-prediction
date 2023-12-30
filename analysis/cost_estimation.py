# %% [markdown]
# ## Cost Estimation
#
# Get a rough estimation of cost by summing up all the tokens for notes in each
# dataset.  This would be the cost of running a Zeroshot Q&A with notes context
# experiment on the whole dataset.
#
# Also computes estimate of duration to generate for each data split based on rate limits.
# %%

import pandas as pd

from llm.experiment import Experiment

pd.options.display.max_columns = None
pd.options.display.max_rows = 100

# GPT4Turbo-1106 & GPT35Turbo-1106 Input token costs
# (output costs are double price, but we typically have much more input than output tokens)
GPT4TURBO_COST_PER_TOKEN = 0.01 / 1000
GPT35TURBO_COST_PER_TOKEN = 0.001 / 1000
# Rate Limits
GPT4_TOKENS_PER_MINUTE = 80000
GPT35_TOKENS_PER_MINUTE = 1200000


print("# ASA-PS Prediction Task")
exp = Experiment(
    experiment_name="asa",
    task="asa",
    note_kind="last10",
)
inference_note_tokens = exp.inference_dataset.NoteTextTokenLength.apply(sum).sum()
fewshot_note_tokens = exp.fewshot_dataset.NoteTextTokenLength.apply(sum).sum()
print(f"Inference GPT4Turbo Cost: ${inference_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT4Turbo Cost: ${fewshot_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Inference GPT35Turbo Cost: ${inference_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT35Turbo Cost: ${fewshot_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(
    "Inference GPT4 Generation Duration: "
    f"{inference_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT4 Generation Duration: "
    f"{fewshot_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Inference GPT35 Generation Duration: "
    f"{inference_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT35 Generation Duration: "
    f"{fewshot_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print("-----")
# %%
print("# Phase 1 PACU Duration Prediction Task")
exp = Experiment(
    experiment_name="phase1_duration",
    task="phase1_duration",
    note_kind="last10",
)
inference_note_tokens = exp.inference_dataset.NoteTextTokenLength.apply(sum).sum()
fewshot_note_tokens = exp.fewshot_dataset.NoteTextTokenLength.apply(sum).sum()
print(f"Inference GPT4Turbo Cost: ${inference_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT4Turbo Cost: ${fewshot_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Inference GPT35Turbo Cost: ${inference_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT35Turbo Cost: ${fewshot_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(
    "Inference GPT4 Generation Duration: "
    f"{inference_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT4 Generation Duration: "
    f"{fewshot_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Inference GPT35 Generation Duration: "
    f"{inference_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT35 Generation Duration: "
    f"{fewshot_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print("-----")
# %%
print("# Hospital Duration Prediction Task")
exp = Experiment(
    experiment_name="hospital_duration",
    task="hospital_duration",
    note_kind="last10",
)
inference_note_tokens = exp.inference_dataset.NoteTextTokenLength.apply(sum).sum()
fewshot_note_tokens = exp.fewshot_dataset.NoteTextTokenLength.apply(sum).sum()
print(f"Inference GPT4Turbo Cost: ${inference_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT4Turbo Cost: ${fewshot_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Inference GPT35Turbo Cost: ${inference_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT35Turbo Cost: ${fewshot_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(
    "Inference GPT4 Generation Duration: "
    f"{inference_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT4 Generation Duration: "
    f"{fewshot_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Inference GPT35 Generation Duration: "
    f"{inference_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT35 Generation Duration: "
    f"{fewshot_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print("-----")
# %%
print("# Hospital Admission Prediction Task")
exp = Experiment(
    experiment_name="hospital_admission",
    task="hospital_admission",
    note_kind="last10",
)
inference_note_tokens = exp.inference_dataset.NoteTextTokenLength.apply(sum).sum()
fewshot_note_tokens = exp.fewshot_dataset.NoteTextTokenLength.apply(sum).sum()
print(f"Inference GPT4Turbo Cost: ${inference_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT4Turbo Cost: ${fewshot_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Inference GPT35Turbo Cost: ${inference_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT35Turbo Cost: ${fewshot_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(
    "Inference GPT4 Generation Duration: "
    f"{inference_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT4 Generation Duration: "
    f"{fewshot_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Inference GPT35 Generation Duration: "
    f"{inference_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT35 Generation Duration: "
    f"{fewshot_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print("-----")
# %%
print("# ICU Duration Prediction Task")
exp = Experiment(
    experiment_name="icu_duration",
    task="icu_duration",
    note_kind="last10",
)
inference_note_tokens = exp.inference_dataset.NoteTextTokenLength.apply(sum).sum()
fewshot_note_tokens = exp.fewshot_dataset.NoteTextTokenLength.apply(sum).sum()
print(f"Inference GPT4Turbo Cost: ${inference_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT4Turbo Cost: ${fewshot_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Inference GPT35Turbo Cost: ${inference_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT35Turbo Cost: ${fewshot_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(
    "Inference GPT4 Generation Duration: "
    f"{inference_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT4 Generation Duration: "
    f"{fewshot_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Inference GPT35 Generation Duration: "
    f"{inference_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT35 Generation Duration: "
    f"{fewshot_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print("-----")
# %%
print("# ICU Admission Prediction Task")
exp = Experiment(
    experiment_name="icu_admission",
    task="icu_admission",
    note_kind="last10",
)
inference_note_tokens = exp.inference_dataset.NoteTextTokenLength.apply(sum).sum()
fewshot_note_tokens = exp.fewshot_dataset.NoteTextTokenLength.apply(sum).sum()
print(f"Inference GPT4Turbo Cost: ${inference_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT4Turbo Cost: ${fewshot_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Inference GPT35Turbo Cost: ${inference_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT35Turbo Cost: ${fewshot_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(
    "Inference GPT4 Generation Duration: "
    f"{inference_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT4 Generation Duration: "
    f"{fewshot_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Inference GPT35 Generation Duration: "
    f"{inference_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT35 Generation Duration: "
    f"{fewshot_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print("-----")
# %%
print("# Unplanned Admission Prediction Task")
exp = Experiment(
    experiment_name="unplanned_admit",
    task="unplanned_admit",
    note_kind="last10",
)
inference_note_tokens = exp.inference_dataset.NoteTextTokenLength.apply(sum).sum()
fewshot_note_tokens = exp.fewshot_dataset.NoteTextTokenLength.apply(sum).sum()
print(f"Inference GPT4Turbo Cost: ${inference_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT4Turbo Cost: ${fewshot_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Inference GPT35Turbo Cost: ${inference_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT35Turbo Cost: ${fewshot_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(
    "Inference GPT4 Generation Duration: "
    f"{inference_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT4 Generation Duration: "
    f"{fewshot_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Inference GPT35 Generation Duration: "
    f"{inference_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT35 Generation Duration: "
    f"{fewshot_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print("-----")
# %%
print("# Hospital Mortality Prediction Task")
exp = Experiment(
    experiment_name="hospital_mortality",
    task="hospital_mortality",
    note_kind="last10",
)
inference_note_tokens = exp.inference_dataset.NoteTextTokenLength.apply(sum).sum()
fewshot_note_tokens = exp.fewshot_dataset.NoteTextTokenLength.apply(sum).sum()
print(f"Inference GPT4Turbo Cost: ${inference_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT4Turbo Cost: ${fewshot_note_tokens * GPT4TURBO_COST_PER_TOKEN:.2f}")
print(f"Inference GPT35Turbo Cost: ${inference_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(f"Fewshot GPT35Turbo Cost: ${fewshot_note_tokens * GPT35TURBO_COST_PER_TOKEN:.2f}")
print(
    "Inference GPT4 Generation Duration: "
    f"{inference_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT4 Generation Duration: "
    f"{fewshot_note_tokens / GPT4_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Inference GPT35 Generation Duration: "
    f"{inference_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print(
    "Fewshot GPT35 Generation Duration: "
    f"{fewshot_note_tokens / GPT35_TOKENS_PER_MINUTE:.2f} minutes"
)
print("-----")
# %%
