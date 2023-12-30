# ruff: noqa: E501
# %% [markdown]
# ### Visualize Different Prompt Strategies
# %%
from pathlib import Path

import pandas as pd

from llm_utils import ProgressBar, read_pandas

pd.options.display.max_columns = None
pd.options.display.max_rows = 100

# %%
# Load Data: Input Prompt Messages, LLM Generations, Prediction Results
task = "asa"
phi_save_dir = save_dir = Path.cwd() / "phi_results" / task
phi_inputs_save_dir = phi_save_dir / "inputs"
phi_outputs_save_dir = phi_save_dir / "outputs"
save_dir = Path.cwd() / "results" / "duration_predictions"
save_dir.mkdir(exist_ok=True, parents=True)

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
    t1 = p.add_task("t1")
    for key in p.track(dict_keys, task_id=t1):
        p.update(t1, description=f"Inputs: {key}")
        inputs_dict |= {
            key: read_pandas(path=phi_inputs_save_dir / f"{key}.feather", set_index="index")
        }
    t2 = p.add_task("t2")
    for key in p.track(dict_keys, task_id=t2):
        p.update(t2, description=f"Outputs: {key}")
        outputs_dict |= {
            key: read_pandas(path=phi_outputs_save_dir / f"{key}.feather", set_index="index")
        }
# %%
# ASA Dataset, Select Specific Case
case_idx = 40
case_predictions = exp_df.loc[case_idx, :]

# Separate Fewshot & Inference Datasets; Create Input & Output DataFrames for Case
fewshot_var_names = ["fewshot_summary", "fewshot_cot_generation"]
inference_var_names = [
    "inference_summary",
    "zeroshot_qa_from_notes",
    "zeroshot_qa_from_notes_summary",
    "fewshot_qa_from_notes_summary",
    "zeroshot_cot_qa_from_notes",
    "zeroshot_cot_qa_from_notes_summary",
    "fewshot_cot_qa_from_notes_summary",
]
fewshot_inputs_dict = {k: v for k, v in inputs_dict.items() if k in fewshot_var_names}
inference_inputs_dict = {k: v for k, v in inputs_dict.items() if k in inference_var_names}
fewshot_outputs_dict = {k: v for k, v in outputs_dict.items() if k in fewshot_var_names}
inference_outputs_dict = {k: v for k, v in outputs_dict.items() if k in inference_var_names}

# Get Inference inputs/outputs for specific case as DataFrame
case_inference_inputs = {k: v.loc[case_idx, :] for k, v, in inference_inputs_dict.items()}
case_inference_inputs = pd.DataFrame(case_inference_inputs)
case_inference_outputs = {k: v.loc[case_idx, :] for k, v, in inference_outputs_dict.items()}
case_inference_outputs = pd.DataFrame(case_inference_outputs)
# Get Fewshot inputs/outputs for specific case as DataFrame
case_fewshot_inputs = {k: v.loc[case_idx, :] for k, v, in fewshot_inputs_dict.items()}
case_fewshot_inputs = pd.DataFrame(case_fewshot_inputs)
case_fewshot_outputs = {k: v.loc[case_idx, :] for k, v, in fewshot_outputs_dict.items()}
case_fewshot_outputs = pd.DataFrame(case_fewshot_outputs)
# %%
# System Message
prompt_strategy = "zeroshot_qa_from_notes"
system_message = case_inference_outputs.loc["system_message", prompt_strategy]
print("---")
print("System Message:")
print(system_message)


# %%
def print_user_and_response_message(df: pd.DataFrame, prompt_strategy: str) -> None:
    user_message = df.loc["user_message", prompt_strategy]
    response_message = df.loc["response_message", prompt_strategy]
    print(f"---({prompt_strategy})---")
    print("User Message:")
    print(user_message)
    print("---")
    print("Response Message:")
    print(response_message)


# %%
# 0-shot | Notes
prompt_strategy = "zeroshot_qa_from_notes"
print_user_and_response_message(df=case_inference_outputs, prompt_strategy=prompt_strategy)
# %%
# 0-shot | Summary
prompt_strategy = "zeroshot_qa_from_notes_summary"
print_user_and_response_message(df=case_inference_outputs, prompt_strategy=prompt_strategy)
# %%
# 10-shot | Summary
prompt_strategy = "fewshot_qa_from_notes_summary"
print_user_and_response_message(df=case_inference_outputs, prompt_strategy=prompt_strategy)
# %%
# 0-shot CoT | Notes
prompt_strategy = "zeroshot_cot_qa_from_notes"
print_user_and_response_message(df=case_inference_outputs, prompt_strategy=prompt_strategy)
# %%
# 0-shot CoT | Summary
prompt_strategy = "zeroshot_cot_qa_from_notes_summary"
print_user_and_response_message(df=case_inference_outputs, prompt_strategy=prompt_strategy)
# %%
# 10-shot | Summary
prompt_strategy = "fewshot_cot_qa_from_notes_summary"
print_user_and_response_message(df=case_inference_outputs, prompt_strategy=prompt_strategy)
# %%
# Summarization Generation
prompt_strategy = "inference_summary"
print_user_and_response_message(df=case_inference_outputs, prompt_strategy=prompt_strategy)
# %%
# CoT Rationale Generation
prompt_strategy = "fewshot_cot_generation"
print_user_and_response_message(df=case_fewshot_outputs, prompt_strategy=prompt_strategy)
# %%
