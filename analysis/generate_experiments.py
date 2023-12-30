# %% [markdown]
# ## Experiments
#
# This notebook contains individual LLM generation experiments
# %%
import pandas as pd

from llm.chat_model import ChatModel
from llm.experiment import Experiment

pd.options.display.max_columns = None
pd.options.display.max_rows = 100

# %%
# python notebooks/generate.py --experiment-name=asa --task=asa --num-fewshot=5 --num-fewshot=10 --num-fewshot=20 --num-concurrent=7 --model=gpt-4-1106
print("ASA-PS Prediction Task")
exp = Experiment(
    experiment_name="asa",
    task="asa",
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    # Optional config to constrain inference dataset size
    # num_inference_examples=250,
    num_concurrent=7,
)
num_fewshot = [5, 10, 20]

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

asa_df = exp.inference_dataset
# %%
print("Phase 1 PACU Duration Prediction Task")
exp = Experiment(
    experiment_name="phase1_duration",
    task="phase1_duration",
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    num_concurrent=5,
)
num_fewshot = [20]

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

phase1_duration_df = exp.inference_dataset
# %%
print("Hospital Duration Prediction Task")
exp = Experiment(
    experiment_name="hospital_duration",
    task="hospital_duration",
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    num_concurrent=5,
)
num_fewshot = [20]

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

hospital_duration_df = exp.inference_dataset
# %%
print("Hospital Admission Prediction Task")
exp = Experiment(
    experiment_name="hospital_admission",
    task="hospital_admission",
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    num_concurrent=5,
)
num_fewshot = [20]

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

hospital_admission_df = exp.inference_dataset
# %%
print("ICU Duration Prediction Task")
exp = Experiment(
    experiment_name="icu_duration",
    task="icu_duration",
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    num_concurrent=5,
)
num_fewshot = [20]

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

icu_duration_df = exp.inference_dataset
# %%
print("ICU Admission Prediction Task")
exp = Experiment(
    experiment_name="icu_admission",
    task="icu_admission",
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    num_concurrent=5,
)
num_fewshot = [20]

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

icu_admission_df = exp.inference_dataset
# %%
print("Unplanned Admission Prediction Task")
exp = Experiment(
    experiment_name="unplanned_admit",
    task="unplanned_admit",
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    num_concurrent=5,
)
num_fewshot = [20]

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

unplanned_admit_df = exp.inference_dataset
# %%
print("Hospital Mortality Prediction Task")
exp = Experiment(
    experiment_name="hospital_mortality",
    task="hospital_mortality",
    note_kind="last10",
    chat_model=ChatModel(model="gpt-4-1106"),
    num_concurrent=5,
)
num_fewshot = [20]

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

hospital_mortality_df = exp.inference_dataset
# %%
