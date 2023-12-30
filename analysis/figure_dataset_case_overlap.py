# %% [markdown]
# ## Explore how much case overlap exists in each of the 7 datasets
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from llm_utils import DataPaths
from make_dataset.dataset import DatasetBuilder

pd.set_option("display.max_columns", None)

# Define Raw Data Paths
paths = DataPaths(project_dir=Path(__file__).parent.parent, data_version=7)

# Load Raw Data
paths.register(name="adt_raw", path=paths.raw / "2023-05-11_ADT.feather")
paths.register(name="case_raw", path=paths.raw / "2023-05-11_Case.feather")
paths.register(name="preanes_notes_raw", path=paths.raw / "2023-05-02_PreAnesNotes.feather")
paths.register(name="last10_notes_raw", path=paths.raw / "2023-05-02_LastTenNotes.feather")
paths.register(name="age_gender", path=paths.raw / "2023-10-01_AgeGender.csv")
paths.register(name="age_sex", path=paths.raw / "2023-10-03_AgeSex.csv")

builder = DatasetBuilder(paths=paths, dataset_version="dataset4")
# %%
# Load Processed Datasets
# NOTE: same cases used for `preanes` and `last10` note types, so we can just
# look at one of them
## ASA
asa_last10_inference = builder.get_dataset(
    dataset_type="asa", note_type="last10", split="inference"
)
asa_last10_fewshot = builder.get_dataset(dataset_type="asa", note_type="last10", split="fewshot")
## Phase 1 Duration
phase1_last10_inference = builder.get_dataset(
    dataset_type="phase1_duration", note_type="last10", split="inference"
)
phase1_last10_fewshot = builder.get_dataset(
    dataset_type="phase1_duration", note_type="last10", split="fewshot"
)
## Hospital Duration
hospitalduration_last10_inference = builder.get_dataset(
    dataset_type="hospital_duration", note_type="last10", split="inference"
)
hospitalduration_last10_fewshot = builder.get_dataset(
    dataset_type="hospital_duration", note_type="last10", split="fewshot"
)
## Hospital Duration
hospitaladmission_last10_inference = builder.get_dataset(
    dataset_type="hospital_admission", note_type="last10", split="inference"
)
hospitaladmission_last10_fewshot = builder.get_dataset(
    dataset_type="hospital_admission", note_type="last10", split="fewshot"
)
## ICU Duration
icuduration_last10_inference = builder.get_dataset(
    dataset_type="icu_duration", note_type="last10", split="inference"
)
icuduration_last10_fewshot = builder.get_dataset(
    dataset_type="icu_duration", note_type="last10", split="fewshot"
)
## ICU Duration
icuadmission_last10_inference = builder.get_dataset(
    dataset_type="icu_admission", note_type="last10", split="inference"
)
icuadmission_last10_fewshot = builder.get_dataset(
    dataset_type="icu_admission", note_type="last10", split="fewshot"
)
## Unplanned Admit
unplannedadmit_last10_inference = builder.get_dataset(
    dataset_type="unplanned_admit", note_type="last10", split="inference"
)
unplannedadmit_last10_fewshot = builder.get_dataset(
    dataset_type="unplanned_admit", note_type="last10", split="fewshot"
)
## Hospital Mortality
hospitalmortality_last10_inference = builder.get_dataset(
    dataset_type="hospital_mortality", note_type="last10", split="inference"
)
hospitalmortality_last10_fewshot = builder.get_dataset(
    dataset_type="hospital_mortality", note_type="last10", split="fewshot"
)
# %%
# Overlap of Inference Datasets
asa_last10_inference
# %%
inference_datasets = {
    "asa": asa_last10_inference,
    "pacu": phase1_last10_inference,
    "hospital_admission": hospitaladmission_last10_inference,
    "hospital_duration": hospitalduration_last10_inference,
    "icu_admission": icuadmission_last10_inference,
    "icu_duration": icuduration_last10_inference,
    "unplanned_admit": unplannedadmit_last10_inference,
    "hospital_mortality": hospitalmortality_last10_inference,
}
inference_proc_ids = {k: set(v.index) for k, v in inference_datasets.items()}

fewshot_datasets = {
    "asa": asa_last10_fewshot,
    "pacu": phase1_last10_fewshot,
    "hospital_admission": hospitaladmission_last10_fewshot,
    "hospital_duration": hospitalduration_last10_fewshot,
    "icu_admission": icuadmission_last10_fewshot,
    "icu_duration": icuduration_last10_fewshot,
    "unplanned_admit": unplannedadmit_last10_fewshot,
    "hospital_mortality": hospitalmortality_last10_fewshot,
}
fewshot_proc_ids = {k: set(v.index) for k, v in fewshot_datasets.items()}


# %%
def get_overlap_count_matrix(proc_ids_dict: dict[str, set]) -> pd.DataFrame:
    # Initialize
    ds_names = proc_ids_dict.keys()
    overlap_cts_df = pd.DataFrame(
        np.zeros((len(ds_names), len(ds_names))).astype(int),
        index=ds_names,
        columns=ds_names,
    )
    # Create Count Matrix
    for ds_name1, ds1_proc_ids in proc_ids_dict.items():
        for ds_name2, ds2_proc_ids in proc_ids_dict.items():
            num_overlap = len(ds1_proc_ids & ds2_proc_ids)
            overlap_cts_df.at[ds_name1, ds_name2] = num_overlap
    return overlap_cts_df


inference_ds_overlap_cts = get_overlap_count_matrix(inference_proc_ids)
fewshot_ds_overlap_cts = get_overlap_count_matrix(fewshot_proc_ids)

dataset_names = [
    "ASA Physical Status",
    "PACU Phase 1 Duration",
    "Hospital Admission",
    "Hospital Duration",
    "ICU Admission",
    "ICU Duration",
    "Unplanned Admission",
    "Hospital Mortality",
]
inference_ds_overlap_cts.index = dataset_names
inference_ds_overlap_cts.columns = dataset_names
fewshot_ds_overlap_cts.index = dataset_names
fewshot_ds_overlap_cts.columns = dataset_names

# %%
# Save Tables
save_dir = Path.cwd() / "results" / "dataset_case_overlap"
save_dir.mkdir(parents=True, exist_ok=True)

inference_ds_overlap_cts.to_csv(save_dir / "inference_case_overlap.csv")
fewshot_ds_overlap_cts.to_csv(save_dir / "fewshot_case_overlap.csv")

# %%


def make_count_matrix_figure(count_matrix: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    # Get Upper Triangle of the matrix as a mask
    mask = np.triu(np.ones_like(count_matrix))
    np.fill_diagonal(mask, False)
    # Create Heatmap, removing upper triangular portion
    ax = sns.heatmap(count_matrix, annot=True, mask=mask, cmap="Blues", cbar=False, fmt="g", ax=ax)
    return ax


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), layout="constrained")
inference_ct_ax = make_count_matrix_figure(inference_ds_overlap_cts, ax=ax[0])
inference_ct_ax.set(title="Inference Datasets Case Overlap")
fewshot_ct_ax = make_count_matrix_figure(fewshot_ds_overlap_cts, ax=ax[1])
fewshot_ct_ax.set(title="Fewshot Datasets Case Overlap")

fig.savefig(save_dir / "dataset_case_overlap.png")

# %%
