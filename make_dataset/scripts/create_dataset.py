# %% [markdown]
## Create Dataset
#
# This script generates 8 different datasets for the following QA Tasks:
# * `asa`
# * `phase1_duration`
# * `hospital_duration`
# * `hospital_admission`
# * `icu_duration`
# * `icu_admission`
# * `unplanned_admit`
# * `hospital_mortality`
#
# These datasets include input data and labels.
# %%
import logging
from pathlib import Path

import pandas as pd
from llm_utils import DataPaths

from make_dataset.dataset import DatasetBuilder

logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

paths = DataPaths(project_dir=Path(__file__).parent.parent.parent, data_version=7)

paths.register(name="adt_raw", path=paths.raw / "2023-05-11_ADT.feather")
paths.register(name="case_raw", path=paths.raw / "2023-05-11_Case.feather")
paths.register(name="preanes_notes_raw", path=paths.raw / "2023-05-02_PreAnesNotes.feather")
paths.register(name="last10_notes_raw", path=paths.raw / "2023-05-02_LastTenNotes.feather")
# %% [markdown]
# ## Build All Datasets for QA Prediction Outcome + Data Split + Note Type Combinations
# %%
builder = DatasetBuilder(paths=paths, dataset_version="dataset4")
# %%
# Encounter Events, Cases, Notes, ADT
E = builder.E
C = E.case
A = E.adt
PN = builder.preanes_notes
L10N = builder.last10_notes
common_note_ids = builder.common_note_proc_ids

# Raw Data Statistics
raw_cases = C.raw_df
print(f"Raw Case Count: {raw_cases.shape[0]}")
print(f"Case Date Range: {raw_cases.DOS.min()} - {raw_cases.DOS.max()}")
print("\n")
raw_adt = A.raw_df
print(f"Raw ADT Count: {raw_adt.shape[0]}")
print(f"ADT Date Range: {raw_adt.EVENT_TIME.min()} - {raw_adt.EVENT_TIME.max()}")
print("\n")
preanes_notes = PN.raw_df
print(f"Raw Preanes Notes Count: {preanes_notes.shape[0]}")
print(
    "Preanes Notes Date Range: "
    f"{preanes_notes.NoteServiceDate.min()} - {preanes_notes.NoteServiceDate.max()}"
)
print("\n")
last10_notes = L10N.raw_df
print(f"Raw Last 10 Notes Count: {last10_notes.shape[0]}")
print(
    "Last 10 Notes Date Range: "
    f"{last10_notes.NoteServiceDate.min()} - {last10_notes.NoteServiceDate.max()}"
)
print("\n")
# Get Final Cases with and without Durations prior to building individual datasets
cases_without_duration = E.case.df.copy()
cases_with_duration = E.case_durations.copy()
patients_without_duration = cases_without_duration.drop_duplicates("PAT_ID")
non_organ_donor_cases_without_duration = patients_without_duration.query(
    "IsOrganDonorCase == False"
)
patients_with_duration = cases_with_duration.drop_duplicates("PAT_ID")
non_organ_donor_cases_with_duration = patients_with_duration.query("IsOrganDonorCase == False")
print(f"Case without Duration Count: {cases_without_duration.shape[0]}")
print(
    f"Case without Duration (Patient:Case uniquification) Count: {len(patients_without_duration)}"  # noqa
)
print(
    "Case without Duration Non-Organ Donors: ",
    len(non_organ_donor_cases_without_duration),
)
print("\n")
print(f"Case with Duration Count: {cases_with_duration.shape[0]}")
print(f"Case with Duration (Patient:Case uniquification) Count: {len(patients_with_duration)}")
print("Case with Duration Non-Organ Donors: ", len(non_organ_donor_cases_with_duration))

# %%
# Create Datasets and Save to disk
logging.info("Create ASA Dataset")
asa = builder.create_dataset(dataset_type="asa")
# %%
logging.info("Create Phase 1 Duration Dataset")
phase1 = builder.create_dataset(dataset_type="phase1_duration")
# %%
logging.info("Create Hospital Duration Dataset")
hospital_duration = builder.create_dataset(dataset_type="hospital_duration")
# %%
logging.info("Create Hospital Admission Dataset")
hospital_admission = builder.create_dataset(dataset_type="hospital_admission")
# %%
logging.info("Create ICU Duration Dataset")
icu_duration = builder.create_dataset(dataset_type="icu_duration")
# %%
logging.info("Create ICU Admission Dataset")
icu_admission = builder.create_dataset(dataset_type="icu_admission")
# %%
logging.info("Create Unplanned Admit Dataset")
unplanned_admit = builder.create_dataset(dataset_type="unplanned_admit")
# %%
logging.info("Create Hospital Mortality Dataset")
hospital_mortality = builder.create_dataset(dataset_type="hospital_mortality")
# %%
