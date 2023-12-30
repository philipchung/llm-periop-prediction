# %% [markdown]
## Create a Mini Dataset for Experimentation
# %%
import logging
from pathlib import Path

import pandas as pd
from llm_utils import DataPaths, save_pandas

from make_dataset.encounter import EncounterEvents

logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

paths = DataPaths(project_dir=Path(__file__).parent.parent.parent, data_version=6)
paths.register(name="adt_raw", path=paths.raw / "2023-03-30_ADT.feather")
paths.register(name="case_raw", path=paths.raw / "2023-03-30_Case.feather")
paths.register(
    name="preanes_notes_raw", path=paths.raw / "2023-05-02_PreAnesNotes.feather"
)
paths.register(
    name="last10_notes_raw", path=paths.raw / "2023-05-02_LastTenNotes.feather"
)
# Load Case & ADT Data
E = EncounterEvents(paths=paths)

# Join Pre-Anesthesia Notes with Labels & Case Data
preanes_data = E.labels.join(E.preanes_notes.df).join(
    E.case.df.loc[:, ["PAT_ID", "SurgService", "ProcedureDescription", "PatientClass"]]
)
# Sample Dataset
df = E.select_cases_per_patient(df=preanes_data, patient_id_var="PAT_ID", max_cases=1)
df = E.sample_inverse_frequency(df=df, col="ASA", n=100)
df
# %%
# Save Mini Dataset
paths.register(name="mini_dataset1", path=paths.processed / "mini_dataset1.feather")
save_pandas(df=df, path=paths.mini_dataset1, compression="zstd")
# %%
