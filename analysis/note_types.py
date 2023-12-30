# %%
import logging
from pathlib import Path

import pandas as pd

from llm_utils import DataPaths, read_pandas, save_pandas

logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

paths = DataPaths(project_dir=Path(__file__).parent.parent, data_version=7)

paths.register(name="adt_raw", path=paths.raw / "2023-05-11_ADT.feather")
paths.register(name="case_raw", path=paths.raw / "2023-05-11_Case.feather")
paths.register(name="preanes_notes_raw", path=paths.raw / "2023-05-02_PreAnesNotes.feather")
paths.register(name="last10_notes_raw", path=paths.raw / "2023-05-02_LastTenNotes.feather")
# %%
# Load Notes
notes_df = read_pandas(paths.last10_notes_raw)
# Table of Note Type x Provider Type
note_count_df = (
    notes_df.loc[:, ["NoteName", "AuthorProviderType"]]
    .value_counts()
    .to_frame()
    .unstack(-1)
    .droplevel(0, axis="columns")
    .fillna(0.0)
)

# Collapse duplicate column categories
note_count_df = (
    note_count_df.assign(
        Fellow=(note_count_df.Fellow + note_count_df.Fellow0124),
        Resident=(note_count_df.Resident + note_count_df.Resident0124),
    )
    .drop(columns=["Fellow0124", "Resident0124"])
    .applymap(lambda x: f"{int(x)}" if x != 0 else "--")
)
note_count_df

# %%
save_dir = paths.project_dir / "notebooks" / "results" / "note_types"
save_pandas(df=note_count_df, path=save_dir / "note_types.csv")
# %%
