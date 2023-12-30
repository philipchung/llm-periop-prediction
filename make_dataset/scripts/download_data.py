# %% [markdown]
## Downloads data from Azure Blob Store to local data directory
# %%
from pathlib import Path

import pandas as pd
from azureml.core import Datastore, Workspace
from llm_utils import DataPaths

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Create Data Paths object to manage local data path references
paths = DataPaths(project_dir=Path(__file__).parent.parent.parent, data_version=7)
# %%
# Get AzureML Workspace and Datastore Reference
ws = Workspace.from_config()
datastore = Datastore.get(ws, datastore_name="llm_asa_los")

# Download Data from Datastore to Local Data Directory
datastore.download(target_path=paths.raw, prefix="")
# %%
# Register Local Data Paths
paths.register(name="adt_raw", path=paths.raw / "2023-05-11_ADT.feather")
paths.register(name="case_raw", path=paths.raw / "2023-05-11_Case.feather")
paths.register(
    name="preanes_notes_raw", path=paths.raw / "2023-05-02_PreAnesNotes.feather"
)
paths.register(
    name="last10_notes_raw", path=paths.raw / "2023-05-02_LastTenNotes.feather"
)

# %%
# Load Data
df = pd.read_feather(paths.case_raw)
df
# %%
