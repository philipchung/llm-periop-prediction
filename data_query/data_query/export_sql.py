# %%
from pathlib import Path

import database_config
import pandas as pd
import sqlalchemy

# Define SQL Database Parameters
server = database_config.server
database = database_config.database
schema = database_config.schema
# Create SQL Engine
engine = sqlalchemy.create_engine(
    f"mssql+pyodbc://{server}/{database}?driver=SQL+Server+Native+Client+11.0"
)
# Create Database Connection
cnx = engine.connect()
# metadata = sqlalchemy.MetaData(bind=engine, schema=schema)
# %%
# Query Cases & PACU Data
case_query = sqlalchemy.text(
    f"""SELECT
        ProcID,
        PAT_ID,
        DOS,
        ASA,
        SurgService,
        ProcedureDescription,
        PatientClass,
        AnesStart,
        InRoom,
        OutOfRoom,
        PACUStart AS InRecovery,
        PACUEnd AS OutOfRecovery,
        InPhase1,
        StartPhase1,
        Phase1Complete,
        OutOfPhase1,
        InPhase2,
        Phase2Complete,
        OutOfPhase2,
        DischargeDisposition,
        PatientLivingStatus,
        PatientDeathDate,
        DeathDtTm
    FROM {database}.{schema}.ChungChatGPTProc
    ORDER BY DOS, ProcedureDescription"""
)
case_df = pd.read_sql_query(sql=case_query, con=cnx)
case_df
# %%
# Query ADT Data
adt_query = sqlalchemy.text(
    f"""SELECT
        ProcID,
        DOS,
        OutOfRoom,
        EVENT_ID,
        SEQ_NUM_IN_ENC,
        EFFECTIVE_TIME,
        EVENT_TIME,
        DELETE_TIME,
        ORIGINAL_EVENT_ID,
        EventType,
        EventSubType,
        Department,
        DepartmentSpecialty,
        LevelOfCare,
        PatientService
    FROM {database}.{schema}.ChungChatGPTADT
    ORDER BY ProcID, SEQ_NUM_IN_ENC, EVENT_TIME"""
)
adt_df = pd.read_sql_query(sql=adt_query, con=cnx)
adt_df
# %%
# Query PreAnesthesia Notes
preanes_notes_query = sqlalchemy.text(
    f"""SELECT
        ProcID,
        NoteID,
        NoteServiceDate,
        ContactEnteredDate,
        NoteName,
        NoteStatus,
        AuthorProviderType,
        NoteText,
        MostRecentNote,
        MultipleNotesForCase
    FROM {database}.{schema}.ChungChatGPTPreAnesNotes
    ORDER BY ProcID
    """
)
preanes_notes_df = pd.read_sql_query(sql=preanes_notes_query, con=cnx)
preanes_notes_df
# %%
# Query Last 10 Notes Before Anesthesia
last10_notes_query = sqlalchemy.text(
    f"""SELECT
        ProcID,
        NoteID,
        NoteServiceDate,
        ContactEnteredDate,
        NoteName,
        NoteStatus,
        AuthorProviderType,
        NoteText
    FROM {database}.{schema}.ChungChatGPTLastTenNotes2
    ORDER BY ProcID
    """
)
last10_notes_df = pd.read_sql_query(sql=last10_notes_query, con=cnx)
last10_notes_df
# %%
# Export Tables to .feather file format using ZSTD compression

current_dir = Path(__file__).parent
data_dir = current_dir.parent.parent / "data"
data_version = 6
raw_data_dir = data_dir / f"v{data_version}" / "raw"
raw_data_dir.mkdir(parents=True, exist_ok=True)

preanes_notes_path = raw_data_dir / "2023-05-02_PreAnesNotes.feather"
preanes_notes_df.to_feather(preanes_notes_path, compression="zstd")

last10_notes_path = raw_data_dir / "2023-05-02_LastTenNotes.feather"
last10_notes_df.to_feather(last10_notes_path, compression="zstd")

case_path = raw_data_dir / "2023-03-30_Case.feather"
case_df.to_feather(case_path, compression="zstd")

adt_path = raw_data_dir / "2023-03-30_ADT.feather"
adt_df.to_feather(adt_path, compression="zstd")

# %%
# Test reading data back into pandas
df = pd.read_feather(last10_notes_path)
df
# %%
