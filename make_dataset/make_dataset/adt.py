import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from llm_utils import (
    DataPaths,
    create_uuid,
    function_with_cache,
    parallel_process,
    read_pandas,
    string_to_datetime_fmt2,
)
from tqdm.auto import tqdm

from make_dataset.pipeline import PipelineMixin, PipelineStep

logger = logging.getLogger(__file__)


@dataclass(kw_only=True)
class ADTEvents(PipelineMixin):
    """Data object to clean and transform hospital encounter Admit/Discharge/Transfer Events."""

    paths: DataPaths
    execute_pipeline: bool = True
    force: bool = False
    # Fields Populated After Running Pipeline
    df: pd.DataFrame | None = None
    raw_df: pd.DataFrame | None = None
    adt_location_df: pd.DataFrame | None = None
    key_info_df: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        "Called upon object instance creation."
        self.create_pipeline()
        # Optionally execute pipeline on initialization
        if self.execute_pipeline:
            self.run_pipeline(
                start_step=0, end_step=4, force=self.force, run_all=self.force
            )

    def create_pipeline(self) -> None:
        self.pipeline = [
            PipelineStep(
                num=0,
                name="load_data",
                data_path=self.paths.adt_raw,
                method=self._load_data,
            ),
            PipelineStep(
                num=1,
                name="clean_data",
                data_path=self.paths.register(
                    self.paths.interim / "adt_cleaned.feather"
                ),
                method=self._clean_data,
            ),
            PipelineStep(
                num=2,
                name="compute_adt_location",
                data_path=self.paths.register(
                    self.paths.interim / "adt_location.feather"
                ),
                method=self._compute_adt_location,
            ),
            PipelineStep(
                num=3, name="drop_adt_cols", data_path=None, method=self._drop_adt_cols
            ),
            # Key Info from ADT Events depends on Case Info, so is run separately
            # after calling `join_case_events_logic`.
            PipelineStep(
                num=4,
                name="compute_key_info_from_adt_events",
                data_path=self.paths.register(
                    self.paths.interim / "adt_key_info.feather"
                ),
                method=self.compute_key_info_from_adt_events,
            ),
        ]

    def run_step_compute_key_info_from_adt_events(self) -> None:
        self.run_pipeline(steps=[4], force=self.force, run_all=self.force)

    def on_run_pipeline_finish(self) -> None:
        # Ensure raw data loaded
        load_data_step = self.get_pipeline_step("load_data")
        if not load_data_step.executed:
            self.raw_df = self.execute_step(load_data_step)

        # Ensure `df` is last executed step (except for key_info_from_adt_events)
        last_executed_step = self.executed_steps()[-1]
        if last_executed_step.name != "compute_key_info_from_adt_events":
            self.df = last_executed_step.result

    def join_case_events_logic(
        self, adt_events_df: pd.DataFrame, case_df: pd.DataFrame
    ) -> pd.DataFrame:
        _df = adt_events_df.copy()
        _case_df = case_df.copy()
        _case_df = _case_df.loc[
            :,
            [
                "AnesStart",
                "InRoom",
                "OutOfRoom",
                "InRecovery",
                "PatientDeathDate",
                "PatientLivingStatus",
            ],
        ]
        _df = _df.merge(right=_case_df, how="inner", left_on="ProcID", right_index=True)
        return _df

    def _load_data(self, data_path: str | Path, **kwargs) -> pd.DataFrame:
        # NOTE: raw data does not have an AdtUUID--this is created in `clean_data` step
        _df = read_pandas(Path(data_path))
        self.df = _df
        self.raw_df = _df
        return _df

    def _clean_data(
        self,
        df: pd.DataFrame | None = None,
        data_path: str | Path | None = None,
        force: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Clean ADT data.

        Args:
            df (pd.DataFrame, optional): Table of input data.
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df if (df is None or df.empty) else df,
            function=self.clean_data_logic,
            cache_path=data_path,
            set_index="AdtUUID",
            force=force,
            **kwargs,
        )
        self.df = _df
        return _df

    def _compute_adt_location(
        self,
        df: pd.DataFrame | None = None,
        data_path: str | Path | None = None,
        force: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Determine Location for each ADT Event.

        Args:
            df (pd.DataFrame, optional): Table of input data.
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df if (df is None or df.empty) else df,
            function=self.compute_adt_location_logic,
            cache_path=data_path,
            set_index="AdtUUID",
            force=force,
            **kwargs,
        )
        self.df = _df
        self.adt_location_df = _df
        return _df

    def _drop_adt_cols(self, **kwargs) -> pd.DataFrame:
        _df = self.df.drop(columns=["EventSubType"])
        self.df = _df
        return _df

    def compute_key_info_from_adt_events(
        self, data_path: str | Path | None = None, force: bool = False
    ) -> pd.DataFrame:
        """For each ProcID, transform all associated ADT events into key info about events.
        The result is that each ProcID has a row of key info about events.

        Args:
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df,
            function=self.compute_key_info_from_adt_events_logic,
            cache_path=data_path,
            set_index="ProcID",
            force=force,
        )
        # NOTE: index for key_info_df is ProcID whereas index for self.df is AdtUUID;
        # they do not have comparable columns
        self.key_info_df = _df
        return _df

    def clean_data_logic(self, adt_events_df: pd.DataFrame) -> pd.DataFrame:
        _df = adt_events_df.copy()
        # Drop cancelled/invalid rows, sort by event sequence number
        _df = _df.loc[_df.SEQ_NUM_IN_ENC.notna()].sort_values(
            by=["ProcID", "DOS", "SEQ_NUM_IN_ENC"]
        )
        # Remove Columns
        for col in [
            "OutOfRoom",
            "EVENT_ID",
            "EVENT_TIME",
            "DELETE_TIME",
            "ORIGINAL_EVENT_ID",
        ]:
            if col in _df.columns:
                _df = _df.drop(columns=[col])

        # Create Unique ADT ID to use as index for each row
        # This ID will be unique as long as the row has a unique combination of
        # ProcID, SEQ_NUM_IN_ENC, EFFECTIVE_TIME, EventType, EventSubType
        tqdm.pandas(desc="Creating AdtUUID")
        adt_uuid = _df.progress_apply(
            lambda row: create_uuid(
                str(row["ProcID"])
                + str(row["SEQ_NUM_IN_ENC"])
                + str(row["EFFECTIVE_TIME"])
                + str(row["EventType"])
                + str(row["EventSubType"])
            ),
            axis=1,
        )
        _df = _df.assign(AdtUUID=adt_uuid).set_index("AdtUUID")

        # Format Data Types
        _df.SEQ_NUM_IN_ENC = _df.SEQ_NUM_IN_ENC.astype(int)
        _df.ProcID = _df.ProcID.astype(int)

        # Format Dates as Pandas Timestamp
        for col in ["EFFECTIVE_TIME"]:
            if col in _df.columns:
                # Skip if column is already Timestamp
                if _df[col].dtype != np.dtype("datetime64[ns]"):
                    tqdm.pandas(desc=f"Format {col} DateTime")
                    _df[col] = _df[col].progress_apply(string_to_datetime_fmt2)
        return _df

    def compute_adt_location_logic(self, adt_events_df: pd.DataFrame) -> pd.DataFrame:
        """Determine Location for each ADT event."""
        _df = adt_events_df.copy()
        # Map values from each column to simplified set
        _df.Department = _df.Department.apply(self.consolidate_department)
        _df.DepartmentSpecialty = _df.DepartmentSpecialty.apply(
            self.consolidate_department_specialty
        )
        _df.LevelOfCare = _df.LevelOfCare.apply(self.consolidate_level_of_care)
        _df.PatientService = _df.PatientService.apply(self.consolidate_patient_service)
        # Determine a location based on the simplified values
        tqdm.pandas(desc="Determining Location")
        location = _df.progress_apply(self.resolve_adt_location, axis=1)
        _df = _df.assign(Location=location)
        return _df

    def consolidate_department(self, text: str) -> str:
        operating_rooms = [
            "UWMC MAIN OR",
            "UWMC ROOSEVELT OR",
            "UWMC NW OR",
            "UWMC NW OSC OR",
            "HMC MAIN OR",
            "HMC EYE INST OR",
        ]
        nora_locations = [
            "UWMC ENDOSCOPY",
            "UWMC NW ENDOSCOPY",
            "HMC ENDOSCOPY 20",
            "UWMC CARDIAC CATH LAB",
            "UWMC ELECTROPHYSIOLOGY LAB",
            "UWMC NW ELECTROPHYSIOLOGY LAB",
            "UWMC NW CARDIAC CATH LAB",
            "HMC ANGIO SUITE 20",
            "UWMC BRONCHOSCOPY",
            "UWMC RAD ULTRASOUND",
            "UWMC RAD CT",
            "UWMC RAD MRI",
            "UWMC RAD IR",
            "UWMC RAD XR",
            "UWMC RAD FLUOROSCOPY",
            "UWMC CC RAD THERAPY CT SIM",
            "UWMC NW RAD IR - CAMPUS",
            "UWMC NW RAD CT",
            "HMC RAD MRI",
            "HMC RAD CT",
            "HMC RAD NUCLEAR MEDICINE",
            "HMC RAD FLUOROSCOPY",
            "HMC RAD ULTRASOUND",
            "HMC ECHOCARDIOGRAPHY LAB 20",
            "HMC BURN CLINIC ANES ASSISTED PROC 20",
        ]
        pacu_locations = [
            "U 2SE PACBDR",
            "UWMC INTERVENTIONAL CARDIAC RECOVERY UNIT",
            "UWMC NW CARDIAC PROC UNIT",
            "UWMC SP RECOVERY",
            "H PAC BDB 20",
            "H PAC BDC 20",
            "UWMC MAIN RECOVERY",
            "UWMC NW MAIN RECOVERY",
            "HMC MAIN RECOVERY",
            "HMC MALENG RECOVERY",
        ]
        icu_locations = [
            "U 5SA",
            "U 5E",
            "U 5SE",
            "U NW ICU",
            "H 9MA 20",
            "H 9MB 20",
            "H 9EA 20",
            "H 9EB 20",
            "H 2WA 20",
            "H 2WC 20",
            "H 2WB 20",
            "H 2EA 20",
        ]
        invalid_locations = [
            "ZZU ICR",
            "ZZU NW OR",
            "ZZU PRE",
            "ZZHMC BURN CLINIC ANES ASSISTED PROC 20",
            "ZZU PAC",
            "ZZU MOR",
            "ZZU NW OSC",
            "ZZU EMERGENCY DEPARTMENT",
            "ZZU BRO",
        ]

        if pd.isna(text):
            return "Unknown"
        elif text in invalid_locations:
            return "Invalid"
        elif text in (*operating_rooms, *nora_locations):
            return "OR"
        elif text in pacu_locations:
            return "PACU"
        elif text in icu_locations:
            return "ICU"
        else:
            return "Floor"

    def consolidate_department_specialty(self, text: str) -> str:
        if pd.isna(text):
            return "Unknown"
        elif text == "Intensive Care":
            return "ICU"
        else:
            return "Floor"

    def consolidate_level_of_care(self, text: str) -> str:
        if pd.isna(text):
            return "Unknown"
        elif text == "Intensive Care":
            return "ICU"
        elif text == "Perioperative":
            return "PACU"
        else:
            return "Floor"

    def consolidate_patient_service(self, text: str) -> str:
        icu_patient_services = [
            # Cardiac ICU
            "CARDIOTHORACIC ICU",
            "CARDIAC ICU - CICU",
            # Coronary Care Unit (CCU) (we treat this as an ICU)
            "CCU CARDIOLOGY CARE",
            "MEDICINE CCU",
            # Surgery/Trauma ICU
            "SURGERY ICU",
            "SICU B",
            "SICU A",
            "S I C U",
            # Onc/BMT ICU
            "HEMATOLOGY ONCOLOGY BMT ICU",
            # Medical ICU
            "MEDICINE ICU",
            "MEDICINE TRANS/ICU",
            # Pediatric ICU
            "MEDICINE ICU - PEDS",
            "MEDICINE TRANS/ICU PEDS",
            "S I C U - PEDS",
            # Covid ICU
            "ICU COVID",
            # Neuro ICU
            "NEUROCRITICAL CARE 1",
            "NEUROCRITICAL CARE 2",
        ]

        if pd.isna(text):
            return "Unknown"
        elif text in icu_patient_services:
            return "ICU"
        else:
            return "Floor"

    def resolve_adt_location(self, row: pd.Series) -> str:
        """Determine location patient is transfered into.
        We will use consistent sources of information first and back-off
        to noisier sources of information to determine patient's location.

        Resolution order:
        1. If Department is "OR" or "PACU", assign location
        2. Otherwise, use Patient Service if "Floor" or "ICU"
        2. Otherwise, use Level Of Care if "Floor" or "ICU"
        3. Otherwise, use both Department & DepartmentSpecialty to determine "Floor" or "ICU".
            In this scenario, we enforce that both must be the same.  If there is a
            discrepancy, then we report "Unknown"

        Args:
            row (pd.Series): pd.Series with indices
                "Department", "DepartmentSpecialty", "LevelOfCare", "PatientService"

        Returns:
            str: Resolved value of location {"OR", "PACU", "Floor", "ICU", "Unknown"}
        """
        department = row.Department
        department_specialty = row.DepartmentSpecialty
        level_of_care = row.LevelOfCare
        patient_service = row.PatientService

        if department == "OR":
            return "OR"
        elif department == "PACU":
            return "PACU"
        else:
            if patient_service == "Floor":
                return "Floor"
            elif patient_service == "ICU":
                return "ICU"
            else:
                if level_of_care == "Floor":
                    return "Floor"
                elif level_of_care == "ICU":
                    return "ICU"
                else:
                    if all([department == "Floor", department_specialty == "Floor"]):
                        return "Floor"
                    elif all([department == "ICU", department_specialty == "ICU"]):
                        return "ICU"
                    else:
                        return "Unknown"

    def compute_key_info_from_adt_events_logic(
        self, adt_events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Transforms table of ADT Events for each patient into a table of key info
        derived from ADT Events for each ProcID.  Input data into this method
        has index `AdtUUID` but output data from this method has index `ProcID`."""
        _df = adt_events_df.copy()

        key_info_from_adt = parallel_process(
            iterable=_df.groupby("ProcID"),
            function=compute_key_info_from_adt_events_wrapper,
            use_args=True,
            desc="Key ADT Info",
        )
        key_info_from_adt = pd.DataFrame(key_info_from_adt).set_index("ProcID")
        return key_info_from_adt


def compute_key_info_from_adt_events_wrapper(
    proc_id: np.intc, adt_events_df: pd.DataFrame
) -> dict:
    key_info_from_adt = compute_key_info_from_adt_events_for_procedure(adt_events_df)
    return {"ProcID": proc_id, **key_info_from_adt}


def compute_key_info_from_adt_events_for_procedure(adt_events_df: pd.DataFrame) -> dict:
    """Given a table of ADT events for a patient who had a procedure,
    compute key ADT intervals that map to specific level-of-care
    in the hospital.  Identify ADT interval where procedure occurred.
    Determine whether procedure has associated ADT events where patient
    is transferred to operating room or procedure area (these are inconsistently present).

    Args:
        adt_events_df (pd.DataFrame): Table of ADT events for a patient associated
            with a single ProcID.  This table should have key columns from CaseEvents joined
            into it before calling this function.  Key columns include "OutOfRoom".

    Returns:
        dict: Dictionary with keys:
            - "AdmitType": Either `Admission` or `Hospital Outpatient`
            - "AdmitTime": Admission datetime
            - "DischargeTime": Discharge datetime
            - "TransferInORTime": Datetime for the ADT event where patient
              is transferred to the OR. `NaT` if cannot be determined.
            - "TransferOutORTime": Datetime for the ADT event where patient
              is transferred out of the OR. `NaT` if cannot be determined.
            - "LocationBeforeProcedure": Location before procedure if HasOutOfRoom,
              otherwise "Unknown"
            - "LocationAfterProcedure": Location after procedure if HasOutOfRoom,
              otherwise "Unknown"
    """
    _df = adt_events_df.copy()

    # Create ADT Intervals Table
    adt_intervals_df = get_adt_intervals(_df)

    # Extract Admit Time (Take last one if multiple)
    admit_events = _df.loc[
        _df.EventType.isin(["Admission", "Hospital Outpatient"])
    ].sort_values(by="EFFECTIVE_TIME")
    has_admit_event = not admit_events.empty
    if has_admit_event:
        admit_event = admit_events.iloc[-1, :]
        admit_type = admit_event.EventType
        admit_time = admit_event.EFFECTIVE_TIME
    else:
        admit_type = None
        admit_time = pd.NaT

    # Extract Discharge Time (Take last one if multiple)
    discharge_events = _df.loc[_df.EventType == "Discharge"].sort_values(
        by="EFFECTIVE_TIME"
    )
    has_discharge_event = not discharge_events.empty
    if has_discharge_event:
        discharge_event = discharge_events.iloc[-1, :]
        discharge_time = (
            discharge_event.EFFECTIVE_TIME if not discharge_event.empty else pd.NaT
        )
    else:
        discharge_time = pd.NaT

    # Extract InRoom Time, OutOfRoom Time, Anesthesia Start Time
    in_room_time = _df.InRoom.iloc[0]
    anes_start = _df.AnesStart.iloc[0]
    out_of_room_time = _df.OutOfRoom.iloc[0]
    in_recovery_time = _df.InRecovery.iloc[0]

    # Get Timepoint in Middle of Case (to determine which ADT time interval corresponds to case)
    case_begin = max(in_room_time, anes_start)
    case_end = min(out_of_room_time, in_recovery_time)
    if pd.notna(case_begin) and pd.notna(case_end):
        case_duration = case_end - case_begin
        middle_of_case = case_begin + case_duration / 2
        case_reference_time = middle_of_case
    elif pd.notna(case_begin):
        case_reference_time = case_begin
    elif pd.notna(case_end):
        case_reference_time = case_end
    else:
        case_reference_time = pd.NaT

    # Get ADT Interval where procedure occurs
    procedure_interval = adt_intervals_df.loc[
        (adt_intervals_df.StartTime < case_reference_time)
        & (case_reference_time <= adt_intervals_df.EndTime)
        & (adt_intervals_df.Location == "OR")
        & (
            adt_intervals_df.StartTime != adt_intervals_df.EndTime
        )  # Exclude 0-duration interval between Transfer Out/In
    ]

    if procedure_interval.empty:
        # No ADT Interval that corresponds to transfer to OR for procedure
        transfer_in_or_time = pd.NaT
        transfer_out_or_time = pd.NaT
        location_before_procedure = "Unknown"
        location_after_procedure = "Unknown"
    else:
        # We have an ADT Interval that corresponds to transfer to OR for procedure
        transfer_in_or_time = procedure_interval.StartTime.iloc[0]
        transfer_out_or_time = procedure_interval.EndTime.iloc[0]

        # Get Event for Procedure Start/End & Earliest/Latest ADT Event
        adt_events_df2 = adt_events_df.loc[
            adt_events_df.EventType.isin(
                ["Admission", "Hospital Outpatient", "Discharge", "Transfer In"]
            )
        ].reset_index()
        procedure_start_event = adt_events_df2.loc[
            adt_events_df2.AdtUUID == procedure_interval.StartAdtUUID.item()
        ]
        procedure_end_event = adt_events_df2.loc[
            adt_events_df2.AdtUUID == procedure_interval.EndAdtUUID.item()
        ]
        procedure_start_idx = procedure_start_event.index.item()
        procedure_end_idx = procedure_end_event.index.item()

        first_idx = adt_events_df2.index.min()
        last_idx = adt_events_df2.index.max()

        # Location Before Procedure
        before_procedure_idx = procedure_start_idx - 1
        if before_procedure_idx < first_idx:
            location_before_procedure = "None"
        else:
            location_before_procedure = adt_events_df2.loc[
                before_procedure_idx
            ].Location
        # Location After Procedure
        after_procedure_idx = procedure_end_idx
        if after_procedure_idx > last_idx:
            location_after_procedure = "None"
        else:
            location_after_procedure = adt_events_df2.loc[after_procedure_idx].Location

    return {
        "AdmitType": admit_type,
        "AdmitTime": admit_time,
        "DischargeTime": discharge_time,
        "TransferInORTime": transfer_in_or_time,
        "TransferOutORTime": transfer_out_or_time,
        "LocationBeforeProcedure": location_before_procedure,
        "LocationAfterProcedure": location_after_procedure,
    }


def deduplicate_adt_events(adt_events_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicates consecutive "Transfer In" ADT events if the events occur
    at the same Location. Note: this does not deduplicate "Admission" or "Discharge"
    ADT events.  Thus the resultant table may still have the same Location between
    "Admission" and "Transfer In" events, or between "Discharge" and "Transfer In" events.

    Args:
        adt_events_df (pd.DataFrame): table of ADT events

    Returns:
        pd.DataFrame: Table of ADT events with consecutive "Transfer In" events
            deduplicated (keeping the first entry) if the events occur at same Location.
    """
    _df = adt_events_df.copy()
    admit_events = _df.loc[_df.EventType.isin(["Admission", "Hospital Outpatient"])]
    discharge_event = _df.loc[_df.EventType.isin(["Discharge"])]
    transfer_ins = _df.loc[_df.EventType.isin(["Transfer In"])]

    # Remove Consecutive Duplicate Locations in "Transfer In" ADT Events
    idx_to_keep: list[str] = []
    last_location: str | None = None
    for row in transfer_ins.itertuples():
        current_location = row.Location
        if last_location is None:  # First item
            idx_to_keep.append(row.Index)
            last_location = current_location
            continue
        else:
            if current_location != last_location:
                idx_to_keep.append(row.Index)
            last_location = current_location

    transfer_ins_deduplicated = transfer_ins.loc[idx_to_keep]

    return pd.concat(
        [transfer_ins_deduplicated, admit_events, discharge_event], axis=0
    ).sort_values(by="EFFECTIVE_TIME")


def adt_events_to_intervals(adt_events_df: pd.DataFrame) -> pd.DataFrame:
    """Given a table where each row is a timestamped ADT events,
    transform it into a table where each row is a time interval between 2
    timestamped ADT events.

    Args:
        adt_events_df (pd.DataFrame): Table of ADT events

    Returns:
        pd.DataFrame: Table of time intervals between ADT events
    """
    _df = adt_events_df.copy()

    intervals_df = pd.DataFrame(
        data={
            "ProcID": _df.ProcID,
            "StartEvent": _df.EventType,
            "EndEvent": _df.EventType.shift(-1),
            "StartTime": _df.EFFECTIVE_TIME,
            "EndTime": _df.EFFECTIVE_TIME.shift(-1),
            "StartAdtUUID": _df.index.to_series(),
            "EndAdtUUID": _df.index.to_series().shift(-1),
            "Location": _df.Location,
        }
    ).reset_index(drop=True)
    intervals_df = intervals_df.iloc[:-1]
    return intervals_df


def get_adt_intervals(
    adt_df: pd.DataFrame,
    proc_id: str | None = None,
    event_types: list[str] = [
        "Admission",
        "Hospital Outpatient",
        "Transfer In",
        "Transfer Out",
        "Discharge",
    ],
) -> pd.DataFrame:
    """Get ADT Time Intervals for Encounter surrounding Procedure with proc_id.

    Args:
        proc_id (str): Unique identifier for procedure.  If provided, then will
            filter `adt_df` to include events for the given proc_id.  If `None`, will
            assume that all of `adt_df` is associated with the same proc_id.
        adt_df (pd.DataFrame): ADT Table.
        event_types (list[str]): Include only ADT Events with event types in this whitelist.
            By default, these include only events that affect location/level-of-care and
            start/end of encounter.

    Returns:
        pd.DataFrame: ADT Time Intervals for Encounter surrounding Procedure.
    """
    _df = adt_df.copy()
    if proc_id:
        _df = _df.loc[proc_id]
    # Filter Event Types
    _df = _df.loc[_df.EventType.isin(event_types)]
    # Deduplicate Consecutive "Transfer In" ADT Events (Drops "Transfer Out")
    _df = deduplicate_adt_events(_df)
    # Create ADT Intervals Table
    adt_intervals_df = adt_events_to_intervals(_df)
    return adt_intervals_df


def post_pacu_end_intervals(
    adt: pd.DataFrame, pacu_end_time: pd.Timestamp | datetime
) -> pd.DataFrame:
    """Get ADT Time Intervals after PACU End.

    Args:
        adt (pd.DataFrame): Pandas dataframe of ADT events for encounter during which
            proc_id occurs.
        pacu_end_time (pd.Timestamp | datetime): Time of PACU End

    Returns:
        pd.DataFrame: Dataframe of ADT time intervals after PACU End
    """
    # The True ICU LOS is when patient is in Location==ICU between PACUEndTime and Discharge.
    _adt = adt.copy()

    # Get ADT Intervals Table
    adt_intervals_df = get_adt_intervals(_adt)

    # Get Only ADT Intervals After PACU End
    post_pacu_intervals = adt_intervals_df.loc[
        adt_intervals_df.StartTime >= pacu_end_time
    ]
    # Interval that contains PACU End
    pacu_end_interval = adt_intervals_df.loc[
        (adt_intervals_df.StartTime < pacu_end_time)
        & (pacu_end_time <= adt_intervals_df.EndTime)
    ]
    return {
        "PostPACUIntervals": post_pacu_intervals,
        "PACUEndInterval": pacu_end_interval,
    }
