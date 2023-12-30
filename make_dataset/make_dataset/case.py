import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from llm_utils import (
    DataPaths,
    function_with_cache,
    parallel_process,
    read_pandas,
    string_to_datetime_fmt2,
)
from tqdm.auto import tqdm

from make_dataset.pipeline import PipelineMixin, PipelineStep

logger = logging.getLogger(__file__)


@dataclass(kw_only=True)
class CaseEvents(PipelineMixin):
    """Data object to clean and transform case & post-operative PACU events."""

    paths: DataPaths
    execute_pipeline: bool = True
    force: bool = False
    # Fields Populated After Running Pipeline
    df: pd.DataFrame | None = None
    raw_df: pd.DataFrame | None = None
    cleaned_df: pd.DataFrame | None = None
    transformed_df: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        "Called upon object instance creation."
        self.create_pipeline()
        # Optionally execute pipeline on initialization
        if self.execute_pipeline:
            self.run_pipeline(force=self.force, run_all=self.force)

    def create_pipeline(self) -> None:
        self.pipeline = [
            PipelineStep(
                num=0,
                name="load_data",
                data_path=self.paths.case_raw,
                method=self._load_data,
            ),
            PipelineStep(
                num=1,
                name="clean_data",
                data_path=self.paths.register(
                    self.paths.interim / "case_cleaned.feather"
                ),
                method=self._clean_data,
            ),
            PipelineStep(
                num=2,
                name="transform_data",
                data_path=self.paths.register(
                    self.paths.interim / "case_transformed.feather"
                ),
                method=self._transform_data,
            ),
        ]

    def on_run_pipeline_finish(self) -> None:
        # Ensure raw data loaded
        load_data_step = self.get_pipeline_step("load_data")
        if not load_data_step.executed:
            self.raw_df = self.execute_step(load_data_step)

        # Ensure `df` is last executed step
        last_executed_step = self.executed_steps()[-1]
        self.df = last_executed_step.result

    def _load_data(self, data_path: str | Path, **kwargs) -> pd.DataFrame:
        _df = read_pandas(Path(data_path)).set_index("ProcID")
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
        """Clean PACU Events data.

        Args:
            df (pd.DataFrame, optional): Table of input data.
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df if (df is None or df.empty) else df,
            function=self.clean_data_logic,
            cache_path=data_path,
            set_index="ProcID",
            force=force,
            **kwargs,
        )
        self.df = _df
        self.cleaned_df = _df
        return _df

    def _transform_data(
        self,
        df: pd.DataFrame | None = None,
        data_path: str | Path | None = None,
        force: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Transform Case data.

        Args:
            df (pd.DataFrame, optional): Table of input data.
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df if (df is None or df.empty) else df,
            function=self.transform_data_logic,
            cache_path=data_path,
            set_index="ProcID",
            force=force,
            **kwargs,
        )
        self.df = _df
        self.transformed_df = _df
        return _df

    def clean_data_logic(self, case_df: pd.DataFrame) -> pd.DataFrame:
        _df = case_df.copy()
        # Remove Columns
        for col in ["DeathDtTm"]:
            if col in _df.columns:
                _df = _df.drop(columns=[col])

        # Format Data Types (index is ProcID)
        _df.index = _df.index.astype(int)

        # Format Dates as Pandas Timestamp
        for col in [
            "AnesStart",
            "InRoom",
            "OutOfRoom",
            "InRecovery",
            "OutOfRecovery",
            "InPhase1",
            "StartPhase1",
            "Phase1Complete",
            "OutOfPhase1",
            "InPhase2",
            "Phase2Complete",
            "OutOfPhase2",
            "PatientDeathDate",
        ]:
            if col in _df.columns:
                # Skip if column is already Timestamp
                if _df[col].dtype != np.dtype("datetime64[ns]"):
                    tqdm.pandas(desc=f"Format {col} DateTime")
                    _df[col] = _df[col].progress_apply(string_to_datetime_fmt2)

        # Remove Cases without ASA-PS
        _df = _df.loc[_df.ASA.notna()]
        # Remove Invalid Alive/Deceased Status Cases
        # Keep only cases where patient is alive & no death date recorded, and
        # cases where patient is deceased & death date is recorded.
        has_death_date = _df.PatientDeathDate.notna()
        valid_alive = _df.loc[(_df.PatientLivingStatus == "Alive") & ~has_death_date]
        valid_deceased = _df.loc[
            (_df.PatientLivingStatus == "Deceased") & has_death_date
        ]
        _df = _df.loc[
            _df.index.isin(
                [*valid_alive.index.tolist(), *valid_deceased.index.tolist()]
            )
        ]

        # Determine Organ Donor Case
        is_organ_donor_case = _df.apply(determine_organ_donor_case, axis=1).rename(
            "IsOrganDonorCase"
        )
        _df = _df.join(is_organ_donor_case)

        # Map ExpectedPatientClass & ActualPatientClass to Admit or Not Admit
        anticipated_admission = _df.ExpectedPatientClass.apply(
            determine_if_expected_admit
        )
        actual_admission = _df.ActualPatientClass.apply(determine_if_actual_admit)
        _df = _df.assign(
            AnticipatedAdmit=anticipated_admission, ActualAdmit=actual_admission
        )
        # Drop cases with invalid ActualPatientClass
        _df = _df.loc[_df.ActualAdmit != "Invalid"]
        # Patients booked to be outpatient surgery, but were admitted
        unplanned_admit = _df.AnticipatedAdmit.eq(False) & _df.ActualAdmit.eq(True)
        # Patients booked to be admitted after surgery, but were outpatient surgery
        unplanned_outpatient = _df.AnticipatedAdmit.eq(True) & _df.ActualAdmit.eq(False)
        _df = _df.assign(
            UnplannedAdmit=unplanned_admit,
            UnplannedOutpatient=unplanned_outpatient,
        )
        return _df

    def transform_data_logic(self, case_df: pd.DataFrame) -> pd.DataFrame:
        _df = case_df.copy()
        _df = self.determine_case_number_in_dataset(case_df=_df)
        return _df

    def filter_cases_by_proc_ids(
        self, case_df: pd.DataFrame, proc_id_whitelist: list[str]
    ) -> pd.DataFrame:
        _df = case_df.copy()
        _df = _df.loc[_df.index.isin(proc_id_whitelist)]
        return _df

    def determine_died_in_hospital_or_case(self, case_df: pd.DataFrame) -> pd.DataFrame:
        _df = case_df.copy()
        expired_in_hospital = _df.apply(died_in_hospital_encounter, axis=1).rename(
            "HasExpiredInHospital"
        )
        expired_in_case = _df.apply(died_in_case, axis=1).rename("HasExpiredInCase")
        _df = _df.join(expired_in_hospital).join(expired_in_case)
        return _df

    def determine_pacu_phases(self, case_df: pd.DataFrame) -> pd.DataFrame:
        _df = case_df.copy()

        def row_generator() -> Iterator[pd.Series]:
            for idx, row in _df.iterrows():
                yield row

        output = parallel_process(
            iterable=row_generator(),
            function=determine_pacu_start_and_end_scenarios2,
            desc="Determine PACU Phase 1/2/Total Start/End/Duration",
        )
        pacu = pd.DataFrame(output, index=_df.index)
        _df = _df.join(pacu)
        return _df

    def validate_pacu_phases(self, case_df: pd.DataFrame) -> pd.DataFrame:
        _df = case_df.copy()
        valid_timestamp_proc_ids = set()
        valid_duration_proc_ids = set()
        for scenario, cases in _df.groupby("Scenario"):
            if scenario == "Phase1AndPhase2":
                timestamp_cols = [
                    "Phase1StartTime",
                    "Phase1EndTime",
                    "Phase2StartTime",
                    "Phase2EndTime",
                ]
                duration_cols = [
                    "Phase1Duration",
                    "Phase2Duration",
                    "PACUDuration",
                ]
                # Valid if Timestamps between Admit & Discharge
                valid_timestamps = drop_invalid_timestamp_cases(
                    case_df=cases, cols=timestamp_cols
                )
                valid_timestamp_proc_ids.update(valid_timestamps.index)
                # Valid if all durations are not NaT
                is_valid_duration = (
                    cases.loc[:, duration_cols]
                    .applymap(lambda x: pd.notna(x))
                    .all(axis=1)
                )
                valid_durations = cases.loc[is_valid_duration]
                valid_duration_proc_ids.update(valid_durations.index)
            elif scenario == "Phase1Only":
                timestamp_cols = [
                    "Phase1StartTime",
                    "Phase1EndTime",
                ]
                duration_cols = [
                    "Phase1Duration",
                    "PACUDuration",
                ]
                # Valid if Timestamps between Admit & Discharge
                valid_timestamps = drop_invalid_timestamp_cases(
                    case_df=cases, cols=timestamp_cols
                )
                valid_timestamp_proc_ids.update(valid_timestamps.index)
                # Valid if all durations are not NaT
                is_valid_duration = (
                    cases.loc[:, duration_cols]
                    .applymap(lambda x: pd.notna(x))
                    .all(axis=1)
                )
                valid_durations = cases.loc[is_valid_duration]
                valid_duration_proc_ids.update(valid_durations.index)
            elif scenario == "Phase2Only":
                timestamp_cols = [
                    "Phase2StartTime",
                    "Phase2EndTime",
                ]
                duration_cols = [
                    "Phase2Duration",
                    "PACUDuration",
                ]
                # Valid if Timestamps between Admit & Discharge
                valid_timestamps = drop_invalid_timestamp_cases(
                    case_df=cases, cols=timestamp_cols
                )
                valid_timestamp_proc_ids.update(valid_timestamps.index)
                # Valid if all durations are not NaT
                is_valid_duration = (
                    cases.loc[:, duration_cols]
                    .applymap(lambda x: pd.notna(x))
                    .all(axis=1)
                )
                valid_durations = cases.loc[is_valid_duration]
                valid_duration_proc_ids.update(valid_durations.index)
            elif scenario == "CannotDetermine":
                # All "CannotDetermine" cases are not valid and not included
                pass
            else:
                raise ValueError(
                    f"Unknown scenario {scenario} in validating PACU phases."
                )
        valid_proc_ids = valid_timestamp_proc_ids & valid_duration_proc_ids
        _df = _df.loc[_df.index.isin(valid_proc_ids)]
        return _df

    def determine_case_number_in_dataset(self, case_df: pd.DataFrame) -> pd.DataFrame:
        _df = case_df.copy()
        _df = parallel_process(
            iterable=_df.groupby("PAT_ID"),
            function=get_case_number_in_dataset,
            use_args=True,
            desc="Assigning Case Number for Patients",
        )
        return pd.concat(_df, axis=0)

    def make_case_booking_strings(self, case_df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Make Case Booking String")
        case_booking_str_df = case_df.progress_apply(make_case_booking_string, axis=1)
        return case_booking_str_df


def make_case_booking_string(series: pd.Series) -> str:
    surg_service = series.SurgService if pd.notna(series.SurgService) else ""
    scheduled_proc = (
        series.ScheduledProcedure if pd.notna(series.ScheduledProcedure) else ""
    )
    proc_desc = (
        series.ProcedureDescription if pd.notna(series.ProcedureDescription) else ""
    )
    diagnosis = series.Diagnosis if pd.notna(series.Diagnosis) else ""
    expected_patient_class = (
        series.ExpectedPatientClass if pd.notna(series.ExpectedPatientClass) else ""
    )

    case_booking_str = (
        "\n\n"
        "Surgery/Procedure Case Booking\n"
        f"Scheduled Procedure: {scheduled_proc}\n"
        f"Procedure Description: {proc_desc}\n"
        f"Diagnosis: {diagnosis}\n"
        f"Service: {surg_service}\n"
        f"Patient Class: {expected_patient_class}\n"
    )
    return case_booking_str


def get_case_number_in_dataset(pat_id: str, case_df: pd.DataFrame) -> pd.DataFrame:
    """For `case_df` that contains all of a patient's cases, chronologically
    number the cases.

    Args:
        pat_id (str): Unique ID to identify patient.
        case_df (pd.DataFrame): Table of cases for patient.

    Returns:
        pd.DataFrame: Table of cases for patient with additional column called
            "CaseNum" which is the chronological count of cases of patients in
            this dataset.  Note that this is not the number of cases for the
            patient's lifetime, but just in our dataset.  It is also not
            the number of cases in a particular hospital encounter.
    """
    _df = (
        case_df.copy()
        .sort_values(by=["AnesStart", "InRoom", "OutOfRoom"])
        .reset_index()
    )
    _df = _df.assign(PAT_ID=pat_id, CaseNum=_df.index).set_index("ProcID")
    return _df


def determine_organ_donor_case(case: pd.Series) -> bool:
    """Given a single case (ProcID), determines whether case is an organ donor case or not.

    Args:
        case (pd.Series): Series of case information, including the columns
            "ASA", "SurgService"

    Returns:
        bool: Whether patient is an organ donor case or not.
    """
    if pd.isna(case.ASA):
        is_organ_donor_case = False
    elif int(case.ASA) == 6:
        is_organ_donor_case = True
    elif case.SurgService == "Organ Donor":
        is_organ_donor_case = True
    else:
        is_organ_donor_case = False
    return is_organ_donor_case


def died_in_hospital_encounter(case: pd.Series) -> bool:
    if pd.notna(case.PatientDeathDate):
        if case.AdmitTime <= case.PatientDeathDate <= case.DischargeTime:
            return True
        else:
            return False
    else:
        return False


def died_in_case(case: pd.Series) -> bool:
    if pd.notna(case.PatientDeathDate):
        case_start = min(case.AnesStart, case.InRoom)
        case_end = max(case.OutOfRoom, case.InRecovery)
        if case_start <= case.PatientDeathDate <= case_end:
            return True
        else:
            return False
    else:
        return False


def determine_pacu_start_and_end_scenarios(case: pd.Series) -> dict:
    """Determines PACU Start & PACU End as well as the specific scenarios used to
    compute PACU Start & PACU End.  Preferentially uses more reliable scenarios
    in this order: Phase1 > Phase2Only > Alt > Phase1AltEnd > Phase1AltStart
    > Phase2AltEnd > Phase2AltStart > CannotDetermine.

    Args:
        case (pd.Series): Series that contains information about a single case.

    Returns:
        dict: Dict with keys that include "PACUStartTime", "PACUEndTime" and the
            "Scenario" used to compute these times.  Also includes keys that
            determine if boundaries for Phase 1 PACU & Phase 2 PACU are present as
            well as other key events that can be used to compute "PACUStartTime"
            and "PACUEndTime".
    """
    # Direct Indicators for PACU Start/End
    has_phase1_start = any(pd.notna(case[x]) for x in ["InPhase1", "StartPhase1"])
    has_phase1_end = any(pd.notna(case[x]) for x in ["Phase1Complete", "OutOfPhase1"])
    has_phase2_start = any(pd.notna(case[x]) for x in ["InPhase2"])
    has_phase2_end = any(pd.notna(case[x]) for x in ["Phase2Complete", "OutOfPhase2"])
    # Indirect Indicators for PACU Start/End
    has_out_of_room = pd.notna(case.OutOfRoom)
    has_in_recovery = pd.notna(case.InRecovery)
    has_out_of_recovery = pd.notna(case.OutOfRecovery)
    has_transfer_out_or = pd.notna(case.TransferOutORTime)
    has_alt_pacu_start = any([has_out_of_room, has_in_recovery])
    has_alt_pacu_end = any([has_out_of_recovery, has_transfer_out_or])

    # Determine PACU Start & End
    if has_phase1_start and has_phase1_end:
        # Cases have Complete Phase1 timestamps, only count Phase1 for PACU LOS
        # and ignore Phase2.  This is common for main OR cases.
        pacu_start_time = min(
            x for x in [case.StartPhase1, case.InPhase1] if pd.notna(x)
        )
        pacu_end_time = min(
            x for x in [case.Phase1Complete, case.OutOfPhase1] if pd.notna(x)
        )
        scenario = "Phase1"
    elif (
        not has_phase1_start
        and not has_phase1_end
        and has_phase2_start
        and has_phase2_end
    ):
        # Cases have no Phase1 timestamps and only Phase 2 timestamps.
        # This is common for brief NORA procedures (IR, MRI, GI, Cath Lab, etc.)
        pacu_start_time = case.InPhase2
        pacu_end_time = min(
            x for x in [case.Phase2Complete, case.OutOfPhase2] if pd.notna(x)
        )
        scenario = "Phase2Only"
    elif (
        not has_phase1_start
        and not has_phase1_end
        and not has_phase2_start
        and not has_phase2_end
    ):
        # Cases have no Phase 1 Start/End or Phase 2 Start/End timestamps
        if has_alt_pacu_start and has_alt_pacu_end:
            # Use Alternative Information that approximates PACU Start/End
            pacu_start_time = min(
                x for x in [case.OutOfRoom, case.InRecovery] if pd.notna(x)
            )
            pacu_end_time = min(
                x for x in [case.OutOfRecovery, case.TransferOutORTime] if pd.notna(x)
            )
            scenario = "Alt"
        else:
            # Alternative Information not available, cannot determine PACU Start/End
            pacu_start_time = pd.NaT
            pacu_end_time = pd.NaT
            scenario = "CannotDetermine"
    elif has_phase1_start and not has_phase1_end and has_alt_pacu_end:
        # Cases have Phase1 Start, but missing Phase1 End.  Use alternatives for PACU End.
        pacu_start_time = min(
            x for x in [case.StartPhase1, case.InPhase1] if pd.notna(x)
        )
        pacu_end_time = min(
            x for x in [case.OutOfRecovery, case.TransferOutORTime] if pd.notna(x)
        )
        scenario = "Phase1AltEnd"
    elif not has_phase1_start and has_phase1_end and has_alt_pacu_start:
        # Cases missing Phase1 Start, but has Phase1 End.  Use alternatives for PACU Start.
        pacu_start_time = min(
            x for x in [case.OutOfRoom, case.InRecovery] if pd.notna(x)
        )
        pacu_end_time = min(
            x for x in [case.Phase1Complete, case.OutOfPhase1] if pd.notna(x)
        )
        scenario = "Phase1AltStart"
    elif has_phase2_start and not has_phase2_end and has_alt_pacu_end:
        # Cases have Phase2 Start, but missing Phase2 End.  Use alternatives for PACU End.
        pacu_start_time = case.InPhase2
        pacu_end_time = min(
            x for x in [case.OutOfRecovery, case.TransferOutORTime] if pd.notna(x)
        )
        scenario = "Phase2AltEnd"
    elif not has_phase2_start and has_phase2_end and has_alt_pacu_start:
        # Cases missing Phase2 Start, but has Phase2 End.  Use alternatives for PACU Start.
        pacu_start_time = min(
            x for x in [case.OutOfRoom, case.InRecovery] if pd.notna(x)
        )
        pacu_end_time = min(
            x for x in [case.Phase2Complete, case.OutOfPhase2] if pd.notna(x)
        )
        scenario = "Phase2AltStart"
    else:
        pacu_start_time = pd.NaT
        pacu_end_time = pd.NaT
        scenario = "CannotDetermine"

    return {
        "PACUStartTime": pacu_start_time,
        "PACUEndTime": pacu_end_time,
        "Scenario": scenario,
        "HasPhase1Start": has_phase1_start,
        "HasPhase1End": has_phase1_end,
        "HasPhase2Start": has_phase2_start,
        "HasPhase2End": has_phase2_end,
        "HasOutOfRoom": has_out_of_room,
        "HasInRecovery": has_in_recovery,
        "HasOutOfRecovery": has_out_of_recovery,
        "HasTransferOutOR": has_transfer_out_or,
    }


def determine_pacu_start_and_end_scenarios2(case: pd.Series) -> dict:
    """Determines PACU Start & End, Phase 1 Start & End, Phase 2 Start & End.
    Cases for which we do not have Phase 1 data AND Phase 2 data are considered
    "CannotDetermine".

    Args:
        case (pd.Series): Series that contains information about a single case.

    Returns:
        dict: Dict with keys that include:
            - "Phase1StartTime"
            - "Phase1EndTime"
            - "Phase2StartTime"
            - "Phase2EndTime"
            - "PACUStartTime"
            - "PACUEndTime"
            - "Phase1Duration"
            - "Phase2Duration"
            - "PACUDuration"
            - "Scenario" = method by which times & durations are computed
    """
    # Direct Indicators for PACU Start/End
    has_phase1_start = any(pd.notna(case[x]) for x in ["InPhase1", "StartPhase1"])
    has_phase1_end = any(pd.notna(case[x]) for x in ["Phase1Complete", "OutOfPhase1"])
    has_phase2_start = any(pd.notna(case[x]) for x in ["InPhase2"])
    has_phase2_end = any(pd.notna(case[x]) for x in ["Phase2Complete", "OutOfPhase2"])
    # Indirect Indicators for PACU Start/End
    has_out_of_room = pd.notna(case.OutOfRoom)
    has_in_recovery = pd.notna(case.InRecovery)
    has_out_of_recovery = pd.notna(case.OutOfRecovery)
    has_transfer_out_or = pd.notna(case.TransferOutORTime)
    any([has_out_of_room, has_in_recovery])
    any([has_out_of_recovery, has_transfer_out_or])

    # Determine PACU Start & End
    if has_phase1_start and has_phase1_end and has_phase2_start and has_phase2_end:
        # Cases have Complete Phase 1 & Phase 2 recovery timestamps.
        # Phase 1 = true recovery period & waiting for full emergence from anesthesia
        # Phase 2 = patient getting dressed, getting discharge instructions
        phase1_start_time = min(
            x for x in [case.StartPhase1, case.InPhase1] if pd.notna(x)
        )
        phase1_end_time = min(
            x for x in [case.Phase1Complete, case.OutOfPhase1] if pd.notna(x)
        )
        phase2_start_time = case.InPhase2
        phase2_end_time = min(
            x for x in [case.Phase2Complete, case.OutOfPhase2] if pd.notna(x)
        )
        pacu_start_time = phase1_start_time
        pacu_end_time = phase2_end_time
        phase1_duration = phase1_end_time - phase1_start_time
        phase2_duration = phase2_end_time - phase2_start_time
        pacu_duration = phase1_duration + phase2_duration
        scenario = "Phase1AndPhase2"
    elif has_phase1_start and has_phase1_end:
        # Cases have Phase 1 timestamps, but no Phase 2 timestamps (rare)
        # We set Phase 2 duration = 0 for these scenarios
        phase1_start_time = min(
            x for x in [case.StartPhase1, case.InPhase1] if pd.notna(x)
        )
        phase1_end_time = min(
            x for x in [case.Phase1Complete, case.OutOfPhase1] if pd.notna(x)
        )
        phase2_start_time = pd.NaT
        phase2_end_time = pd.NaT
        pacu_start_time = phase1_start_time
        pacu_end_time = phase1_end_time
        phase1_duration = phase1_end_time - phase1_start_time
        phase2_duration = pd.Timedelta(minutes=0)
        pacu_duration = phase1_duration + phase2_duration
        scenario = "Phase1Only"
    elif has_phase2_start and has_phase2_end:
        # Cases have Phase 2 timestamps, but no Phase 1 timestamps.
        # We set Phase 1 duration = 0 for these scenarios
        # If patients are directly transferred to ICU, PACU duration is irrelevant
        # If patient is not directly transferred to ICU, these are almost all NORA
        # cases where patient arrives in PACU already in Phase 2 recovery.
        phase1_start_time = pd.NaT
        phase1_end_time = pd.NaT
        phase2_start_time = case.InPhase2
        phase2_end_time = min(
            x for x in [case.Phase2Complete, case.OutOfPhase2] if pd.notna(x)
        )
        pacu_start_time = phase2_start_time
        pacu_end_time = phase2_end_time
        phase1_duration = pd.Timedelta(minutes=0)
        phase2_duration = phase2_end_time - phase2_start_time
        pacu_duration = phase1_duration + phase2_duration
        scenario = "Phase2Only"
    else:
        # Cases don't have Phase 1 or Phase 2 timestamps.
        # Out of caution, we just don't use these cases.
        phase1_start_time = pd.NaT
        phase1_end_time = pd.NaT
        phase2_start_time = pd.NaT
        phase2_end_time = pd.NaT
        pacu_start_time = pd.NaT
        pacu_end_time = pd.NaT
        phase1_duration = pd.NaT
        phase2_duration = pd.NaT
        pacu_duration = pd.NaT
        scenario = "CannotDetermine"

    return {
        "Phase1StartTime": phase1_start_time,
        "Phase1EndTime": phase1_end_time,
        "Phase2StartTime": phase2_start_time,
        "Phase2EndTime": phase2_end_time,
        "PACUStartTime": pacu_start_time,
        "PACUEndTime": pacu_end_time,
        "Phase1Duration": phase1_duration,
        "Phase2Duration": phase2_duration,
        "PACUDuration": pacu_duration,
        "Scenario": scenario,
    }


def valid_timestamp(
    timestamp: pd.Timedelta | timedelta,
    upper_datetime: pd.Timedelta | timedelta,
    lower_datetime: pd.Timedelta | timedelta,
) -> bool | None:
    if pd.notna(timestamp):
        if lower_datetime <= timestamp <= upper_datetime:
            return True
        else:
            return False
    else:
        return None


def valid_timestamps_in_series(
    case: pd.Series,
    cols: list[str],
    lower_datetime: pd.Timedelta | timedelta,
    upper_datetime: pd.Timedelta | timedelta,
) -> pd.Series:
    is_valid = case.loc[cols].apply(
        lambda ts: valid_timestamp(
            timestamp=ts, lower_datetime=lower_datetime, upper_datetime=upper_datetime
        )
    )
    return is_valid


def check_for_valid_timestamps(
    case_df: pd.DataFrame,
    cols: list[str] = [
        "AnesStart",
        "InRoom",
        "OutOfRoom",
        "InRecovery",
        "OutOfRecovery",
        "InPhase1",
        "StartPhase1",
        "Phase1Complete",
        "OutOfPhase1",
        "InPhase2",
        "Phase2Complete",
        "OutOfPhase2",
        "OutOfRoomTime",
        "TransferOutORTime",
    ],
) -> pd.DataFrame:
    """Checks timestamps to see if they are within Admit and Discharge DateTime.

    Args:
        case_df (pd.DataFrame): Table of case data, which includes columns with timestamps
            to check
        cols (list[str], optional): Column names to check.

    Returns:
        pd.DataFrame: Table with selected columns will value in each cell indicating
            if data is valid or not.
    """
    _df = case_df.copy()
    col_str = ", ".join(cols)
    tqdm.pandas(desc=f"Validating: {col_str}")
    is_valid_df = _df.progress_apply(
        lambda row: valid_timestamps_in_series(
            case=row,
            cols=cols,
            lower_datetime=row.AdmitTime,
            upper_datetime=row.DischargeTime,
        ),
        axis=1,
    )
    return is_valid_df


def drop_invalid_timestamp_cases(
    case_df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    is_valid = check_for_valid_timestamps(case_df=case_df, cols=cols)
    # Fill missing timestamps (NaT/None) as True.
    is_valid_or_missing = is_valid.fillna(True)
    # Only keep cases where all timestamps are valid or missing.
    valid_cases = is_valid_or_missing.loc[is_valid_or_missing.all(axis=1)]
    return valid_cases


def determine_if_expected_admit(expected_patient_class: str) -> bool:
    if expected_patient_class is None:
        return None
    elif expected_patient_class in ("Inpatient", "Surgery Admit", "Emergency"):
        return True
    elif expected_patient_class in ("Outpatient", "Surgery Overnight Stay"):
        return False
    else:
        raise ValueError(
            f"Unknown value {expected_patient_class} for `expected_patient_class`."
        )


def determine_if_actual_admit(actual_patient_class: str) -> bool | str:
    if actual_patient_class is None:
        return None
    elif actual_patient_class in ("Inpatient", "Surgery Admit", "Emergency", "Newborn"):
        return True
    elif actual_patient_class in (
        "Outpatient",
        "Observation",
        "Surgery Overnight Stay",
    ):
        return False
    elif actual_patient_class in ("Deceased - Organ Donor"):
        return None
    elif actual_patient_class in (
        "Series - Misc Svcs",
        "Procedural Series",
        "Specimen",
    ):
        return "Invalid"
    else:
        raise ValueError(
            f"Unknown value {actual_patient_class} for `actual_patient_class`."
        )
