from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Generator

import pandas as pd
from llm_utils import (
    DataPaths,
    function_with_cache,
    parallel_process,
)

from make_dataset.adt import ADTEvents, post_pacu_end_intervals
from make_dataset.case import CaseEvents


@dataclass
class EncounterEvents:
    """Data object to transform post-operative PACU and ADT events into length of stay
    measurements: PACU Length of Stay, Hospital Length of Stay, ICU Length of Stay.

    This class inherits from DatasetSampler which acts as a mix-in and provides
    methods for data sampling.
    """

    paths: DataPaths
    adt: ADTEvents = field(init=False)
    case: CaseEvents = field(init=False)
    force: bool = False

    def __post_init__(self) -> None:
        "Called upon object instance creation."
        interim = self.paths.interim
        self.durations_cache_path = self.paths.register(
            interim / "pacu_hospital_icu_durations.feather"
        )

        # Load ADT & Case, Basic Data Cleaning
        self.adt = ADTEvents(paths=self.paths, force=self.force)
        self.case = CaseEvents(paths=self.paths, force=self.force)
        # Compute key info from ADT events, which requires some of the event timestamps
        # from Case events table.
        self.adt.df = self.adt.join_case_events_logic(
            adt_events_df=self.adt.adt_location_df, case_df=self.case.df
        )
        self.adt.run_step_compute_key_info_from_adt_events()
        # Join ADT Key Info to Case Events
        self.case.df = self.case.df.join(other=self.adt.key_info_df)
        # Drop ProcIDs that are not in both Case & ADT data
        self.proc_id_intersect = set(self.case.df.index) & set(
            self.adt.key_info_df.index
        )
        self.case.df = self.case.filter_cases_by_proc_ids(
            case_df=self.case.df, proc_id_whitelist=list(self.proc_id_intersect)
        )
        self.case.df = self.case.df.astype({"PAT_ID": str, "ASA": int})
        # Determine if Patient Died in Hospital Encounter
        self.case.df = self.case.determine_died_in_hospital_or_case(self.case.df)
        # Remove Cases with no Admit or Discharge ADT events.  This ensures that all
        # cases are retrospective (i.e. no currently hospitalized patients included)
        # and also have an associated hospital encounter where we can compute
        # length-of-stay metrics.  Note that generally even outpatient hospital surgery
        # and overnight stay cases have an Admit and Discharge ADT event.
        self.case.df = self.case.df.loc[
            self.case.df.AdmitTime.notna() & self.case.df.DischargeTime.notna()
        ]
        # Coerce Data Types and Fill Missing Values
        self.case.df = self.format_case_data(self.case.df)
        # Get PACU, Hospital, ICU Durations
        self.case_durations = self._get_durations(self.case.df)

        # Expose dataframes as properties of EncounterEvents
        self.adt_df = self.adt.df
        self.case_df = self.case.df

    def _get_durations(
        self, cases: pd.DataFrame, force: bool = False, **kwargs
    ) -> pd.DataFrame:
        _df = function_with_cache(
            input_data=cases,
            function=self.get_durations,
            cache_path=self.durations_cache_path,
            set_index="ProcID",
            force=force,
            **kwargs,
        )
        return _df

    def get_durations(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Get PACU, Hospital, ICU Duration for Cases

        This method drops all cases with invalid computed durations."""
        _cases = cases.copy()
        # Partition Organ Donor vs. Non-Organ Donor Cases
        organ_donor_cases = _cases.loc[_cases.IsOrganDonorCase]
        non_organ_donor_cases = _cases.loc[_cases.IsOrganDonorCase.eq(False)]
        # Get PACU Durations
        non_organ_donor_cases = self.determine_pacu_duration(non_organ_donor_cases)
        # NOTE: Hospital Duration depends on knowing PACU End time
        # Hospital Length of Stay (using ADT movement events)
        non_organ_donor_cases = self.determine_hospital_stay_duration(
            non_organ_donor_cases
        )
        # ICU Length of Stay (using ADT movement events)
        non_organ_donor_cases = self.determine_icu_stay_duration(
            non_organ_donor_cases, adt_df=self.adt.df
        )
        # Hospital & ICU Length of Stay (using census events)
        non_organ_donor_cases = self.determine_hospital_icu_stay_duration_from_census(
            case_df=non_organ_donor_cases, adt_df=self.adt.df
        )
        # Drop Cases with discordance between Hospital Length of Stay Methods
        non_organ_donor_cases = self.drop_cases_with_discrepant_hospital_duration(
            non_organ_donor_cases
        )
        # Drop Cases with discordance between ICU Length of Stay Methods
        non_organ_donor_cases = self.drop_cases_with_discrepant_icu_duration(
            non_organ_donor_cases
        )
        # Combine Organ Donor & Non-Organ Donor Cases
        return self.combine_cases(
            organ_donor_cases=organ_donor_cases,
            non_organ_donor_cases=non_organ_donor_cases,
        )

    def format_case_data(self, cases: pd.DataFrame) -> pd.DataFrame:
        "Cleanup Case Data, Fill Missing Data, Coerce Data Types"
        _cases = cases.copy()

        # Fill Missing Columns
        cols_to_fill_with_empty_str = [
            "SurgService",
            "ScheduledProcedure",
            "ProcedureDescription",
            "Diagnosis",
            "ExpectedPatientClass",
            "ActualPatientClass",
        ]
        for col in _cases.columns:
            if col in cols_to_fill_with_empty_str:
                _cases[col] = _cases[col].fillna("")
            if col in ("AnesType"):
                _cases[col] = _cases[col].fillna("unknown")

        # NOTE: All cases have valid values for columns `PAT_ID`` and `IsOrganDonorCase`

        # Coerce Data Types if Column is in labels
        col_dtypes_dict = {
            "PAT_ID": str,
            "SurgService": str,
            "ScheduledProcedure": str,
            "ProcedureDescription": str,
            "Diagnosis": str,
            "ExpectedPatientClass": str,
            "ActualPatientClass": str,
            "ASA": int,
            "Scenario": str,
            "LocationAfterProcedure": str,
            "AnticipatedAdmit": bool,
            "ActualAdmit": bool,
        }
        col_dtypes_dict = {
            k: v for k, v in col_dtypes_dict.items() if k in _cases.columns
        }
        _cases = _cases.astype(col_dtypes_dict)
        return _cases

    def combine_cases(
        self,
        organ_donor_cases: pd.DataFrame,
        non_organ_donor_cases: pd.DataFrame,
    ) -> pd.DataFrame:
        """Case dataset with non-organ donor cases with valid length-of-stay durations
        combined with ASA 6 organ donor cases."""
        # Enforce values for Organ Donor Cases.
        # - Organ Donor patients are dead prior to entering procedure, so all
        #   length of stay durations should be 0.
        # - Organ Donor cases are ASA 6 by definition.  Sometimes the anesthesiologist
        #   will make the wrong assignment here, so we force ASA=6.
        # - Organ Donor cases by definition are dead.  We force HasExpiredInHospital=True
        #   because death declaration is mandatory for this case to occur.  This may
        #   not be properly recorded in the EHR data.
        organ_donor_cases = organ_donor_cases.assign(
            Scenario="OrganDonor",
            Phase1StartTime=pd.NaT,
            Phase1EndTime=pd.NaT,
            Phase1Duration=pd.Timedelta(seconds=0),
            Phase2StartTime=pd.NaT,
            Phase2EndTime=pd.NaT,
            Phase2Duration=pd.Timedelta(seconds=0),
            PACUStartTime=pd.NaT,
            PACUEndTime=pd.NaT,
            PACUDuration=pd.Timedelta(seconds=0),
            HospitalDuration=pd.Timedelta(seconds=0),
            HospitalDuration2=pd.Timedelta(seconds=0),
            ICUDuration=pd.Timedelta(seconds=0),
            ICUDuration2=pd.Timedelta(seconds=0),
            ASA=6,
            HasExpiredInHospital=True,
        )
        # Combine Organ Donor & Non-Organ Donor Cases
        all_cases = pd.concat(
            [
                non_organ_donor_cases,
                organ_donor_cases,
            ],
            axis=0,
        )
        return all_cases

    def determine_length_of_stay(self, case_df: pd.DataFrame) -> pd.DataFrame:
        _df = case_df.copy()
        # Hospital Length of Stay
        _df = self.determine_hospital_stay_duration(_df)
        # ICU Length of Stay
        _df = self.determine_icu_stay_duration(case_df=_df, adt_df=self.adt.df)
        # Hospital & ICU Length of Stay (using census events)
        _df = self.determine_hospital_icu_stay_duration_from_census(
            case_df=_df, adt_df=self.adt.df
        )
        return _df

    def determine_pacu_duration(self, case_df: pd.DataFrame) -> pd.DataFrame:
        """Compute PACU Duration and remove cases with invalid durations.

        Args:
            case_df (pd.DataFrame): Table cases that contains the columns
                "PACUStartTime" and "PACUEndTime".

        Returns:
            pd.DataFrame: Table of cases with PACU Durations
        """
        _df = case_df.copy()
        # Determine PACU Start, End & Durations
        _df = self.case.determine_pacu_phases(_df)
        # Remove Cases where PACU Event Timestamps do not fall within Admit & Discharge Timestamps
        _df = self.case.validate_pacu_phases(_df)
        # Remove cases with PACU Durations <0 seconds & >1 day
        _df = _df.loc[
            (_df.Phase1Duration >= timedelta(seconds=0))
            & (_df.Phase1Duration < timedelta(days=1))
        ]
        _df = _df.loc[
            (_df.Phase2Duration >= timedelta(seconds=0))
            & (_df.Phase2Duration < timedelta(days=1))
        ]
        _df = _df.loc[
            (_df.PACUDuration >= timedelta(seconds=0))
            & (_df.PACUDuration < timedelta(days=1))
        ]
        return _df

    def drop_cases_with_discrepant_hospital_duration(
        self, cases: pd.DataFrame
    ) -> pd.DataFrame:
        """Hospital Duration & ICU Duration can be computed using either ADT Orders or
        ADT Census Events. In some cases, there is a large discrepancy between the 2 methods.
        We drop these cases for quality control and we keep only cases where there is a
        <24 hour discrepancy between the 2 methods.
        """
        _cases = cases.copy()
        hospital_stay_delta = abs(_cases.HospitalDuration - _cases.HospitalDuration2)
        hospital_stay_concordant = hospital_stay_delta < pd.Timedelta(hours=24)
        return _cases.loc[hospital_stay_concordant]

    def drop_cases_with_discrepant_icu_duration(
        self, cases: pd.DataFrame
    ) -> pd.DataFrame:
        """ICU Duration can be computed using either ADT Orders or ADT Census Events.
        In some cases, there is a large discrepancy between the 2 methods.
        We drop these cases for quality control and we keep only cases where there is a
        <24 hour discrepancy between the 2 methods."""
        _cases = cases.copy()
        icu_stay_delta = abs(_cases.ICUDuration - _cases.ICUDuration2)
        icu_stay_concordant = icu_stay_delta < pd.Timedelta(hours=24)
        return _cases.loc[icu_stay_concordant]

    def determine_hospital_stay_duration(self, case_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Hospital Stay Duration using PACUEndTime and DischargeTime,
        and remove cases with invalid durations.

        Args:
            case_df (pd.DataFrame): Table of cases that contains the columns
                "DischargeTime" and "PACUEndTime".

        Returns:
            pd.DataFrame: Table of cases with Hospital Stay Durations
        """
        _df = case_df.copy()
        hospital_duration = (_df.DischargeTime - _df.PACUEndTime).rename(
            "HospitalDuration"
        )
        _df = _df.join(hospital_duration)
        # Remove cases with Hospital Stay Durations less than 0 seconds
        _df = _df.loc[_df.HospitalDuration >= timedelta(seconds=0)]
        return _df

    def determine_icu_stay_duration(
        self, case_df: pd.DataFrame, adt_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute ICU Stay Duration using PACUEndTime, DischargeTime, and ADT Intervals,
        and remove cases with invalid durations.

        Args:
            case_df (pd.DataFrame): Table of cases with all proc_id.
            adt_df (pd.DataFrame): Table of ADT events for all proc_id.

        Returns:
            pd.DataFrame: Table of cases with ICU Stay Durations
        """
        _case = case_df.copy()
        _adt = adt_df.copy()

        def generator(
            case_df: pd.DataFrame = _case, adt_df: pd.DataFrame = _adt
        ) -> Generator[dict[str, Any], None, None]:
            for proc_id, case in case_df.iterrows():
                adt = adt_df.loc[adt_df.ProcID == proc_id]
                yield {"proc_id": proc_id, "case": case, "adt": adt}

        # Grab ADT for each proc_id
        icu_duration = parallel_process(
            iterable=generator(case_df=_case, adt_df=_adt),
            function=compute_icu_los,
            use_kwargs=True,
            desc="Determining ICU LOS",
        )
        icu_duration = pd.DataFrame(icu_duration).set_index("ProcID")
        _case = _case.join(icu_duration)
        return _case

    def determine_hospital_icu_stay_duration_from_census(
        self, case_df: pd.DataFrame, adt_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute ICU Duration using ADT census events.  The resolution of ICU length of stay
        is less than using all ADT transfer in/out events, but may be more similar to how
        clinicians think about ICU days.

        Args:
            case_df (pd.DataFrame): Table of cases (1 row for each ProcID)
            adt_df (pd.DataFrame): Table of ADT events (multiple rows for each ProcID)

        Returns:
            pd.DataFrame: Table of cases with ICU Stay Durations, determined by counting
                census ADT events where location is ICU.
        """
        _case = case_df.copy()
        _adt = adt_df.copy()

        def generator(
            case_df: pd.DataFrame = _case, adt_df: pd.DataFrame = _adt
        ) -> Generator[dict[str, Any], None, None]:
            for proc_id, case in case_df.iterrows():
                adt = adt_df.loc[adt_df.ProcID == proc_id]
                yield {"proc_id": proc_id, "case": case, "adt": adt}

        # Grab ADT for each proc_id
        icu_duration = parallel_process(
            iterable=generator(case_df=_case, adt_df=_adt),
            function=compute_hosp_icu_los_from_census,
            use_kwargs=True,
            desc="Getting Post-Procedure Census Events",
        )
        icu_duration = pd.DataFrame(icu_duration).set_index("ProcID")
        _case = _case.join(icu_duration)
        return _case


def compute_icu_los(proc_id: str, case: pd.Series, adt: pd.DataFrame) -> dict:
    """Compute ICU LOS for a single case.

    Args:
        proc_id (str): Unique identifier for procedure.
        case (pd.Series): Pandas series of case info for the proc_id.
        adt (pd.DataFrame): Pandas dataframe of ADT events for encounter where
            proc_id occurs.

    Returns:
        dict: Dictionary with ProcID & ICU length of stay timedelta.
    """
    _adt = adt.copy()
    pacu_end_time = case["PACUEndTime"]
    # ADT Intervals that occur after PACU End
    d = post_pacu_end_intervals(adt=_adt, pacu_end_time=pacu_end_time)
    pacu_end_interval = d["PACUEndInterval"]
    post_pacu_intervals = d["PostPACUIntervals"]

    if post_pacu_intervals.empty:
        # No ADT intervals after PACU End
        icu_duration = pd.Timedelta(seconds=0)
    else:
        # Get Durations for ADT Intervals
        durations = post_pacu_intervals.EndTime - post_pacu_intervals.StartTime
        post_pacu_intervals = post_pacu_intervals.assign(Duration=durations)
        icu_intervals = post_pacu_intervals.loc[post_pacu_intervals.Location == "ICU"]
        icu_duration = icu_intervals.Duration.sum()

        # If First Post-PACU ADT Location is ICU, we get duration between
        # PACU End and return to ICU and include it as part of ICU duration.
        # This accounts for scenario where procedure occurred but there is no
        # corresponding ADT event transferring patient to OR (case.HasTransferOR == False).
        # Then part of the ADT interval where procedure occurs should count
        # toward ICU stay if patient was in ICU at that time.
        if post_pacu_intervals.Location.iloc[0] == "ICU":
            # Get Time from PACU End to Start of First Post Procedure ADT Interval
            pacu_to_icu = pacu_end_interval.EndTime.iloc[0] - pacu_end_time
        else:
            pacu_to_icu = pd.Timedelta(seconds=0)

        icu_duration = icu_duration + pacu_to_icu

    return {
        "ProcID": proc_id,
        "ICUDuration": icu_duration,
    }


def compute_hosp_icu_los_from_census(
    proc_id: str, case: pd.Series, adt: pd.DataFrame
) -> dict:
    """Compute Hospital & ICU LOS for a single case using ADT Census events.

    Args:
        proc_id (str): Unique identifier for procedure.
        case (pd.Series): Pandas series of case info for the proc_id.
        adt (pd.DataFrame): Pandas dataframe of ADT events for encounter where
            proc_id occurs.

    Returns:
        dict: Dictionary with ProcID, Hospital Length of Stay, ICU Length of Stay.
    """
    pacu_end_time = case["PACUEndTime"]
    _adt = adt.copy()
    # Filter Event Types
    event_types = ["Census", "Discharge"]
    _adt = _adt.loc[_adt.EventType.isin(event_types)]
    # Get ADT Census Events that Occur After PACU End
    post_procedure_adt = _adt.loc[_adt.EFFECTIVE_TIME >= pacu_end_time]
    # Compute Hospital Length of Stay
    hosp_adt = post_procedure_adt.copy()
    if hosp_adt.empty:
        hosp_duration = pd.Timedelta(seconds=0)
    else:
        hosp_census_events = hosp_adt.loc[hosp_adt.EventType == "Census"]
        hosp_days = pd.Timedelta(days=len(hosp_census_events))
        discharge_event = hosp_adt.loc[hosp_adt.EventType == "Discharge"]

        if discharge_event.empty:
            hosp_duration = hosp_days
        else:
            # If discharge from hospital and there has not been a census event
            # registered for that day, then we count the partial day up until discharge time
            discharge_time = discharge_event.EFFECTIVE_TIME.item()
            if discharge_time.date in hosp_census_events.EFFECTIVE_TIME.tolist():
                discharge_day_midnight = pd.Timestamp(
                    year=discharge_time.year,
                    month=discharge_time.month,
                    day=discharge_time.day,
                )
                discharge_day_duration = discharge_time - discharge_day_midnight
                hosp_duration = hosp_days + discharge_day_duration
            else:
                # Census event registered on day of discharge, so the discharge date
                # is already counted as a hospital stay day.
                hosp_duration = hosp_days

    # Compute ICU Length of Stay
    # Get Only ICU Events
    icu_adt = post_procedure_adt.loc[post_procedure_adt.Location == "ICU"]
    if icu_adt.empty:
        icu_duration = pd.Timedelta(seconds=0)
    else:
        icu_census_events = icu_adt.loc[icu_adt.EventType == "Census"]
        icu_days = pd.Timedelta(days=len(icu_census_events))
        discharge_event = icu_adt.loc[icu_adt.EventType == "Discharge"]

        if discharge_event.empty:
            icu_duration = icu_days
        else:
            # If direct discharge from ICU and there has not been a census event
            # registered for that day, then we count the partial day up until discharge time
            discharge_time = discharge_event.EFFECTIVE_TIME.item()
            if discharge_time.date in icu_census_events.EFFECTIVE_TIME.tolist():
                discharge_day_midnight = pd.Timestamp(
                    year=discharge_time.year,
                    month=discharge_time.month,
                    day=discharge_time.day,
                )
                discharge_day_duration = discharge_time - discharge_day_midnight
                icu_duration = icu_days + discharge_day_duration
            else:
                # Census event registered on day of discharge, so the discharge date
                # is already counted as an ICU stay day
                icu_duration = icu_days

    return {
        "ProcID": proc_id,
        "HospitalDuration2": hosp_duration,
        "ICUDuration2": icu_duration,
    }
