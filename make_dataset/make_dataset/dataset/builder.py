from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from llm_utils import (
    DataPaths,
    read_pandas,
    save_pandas,
    timedelta2days,
    timedelta2minutes,
)

from make_dataset.dataset.sampler import DatasetSampler
from make_dataset.encounter import EncounterEvents
from make_dataset.notes import LastTenNote, PreAnesNote

builder_logger = logging.getLogger(name="BuilderLogger")


@dataclass(kw_only=True)
class DatasetBundle:
    preanes_inference: pd.DataFrame | None = None
    preanes_fewshot: pd.DataFrame | None = None
    last10_inference: pd.DataFrame | None = None
    last10_fewshot: pd.DataFrame | None = None


@dataclass(kw_only=True)
class DatasetBuilder(DatasetSampler):
    """Builder Class used to construct a datasets.

    `create_dataset` will create a unique dataset based on:
    - dataset_type: "asa", "phase1_duration", "hospital_duration", "icu_duration",
        "unplanned_admit", "hospital_mortality"

    The resultant dataset will be cached on disk for fast repeat access.  Calling
    `create_dataset` again with the same input arguments will simply load the cached
    version of the dataset.
    """

    paths: DataPaths
    dataset_version: int | str
    force: bool = False
    E: EncounterEvents = field(init=False)
    preanes_notes: PreAnesNote = field(init=False)
    last10_notes: LastTenNote = field(init=False)
    common_note_proc_ids: set = field(init=False)

    def __post_init__(self) -> None:
        "Called upon object instance creation."
        builder_logger.info("Loading EncounterEvents.")
        self.E = EncounterEvents(paths=self.paths, force=self.force)
        builder_logger.info("Loading Anesthesia Preoperative Evaluation Notes")
        self.preanes_notes = PreAnesNote(paths=self.paths, concatenate_notes=False)
        builder_logger.info("Loading Last 10 Clinical Notes")
        self.last10_notes = LastTenNote(paths=self.paths, concatenate_notes=False)
        # Select subset of notes
        self.select_note_ids()

    def select_note_ids(self) -> None:
        "Select subset of cases & notes which will be included in dataset."
        # Remove cases where the sum of all note text <= 100 tokens
        # Many short notes do not have adequate clinical content and many
        # are simply attestations or a physician documenting that they reviewed
        # a paper record or chart in another system, etc.
        preanes_notes = self.preanes_notes.df
        builder_logger.info(f"Starting Preanes Note Count: {preanes_notes.shape[0]}")
        total_preanes_note_token_length = preanes_notes.NoteTextTokenLength.apply(sum)
        preanes_notes = preanes_notes.loc[total_preanes_note_token_length > 100]
        builder_logger.info(
            f"Preanes Note Count (remove total note text < 100 tokens): {preanes_notes.shape[0]}"
        )

        last10_notes = self.last10_notes.df
        builder_logger.info(f"Starting Last10 Note Count: {last10_notes.shape[0]}")
        total_last10_note_token_length = last10_notes.NoteTextTokenLength.apply(sum)
        last10_notes = last10_notes.loc[total_last10_note_token_length > 100]
        builder_logger.info(
            f"Last10 Note Count (remove total note text < 100 tokens): {last10_notes.shape[0]}"
        )
        # Get ProcIDs where we have both kinds of notes
        self.common_note_proc_ids = set(preanes_notes.index) & set(last10_notes.index)
        builder_logger.info(f"Common Note ProcID Count: {len(self.common_note_proc_ids)}")

    def create_dataset(
        self,
        dataset_type: str,
        n: int | None = 1250,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        "Dataset constructor method."
        # Create Dataset of Cases
        match dataset_type:
            case "asa":
                make_dataset_fn = self.select_asa_cases
            case "phase1_duration":
                make_dataset_fn = self.select_phase1_duration_cases
            case "hospital_duration":
                make_dataset_fn = self.select_hospital_duration_cases
            case "hospital_admission":
                make_dataset_fn = self.select_hospital_admission_cases
            case "icu_duration":
                make_dataset_fn = self.select_icu_duration_cases
            case "icu_admission":
                make_dataset_fn = self.select_icu_admission_cases
            case "unplanned_admit":
                make_dataset_fn = self.select_unplanned_admit_cases
            case "hospital_mortality":
                make_dataset_fn = self.select_hospital_mortality_cases

        df = make_dataset_fn(n=n, seed=seed).set_index("ProcID")

        # Join Case Info & Notes
        preanes_df = df.join(self.preanes_notes.df, how="left")
        last10_df = df.join(self.last10_notes.df, how="left")

        # Partition 80% to Inference Dataset, 20% to Fewshot Dataset
        preanes_inference = preanes_df.sample(frac=0.8, random_state=seed)
        preanes_fewshot = preanes_df.loc[~preanes_df.index.isin(preanes_inference.index)]
        last10_inference = last10_df.loc[preanes_inference.index]
        last10_fewshot = last10_df.loc[~last10_df.index.isin(last10_inference.index)]
        builder_logger.info(f"Preanes Inference Split: {len(preanes_inference)}")
        builder_logger.info(f"Preanes Fewshot Split: {len(preanes_fewshot)}")
        builder_logger.info(f"Last10 Inference Split: {len(last10_inference)}")
        builder_logger.info(f"Last10 Fewshot Split: {len(last10_fewshot)}")

        # Save Dataset
        preanes_inference_path = self.make_dataset_path(
            dataset_type=dataset_type, split="inference", note_type="preanes"
        )
        preanes_fewshot_path = self.make_dataset_path(
            dataset_type=dataset_type, split="fewshot", note_type="preanes"
        )
        last10_inference_path = self.make_dataset_path(
            dataset_type=dataset_type, split="inference", note_type="last10"
        )
        last10_fewshot_path = self.make_dataset_path(
            dataset_type=dataset_type, split="fewshot", note_type="last10"
        )
        save_pandas(df=preanes_inference, path=preanes_inference_path)
        save_pandas(df=preanes_fewshot, path=preanes_fewshot_path)
        save_pandas(df=last10_inference, path=last10_inference_path)
        save_pandas(df=last10_fewshot, path=last10_fewshot_path)
        return DatasetBundle(
            preanes_inference=preanes_inference,
            preanes_fewshot=preanes_fewshot,
            last10_inference=last10_inference,
            last10_fewshot=last10_fewshot,
        )

    def make_dataset_path(
        self,
        dataset_type: str,
        split: str,
        note_type: str,
    ) -> Path:
        "Convenience method for building path to save data to disk."
        return (
            self.paths.processed
            / f"{self.dataset_version}"
            / f"{dataset_type}-{note_type}-{split}.feather"
        )

    def get_dataset(
        self,
        path: str | Path | None = None,
        dataset_type: str | None = None,
        note_type: str | None = None,
        split: str | None = None,
        set_index: str = "ProcID",
    ) -> pd.DataFrame:
        """Read existing dataset on disk.

        Args:
            path (str | Path | None, optional): Path to dataset on disk. If not specified,
                will be inferred from `dataset_type` and `note_type`.
            dataset_type (str | None, optional): Outcome variable.
            note_type (str | None, optional): Note type.
            set_index (str): Column variable to use as index.

        Returns:
            pd.DataFrame: Table that was persisted on disk
        """
        if path is None:
            if any(
                x is None
                for x in [
                    dataset_type,
                    note_type,
                    split,
                ]
            ):
                raise ValueError(
                    "Must provide full `path` or `dataset_type`, `split`, `note_type`."
                )
            path = self.make_dataset_path(
                dataset_type=dataset_type,
                note_type=note_type,
                split=split,
            )
        else:
            path = Path(path)
        return read_pandas(path=path, set_index=set_index)

    @staticmethod
    def select_cols(label_name: str) -> list[str]:
        "Columns from Case DataFrames for each Dataset."
        # Case Metadata
        shared_cols = [
            "PAT_ID",
            "ASA",
            "AnesType",
            "SurgService",
            "ScheduledProcedure",
            "ProcedureDescription",
            "Diagnosis",
            "ExpectedPatientClass",
            "ActualPatientClass",
            "IsOrganDonorCase",
            "LocationAfterProcedure",
        ]
        duration_cols = [
            "Scenario",
            "Phase1Duration",
            "HospitalDuration",
            "HospitalDuration2",
            "ICUDuration",
            "ICUDuration2",
        ]
        unplanned_admit_cols = [
            "AnticipatedAdmit",
            "ActualAdmit",
            "UnplannedAdmit",
            "UnplannedOutpatient",
        ]
        hospital_mortality_cols = ["HasExpiredInCase", "HasExpiredInHospital"]
        match label_name:
            case "asa" | "unplanned_admit" | "hospital_mortality":
                return shared_cols + unplanned_admit_cols + hospital_mortality_cols
            case "phase1_duration" | "hospital_duration" | "icu_duration":
                return shared_cols + unplanned_admit_cols + hospital_mortality_cols + duration_cols

    def select_asa_cases(
        self,
        n: int | None = 1250,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Make Dataset with ASA-PS Label.

        ASA-PS is not uniformly distributed in the population.  Less than 1% of patients
        are ASA 5 or ASA 6. Most patients are ASA 2 and 3. A smaller fraction are ASA 1 and 4.
        To get an equal representation of ASA-PS in this dataset, we use inverse frequency
        sampling of ASA-PS to get a roughly equal number of cases in each of the
        ASA-PS categories, though there will still be much fewer ASA 5 and 6 since
        these cases are so rare that there are only a handful in the entire original dataset.

        Organ Donor cases are not removed.
        """
        case_df = self.E.case.df.copy()
        # Select Columns
        df = case_df.loc[:, self.select_cols("asa")].rename(
            columns={
                "HasExpiredInHospital": "HospitalMortality",
                "HasExpiredInCase": "CaseMortality",
            }
        )
        builder_logger.info(f"Starting Num Cases: {df.shape[0]}")
        # Filter by ProcIDs where we have Preanes & Last10 Notes
        df = df.loc[list(set(df.index) & self.common_note_proc_ids)]
        builder_logger.info(
            f"Num Cases (filter by ProcIDs where we have PreAnes & Last10 Notes): {df.shape[0]}"
        )
        # Separate Rare Outcomes
        asa6 = df.query("ASA == 6")
        asa5 = df.query("ASA == 5")
        others = df.query("~ASA.isin([5, 6])")
        # Enforce 1 Case per Patient, Maximizing Rare Cases Kept
        asa5_filtered = asa5.query(f"~PAT_ID.isin({asa6.PAT_ID.tolist()})")
        others_filtered = others.query(
            f"~PAT_ID.isin({asa6.PAT_ID.tolist()}) & ~PAT_ID.isin({asa5.PAT_ID.tolist()})"
        )
        df = pd.concat([others_filtered, asa5_filtered, asa6], axis=0)
        df = self.select_cases_per_patient(df=df, max_cases=1, seed=seed)
        builder_logger.info(f"1 Case Per Patient. Num Cases: {df.shape[0]}")
        # Downsample Cases to Target Number
        if n is not None:
            # Downsample
            df = self.sample_inverse_frequency(df=df, col="ASA", n=n, seed=seed)
            builder_logger.info(f"After Downsample. Num Cases: {df.shape[0]}")
        return df

    def select_phase1_duration_cases(
        self,
        n: int | None = 1250,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Make Dataset with Phase 1 Duration Label.

        Most (but not all patients) require some Phase 1 recovery. This is common for patients
        who are recovering from OR surgery, but in NORA locations such as GI and IR, many
        patients who receive MAC/sedation may essentially complete Phase 1 recovery by the
        time they arrive in PACU/recovery area and move directly to Phase 2 recovery. Other
        patients who are severly ill will skip the  PACU/recovery area and be transferred
        directly to the ICU.  There are also patients on the long tail who require a
        long amount of time in PACU/recovery. To adequately represent all these kinds of
        patients in the dataset, we bin the Phase1 PACU duration into 10-percentile bins
        and sample a roughly equal number of cases in each of these bins.

        Organ Donor cases are removed.
        """
        # Get Cases w/ PACU Durations
        df = self.E.case_durations.copy()
        # Select Columns
        df = df.loc[:, self.select_cols("phase1_duration")].rename(
            columns={
                "HasExpiredInHospital": "HospitalMortality",
                "HasExpiredInCase": "CaseMortality",
            }
        )
        builder_logger.info(f"Starting Num Cases: {df.shape[0]}")
        # Filter by ProcIDs where we have Preanes & Last10 Notes
        df = df.loc[list(set(df.index) & self.common_note_proc_ids)]
        builder_logger.info(
            f"Num Cases (filter by ProcIDs where we have PreAnes & Last10 Notes): {df.shape[0]}"
        )
        # Convert to Timedelta to Minutes & Days
        df = df.assign(
            Phase1Duration=df["Phase1Duration"].apply(timedelta2minutes),
            HospitalDuration=df["HospitalDuration"].apply(timedelta2days),
            HospitalDuration2=df["HospitalDuration2"].apply(timedelta2days),
            HospitalAdmission=df["HospitalDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
            ICUDuration=df["ICUDuration"].apply(timedelta2days),
            ICUDuration2=df["ICUDuration2"].apply(timedelta2days),
            ICUAdmission=df["ICUDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
        )
        # Remove Organ Donor Cases
        df = df.loc[~df.IsOrganDonorCase]
        builder_logger.info(f"Remove Organ Donor Cases. Num Cases: {df.shape[0]}")
        # Enforce 1 Case per Patient
        df = self.select_cases_per_patient(df=df, max_cases=1, seed=seed)
        builder_logger.info(f"1 Case Per Patient. Num Cases: {df.shape[0]}")
        # Downsample Cases to Target Number
        if n is not None:
            # Bin Durations based on Percentiles
            num_bins = 20
            col = df.Phase1Duration
            bin_boundaries = np.linspace(start=0, stop=1, num=num_bins + 1)
            percentiles = col.quantile(bin_boundaries)
            # Create Bins (duplicate bin boundaries combined)
            bins = pd.cut(
                col,
                bins=percentiles,
                include_lowest=True,
                duplicates="drop",
            )
            # Use Bins as Group/Class identities for Inverse Frequency Sampling
            df = df.assign(Group=bins)
            df = self.sample_inverse_frequency(df=df, col="Group", n=n, seed=seed).drop(
                columns="Group"
            )
            builder_logger.info(f"After Downsample. Num Cases: {df.shape[0]}")
        return df

    def select_hospital_duration_cases(
        self,
        n: int | None = 1250,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Make Dataset with Hospital Duration Label.

        Not all patients get admitted after surgery, and there is also a long tail with
        majority of hospital admitted patients getting discharged within a few days and
        a small number staying hospital for weeks/months.  We balance the representation
        across all durations by splitting the durations into percentile bins and then
        using inverse frequency weighting to downsample the population of cases.

        Organ Donor cases are removed.
        """
        # Get Cases w/ Hospital Durations
        df = self.E.case_durations.copy()
        # Select Columns
        df = df.loc[:, self.select_cols("hospital_duration")].rename(
            columns={
                "HasExpiredInHospital": "HospitalMortality",
                "HasExpiredInCase": "CaseMortality",
            }
        )
        builder_logger.info(f"Starting Num Cases: {df.shape[0]}")
        # Filter by ProcIDs where we have Preanes & Last10 Notes
        df = df.loc[list(set(df.index) & self.common_note_proc_ids)]
        builder_logger.info(
            f"Num Cases (filter by ProcIDs where we have PreAnes & Last10 Notes): {df.shape[0]}"
        )
        # Convert to Timedelta to Minutes & Days
        df = df.assign(
            Phase1Duration=df["Phase1Duration"].apply(timedelta2minutes),
            HospitalDuration=df["HospitalDuration"].apply(timedelta2days),
            HospitalDuration2=df["HospitalDuration2"].apply(timedelta2days),
            HospitalAdmission=df["HospitalDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
            ICUDuration=df["ICUDuration"].apply(timedelta2days),
            ICUDuration2=df["ICUDuration2"].apply(timedelta2days),
            ICUAdmission=df["ICUDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
        )
        # Remove Organ Donor Cases
        df = df.loc[~df.IsOrganDonorCase]
        builder_logger.info(f"Remove Organ Donor Cases. Num Cases: {df.shape[0]}")
        # Enforce 1 Case per Patient
        df = self.select_cases_per_patient(df=df, max_cases=1, seed=seed)
        builder_logger.info(f"1 Case Per Patient. Num Cases: {df.shape[0]}")
        # Downsample Cases to Target Number
        if n is not None:
            # Bin Durations based on Percentiles
            num_bins = 100
            col = df.HospitalDuration2
            bin_boundaries = np.linspace(start=0, stop=1, num=num_bins + 1)
            percentiles = col.quantile(bin_boundaries)
            # Create Bins (duplicate bin boundaries combined)
            bins = pd.cut(
                col,
                bins=percentiles,
                include_lowest=True,
                duplicates="drop",
            )
            # Use Bins as Group/Class identities for Inverse Frequency Sampling
            df = df.assign(Group=bins)
            df = self.sample_inverse_frequency(df=df, col="Group", n=n, seed=seed).drop(
                columns="Group"
            )
            builder_logger.info(f"After Downsample. Num Cases: {df.shape[0]}")
        return df

    def select_hospital_admission_cases(
        self,
        n: int | None = 1250,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Make Dataset with Hospital Admission Label.  Unlike Hospital Duration,
        this is a Boolean yes/no dataset for Hospital Admission, and the dataset
        construction is different as we are asking a slightly different question.

        Not all patients get admitted after surgery.
        To have a balanced representation, we determine whether patient is admitted or not
        as indicated by having a non-zero hospital admission duration and take the
        cross product with PatientClass which can be: {"inpatient", "outpatient",
        "surgery admit", "overnight stay", "observation", "emergency"}.
        We then inverse frequency sample this cross product which jointy balances
        both hospital admission and PatientClass.

        Organ Donor cases are removed.
        """
        # Get Cases w/ Hospital Durations
        df = self.E.case_durations.copy()
        # Select Columns
        df = df.loc[:, self.select_cols("hospital_duration")].rename(
            columns={
                "HasExpiredInHospital": "HospitalMortality",
                "HasExpiredInCase": "CaseMortality",
            }
        )
        builder_logger.info(f"Starting Num Cases: {df.shape[0]}")
        # Filter by ProcIDs where we have Preanes & Last10 Notes
        df = df.loc[list(set(df.index) & self.common_note_proc_ids)]
        builder_logger.info(
            f"Num Cases (filter by ProcIDs where we have PreAnes & Last10 Notes): {df.shape[0]}"
        )
        # Convert to Timedelta to Minutes & Days
        df = df.assign(
            Phase1Duration=df["Phase1Duration"].apply(timedelta2minutes),
            HospitalDuration=df["HospitalDuration"].apply(timedelta2days),
            HospitalDuration2=df["HospitalDuration2"].apply(timedelta2days),
            HospitalAdmission=df["HospitalDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
            ICUDuration=df["ICUDuration"].apply(timedelta2days),
            ICUDuration2=df["ICUDuration2"].apply(timedelta2days),
            ICUAdmission=df["ICUDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
        )
        # Remove Organ Donor Cases
        df = df.loc[~df.IsOrganDonorCase]
        builder_logger.info(f"Remove Organ Donor Cases. Num Cases: {df.shape[0]}")
        # Enforce 1 Case per Patient
        df = self.select_cases_per_patient(df=df, max_cases=1, seed=seed)
        builder_logger.info(f"1 Case Per Patient. Num Cases: {df.shape[0]}")
        # Downsample Cases to Target Number
        if n is not None:
            # Inverse Frequency Sampling of Patient Class x Hospital Admission
            def assign_hospital_admit_group(case: pd.Series) -> str:
                return f"{case.ActualPatientClass}-{case.HospitalAdmission}"

            df = df.assign(Group=df.apply(assign_hospital_admit_group, axis=1))
            df = self.sample_inverse_frequency(df=df, col="Group", n=n, seed=seed).drop(
                columns="Group"
            )
            builder_logger.info(f"After Downsample. Num Cases: {df.shape[0]}")
        return df

    def select_icu_duration_cases(
        self,
        n: int | None = 1250,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Make Dataset with ICU Duration Label.

        ICU admission is a rare event that only occurs in ~1.6% cases.  This dataset
        aims to study prediction performance of ICU duration, so we will remove all
        non-ICU admission cases and only consider cases with ICU admission. Prediction
        of admission to ICU or not will utilize a separate dataset `icu_admission_cases`.

        Organ Donor cases are removed.
        """
        # Get Cases w/ ICU Durations
        df = self.E.case_durations.copy()
        # Select Columns
        df = df.loc[:, self.select_cols("icu_duration")].rename(
            columns={
                "HasExpiredInHospital": "HospitalMortality",
                "HasExpiredInCase": "CaseMortality",
            }
        )
        builder_logger.info(f"Starting Num Cases: {df.shape[0]}")
        # Filter by ProcIDs where we have Preanes & Last10 Notes
        df = df.loc[list(set(df.index) & self.common_note_proc_ids)]
        builder_logger.info(
            f"Num Cases (filter by ProcIDs where we have PreAnes & Last10 Notes): {df.shape[0]}"
        )
        # Convert to Timedelta to Minutes & Days
        df = df.assign(
            Phase1Duration=df["Phase1Duration"].apply(timedelta2minutes),
            HospitalDuration=df["HospitalDuration"].apply(timedelta2days),
            HospitalDuration2=df["HospitalDuration2"].apply(timedelta2days),
            HospitalAdmission=df["HospitalDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
            ICUDuration=df["ICUDuration"].apply(timedelta2days),
            ICUDuration2=df["ICUDuration2"].apply(timedelta2days),
            ICUAdmission=df["ICUDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
        )
        # Remove Organ Donor Cases
        df = df.loc[~df.IsOrganDonorCase]
        builder_logger.info(f"Remove Organ Donor Cases. Num Cases: {df.shape[0]}")

        # Partition by ICU vs. No ICU stay cases
        icu_stay = df.loc[df.ICUDuration2 > 0]
        no_icu_stay = df.loc[df.ICUDuration2 == 0]
        builder_logger.info(f"ICU Num Cases: {icu_stay.shape[0]}")
        builder_logger.info(f"Non-ICU Num Cases: {no_icu_stay.shape[0]}")
        # Enforce 1 Case per Patient & Maximize ICU Cases
        # Enforce 1 Case per Patient in ICU case subgroup
        icu_stay = self.select_cases_per_patient(df=icu_stay, max_cases=1, seed=seed)
        # Remove those Patients from Non-ICU case subgroup
        no_icu_stay = no_icu_stay.query("~PAT_ID.isin(@icu_stay.PAT_ID.unique())")
        # Enforce 1 Case per Patient in Non-ICU case subgroup
        no_icu_stay = self.select_cases_per_patient(df=no_icu_stay, max_cases=1, seed=seed)
        builder_logger.info(f"1 Case Per Patient. ICU Num Cases: {icu_stay.shape[0]}")
        builder_logger.info(f"1 Case Per Patient. Non-ICU Num Cases: {no_icu_stay.shape[0]}")
        # Downsample Cases to Target Number
        if n is not None:
            # Sample half of target in each class if data allows, maintain 50-50 ratio
            target_n_pos = int(n / 2)
            if len(icu_stay) > target_n_pos:
                icu_stay = icu_stay.sample(n=target_n_pos, random_state=seed)
            n_pos = len(icu_stay)
            n_neg = n_pos
            no_icu_stay = no_icu_stay.sample(n=n_neg, random_state=seed)
            df = pd.concat([icu_stay, no_icu_stay])
            # Shuffle Dataset
            df = df.sample(frac=1.0, random_state=seed)
            builder_logger.info(f"After Downsample. Num Cases: {df.shape[0]}")
        return df

    def select_icu_admission_cases(
        self,
        n: int | None = 1250,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Make Dataset with ICU Admission Label. Unlike ICU Duration,
        this is a Boolean yes/no dataset for ICU Admission.

        ICU admission is a rare event that only occurs in ~1.6% cases.  To study
        this population, we create an enriched dataset wtih 50% containing ICU admission
        cases and 50% containing non-ICU admission cases.

        Organ Donor cases are removed.
        """
        # Get Cases w/ ICU Durations
        df = self.E.case_durations.copy()
        # Select Columns
        df = df.loc[:, self.select_cols("icu_duration")].rename(
            columns={
                "HasExpiredInHospital": "HospitalMortality",
                "HasExpiredInCase": "CaseMortality",
            }
        )
        builder_logger.info(f"Starting Num Cases: {df.shape[0]}")
        # Filter by ProcIDs where we have Preanes & Last10 Notes
        df = df.loc[list(set(df.index) & self.common_note_proc_ids)]
        builder_logger.info(
            f"Num Cases (filter by ProcIDs where we have PreAnes & Last10 Notes): {df.shape[0]}"
        )
        # Convert to Timedelta to Minutes & Days
        df = df.assign(
            Phase1Duration=df["Phase1Duration"].apply(timedelta2minutes),
            HospitalDuration=df["HospitalDuration"].apply(timedelta2days),
            HospitalDuration2=df["HospitalDuration2"].apply(timedelta2days),
            HospitalAdmission=df["HospitalDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
            ICUDuration=df["ICUDuration"].apply(timedelta2days),
            ICUDuration2=df["ICUDuration2"].apply(timedelta2days),
            ICUAdmission=df["ICUDuration2"]
            .apply(timedelta2days)
            .apply(lambda x: True if x > 0 else False),
        )
        # Remove Organ Donor Cases
        df = df.loc[~df.IsOrganDonorCase]
        builder_logger.info(f"Remove Organ Donor Cases. Num Cases: {df.shape[0]}")
        # Partition by ICU vs. No ICU stay cases
        icu_stay = df.loc[df.ICUDuration2 > 0]
        no_icu_stay = df.loc[df.ICUDuration2 == 0]
        builder_logger.info(f"ICU Num Cases: {icu_stay.shape[0]}")
        builder_logger.info(f"Non-ICU Num Cases: {no_icu_stay.shape[0]}")
        # Enforce 1 Case per Patient & Maximize ICU Cases
        # Enforce 1 Case per Patient in ICU case subgroup
        icu_stay = self.select_cases_per_patient(df=icu_stay, max_cases=1, seed=seed)
        # Remove those Patients from Non-ICU case subgroup
        no_icu_stay = no_icu_stay.query("~PAT_ID.isin(@icu_stay.PAT_ID.unique())")
        # Enforce 1 Case per Patient in Non-ICU case subgroup
        no_icu_stay = self.select_cases_per_patient(df=no_icu_stay, max_cases=1, seed=seed)
        builder_logger.info(f"1 Case Per Patient. ICU Num Cases: {icu_stay.shape[0]}")
        builder_logger.info(f"1 Case Per Patient. Non-ICU Num Cases: {no_icu_stay.shape[0]}")
        # Downsample Cases to Target Number
        if n is not None:
            # Sample half of target in each class if data allows, maintain 50-50 ratio
            target_n_pos = int(n / 2)
            if len(icu_stay) > target_n_pos:
                icu_stay = icu_stay.sample(n=target_n_pos, random_state=seed)
            n_pos = len(icu_stay)
            n_neg = n_pos
            no_icu_stay = no_icu_stay.sample(n=n_neg, random_state=seed)
            df = pd.concat([icu_stay, no_icu_stay])
            # Shuffle Dataset
            df = df.sample(frac=1.0, random_state=seed)
            builder_logger.info(f"After Downsample. Num Cases: {df.shape[0]}")
        return df

    def select_unplanned_admit_cases(
        self,
        n: int | None = 1250,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Make Dataset with Unplanned Admit Label.

        Unplanned admission is a rare event that only occurs in ~1.3% cases.  To study
        this population, we create an enriched dataset wtih 50% containing unplanned
        admits and 50% without unplanned admits.

        Organ Donor cases are removed.
        """
        case_df = self.E.case.df.copy()
        # Select Columns
        df = case_df.loc[:, self.select_cols("unplanned_admit")].rename(
            columns={
                "HasExpiredInHospital": "HospitalMortality",
                "HasExpiredInCase": "CaseMortality",
            }
        )
        builder_logger.info(f"Starting Num Cases: {df.shape[0]}")
        # Filter by ProcIDs where we have Preanes & Last10 Notes
        df = df.loc[list(set(df.index) & self.common_note_proc_ids)]
        builder_logger.info(
            f"Num Cases (filter by ProcIDs where we have PreAnes & Last10 Notes): {df.shape[0]}"
        )
        # Remove Organ Donor Cases
        df = df.loc[~df.IsOrganDonorCase]
        builder_logger.info(f"Remove Organ Donor Cases. Num Cases: {df.shape[0]}")
        # Partition by Unplanned Admit vs. No Unplanned Admit cases
        unplanned_admit = df.loc[df.UnplannedAdmit]
        no_unplanned_admit = df.loc[~df.UnplannedAdmit]
        builder_logger.info(f"Unplanned Admit Num Cases: {unplanned_admit.shape[0]}")
        builder_logger.info(f"Not Unplanned Admit Num Cases: {no_unplanned_admit.shape[0]}")
        # Enforce 1 Case per Patient & Maximize Unplanned Admit Cases
        # Enforce 1 Case per Patient in Unplanned Admit case subgroup
        unplanned_admit = self.select_cases_per_patient(df=unplanned_admit, max_cases=1, seed=seed)
        # Remove those Patients from Not Unplanned Admit case subgroup
        no_unplanned_admit = no_unplanned_admit.query(
            "~PAT_ID.isin(@unplanned_admit.PAT_ID.unique())"
        )
        # Enforce 1 Case per Patient in Not Unplanned Admit case subgroup
        no_unplanned_admit = self.select_cases_per_patient(
            df=no_unplanned_admit, max_cases=1, seed=seed
        )
        builder_logger.info(
            f"1 Case Per Patient. Unplanned Admit Num Cases: {unplanned_admit.shape[0]}"
        )
        builder_logger.info(
            f"1 Case Per Patient. Not Unplanned Admit Num Cases: {no_unplanned_admit.shape[0]}"
        )
        # Downsample Cases to Target Number
        if n is not None:
            # Sample half of target in each class if data allows, maintain 50-50 ratio
            target_n_pos = int(n / 2)
            if len(unplanned_admit) > target_n_pos:
                unplanned_admit = unplanned_admit.sample(n=target_n_pos, random_state=seed)
            n_pos = len(unplanned_admit)
            n_neg = n_pos
            no_unplanned_admit = no_unplanned_admit.sample(n=n_neg, random_state=seed)
            df = pd.concat([unplanned_admit, no_unplanned_admit])
            # Shuffle Dataset
            df = df.sample(frac=1.0, random_state=seed)
            builder_logger.info(f"After Downsample. Num Cases: {df.shape[0]}")
        return df

    def select_hospital_mortality_cases(
        self,
        n: int | None = 1250,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Make Dataset with Hospital Mortality Label.

        Mospital mortality is a rare event that only occurs in ~1.5% cases.  To study
        this population, we create an enriched dataset wtih 50% containing hospital mortality
        cases and 50% without hospital mortality.

        Organ Donor cases are removed.
        """
        case_df = self.E.case.df.copy()
        # Select Columns
        df = case_df.loc[:, self.select_cols("hospital_mortality")].rename(
            columns={
                "HasExpiredInHospital": "HospitalMortality",
                "HasExpiredInCase": "CaseMortality",
            }
        )
        builder_logger.info(f"Starting Num Cases: {df.shape[0]}")
        # Filter by ProcIDs where we have Preanes & Last10 Notes
        df = df.loc[list(set(df.index) & self.common_note_proc_ids)]
        builder_logger.info(
            f"Num Cases (filter by ProcIDs where we have PreAnes & Last10 Notes): {df.shape[0]}"
        )
        # Remove Organ Donor Cases
        df = df.loc[~df.IsOrganDonorCase]
        builder_logger.info(f"Remove Organ Donor Cases. Num Cases: {df.shape[0]}")
        # Partition by Hospital Mortality vs. No Hospital Mortality
        hospital_mortality = df.loc[df.HospitalMortality]
        no_hospital_mortality = df.loc[~df.HospitalMortality]
        builder_logger.info(f"Hospital Mortality Num Cases: {hospital_mortality.shape[0]}")
        builder_logger.info(f"Not Hospital Mortality Num Cases: {no_hospital_mortality.shape[0]}")
        # Enforce 1 Case per Patient & Maximize Hospital Mortality Cases
        # Enforce 1 Case per Patient in Hospital Mortality case subgroup
        hospital_mortality = self.select_cases_per_patient(
            df=hospital_mortality, max_cases=1, seed=seed
        )
        # Remove those Patients from Not Hospital Mortality case subgroup
        no_hospital_mortality = no_hospital_mortality.query(
            "~PAT_ID.isin(@hospital_mortality.PAT_ID.unique())"
        )
        # Enforce 1 Case per Patient in Not Hospital Mortality case subgroup
        no_hospital_mortality = self.select_cases_per_patient(
            df=no_hospital_mortality, max_cases=1, seed=seed
        )
        builder_logger.info(
            f"1 Case Per Patient. Hospital Mortality Num Cases: {hospital_mortality.shape[0]}"
        )
        builder_logger.info(
            f"1 Case Per Patient. Not Hospital Mortality Num Cases: {no_hospital_mortality.shape[0]}"  # noqa: E501
        )
        # Downsample Cases to Target Number
        if n is not None:
            # Sample half of target in each class if data allows, maintain 50-50 ratio
            target_n_pos = int(n / 2)
            if len(hospital_mortality) > target_n_pos:
                hospital_mortality = hospital_mortality.sample(n=target_n_pos, random_state=seed)
            n_pos = len(hospital_mortality)
            n_neg = n_pos
            no_hospital_mortality = no_hospital_mortality.sample(n=n_neg, random_state=seed)
            df = pd.concat([hospital_mortality, no_hospital_mortality])
            # Shuffle Dataset
            df = df.sample(frac=1.0, random_state=seed)
            builder_logger.info(f"After Downsample. Num Cases: {df.shape[0]}")
        return df
