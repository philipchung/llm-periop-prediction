# %% [markdown]
# ## Script for Table 1: Dataset Demographics & Outcome Variable Distribution
# %%
from pathlib import Path

import pandas as pd

from llm_utils import DataPaths, read_pandas
from make_dataset.dataset import DatasetBuilder

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Define Raw Data Paths
paths = DataPaths(project_dir=Path(__file__).parent.parent, data_version=7)

# Load Raw Data
paths.register(name="adt_raw", path=paths.raw / "2023-05-11_ADT.feather")
paths.register(name="case_raw", path=paths.raw / "2023-05-11_Case.feather")
paths.register(name="preanes_notes_raw", path=paths.raw / "2023-05-02_PreAnesNotes.feather")
paths.register(name="last10_notes_raw", path=paths.raw / "2023-05-02_LastTenNotes.feather")
paths.register(name="age_gender", path=paths.raw / "2023-10-01_AgeGender.csv")
paths.register(name="age_sex", path=paths.raw / "2023-10-03_AgeSex.csv")
# %%
# Load Age & Sex Demographics
age_gender = read_pandas(paths.age_gender).set_index("ProcID")
age_sex = read_pandas(paths.age_sex).set_index("ProcID")

age = age_gender.loc[:, "PatientAgeYears"].rename({"PatientAgeYears": "Age"})


def determine_sex(row: pd.Series) -> str:
    """Determine Biological Gender and sort into 3 categories: Male, Female, Other/Unspecificied.
    "PatientGender" is most complete data.  "SexAssignedAtBirth" contains many missing values.

    We don't use "GenderIdentity" because we are interested in Biological Gender.

    Prioritize information: PatientGender > SexAssignedAtBirth.
    """
    gender = row.PatientGender
    birth_sex = row.SexAssignedAtBirth

    match gender:
        case "F":
            return "Female"
        case "M":
            return "Male"
        case _:
            match birth_sex:
                case "F":
                    return "Female"
                case "M":
                    return "Male"
                case _:
                    return "Other or Unknown"


sex = age_sex.apply(determine_sex, axis=1).rename("Sex")
age_sex = age.rename("Age").to_frame().join(sex)
# %%
builder = DatasetBuilder(paths=paths, dataset_version="dataset4")
# %%
# Load Processed Datasets
## ASA
asa_last10_inference = builder.get_dataset(
    dataset_type="asa", note_type="last10", split="inference"
)
asa_last10_fewshot = builder.get_dataset(dataset_type="asa", note_type="last10", split="fewshot")
asa_preanes_inference = builder.get_dataset(
    dataset_type="asa", note_type="preanes", split="inference"
)
asa_preanes_fewshot = builder.get_dataset(dataset_type="asa", note_type="preanes", split="fewshot")
## Phase 1 Duration
phase1_last10_inference = builder.get_dataset(
    dataset_type="phase1_duration", note_type="last10", split="inference"
)
phase1_last10_fewshot = builder.get_dataset(
    dataset_type="phase1_duration", note_type="last10", split="fewshot"
)
phase1_preanes_inference = builder.get_dataset(
    dataset_type="phase1_duration", note_type="preanes", split="inference"
)
phase1_preanes_fewshot = builder.get_dataset(
    dataset_type="phase1_duration", note_type="preanes", split="fewshot"
)
## Hospital Duration
hospitalduration_last10_inference = builder.get_dataset(
    dataset_type="hospital_duration", note_type="last10", split="inference"
)
hospitalduration_last10_fewshot = builder.get_dataset(
    dataset_type="hospital_duration", note_type="last10", split="fewshot"
)
hospitalduration_preanes_inference = builder.get_dataset(
    dataset_type="hospital_duration", note_type="preanes", split="inference"
)
hospitalduration_preanes_fewshot = builder.get_dataset(
    dataset_type="hospital_duration", note_type="preanes", split="fewshot"
)
## Hospital Duration
hospitaladmission_last10_inference = builder.get_dataset(
    dataset_type="hospital_admission", note_type="last10", split="inference"
)
hospitaladmission_last10_fewshot = builder.get_dataset(
    dataset_type="hospital_admission", note_type="last10", split="fewshot"
)
hospitaladmission_preanes_inference = builder.get_dataset(
    dataset_type="hospital_admission", note_type="preanes", split="inference"
)
hospitaladmission_preanes_fewshot = builder.get_dataset(
    dataset_type="hospital_admission", note_type="preanes", split="fewshot"
)
## ICU Duration
icuduration_last10_inference = builder.get_dataset(
    dataset_type="icu_duration", note_type="last10", split="inference"
)
icuduration_last10_fewshot = builder.get_dataset(
    dataset_type="icu_duration", note_type="last10", split="fewshot"
)
icuduration_preanes_inference = builder.get_dataset(
    dataset_type="icu_duration", note_type="preanes", split="inference"
)
icuduration_preanes_fewshot = builder.get_dataset(
    dataset_type="icu_duration", note_type="preanes", split="fewshot"
)
## ICU Duration
icuadmission_last10_inference = builder.get_dataset(
    dataset_type="icu_admission", note_type="last10", split="inference"
)
icuadmission_last10_fewshot = builder.get_dataset(
    dataset_type="icu_admission", note_type="last10", split="fewshot"
)
icuadmission_preanes_inference = builder.get_dataset(
    dataset_type="icu_admission", note_type="preanes", split="inference"
)
icuadmission_preanes_fewshot = builder.get_dataset(
    dataset_type="icu_admission", note_type="preanes", split="fewshot"
)
## Unplanned Admit
unplannedadmit_last10_inference = builder.get_dataset(
    dataset_type="unplanned_admit", note_type="last10", split="inference"
)
unplannedadmit_last10_fewshot = builder.get_dataset(
    dataset_type="unplanned_admit", note_type="last10", split="fewshot"
)
unplannedadmit_preanes_inference = builder.get_dataset(
    dataset_type="unplanned_admit", note_type="preanes", split="inference"
)
unplannedadmit_preanes_fewshot = builder.get_dataset(
    dataset_type="unplanned_admit", note_type="preanes", split="fewshot"
)
## Hospital Mortality
hospitalmortality_last10_inference = builder.get_dataset(
    dataset_type="hospital_mortality", note_type="last10", split="inference"
)
hospitalmortality_last10_fewshot = builder.get_dataset(
    dataset_type="hospital_mortality", note_type="last10", split="fewshot"
)
hospitalmortality_preanes_inference = builder.get_dataset(
    dataset_type="hospital_mortality", note_type="preanes", split="inference"
)
hospitalmortality_preanes_fewshot = builder.get_dataset(
    dataset_type="hospital_mortality", note_type="preanes", split="fewshot"
)
# %% [markdown]
# ### Table 1: Dataset Characteristics
# %%


def make_table_for_dataset(
    last10: pd.DataFrame, preanes: pd.DataFrame, age_sex: pd.DataFrame
) -> pd.DataFrame:
    ds = last10.join(age_sex)
    preanes_ds = preanes.join(age_sex)

    def format_count_percent(ct, pct) -> str:
        return f"{ct} ({pct:.1%})"

    def compute_count_percent(s: pd.Series) -> pd.Series:
        cts = s.value_counts()
        pcts = s.value_counts(normalize=True)
        return cts.combine(pcts, func=format_count_percent).sort_index()

    age_mean = ds.Age.mean()
    age_std = ds.Age.std()
    sex = compute_count_percent(ds.Sex)
    sex.index = [("Sex, no. (%)", x) for x in sex.index]
    ## Demographics
    demographics = {
        ("Demographics", "Case Counts, no. (%)"): len(ds.index.unique()),
        ("Demographics", "Age, mean (std)"): f"{age_mean:.1f} ({age_std:.1f})",
        **sex,
    }

    ## Anesthesia Type
    # Coerce "labor analgesia" to regional--these are labor analgesia converted to cesarean cases
    ds = ds.assign(
        AnesType=ds.AnesType.apply(lambda x: "regional" if x == "labor analgesia" else x)
    )
    anes_type = compute_count_percent(ds.AnesType).rename(
        index={
            "general": "General",
            "regional": "Regional",
            "labor analgesia": "Labor Analgesia",
            "unknown": "Unknown",
        }
    )
    anes_type.index = [("Anesthesia Type, no. (%)", x) for x in anes_type.index]

    ## ASA-PS
    asa_class = compute_count_percent(ds.ASA).sort_index()
    asa_class.index = [("ASA Physical Status Classification, no. (%)", x) for x in asa_class.index]

    ## Patient Class - Expected & Actual
    expected_patient_class = (
        compute_count_percent(ds.ExpectedPatientClass)
        .rename(index={"": "Unspecified"})
        .sort_index()
    )
    expected_patient_class.index = [
        ("Expected Patient Class, no. (%)", x) for x in expected_patient_class.index
    ]
    actual_patient_class = (
        compute_count_percent(ds.ActualPatientClass).rename(index={"": "Unspecified"}).sort_index()
    )
    actual_patient_class.index = [
        ("Actual Patient Class, no. (%)", x) for x in actual_patient_class.index
    ]

    ## Phase 1 PACU Duration
    if "Phase1Duration" in ds.columns:
        median = ds.Phase1Duration.median().astype(int)
        quartile1 = ds.Phase1Duration.quantile(0.25).astype(int)
        quartile3 = ds.Phase1Duration.quantile(0.75).astype(int)
        phase1_duration = f"{median} ({quartile1}, {quartile3})"
    else:
        phase1_duration = "--"

    ## Hospital Stay Duration
    if "HospitalDuration2" in ds.columns:
        median = ds.HospitalDuration2.median().astype(float)
        quartile1 = ds.HospitalDuration2.quantile(0.25).astype(float)
        quartile3 = ds.HospitalDuration2.quantile(0.75).astype(float)
        hospital_duration = f"{median:.2f} ({quartile1:.2f}, {quartile3:.2f})"
    else:
        hospital_duration = "--"

    ## ICU Stay Duration
    if "ICUDuration2" in ds.columns:
        median = ds.ICUDuration2.median().astype(float)
        quartile1 = ds.ICUDuration2.quantile(0.25).astype(float)
        quartile3 = ds.ICUDuration2.quantile(0.75).astype(float)
        icu_duration = f"{median:.2f} ({quartile1:.2f}, {quartile3:.2f})"
    else:
        icu_duration = "--"

    durations = {
        (
            "PACU Length of Stay, median (IQR)",
            "Phase 1 PACU Duration, minutes",
        ): phase1_duration,
        (
            "Hospital Length of Stay, median (IQR)",
            "Hospital Admission Duration, days",
        ): hospital_duration,
        (
            "ICU Length of Stay, median (IQR)",
            "ICU Duration, days",
        ): icu_duration,
    }

    ## Hospital Admission
    if "HospitalAdmission" in ds.columns:
        hospital_admission = compute_count_percent(ds.HospitalAdmission)
        hospital_admission = {
            ("Hospital Admission, no. (%)", "Yes"): hospital_admission.at[True]
            if True in hospital_admission.index
            else "--",
            ("Hospital Admission, no. (%)", "No"): hospital_admission.at[False]
            if False in hospital_admission.index
            else "--",
        }
    else:
        hospital_admission = {
            ("Hospital Admission, no. (%)", "Yes"): "--",
            ("Hospital Admission, no. (%)", "No"): "--",
        }

    ## ICU Admission
    if "ICUAdmission" in ds.columns:
        icu_admission = compute_count_percent(ds.ICUAdmission)
        icu_admission = {
            ("ICU Admission, no. (%)", "Yes"): icu_admission.at[True]
            if True in icu_admission.index
            else "--",
            ("ICU Admission, no. (%)", "No"): icu_admission.at[False]
            if False in icu_admission.index
            else "--",
        }
    else:
        icu_admission = {
            ("ICU Admission, no. (%)", "Yes"): "--",
            ("ICU Admission, no. (%)", "No"): "--",
        }

    ## Unplanned Admit
    unplanned_admit = compute_count_percent(ds.UnplannedAdmit)
    unplanned_admit = {
        ("Unplanned Admission, no. (%)", "Yes"): unplanned_admit.at[True]
        if True in unplanned_admit.index
        else "--",
        ("Unplanned Admission, no. (%)", "No"): unplanned_admit.at[False]
        if False in unplanned_admit.index
        else "--",
    }

    ## Hospital Mortality
    hospital_mortality = compute_count_percent(ds.HospitalMortality)
    hospital_mortality = {
        ("Hospital Mortality, no. (%)", "Yes"): hospital_mortality.at[True]
        if True in hospital_mortality.index
        else "--",
        ("Hospital Mortality, no. (%)", "No"): hospital_mortality.at[False]
        if False in hospital_mortality.index
        else "--",
    }

    ## Surgery Service
    def categorize_cardiovascular_cath_lab(surg_service: str) -> str:
        if surg_service in (
            "Cathlab",
            "Cardiovascular",
            "Electrophysiology",
            "Laser Lead Extraction",
            "Cardiology",
        ):
            return "Cardiovascular Cath Lab"
        else:
            return surg_service

    def categorize_gi_ir(surg_service: str, procedure_description: str) -> str:
        if surg_service == "":
            if any(
                x in procedure_description
                for x in [
                    "EGD",
                    "COLON",
                    "ERCP",
                    "EUS",
                    "FLEX SIG",
                    "ILEOSCOPY",
                    "ILEO",
                    "POUCHOSCOPY",
                ]
            ):
                return "Gastroenterology"
            else:
                return "Interventional Radiology"
        else:
            return surg_service

    def clean_surgery_services(surg_service: str) -> str:
        if surg_service in ("Gynecology", "Gynecologic Oncology"):
            return "Gynecologic Surgery"
        elif surg_service in ("Hand Surgery", "Orthopedics"):
            return "Orthopedic Surgery"
        elif surg_service in ("Burns"):
            return "Burn Surgery"
        elif surg_service in ("Oral-Maxillofacial"):
            return "Oral-Maxillofacial Surgery"
        elif surg_service in ("Plastics"):
            return "Plastic Surgery"
        elif surg_service in ("Vascular"):
            return "Vascular Surgery"
        else:
            return surg_service

    # Combine Cath Lab Services
    ds = ds.assign(SurgService=ds.SurgService.apply(categorize_cardiovascular_cath_lab))
    ds = ds.assign(
        SurgService=ds.apply(
            lambda row: categorize_gi_ir(row.SurgService, row.ProcedureDescription),
            axis=1,
        )
    )
    ds = ds.assign(SurgService=ds.SurgService.apply(clean_surgery_services))
    surg_service = compute_count_percent(ds.SurgService).rename(
        index={"": "NORA (Gastroenterology, Interventional Radiology, Radiology)"}
    )
    surg_service.index = [("Surgery/Proceduralist Service, no. (%)", x) for x in surg_service.index]

    ## Note Statistics
    def get_median_iqr_str(s: pd.Series) -> str:
        stats = s.describe()
        median = stats.loc["50%"].astype(int)
        quartile1 = stats.loc["25%"].astype(int)
        quartile3 = stats.loc["75%"].astype(int)
        return f"{median} ({quartile1}, {quartile3})"

    last10_notes_ct = ds.NoteID.apply(len)
    last10_notes_token_length = ds.NoteTextTokenLength.apply(sum)
    preanes_notes_token_length = preanes_ds.NoteTextTokenLength.apply(lambda x: x[0])
    num_notes = {
        (
            "Note Characteristics per Case, median (IQR)",
            "Most Recent Clinical Notes, note count",
        ): get_median_iqr_str(last10_notes_ct),
        (
            "Note Characteristics per Case, median (IQR)",
            "Most Recent Clinical Notes, total number of tokens",
        ): get_median_iqr_str(last10_notes_token_length),
        (
            "Note Characteristics per Case, median (IQR)",
            "Anesthesia Preoperative Evaluation Note, total number of tokens",
        ): get_median_iqr_str(preanes_notes_token_length),
    }

    # Create Table
    table = pd.Series(
        {
            # Demographic Statistics
            **demographics,
            **anes_type.to_dict(),
            **surg_service.to_dict(),
            **expected_patient_class.to_dict(),
            **actual_patient_class.to_dict(),
            # Note Statisticis
            **num_notes,
            # Outcome Variable Statistics
            **asa_class.to_dict(),
            **durations,
            **hospital_admission,
            **icu_admission,
            **unplanned_admit,
            **hospital_mortality,
        }
    )
    return table


table_asa_inference = make_table_for_dataset(
    last10=asa_last10_inference, preanes=asa_preanes_inference, age_sex=age_sex
)
table_asa_fewshot = make_table_for_dataset(
    last10=asa_last10_fewshot, preanes=asa_preanes_fewshot, age_sex=age_sex
)
table_phase1_inference = make_table_for_dataset(
    last10=phase1_last10_inference,
    preanes=phase1_preanes_inference,
    age_sex=age_sex,
)
table_phase1_fewshot = make_table_for_dataset(
    last10=phase1_last10_fewshot, preanes=phase1_preanes_fewshot, age_sex=age_sex
)
table_hospitaladmission_inference = make_table_for_dataset(
    last10=hospitaladmission_last10_inference,
    preanes=hospitaladmission_preanes_inference,
    age_sex=age_sex,
)
table_hospitaladmission_fewshot = make_table_for_dataset(
    last10=hospitaladmission_last10_fewshot,
    preanes=hospitaladmission_preanes_fewshot,
    age_sex=age_sex,
)
table_hospitalduration_inference = make_table_for_dataset(
    last10=hospitalduration_last10_inference,
    preanes=hospitalduration_preanes_inference,
    age_sex=age_sex,
)
table_hospitalduration_fewshot = make_table_for_dataset(
    last10=hospitalduration_last10_fewshot,
    preanes=hospitalduration_preanes_fewshot,
    age_sex=age_sex,
)
table_icuadmission_inference = make_table_for_dataset(
    last10=icuadmission_last10_inference,
    preanes=icuadmission_preanes_inference,
    age_sex=age_sex,
)
table_icuadmission_fewshot = make_table_for_dataset(
    last10=icuadmission_last10_fewshot,
    preanes=icuadmission_preanes_fewshot,
    age_sex=age_sex,
)
table_icuduration_inference = make_table_for_dataset(
    last10=icuduration_last10_inference,
    preanes=icuduration_preanes_inference,
    age_sex=age_sex,
)
table_icuduration_fewshot = make_table_for_dataset(
    last10=icuduration_last10_fewshot,
    preanes=icuduration_preanes_fewshot,
    age_sex=age_sex,
)
table_unplannedadmit_inference = make_table_for_dataset(
    last10=unplannedadmit_last10_inference,
    preanes=unplannedadmit_preanes_inference,
    age_sex=age_sex,
)
table_unplannedadmit_fewshot = make_table_for_dataset(
    last10=unplannedadmit_last10_fewshot,
    preanes=unplannedadmit_preanes_fewshot,
    age_sex=age_sex,
)
table_hospitalmortality_inference = make_table_for_dataset(
    last10=hospitalmortality_last10_inference,
    preanes=hospitalmortality_preanes_inference,
    age_sex=age_sex,
)
table_hospitalmortality_fewshot = make_table_for_dataset(
    last10=hospitalmortality_last10_fewshot,
    preanes=hospitalmortality_preanes_fewshot,
    age_sex=age_sex,
)
# Combine all dataset tables
# NOTE: ICU Admission & ICU Duration datasets are identical; others are different
combined_table = pd.DataFrame(
    {
        ("ASA", "Inference"): table_asa_inference,
        ("ASA", "Fewshot"): table_asa_fewshot,
        ("PACU Phase 1 Duration", "Inference"): table_phase1_inference,
        ("PACU Phase 1 Duration", "Fewshot"): table_phase1_fewshot,
        ("Hospital Admission", "Inference"): table_hospitaladmission_inference,
        ("Hospital Admission", "Fewshot"): table_hospitaladmission_fewshot,
        ("Hospital Duration", "Inference"): table_hospitalduration_inference,
        ("Hospital Duration", "Fewshot"): table_hospitalduration_fewshot,
        ("ICU Admission & ICU Duration", "Inference"): table_icuadmission_inference,
        ("ICU Admission & ICU Duration", "Fewshot"): table_icuadmission_fewshot,
        # ("ICU Duration", "Inference"): table_icuduration_inference,
        # ("ICU Duration", "Fewshot"): table_icuduration_fewshot,
        ("Unplanned Admit", "Inference"): table_unplannedadmit_inference,
        ("Unplanned Admit", "Fewshot"): table_unplannedadmit_fewshot,
        ("Hospital Mortality", "Inference"): table_hospitalmortality_inference,
        ("Hospital Mortality", "Fewshot"): table_hospitalmortality_fewshot,
    }
).fillna("--")
# %%
# Manually Re-order Top Level Index
row_order = [
    # Demographics
    ("Demographics", "Case Counts, no. (%)"),
    ("Demographics", "Age, mean (std)"),
    ("Sex, no. (%)", "Female"),
    ("Sex, no. (%)", "Male"),
    ("Sex, no. (%)", "Other or Unknown"),
    # Case Info
    ("Anesthesia Type, no. (%)", "General"),
    ("Anesthesia Type, no. (%)", "MAC"),
    ("Anesthesia Type, no. (%)", "Regional"),
    ("Actual Patient Class, no. (%)", "Deceased - Organ Donor"),
    ("Actual Patient Class, no. (%)", "Emergency"),
    ("Actual Patient Class, no. (%)", "Inpatient"),
    ("Actual Patient Class, no. (%)", "Observation"),
    ("Actual Patient Class, no. (%)", "Outpatient"),
    ("Actual Patient Class, no. (%)", "Surgery Admit"),
    ("Actual Patient Class, no. (%)", "Surgery Overnight Stay"),
    ("Expected Patient Class, no. (%)", "Emergency"),
    ("Expected Patient Class, no. (%)", "Inpatient"),
    ("Expected Patient Class, no. (%)", "Outpatient"),
    ("Expected Patient Class, no. (%)", "Surgery Admit"),
    ("Expected Patient Class, no. (%)", "Surgery Overnight Stay"),
    ("Expected Patient Class, no. (%)", "Unspecified"),
    ("Surgery/Proceduralist Service, no. (%)", "Burn Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Cardiovascular Cath Lab"),
    ("Surgery/Proceduralist Service, no. (%)", "Cardiovascular Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Gastroenterology"),
    ("Surgery/Proceduralist Service, no. (%)", "General Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Gynecologic Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Hematology/Oncology"),
    ("Surgery/Proceduralist Service, no. (%)", "Interventional Radiology"),
    ("Surgery/Proceduralist Service, no. (%)", "Neurosurgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Obstetrics"),
    ("Surgery/Proceduralist Service, no. (%)", "Ophthalmology"),
    ("Surgery/Proceduralist Service, no. (%)", "Oral-Maxillofacial Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Organ Donor"),
    ("Surgery/Proceduralist Service, no. (%)", "Orthopedic Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Otolaryngology"),
    ("Surgery/Proceduralist Service, no. (%)", "Plastic Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Podiatry"),
    ("Surgery/Proceduralist Service, no. (%)", "Thoracic Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Transplant Surgery"),
    ("Surgery/Proceduralist Service, no. (%)", "Urology"),
    ("Surgery/Proceduralist Service, no. (%)", "Vascular Surgery"),
    # Note Characteristics
    (
        "Note Characteristics per Case, median (IQR)",
        "Anesthesia Preoperative Evaluation Note, total number of tokens",
    ),
    (
        "Note Characteristics per Case, median (IQR)",
        "Most Recent Clinical Notes, note count",
    ),
    (
        "Note Characteristics per Case, median (IQR)",
        "Most Recent Clinical Notes, total number of tokens",
    ),
    # Outcome Variables
    ("ASA Physical Status Classification, no. (%)", 1),
    ("ASA Physical Status Classification, no. (%)", 2),
    ("ASA Physical Status Classification, no. (%)", 3),
    ("ASA Physical Status Classification, no. (%)", 4),
    ("ASA Physical Status Classification, no. (%)", 5),
    ("ASA Physical Status Classification, no. (%)", 6),
    ("PACU Length of Stay, median (IQR)", "Phase 1 PACU Duration, minutes"),
    ("Hospital Admission, no. (%)", "Yes"),
    ("Hospital Admission, no. (%)", "No"),
    (
        "Hospital Length of Stay, median (IQR)",
        "Hospital Admission Duration, days",
    ),
    ("ICU Admission, no. (%)", "Yes"),
    ("ICU Admission, no. (%)", "No"),
    ("ICU Length of Stay, median (IQR)", "ICU Duration, days"),
    ("Unplanned Admission, no. (%)", "Yes"),
    ("Unplanned Admission, no. (%)", "No"),
    ("Hospital Mortality, no. (%)", "Yes"),
    ("Hospital Mortality, no. (%)", "No"),
]

combined_table = combined_table.loc[row_order, :]
combined_table
# %%
# Save Tables
save_dir = Path.cwd() / "results" / "table1"
save_dir.mkdir(parents=True, exist_ok=True)

combined_table.to_csv(save_dir / "Table1.csv")
# %%
