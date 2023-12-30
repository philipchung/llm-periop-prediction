# ruff: noqa: E501

import json
import math
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from llm_utils import (
    get_system_message_length,
    num_tokens_from_string,
    timedelta2days,
    timedelta2minutes,
)

from llm.chat_model import Message

## System Message Prompt Templates Defining Different Tasks
SYSTEM_MESSAGE = "You are a physician working in a hospital surgery center who is assessing patients to determine their outcome after a procedure."

## Questions
ASA_QUESTION = "What is the patient's ASA Physical Status Classification?"
PHASE1_DURATION_QUESTION = "Predict the patient's Post-Anesthesia Care Unit (PACU) Phase 1 Recovery Duration of Stay in minutes."
PHASE2_DURATION_QUESTION = "Predict the patient's Post-Anesthesia Care Unit (PACU) Phase 2 Recovery Duration of Stay in minutes."
PACU_DURATION_QUESTION = "Predict the patient's Post-Anesthesia Care Unit (PACU) Total Recovery Duration of Stay in minutes."
HOSPITAL_DURATION_QUESTION = "Predict the patient's Total Hospital Duration of Stay in days."
HOSPITAL_ADMISSION_QUESTION = (
    "Predict whether the patient will be admitted to the hospital after the procedure."
)
ICU_DURATION_QUESTION = "Predict the patient's Intensive Care Unit (ICU) Duration of Stay in days."
ICU_ADMISSION_QUESTION = "Predict whether the patient will be admitted to the Intensive Care Unit (ICU) after the procedure."
UNPLANNED_ADMIT_QUESTION = "Predict whether the patient is likely to have an unplanned hospital admission after the procedure."
HOSPITAL_MORTALITY_QUESTION = (
    "Predict whether the patient is likely to die during this hospitalization."
)

## String Keys in JSON Response Template
PATIENT_SUMMARY_KEY = "Brief Patient Summary"
ANSWER_KEY = "Answer"
EXPLANATION_KEY = "Explanation"
EXPLANATION_COT_KEY = "Step By Step Explanation"

## User Message Prompt Template Chunks
## Summarization
SUMMARIZE_INSTRUCTION = """\
You are given information from the patient's medical record. Summarize this information, making sure to include the most important positive clinical findings."""

SUMMARIZE_RESPONSE_TEMPLATE = """\
JSON Response:
{json_template}""".format(
    json_template=json.dumps(
        {PATIENT_SUMMARY_KEY: "<str>"},
        indent=4,
    )
)

## Chain-of-Thought Generation from Patient Summary & Answer
COT_GENERATION_INSTRUCTION = """\
You are given a task, the answer, and context which contains information from the proposed procedure and patient's medical record. Provide the logical reasoning steps that lead to the answer using information from the proposed procedure and patient's medical record."""


COT_GENERATION_RESPONSE_TEMPLATE = """\
JSON Response:
{json_template}""".format(
    json_template=json.dumps(
        {
            EXPLANATION_COT_KEY: "<str>",
        },
        indent=4,
    )
)

## Q&A Instruction
QA_INSTRUCTION = """\
You are given a task and context. The context contains information from the proposed procedure and patient's medical record. Assess the patient in the context of the proposed procedure and then provide an answer."""

## Response Instruction
RESPONSE_INSTRUCTION = """\
Give your response in JSON format using the provided template. The desired response type is provided in angle brackets < >. For example, <int> means to provide an integer response. Provide a single value response without ranges."""
COT_RESPONSE_INSTRUCTION = """\
Think step by step and give your response in JSON format using the provided template. The desired response type is provided in angle brackets < >. For example, <int> means to provide an integer response. Provide a single value response without ranges."""


def qa_response_template(answer_type: str = "int") -> str:
    return """\
JSON Response:
{json_template}""".format(
        json_template=json.dumps(
            {
                ANSWER_KEY: f"<{answer_type}>",
                EXPLANATION_KEY: "<str>",
            },
            indent=4,
        )
    )


def cot_qa_response_template(answer_type: str = "int") -> str:
    return """\
JSON Response:
{json_template}""".format(
        json_template=json.dumps(
            {
                EXPLANATION_COT_KEY: "<str>",
                ANSWER_KEY: f"<{answer_type}>",
            },
            indent=4,
        )
    )


## Few-shot Demonstrations followed by Question-Answer Task
FEWSHOT_QA_INSTRUCTION = """\
You are given examples of task, context, and answer. The context contains information from the proposed procedure and patient's medical record which can be used to determine the answer."""

## Few-shot QA+CoT Demonstrations followed by Question-Answer w/ Induction of CoT Reasoning
FEWSHOT_COT_QA_INSTRUCTION = """\
You are given examples of task, context, logical reasoning, and answer. The context contains information from the proposed procedure and patient's medical record which can be used to determine the answer. The logical reasoning contains a step by step explanation leading to the answer."""


@dataclass(kw_only=True)
class PromptComposer:
    """Compose case notes into prompt messages that can be passed into the Chat Model."""

    task: str
    # Determined by task
    question: str = field(init=False)
    label_name: str = field(init=False)

    system_message: str = SYSTEM_MESSAGE
    # `token_limit` is max number of tokens allowed for Message.messages
    token_limit: int | None = None
    # `user_message_token_limit` is computed upon init
    user_message_token_limit: int | None = None

    def __post_init__(self) -> None:
        self.set_question_and_label_name_from_task()
        self.set_token_limit()

    def set_question_and_label_name_from_task(self) -> None:
        "Set `question` and `label_name` attributes based on `task` attribute."
        match self.task:
            case "summarize":
                self.question = None
                self.label_name = self.task
            case "asa":
                self.question = ASA_QUESTION
                self.label_name = self.task
            case "phase1_duration":
                self.question = PHASE1_DURATION_QUESTION
                self.label_name = self.task
            case "phase2_duration":
                self.question = PHASE2_DURATION_QUESTION
                self.label_name = self.task
            case "pacu_duration":
                self.question = PACU_DURATION_QUESTION
                self.label_name = self.task
            case "hospital_duration":
                self.question = HOSPITAL_DURATION_QUESTION
                self.label_name = self.task
            case "hospital_admission":
                self.question = HOSPITAL_ADMISSION_QUESTION
                self.label_name = self.task
            case "icu_duration":
                self.question = ICU_DURATION_QUESTION
                self.label_name = self.task
            case "icu_admission":
                self.question = ICU_ADMISSION_QUESTION
                self.label_name = self.task
            case "unplanned_admit":
                self.question = UNPLANNED_ADMIT_QUESTION
                self.label_name = self.task
            case "hospital_mortality":
                self.question = HOSPITAL_MORTALITY_QUESTION
                self.label_name = self.task
            case _:
                warnings.warn(
                    f"Unknown `task`: {self.task}. "
                    "Setting `question` and `label` both to `None`."
                )
                return None, None

    def set_token_limit(self) -> None:
        "Determines token limit for composing User Message in the prompts by accounting."
        if self.token_limit is not None:
            system_message_num_tokens = get_system_message_length(self.system_message)
            self.user_message_token_limit = self.token_limit - system_message_num_tokens

    def summarize(self, case: pd.Series) -> Message:
        "Create prompt to make summary from notes."
        # Format Messages
        note_text, notes_df = format_all_notes_for_case(
            case=case, token_limit=self.user_message_token_limit
        )
        # Instruction & Response Specification
        instruction = SUMMARIZE_INSTRUCTION
        response_instruction = RESPONSE_INSTRUCTION
        json_response_template = SUMMARIZE_RESPONSE_TEMPLATE
        user_message = f"""\
{instruction}

{note_text}

{response_instruction}
{json_response_template}"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        # Save case and all associated info as a JSON-parsable metadata dict
        metadata = {
            "case": json.loads(case.to_json(orient="index")),
            "notes": json.loads(notes_df.to_json(orient="records")),
            "question": None,
            "label": None,
            "label_type": None,
            "fewshot_cases": None,
        }
        return Message(messages=messages, metadata=metadata)

    def generate_cot_with_summary_context(self, case: pd.Series) -> Message:
        "Create prompt to derive a chain-of-thought rationale."
        # Case Info
        scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
        procedure_description = (
            case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
        )
        diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
        surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
        # Case Notes
        summary = case.Summary
        # Question & Answer
        question = self.question
        label, label_type = get_label_from_case(case=case, label_name=self.label_name)
        answer = str(label)
        # Instruction & Reponse Specification
        instruction = COT_GENERATION_INSTRUCTION
        json_response_template = COT_GENERATION_RESPONSE_TEMPLATE
        # Format Messages
        user_message = f"""\
{instruction}

Task: {question}
Answer: {answer}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Summary:
{summary}

{json_response_template}"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        # Save case and all associated info as a JSON-parsable metadata dict
        metadata = {
            "case": json.loads(case.to_json(orient="index")),
            "notes": None,  # No notes used, only summaries
            "question": question,
            "label": label,
            "label_type": label_type,
            "fewshot_cases": None,
        }
        return Message(messages=messages, metadata=metadata)

    def zeroshot_qa_with_notes_context(self, case: pd.Series) -> Message:
        "Create prompt for Zeroshot Q&A using original notes associated with case."
        # Case Info
        scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
        procedure_description = (
            case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
        )
        diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
        surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
        # Case Notes
        note_text, notes_df = format_all_notes_for_case(
            case=case, token_limit=self.user_message_token_limit
        )
        # Question & Answer
        question = self.question
        label, label_type = get_label_from_case(case=case, label_name=self.label_name)
        # Instruction & Response Specification
        instruction = QA_INSTRUCTION
        response_instruction = RESPONSE_INSTRUCTION
        json_response_template = qa_response_template(answer_type=label_type)
        # Format Messages
        user_message = f"""\
{instruction}

Task: {question}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Notes:
{note_text}

{response_instruction}
{json_response_template}"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        # Save case and all associated info as a JSON-parsable metadata dict
        metadata = {
            "case": json.loads(case.to_json(orient="index")),
            "notes": json.loads(notes_df.to_json(orient="records")),
            "question": question,
            "label": label,
            "label_type": label_type,
            "fewshot_cases": None,
        }
        return Message(messages=messages, metadata=metadata)

    def zeroshot_qa_with_summary_context(self, case: pd.Series) -> Message:
        """Create prompt for Zeroshot Q&A using summary of notes associated with case.
        The `case` series is expected to have the index `Summary` which corresponds to the
        summary of notes for that case."""
        # Case Info
        scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
        procedure_description = (
            case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
        )
        diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
        surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
        # Case Notes
        summary = case.Summary
        # Question & Answer
        question = self.question
        label, label_type = get_label_from_case(case=case, label_name=self.label_name)
        # Instruction & Response Specification
        instruction = QA_INSTRUCTION
        response_instruction = RESPONSE_INSTRUCTION
        json_response_template = qa_response_template(answer_type=label_type)
        # Format Messages
        user_message = f"""\
{instruction}

Task: {question}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Summary:
{summary}

{response_instruction}
{json_response_template}"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        # Save case and all associated info as a JSON-parsable metadata dict
        metadata = {
            "case": json.loads(case.to_json(orient="index")),
            "notes": None,  # No notes used, only summaries
            "question": question,
            "label": label,
            "label_type": label_type,
            "fewshot_cases": None,
        }
        return Message(messages=messages, metadata=metadata)

    def zeroshot_cot_qa_with_notes_context(self, case: pd.Series) -> Message:
        "Create prompt for Zeroshot CoT Q&A using original notes associated with case."
        # Case Info
        scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
        procedure_description = (
            case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
        )
        diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
        surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
        # Case Notes
        note_text, notes_df = format_all_notes_for_case(
            case=case, token_limit=self.user_message_token_limit
        )
        # Question & Answer
        question = self.question
        label, label_type = get_label_from_case(case=case, label_name=self.label_name)
        # Instruction & Response Specification
        instruction = QA_INSTRUCTION
        response_instruction = COT_RESPONSE_INSTRUCTION
        json_response_template = cot_qa_response_template(answer_type=label_type)
        # Format Messages
        user_message = f"""\
{instruction}

Task: {question}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Notes:
{note_text}

{response_instruction}
{json_response_template}"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        # Save case and all associated info as a JSON-parsable metadata dict
        metadata = {
            "case": json.loads(case.to_json(orient="index")),
            "notes": json.loads(notes_df.to_json(orient="records")),
            "question": question,
            "label": label,
            "label_type": label_type,
            "fewshot_cases": None,
        }
        return Message(messages=messages, metadata=metadata)

    def zeroshot_cot_qa_with_summary_context(self, case: pd.Series) -> Message:
        """Create prompt for Zeroshot CoT Q&A using summary of notes associated with case.
        The `case` series is expected to have the index `Summary` which corresponds to the
        summary of notes for that case."""
        # Case Info
        scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
        procedure_description = (
            case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
        )
        diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
        surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
        # Case Notes
        summary = case.Summary
        # Question & Answer
        question = self.question
        label, label_type = get_label_from_case(case=case, label_name=self.label_name)
        # Instruction & Response Specification
        instruction = QA_INSTRUCTION
        response_instruction = COT_RESPONSE_INSTRUCTION
        json_response_template = cot_qa_response_template(answer_type=label_type)
        # Format Messages
        user_message = f"""\
{instruction}

Task: {question}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Summary:
{summary}

{response_instruction}
{json_response_template}"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        # Save case and all associated info as a JSON-parsable metadata dict
        metadata = {
            "case": json.loads(case.to_json(orient="index")),
            "notes": None,  # No notes used, only summaries
            "question": question,
            "label": label,
            "label_type": label_type,
            "fewshot_cases": None,
        }
        return Message(messages=messages, metadata=metadata)

    def fewshot_qa_with_summary_context(
        self,
        case: pd.Series,
        fewshot_cases: pd.DataFrame,
        num_fewshot: int = 10,
        seed: int = 42,
        dynamic_seed: bool = True,
        balance_fewshot_classes: bool = True,
        num_bins: int = 4,
    ) -> str:
        """Create prompt for Fewshot Q&A using summary of notes associated with case.
        The `case` series is expected to have the index `Summary` which corresponds to the
        summary of notes for that case.  Similarly, each row in `fewshot_cases` is expected
        to have the column `Summary` which corresponds to the summary of notes for the
        fewshot case."""
        # Case Info
        scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
        procedure_description = (
            case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
        )
        diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
        surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
        # Case Notes
        summary = case.Summary
        # Question & Answer
        question = self.question
        label, label_type = get_label_from_case(case=case, label_name=self.label_name)
        # Instruction & Response Specification
        instruction = FEWSHOT_QA_INSTRUCTION
        response_instruction = RESPONSE_INSTRUCTION
        json_response_template = qa_response_template(answer_type=label_type)
        # Fewshot Examples
        seed = seed + int(case.ProcID) if dynamic_seed else seed
        fewshot_cases_df = select_fewshot_examples(
            fewshot_data=fewshot_cases,
            n_fewshot=num_fewshot,
            label_name=self.label_name,
            seed=seed,
            balance_fewshot_classes=balance_fewshot_classes,
            num_bins=num_bins,
        )
        fewshot_string_fn = partial(
            make_fewshot_examples_string,
            question=question,
            label_name=self.label_name,
        )
        fewshot_case_strings = fewshot_cases_df.apply(fewshot_string_fn, axis="columns")
        fewshot_case_strings = "\n\n".join(fewshot_case_strings)
        # Format Messages
        user_message = f"""\
{instruction}

{fewshot_case_strings}

Task: {question}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Summary:
{summary}

{response_instruction}
{json_response_template}"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        # Save case and all associated info as a JSON-parsable metadata dict
        metadata = {
            "case": json.loads(case.to_json(orient="index")),
            "notes": None,  # No notes used, only summaries
            "question": question,
            "label": label,
            "label_type": label_type,
            "fewshot_cases": json.loads(fewshot_cases_df.to_json(orient="records")),
        }
        return Message(messages=messages, metadata=metadata)

    def fewshot_cot_qa_with_summary_context(
        self,
        case: pd.Series,
        fewshot_cases: pd.DataFrame,
        num_fewshot: int = 10,
        seed: int = 42,
        dynamic_seed: bool = True,
        balance_fewshot_classes: bool = True,
        num_bins: int = 4,
    ) -> str:
        """Create prompt for Fewshot Q&A using summary of notes associated with case.
        The `case` series is expected to have the indices `Summary` which corresponds to the
        summary of notes for that case.  Similarly, each row in `fewshot_cases` is expected
        to have the column `Summary` which corresponds to the summary of notes for the
        fewshot case and `Rationale` which corresponds to a generated chain-of-thought
        rationale."""
        # Case Info
        scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
        procedure_description = (
            case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
        )
        diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
        surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
        # Case Notes
        summary = case.Summary
        # Question & Answer
        question = self.question
        label, label_type = get_label_from_case(case=case, label_name=self.label_name)
        # Instruction & Response Specification
        instruction = FEWSHOT_COT_QA_INSTRUCTION
        response_instruction = COT_RESPONSE_INSTRUCTION
        json_response_template = cot_qa_response_template(answer_type=label_type)
        # Fewshot Examples
        seed = seed + int(case.ProcID) if dynamic_seed else seed
        fewshot_cases_df = select_fewshot_examples(
            fewshot_data=fewshot_cases,
            n_fewshot=num_fewshot,
            label_name=self.label_name,
            seed=seed,
            balance_fewshot_classes=balance_fewshot_classes,
            num_bins=num_bins,
        )
        fewshot_string_fn = partial(
            make_fewshot_cot_examples_string,
            question=question,
            label_name=self.label_name,
        )
        fewshot_case_strings = fewshot_cases_df.apply(fewshot_string_fn, axis="columns")
        fewshot_case_strings = "\n\n".join(fewshot_case_strings)
        # Format Messages
        user_message = f"""\
{instruction}

{fewshot_case_strings}

Task: {question}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Summary:
{summary}

{response_instruction}
{json_response_template}"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        # Save case and all associated info as a JSON-parsable metadata dict
        metadata = {
            "case": json.loads(case.to_json(orient="index")),
            "notes": None,  # No notes used, only summaries
            "question": question,
            "label": label,
            "label_type": label_type,
            "fewshot_cases": json.loads(fewshot_cases_df.to_json(orient="records")),
        }
        return Message(messages=messages, metadata=metadata)


## Utilities for composing few-shot examples


def make_fewshot_examples_string(
    case: pd.Series,
    question: str = "",
    label_name: str = "",
) -> str:
    """Template for a single few-shot example with task, context, answer.
    The `case` must contain index `Summary`.
    For n-shot demonstrations, use this function on each of n examples and
    stitch together into a string.
    """
    # Case Info
    scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
    procedure_description = case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
    diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
    surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
    # Case Notes
    summary = case.Summary
    # Question & Answer
    question = str(question)
    label, label_type = get_label_from_case(case=case, label_name=label_name)
    answer = str(label)
    json_response = json.dumps(
        {
            ANSWER_KEY: answer,
        },
        indent=4,
    )
    # Format Example String
    fewshot_examples_string = f"""\
Task: {question}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Summary:
{summary}
JSON Response:
{json_response}"""
    return fewshot_examples_string


def make_fewshot_cot_examples_string(
    case: pd.Series,
    question: str = "",
    label_name: str = "",
) -> str:
    """Template for a single few-shot example with task, context, answer.
    The `case` must contain index `Summary` and `Rationale`.
    For n-shot demonstrations, use this function on each of n examples and
    stitch together into a string.
    """
    # Case Info
    scheduled_procedure = case.ScheduledProcedure if pd.notna(case.ScheduledProcedure) else ""
    procedure_description = case.ProcedureDescription if pd.notna(case.ProcedureDescription) else ""
    diagnosis = case.Diagnosis if pd.notna(case.Diagnosis) else ""
    surgery_service = case.SurgService if pd.notna(case.SurgService) else ""
    # Case Notes
    summary = case.Summary
    rationale = case.Rationale
    # Question & Answer
    question = str(question)
    label, label_type = get_label_from_case(case=case, label_name=label_name)
    answer = str(label)
    json_response = json.dumps(
        {
            EXPLANATION_COT_KEY: rationale,
            ANSWER_KEY: answer,
        },
        indent=4,
    )
    # Format Example String
    fewshot_examples_string = f"""\
Task: {question}
Context:
Procedure: {scheduled_procedure}
Procedure Description: {procedure_description}
Diagnosis: {diagnosis}
Provider Service: {surgery_service}
Medical Record Summary:
{summary}
JSON Response:
{json_response}"""
    return fewshot_examples_string


## Utilities for formatting notes for prompts


def note_dataframe(case: pd.Series | tuple) -> pd.DataFrame:
    "For each case, unpack the associated notes and explode them into a notes dataframe."
    if isinstance(case, tuple):
        case = pd.Series(case)
    notes = pd.DataFrame(
        case.loc[
            [
                "NoteServiceDate",
                "ContactEnteredDate",
                "NoteName",
                "NoteStatus",
                "AuthorProviderType",
                "NoteText",
                "NoteTextTokenLength",
            ]
        ].to_dict()
        # Sort in chronological order
    ).sort_values(by=["NoteServiceDate", "ContactEnteredDate"])
    return notes


def format_note_with_header(note: pd.Series) -> str:
    "Add a single-line note header that includes note type, author type, and note datetime."
    header = f"{note.NoteName} written by {note.AuthorProviderType} at {note.NoteServiceDate}:"
    note_text = note.NoteText
    formatted_note = f"{header}\n{note_text}"
    return formatted_note


def format_all_notes_for_case(
    case: pd.Series, note_delimiter: str = "\n\n", token_limit: int | None = None
) -> tuple[str, pd.DataFrame]:
    "Get all notes for case, drop ones that don't fit in token limit, and join as a single string."
    # Get notes info as a dataframe
    notes = note_dataframe(case).sort_values(by="NoteServiceDate", ascending=True)
    # Format notes and note metadata into a single string per note
    notes = notes.assign(FormattedNote=notes.apply(format_note_with_header, axis="columns"))
    # Compute Token Length for Formatted Note
    token_ct_fn = partial(num_tokens_from_string, encoding_name="cl100k_base")
    notes = notes.assign(FormattedNoteTokenLength=notes.FormattedNote.apply(token_ct_fn))
    # Add tokens from delimiter to all except most recent note
    note_delimiter_num_tokens = token_ct_fn(note_delimiter)
    additional_num_tokens = pd.Series([note_delimiter_num_tokens] * (len(notes) - 1) + [0])
    notes = notes.assign(
        FormattedNoteTokenLength=notes.FormattedNoteTokenLength + additional_num_tokens
    )

    # Drop oldest notes that don't fit into input context
    if token_limit is not None:
        # Get cumulative sum starting from the most recent note and going backward
        notes = notes.assign(ReverseCumSum=notes.FormattedNoteTokenLength[::-1].cumsum()[::-1])
        # Remove oldest notes that would exceed input context
        notes = notes.loc[notes.ReverseCumSum <= token_limit]
    output_string = note_delimiter.join(notes.FormattedNote)

    # Return output string and final notes dataframe used to construct output string
    return output_string, notes


## Utilities for getting label value and type from case


def get_label_from_case(case: pd.Series, label_name: str) -> tuple[Any, str]:
    "Return tuple (label, label_type,)"
    match label_name:
        case "summarize":
            return None, None
        case "asa":
            label = int(case.ASA)
            return label, "int"
        case "phase1_duration":
            if isinstance(case.Phase1Duration, pd.Timedelta):
                label = timedelta2minutes(case.Phase1Duration)
            else:
                label = case.Phase1Duration
            return int(label), "int"
        case "phase2_duration":
            if isinstance(case.Phase2Duration, pd.Timedelta):
                label = timedelta2minutes(case.Phase2Duration)
            else:
                label = case.Phase2Duration
            return int(label), "int"
        case "pacu_duration":
            if isinstance(case.PACUDuration, pd.Timedelta):
                label = timedelta2minutes(case.PACUDuration)
            else:
                label = case.PACUDuration
            return int(label), "int"
        case "hospital_duration":
            if isinstance(case.HospitalDuration2, pd.Timedelta):
                label = timedelta2days(case.HospitalDuration2)
            else:
                label = case.HospitalDuration2
            return int(label), "int"
        case "hospital_admission":
            return case.HospitalAdmission, "bool"
        case "icu_duration":
            if isinstance(case.ICUDuration2, pd.Timedelta):
                label = timedelta2days(case.ICUDuration2)
            else:
                label = case.ICUDuration2
            return int(label), "int"
        case "icu_admission":
            return case.ICUAdmission, "bool"
        case "unplanned_admit":
            return case.UnplannedAdmit, "bool"
        case "hospital_mortality":
            return case.HospitalMortality, "bool"
        case _:
            warnings.warn(f"Unknown `label_name`: {label_name}. Setting label to `None`.")
            return None, None


## Utilities for constructing few-shot prompts


def select_fewshot_examples(
    fewshot_data: pd.DataFrame,
    n_fewshot: int = 10,
    seed: int = 42,
    balance_fewshot_classes: bool = True,
    label_name: str | None = None,
    num_bins: int = 4,
) -> pd.DataFrame:
    """Controls behavior in how fewshot examples are selected from `fewshot_data`.

    If `label_name` is categorical or boolean, this method counts the prevalence of
    each class and reduces the dominant classes until reaching `n_fewshot`.
    If `label_name` is continuous or datetime, this method will first bin the
    variable into quantiles and then treat each quantile as a class, reducing the dominant
    classes until reaching `n_fewshot`. By default continuous variables are binned into
    quartiles.

    The balancing technique used here cannot compensate for an imbalance where one
    or more classes are underrepresented as we are removing majority class examples and
    replacing them with minority class examples (to the extent that minority class
    examples are available in `fewshot_data`). There is no oversampling or data augmentation
    of the minority class. This ensures that we remain faithful to the examples present
    in the fewshot data.

    Args:
        fewshot_data (pd.DataFrame): Dataframe of fewshot examples.
        n_fewshot (int, optional): Number of fewshot examples to sample. Defaults to 10.
        seed (int, optional): Random number generator seed. Defaults to 42.
        balance_fewshot_classes (bool, optional): Whether to balance fewshot classes
            or not.  If False, the frequency of fewshot examples returned reflect
            the class frequency of labels in `fewshot_data`. If True, a best attempt
            effort is made to balance label representation in the fewshot examples by
            downsampling common class labels and upsampling minority class labels.
            Defaults to True.
        label_name (str | None, optional): String name of label {"asa", "phase1_duration",
            "phase2_duration", "pacu_duration", "hospital_duration", "hospital_admission",
            "icu_duration", "icu_admission", "unplanned_admit", "hospital_mortality"}
        num_bins (int, optional): Only used for continuous-valued labels to bin
            continuous values into groups upon which class balancing is performed.
            This argument determines the number of groups. If labels are categorical,
            this argument has no effect and the inherent number of classes used
            in categorical data is used instead. Defaults to 4.

    Returns:
        pd.DataFrame: Dataframe of `n_fewshot` examples sampled from `fewshot_data`.
    """
    _fewshot_data = fewshot_data.copy()
    ## Shuffle Fewshot dataset
    if seed is not None:
        _fewshot_data = _fewshot_data.sample(frac=1, random_state=seed)

    ## Balance Representation of Each Class in Fewshot Examples by dropping
    # samples in overrepresented classes. Only works properly if `fewshot_data`
    # is much larger than `n_fewshot`.
    # Otherwise, default class distribution in `fewshot_data`` is used.
    if balance_fewshot_classes:
        match label_name:
            case "asa":
                label_col_name = "ASA"
            case "phase1_duration":
                label_col_name = "Phase1Duration"
            case "phase2_duration":
                label_col_name = "Phase2Duration"
            case "pacu_duration":
                label_col_name = "PACUDuration"
            case "hospital_duration":
                label_col_name = "HospitalDuration2"
            case "hospital_admission":
                label_col_name = "HospitalAdmission"
            case "icu_duration":
                label_col_name = "ICUDuration2"
            case "icu_admission":
                label_col_name = "ICUAdmission"
            case "unplanned_admit":
                label_col_name = "UnplannedAdmit"
            case "hospital_mortality":
                label_col_name = "HospitalMortality"
            case _:
                warnings.warn(
                    f"Unknown `label_name`: {label_name}. " "Cannot balance fewshot classes."
                )
                label_col_name = ""
        if label_col_name in ("ASA", "UnplannedAdmit", "HospitalMortality"):
            # Get approximate desired counts for each class
            # (not exact but accurate within the number of classes available for label)
            class_values, class_counts = get_class_counts_after_balancing(
                values=_fewshot_data[label_col_name].tolist(),
                desired_num_samples=n_fewshot,
            )
            # Create Balanced `fewshot_data` by subsampling each class
            sampled_class_dfs = []
            for class_value, class_count in zip(class_values, class_counts):
                class_df = _fewshot_data.query(f"{label_col_name} == {class_value}")
                num_class_in_data = len(class_df)
                class_df = class_df.sample(n=min(class_count, num_class_in_data), random_state=seed)
                sampled_class_dfs += [class_df]
            # Replace `fewshot_data` with new class-balanced version
            # Because of rounding, this does not always leave us with exactly
            # `n_fewshot`, so we still need to select exactly the first `n_fewshot`
            # examples below.
            _fewshot_data = pd.concat(sampled_class_dfs, axis=0)
        elif label_col_name in (
            "Phase1Duration",
            "Phase2Duration",
            "PACUDuration",
            "ICUDuration2",
            "HospitalDuration2",
        ):
            durations = _fewshot_data[label_col_name]
            if label_col_name in ("ICUDuration2", "HospitalDuration2"):
                if isinstance(durations.iloc[0], pd.Timedelta):
                    durations = durations.apply(timedelta2days)
            elif label_col_name in ("Phase1Duration", "Phase2Duration", "PACUDuration"):
                if isinstance(durations.iloc[0], pd.Timedelta):
                    durations = durations.apply(timedelta2minutes)
            # Categorize Durations into Groups by Creating bins from dynamic range of values
            bin_boundaries = np.linspace(
                start=durations.min(), stop=durations.max(), num=num_bins + 1
            )
            group_labels = list(range(1, num_bins + 1))
            groups = pd.cut(
                durations,
                bins=bin_boundaries,
                labels=group_labels,
                include_lowest=True,
            ).rename("Group")
            # Merge Quantile Info
            _fewshot_data = _fewshot_data.join(groups)
            # Get approximate desired counts for each class
            # (not exact but accurate within the number of classes available for label)
            class_values, class_counts = get_class_counts_after_balancing(
                values=_fewshot_data["Group"],
                desired_num_samples=n_fewshot,
            )
            # Create Balanced `fewshot_data` by subsampling each class
            sampled_class_dfs = []
            for class_value, class_count in zip(class_values, class_counts):
                class_df = _fewshot_data.query(f"Group == {class_value}")
                num_class_in_data = len(class_df)
                class_df = class_df.sample(n=min(class_count, num_class_in_data), random_state=seed)
                sampled_class_dfs += [class_df]
            # Replace `fewshot_data` with new class-balanced version
            # Because of rounding, this does not always leave us with exactly
            # `n_fewshot`, so we still need to select exactly the first `n_fewshot`
            # examples below.
            _fewshot_data = pd.concat(sampled_class_dfs, axis=0).drop(columns="Group")

    ## Shuffle Fewshot dataset
    if seed is not None:
        _fewshot_data = _fewshot_data.sample(frac=1, random_state=seed)

    ## Select Max Number of Fewshot examples to use
    if len(_fewshot_data) > n_fewshot:
        _fewshot_data = _fewshot_data.iloc[:n_fewshot, :]
    return _fewshot_data


def get_class_counts_after_balancing(values: list, desired_num_samples: int) -> tuple[list, list]:
    """Identify number of samples per class for balanced representation and then
    attempts to match this number with the given data.  This is done by redistributing
    samples in overrepresented classes to samples in underrepresented classes within
    the constraints of the number of actual samples available in each class in the dataset.

    Args:
        values (list): All label values in dataset (includes all classes)
        desired_num_samples (int): Final samples desired from dataset.

    Returns:
        tuple[list, list]: First list contains the number of unique class label values.
            Second list contains the corresponding count of examples in `values` for
            each class label. The counts in the second list may be slightly over the
            `desired_num_samples` based on the heuristic used for balancing classes.
    """
    class_values, class_counts = np.unique(values, return_counts=True)
    num_classes = len(class_values)
    ideal_samples_per_class = math.ceil(desired_num_samples / num_classes)
    # Determine which classes have fewer or more than `ideal_samples_per_class`
    classes_below_ideal = class_counts <= ideal_samples_per_class
    classes_above_ideal = class_counts > ideal_samples_per_class
    # Redistribute unused budget of samples from `classes_below_ideal` to `classes_above_ideal`:
    total_samples_in_below_ideal_classes = class_counts[classes_below_ideal]
    unused_budget_in_below_ideal_classes = sum(
        ideal_samples_per_class - total_samples_in_below_ideal_classes
    )
    num_classes_above_ideal = classes_above_ideal.sum()
    if num_classes_above_ideal == 0:
        warnings.warn(
            "Cannot perform balancing because all classes "
            "have below the ideal number of samples per class. "
            "Either increase size of dataset used for fewshot examples "
            "or choose fewer examples for in-context demonstrations "
            "or disable class balancing."
        )
        return class_values, class_counts
    else:
        extra_samples_for_above_ideal = classes_above_ideal * math.ceil(
            unused_budget_in_below_ideal_classes / num_classes_above_ideal
        )
    balanced_class_counts = (
        classes_below_ideal * class_counts
        + classes_above_ideal * ideal_samples_per_class
        + extra_samples_for_above_ideal
    )
    # NOTE: Allowed_class_counts may be slightly over the desired_num_samples
    return class_values, balanced_class_counts
