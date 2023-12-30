import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from llm_utils import DataPaths, read_pandas, sidethread_event_loop_async_runner

from llm.chat_model import Bundle, ChatCompletionResponseType, ChatModel, Message, MessagesType
from llm.database import Database, PerioperativePrediction
from llm.prompts import (
    ANSWER_KEY,
    EXPLANATION_COT_KEY,
    EXPLANATION_KEY,
    PATIENT_SUMMARY_KEY,
    PromptComposer,
)

logger = logging.getLogger("ExperimentLogger")


GPT4TURBO_MAX_TOKENS = 128000
GPT35TURBO_MAX_TOKENS = 16384


@dataclass(kw_only=True)
class CaseBundle:
    "Package input and output messages together."
    index: int
    input: Message
    output: PerioperativePrediction


@dataclass(kw_only=True)
class Experiment:
    """
    Encapsulates entire experiment and loads dataset specific for task.
    """

    ## Experimental Config (required)
    experiment_name: str
    """task: {"asa", "phase1_duration", "hospital_duration", "hospital_admission", 
        "icu_duration", "icu_admission", "unplanned_admit", "hospital_mortality"}"""
    task: str
    '''note_kind: "last10", "preanes"'''
    note_kind: str

    ## Optional Experimental Config (Useful for debugging/prelim experiments)
    num_inference_examples: int | None = None
    num_fewshot_examples: int | None = None

    ## Optional for Async/Concurrent API calls
    """num_concurrent: max number of concurrent LLM Generation API Calls"""
    num_concurrent: int = 5
    """num_retries: max number of retries if fail validation or LLM generation failed,
        after which LLM ChatCompletion Generation returns `None`."""
    num_retries: int = 5

    ## Database & Path Objects
    db: Database = field(init=False)
    project_dir: str | Path = Path(__file__).parent.parent.parent
    data_version: int = 7
    dataset_name: str = "dataset4"
    paths: DataPaths = field(init=False)

    ## Loaded Datasets (determined by experiment config)
    inference_dataset: pd.DataFrame | None = None
    fewshot_dataset: pd.DataFrame | None = None
    raw_inference_dataset: pd.DataFrame | None = None  # unmodified dataset
    raw_fewshot_dataset: pd.DataFrame | None = None  # unmodified dataset
    inference_dataset_name: str | None = None
    fewshot_dataset_name: str | None = None

    ## Chat Model
    chat_model: ChatModel = ChatModel()
    """expected_response_tokens: number of tokens to reserve for response.  This must be
        an adequate number; if response JSON is truncated, then it is not parsable.
        This value needs to be manually set."""
    expected_response_tokens: int = 500
    """token_limit: the max number of tokens available to the PromptComposer for
        composing prompts. This is the max token limit of the `ChatModel.model` with
        the `expected_response_tokens` subtracted from the amount."""
    token_limit: int | None = None

    ## Experiment Inputs & Outputs
    inputs: dict[str, Any] = field(default_factory=lambda: {})
    outputs: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        # Initialize Database
        self.db = Database(echo=False)
        # Define Raw Data Paths
        self.setup_filepaths()
        # Load Datasets for Task
        self.load_task_datasets()
        # Set Token Limits
        self.set_token_limit()
        # Checks
        if self.num_concurrent == 0:
            warnings.warn("Invalid `num_concurrent`.  Setting num_concurrent=1.")
            self.num_concurrent = 1

    def setup_filepaths(self) -> None:
        "Get filepath references to all files in dataset."
        self.paths = DataPaths(project_dir=self.project_dir, data_version=self.data_version)
        # Get file stems (name w/o file extension) in dataset folder
        dataset_path = self.paths.processed / self.dataset_name
        file_stems = [x.stem for x in dataset_path.glob("*.feather")]
        # Register all the paths
        for file_stem in file_stems:
            self.paths.register(name=file_stem, path=dataset_path / f"{file_stem}.feather")

    def load_task_datasets(self) -> None:
        "Loads dataset for task."
        # Load Inference Data
        self.inference_dataset_name = f"{self.task}-{self.note_kind}-inference"
        self.inference_dataset = read_pandas(self.paths[self.inference_dataset_name])
        self.raw_inference_dataset = self.inference_dataset.copy()
        if self.num_inference_examples:
            self.inference_dataset = self.inference_dataset.iloc[: self.num_inference_examples, :]
        # Load Fewshot Data
        self.fewshot_dataset_name = f"{self.task}-{self.note_kind}-fewshot"
        self.fewshot_dataset = read_pandas(self.paths[self.fewshot_dataset_name])
        self.raw_fewshot_dataset = self.fewshot_dataset.copy()
        if self.num_fewshot_examples:
            self.fewshot_dataset = self.fewshot_dataset.iloc[: self.num_fewshot_examples, :]

    def set_token_limit(self) -> None:
        """Determines token limit available to the PromptComposer which will generate
        the final prompt. It is up to the PromptComposer to account for other tokens
        (e.g. system message, note header metadata tokens, additional tokens
        used in message formatting.)
        Set `self.expected_response_tokens` to larger value if having length limit issues
        since this will add an extra buffer of tokens."""
        if self.token_limit is None:
            match self.chat_model.model:
                case "gpt-35-turbo-1106":
                    self.token_limit = GPT35TURBO_MAX_TOKENS - self.expected_response_tokens
                case "gpt-4-1106":
                    self.token_limit = GPT4TURBO_MAX_TOKENS - self.expected_response_tokens
                case _:
                    warnings.warn(
                        "Unknown model {self.chat_model.model}, "
                        "setting `token_limit` to 128k - `expected_response_tokens`."
                    )
                    self.token_limit = GPT4TURBO_MAX_TOKENS - self.expected_response_tokens

    def retrieve_from_db(
        self, messages: Message, force: bool = False
    ) -> list[PerioperativePrediction | None]:
        """Compares hash of input messages + model with database to see if
        prior LLM generation result was cached.

        Args:
            messages (Message): Message object (wrapper around messages and metadata)

        Returns:
            list[PerioperativePrediction | None]: PerioperativePrediction from
                database if exists, or returns `None`
        """
        if force:
            return None
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        user_message = [msg for msg in messages.messages if msg["role"] == "user"][0]["content"]
        input_hash = PerioperativePrediction.compute_input_hash(
            model_name=self.chat_model.model,
            system_message=pc.system_message,
            user_message=user_message,
        )
        pp_list = self.db.select_where_equals(
            experiment_name=self.experiment_name, input_hash=input_hash
        )
        # Return existing PerioperativePrediction if exists in database.
        if bool(pp_list):
            pp = pp_list[0]
            logger.debug(f"Retrieving existing: {pp}")
            return pp
        else:
            return None

    # Format answer from LLM response
    def format_answer(self, text: str) -> int | float | bool | None:
        "Format answer based on task."
        return PerioperativePrediction.format_text(text=text, task_name=self.task)

    # Validate answer from LLM response
    def summary_validation_callback(
        self,
        messages: MessagesType,
        response: Bundle,
    ) -> bool:
        """Validation of generated summary in the LLM response. Assumes output type is Bundle.
        Returns `True` to accept response and `False` to reject response."""
        # Try to format answer and check resulting data type.
        # If fails, then reject the response.
        try:
            # Extract answer from Response
            json_response = json.loads(response.response_message)
            json_response[PATIENT_SUMMARY_KEY]
            # If no failure in extracting summary from JSON, then validation succeed
            return True
        except Exception as e:
            warnings.warn(
                f"Error in validation_callback: {e}. Triggering validation fail.\n"
                f"Message: {messages}\n"
                f"Response: {response}"
            )
            return False

    def generate_cot_rationale_validation_callback(
        self,
        messages: MessagesType,
        response: Bundle,
    ) -> bool:
        """Validation of generated CoT rationale in the LLM response. Assumes output type is Bundle.
        Returns `True` to accept response and `False` to reject response."""
        # Try to format answer and check resulting data type.
        # If fails, then reject the response.
        try:
            # Extract answer from Response
            json_response = json.loads(response.response_message)
            json_response[EXPLANATION_COT_KEY]
            # If no failure in extracting CoT Rationale from JSON, then validation succeed
            return True
        except Exception as e:
            warnings.warn(
                f"Error in validation_callback: {e}. Triggering validation fail.\n"
                f"Message: {messages}\n"
                f"Response: {response}"
            )
            return False

    def qa_validation_callback(
        self,
        messages: MessagesType,
        response: Bundle,
    ) -> bool:
        """Validation of answer in the LLM response. Assumes output type is Bundle.
        Returns `True` to accept response and `False` to reject response."""
        # Try to format answer and check resulting data type.
        # If fails, then reject the response.
        try:
            # Extract answer from Response
            json_response = json.loads(response.response_message)
            answer = json_response[ANSWER_KEY]
            formatted_answer = self.format_answer(answer)
            # If formatted answer is in desired data type (and not `None`),
            # then accept. Otherwise, reject.
            match self.task:
                case "asa":
                    if isinstance(formatted_answer, int):
                        return True
                case (
                    "phase1_duration"
                    | "phase2_duration"
                    | "pacu_duration"
                    | "icu_duration"
                    | "hospital_duration"
                ):
                    if isinstance(formatted_answer, (int, float)):
                        return True
                case (
                    "hospital_admission"
                    | "icu_admission"
                    | "unplanned_admit"
                    | "hospital_mortality"
                ):
                    if isinstance(formatted_answer, bool):
                        return True
                case _:
                    raise ValueError(f"Invalid `task_name`: {self.task}")
            return False
        except Exception as e:
            warnings.warn(
                f"Error in validation_callback: {e}. Triggering validation fail.\n"
                f"Message: {messages}\n"
                f"Response: {response}"
            )
            return False

    # Notes Summary Generation
    def generate_notes_summary_for_inference_dataset(self, force: bool = False) -> list[CaseBundle]:
        "Generate summaries for inference dataset."
        inference_summary_cb = self.generate_notes_summary(
            cases=self.inference_dataset, force=force, description="Summarize Inference Dataset"
        )
        inference_summary_inputs = pd.DataFrame(cb.input for cb in inference_summary_cb)
        inference_summary_outputs = pd.DataFrame(cb.output for cb in inference_summary_cb)
        self.inputs["inference_summary"] = inference_summary_inputs
        self.outputs["inference_summary"] = inference_summary_outputs
        # Add summaries to dataset
        df = inference_summary_outputs
        self.inference_dataset = self.inference_dataset.assign(Summary=df.summary)
        return self.inference_dataset

    def generate_notes_summary_for_fewshot_dataset(self, force: bool = False) -> list[CaseBundle]:
        "Generate summaries for fewshot dataset."
        fewshot_summary_cb = self.generate_notes_summary(
            cases=self.fewshot_dataset, force=force, description="Summarize Fewshot Dataset"
        )
        fewshot_summary_inputs = pd.DataFrame(cb.input for cb in fewshot_summary_cb)
        fewshot_summary_outputs = pd.DataFrame(cb.output for cb in fewshot_summary_cb)
        self.inputs["fewshot_summary"] = fewshot_summary_inputs
        self.outputs["fewshot_summary"] = fewshot_summary_outputs
        # Add summaries to dataset
        df = fewshot_summary_outputs
        self.fewshot_dataset = self.fewshot_dataset.assign(Summary=df.summary)
        return self.fewshot_dataset

    def generate_notes_summary(
        self, cases: pd.Series | pd.DataFrame, force: bool = False, description: str = "Summarize"
    ) -> list[CaseBundle]:
        """Generate summary for notes.
        If previously generated for input message + model, will instead retrieve from database.

        Args:
            cases (pd.Series | pd.DataFrame): Dataframe with each row as a case.
            force (bool, optional): Force LLM generation for all messages even if
                previously cached response exists. Defaults to False.
            description (str, optional): Description for progress bar.

        Returns:
            list[CaseBundle]: List of CaseBundle items which package input Message and output
                PerioperativePredictions objects together.
        """
        if isinstance(cases, pd.Series):
            cases = cases.to_frame()

        # Create Prompts for all cases
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        messages_list = [pc.summarize(case=case) for _, case in cases.iterrows()]
        # Check to see if prior LLM generation for message exists in database
        result_list = [
            self.retrieve_from_db(messages=messages, force=force) for messages in messages_list
        ]
        # Package Message & PerioperativePrediction together to jointly iterate
        case_bundles = [
            CaseBundle(index=idx, input=message, output=pp)
            for idx, message, pp in zip(range(len(messages_list)), messages_list, result_list)
        ]
        # Identify cases where we didn't retrieve a PerioperativePrediction from database
        cases_needing_generation = [x for x in case_bundles if x.output is None]
        messages_list_to_generate = [x.input for x in cases_needing_generation]
        if not bool(messages_list_to_generate):
            return case_bundles
        else:
            # Unpack Response Bundles into PerioperativePrediction objects
            def bundle_to_pp_callback(
                messages: Message, response: ChatCompletionResponseType
            ) -> PerioperativePrediction:
                "Unpack response bundle, format as PerioperativePrediction, save in Database."
                # Unpack Message
                msgs = messages.messages
                system_message = [m for m in msgs if m["role"] == "system"][0]["content"]
                user_message = [m for m in msgs if m["role"] == "user"][0]["content"]
                # Unpack Metadata from Message
                metadata = messages.metadata
                proc_id = metadata["case"]["ProcID"]
                label = metadata["label"]
                label_type = metadata["label_type"]
                fewshot_cases = metadata["fewshot_cases"]
                num_fewshot = len(fewshot_cases) if fewshot_cases is not None else None
                pp_kwargs = {
                    "experiment_name": self.experiment_name,
                    "task_name": self.task,
                    "task_strategy": "summarize",
                    "shot_kind": "zeroshot",
                    "num_fewshot": num_fewshot,
                    "note_kind": self.note_kind,
                    "note_summary": False,
                    "dataset_name": self.inference_dataset_name,
                    "model_name": self.chat_model.model,
                    "proc_id": proc_id,
                    "label": label,
                    "label_type": label_type,
                    "metadata_dict": metadata,
                    "system_message": system_message,
                    "user_message": user_message,
                }
                bundle = response
                if bundle is None or bundle.num_retries == 0:
                    # LLM Generation/Validation failed across all retries.
                    # Return PerioperativePrediction with no answers
                    # and do not insert into database.  Raw response_dict still returned.
                    if bundle is not None:
                        pp_kwargs |= {"response_dict": bundle.response_dict}
                    pp = PerioperativePrediction(**pp_kwargs)
                    return pp
                else:
                    # Unpack Response
                    json_response = json.loads(bundle.response_message)
                    summary = json_response[PATIENT_SUMMARY_KEY]
                    # Format as PerioperativePrediction
                    pp = PerioperativePrediction(
                        **pp_kwargs,
                        summary=summary,
                        answer=None,
                        explanation=None,
                        response_message=bundle.response_message,
                        response_model_name=bundle.model,
                        response_id=bundle.id,
                        system_fingerprint=bundle.system_fingerprint,
                        created_time=bundle.created_time,
                        finish_reason=bundle.finish_reason,
                        response_dict=bundle.response_dict,
                    )
                    # Insert into Database
                    self.db.insert(pp, dry_run=False, expire_on_commit=False)
                    logger.debug(f"Generated and inserted into database: {pp}")
                    return pp

            # Generate LLM Output
            if self.num_concurrent == 1:
                response_bundles = self.chat_model.chat_completions(
                    messages_list=messages_list_to_generate,
                    num_retries=self.num_retries,
                    output_format="bundle",
                    validation_callback=self.summary_validation_callback,
                    callback=bundle_to_pp_callback,
                    description=description,
                )
            else:
                response_bundles = sidethread_event_loop_async_runner(
                    async_function=self.chat_model.async_chat_completions(
                        messages_list=messages_list_to_generate,
                        num_concurrent=self.num_concurrent,
                        num_retries=self.num_retries,
                        output_format="bundle",
                        validation_callback=self.summary_validation_callback,
                        callback=bundle_to_pp_callback,
                        description=description,
                    )
                )
            # Get PerioperativePredictions from LLM generations
            generated_pp_list = [b.callback_response for b in response_bundles]
            # Replace `None` output values in case bundles with PerioperativePredictions
            for case_bundle, pp in zip(cases_needing_generation, generated_pp_list):
                case_bundle.output = pp
            # Return All PerioperativePredictions (Retrieved from DB + LLM generated)
            return case_bundles

    # Chain-of-Thought Rationale Generation
    def generate_cot_rationale_for_fewshot_dataset(self, force: bool = False) -> list[CaseBundle]:
        "Generate chain-of-thought rationale for fewshot dataset."
        fewshot_cot_generation_cb = self.generate_cot_rationale(
            cases=self.fewshot_dataset,
            force=force,
            description="Generate CoT Rationale for Fewshot Dataset",
        )
        fewshot_cot_generation_inputs = pd.DataFrame(cb.input for cb in fewshot_cot_generation_cb)
        fewshot_cot_generation_outputs = pd.DataFrame(cb.output for cb in fewshot_cot_generation_cb)
        self.inputs["fewshot_cot_generation"] = fewshot_cot_generation_inputs
        self.outputs["fewshot_cot_generation"] = fewshot_cot_generation_outputs
        # Add CoT Rationales to dataset
        df = fewshot_cot_generation_outputs
        self.fewshot_dataset = self.fewshot_dataset.assign(Rationale=df.explanation)
        return self.fewshot_dataset

    def generate_cot_rationale(
        self,
        cases: pd.Series | pd.DataFrame,
        force: bool = False,
        description: str = "Generate CoT Rationale",
    ) -> list[CaseBundle]:
        """Generate Chain-of-Thought Rationale for notes.
        If previously generated for input message + model, will instead retrieve from database.

        Args:
            cases (pd.Series | pd.DataFrame): Dataframe with each row as a case.
            force (bool, optional): Force LLM generation for all messages even if
                previously cached response exists. Defaults to False.
            description (str, optional): Description for progress bar.

        Returns:
            list[CaseBundle]: List of CaseBundle items which package input Message and output
                PerioperativePredictions objects together.
        """
        if isinstance(cases, pd.Series):
            cases = cases.to_frame()
        if "Summary" not in cases.columns:
            raise ValueError("Argument `cases` must contain column `Summary`.")

        # Create Prompts for all cases
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        messages_list = [
            pc.generate_cot_with_summary_context(case=case) for _, case in cases.iterrows()
        ]
        # Check to see if prior LLM generation for message exists in database
        result_list = [
            self.retrieve_from_db(messages=messages, force=force) for messages in messages_list
        ]
        # Package Message & PerioperativePrediction together to jointly iterate
        case_bundles = [
            CaseBundle(index=idx, input=message, output=pp)
            for idx, message, pp in zip(range(len(messages_list)), messages_list, result_list)
        ]
        # Identify cases where we didn't retrieve a PerioperativePrediction from database
        cases_needing_generation = [x for x in case_bundles if x.output is None]
        messages_list_to_generate = [x.input for x in cases_needing_generation]
        if not bool(messages_list_to_generate):
            return case_bundles
        else:
            # Unpack Response Bundles into PerioperativePrediction objects
            def bundle_to_pp_callback(
                messages: Message, response: ChatCompletionResponseType
            ) -> PerioperativePrediction:
                "Unpack response bundle, format as PerioperativePrediction, save in Database."
                # Unpack Message
                msgs = messages.messages
                system_message = [m for m in msgs if m["role"] == "system"][0]["content"]
                user_message = [m for m in msgs if m["role"] == "user"][0]["content"]
                # Unpack Metadata from Message
                metadata = messages.metadata
                proc_id = metadata["case"]["ProcID"]
                label = metadata["label"]
                label_type = metadata["label_type"]
                fewshot_cases = metadata["fewshot_cases"]
                num_fewshot = len(fewshot_cases) if fewshot_cases is not None else None
                pp_kwargs = {
                    "experiment_name": self.experiment_name,
                    "task_name": self.task,
                    "task_strategy": "cot_generation",
                    "shot_kind": "zeroshot",
                    "num_fewshot": num_fewshot,
                    "note_kind": self.note_kind,
                    "note_summary": True,
                    "dataset_name": self.inference_dataset_name,
                    "model_name": self.chat_model.model,
                    "proc_id": proc_id,
                    "label": label,
                    "label_type": label_type,
                    "metadata_dict": metadata,
                    "system_message": system_message,
                    "user_message": user_message,
                }
                bundle = response
                if bundle is None or bundle.num_retries == 0:
                    # LLM Generation/Validation failed across all retries.
                    # Return PerioperativePrediction with no answers
                    # and do not insert into database.  Raw response_dict still returned.
                    if bundle is not None:
                        pp_kwargs |= {"response_dict": bundle.response_dict}
                    pp = PerioperativePrediction(**pp_kwargs)
                    return pp
                else:
                    # Unpack Response
                    bundle = response
                    generated_cot = json.loads(bundle.response_message)[EXPLANATION_COT_KEY]
                    # Format as PerioperativePrediction
                    pp = PerioperativePrediction(
                        **pp_kwargs,
                        summary=None,
                        answer=None,
                        explanation=generated_cot,
                        response_message=bundle.response_message,
                        response_model_name=bundle.model,
                        response_id=bundle.id,
                        system_fingerprint=bundle.system_fingerprint,
                        created_time=bundle.created_time,
                        finish_reason=bundle.finish_reason,
                        response_dict=bundle.response_dict,
                    )
                    # Insert into Database
                    self.db.insert(pp, dry_run=False, expire_on_commit=False)
                    logger.debug(f"Generated and inserted into database: {pp}")
                    return pp

            # Generate LLM Output
            if self.num_concurrent == 1:
                response_bundles = self.chat_model.chat_completions(
                    messages_list=messages_list_to_generate,
                    num_retries=self.num_retries,
                    output_format="bundle",
                    validation_callback=self.generate_cot_rationale_validation_callback,
                    callback=bundle_to_pp_callback,
                    description=description,
                )
            else:
                response_bundles = sidethread_event_loop_async_runner(
                    async_function=self.chat_model.async_chat_completions(
                        messages_list=messages_list_to_generate,
                        num_concurrent=self.num_concurrent,
                        num_retries=self.num_retries,
                        output_format="bundle",
                        validation_callback=self.generate_cot_rationale_validation_callback,
                        callback=bundle_to_pp_callback,
                        description=description,
                    )
                )
            # Get PerioperativePredictions from LLM generations
            generated_pp_list = [b.callback_response for b in response_bundles]
            # Replace `None` output values in case bundles with PerioperativePredictions
            for case_bundle, pp in zip(cases_needing_generation, generated_pp_list):
                case_bundle.output = pp
            # Return All PerioperativePredictions (Retrieved from DB + LLM generated)
            return case_bundles

    # Zeroshot Q&A from Notes
    def generate_zeroshot_qa_from_notes_for_inference_dataset(
        self, force: bool = False
    ) -> list[CaseBundle]:
        "Generate Zeroshot Q&A from notes for inference dataset."
        zeroshot_qa_from_notes_cb = self.generate_zeroshot_qa_from_notes(
            cases=self.inference_dataset, force=force
        )
        zeroshot_qa_from_notes_inputs = pd.DataFrame(cb.input for cb in zeroshot_qa_from_notes_cb)
        zeroshot_qa_from_notes_outputs = pd.DataFrame(cb.output for cb in zeroshot_qa_from_notes_cb)
        self.inputs["zeroshot_qa_from_notes"] = zeroshot_qa_from_notes_inputs
        self.outputs["zeroshot_qa_from_notes"] = zeroshot_qa_from_notes_outputs
        # Add zeroshot answer, explanation & label to dataset
        df = zeroshot_qa_from_notes_outputs
        self.inference_dataset = self.inference_dataset.assign(
            Label=df.label.apply(self.format_answer),
            LabelType=df.label_type,
            ZeroshotFromNotesAnswer=df.answer.apply(self.format_answer),
            ZeroshotFromNotesExplanation=df.explanation,
        )
        return self.inference_dataset

    def generate_zeroshot_qa_from_notes(
        self,
        cases: pd.Series | pd.DataFrame,
        force: bool = False,
        description: str = "Zeroshot QA from Notes",
    ) -> list[CaseBundle]:
        """Generate Zeroshot Q&A from notes context.
        If previously generated for input message + model, will instead retrieve from database.

        Args:
            cases (pd.Series | pd.DataFrame): Dataframe with each row as a case.
            force (bool, optional): Force LLM generation for all messages even if
                previously cached response exists. Defaults to False.
            description (str, optional): Description for progress bar.

        Returns:
            list[CaseBundle]: List of CaseBundle items which package input Message and output
                PerioperativePredictions objects together.
        """
        if isinstance(cases, pd.Series):
            cases = cases.to_frame()

        # Create Prompts for all cases
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        messages_list = [
            pc.zeroshot_qa_with_notes_context(case=case) for _, case in cases.iterrows()
        ]
        # Check to see if prior LLM generation for message exists in database
        result_list = [
            self.retrieve_from_db(messages=messages, force=force) for messages in messages_list
        ]
        # Package Message & PerioperativePrediction together to jointly iterate
        case_bundles = [
            CaseBundle(index=idx, input=message, output=pp)
            for idx, message, pp in zip(range(len(messages_list)), messages_list, result_list)
        ]
        # Identify cases where we didn't retrieve a PerioperativePrediction from database
        cases_needing_generation = [x for x in case_bundles if x.output is None]
        messages_list_to_generate = [x.input for x in cases_needing_generation]
        if not bool(messages_list_to_generate):
            return case_bundles
        else:
            # Unpack Response Bundles into PerioperativePrediction objects
            def bundle_to_pp_callback(
                messages: Message, response: ChatCompletionResponseType
            ) -> PerioperativePrediction:
                "Unpack response bundle, format as PerioperativePrediction, save in Database."
                # Unpack Message
                msgs = messages.messages
                system_message = [m for m in msgs if m["role"] == "system"][0]["content"]
                user_message = [m for m in msgs if m["role"] == "user"][0]["content"]
                # Unpack Metadata from Message
                metadata = messages.metadata
                proc_id = metadata["case"]["ProcID"]
                label = metadata["label"]
                label_type = metadata["label_type"]
                fewshot_cases = metadata["fewshot_cases"]
                num_fewshot = len(fewshot_cases) if fewshot_cases is not None else None
                pp_kwargs = {
                    "experiment_name": self.experiment_name,
                    "task_name": self.task,
                    "task_strategy": "qa",
                    "shot_kind": "zeroshot",
                    "num_fewshot": num_fewshot,
                    "note_kind": self.note_kind,
                    "note_summary": False,
                    "dataset_name": self.inference_dataset_name,
                    "model_name": self.chat_model.model,
                    "proc_id": proc_id,
                    "label": label,
                    "label_type": label_type,
                    "metadata_dict": metadata,
                    "system_message": system_message,
                    "user_message": user_message,
                    # TODO: save bundle.response_dict here, but leave all other fields null
                }
                bundle = response
                if bundle is None or bundle.num_retries == 0:
                    # LLM Generation/Validation failed across all retries.
                    # Return PerioperativePrediction with no answers
                    # and do not insert into database.  Raw response_dict still returned.
                    if bundle is not None:
                        pp_kwargs |= {"response_dict": bundle.response_dict}
                    pp = PerioperativePrediction(**pp_kwargs)
                    return pp
                else:
                    # Unpack Response
                    json_response = json.loads(bundle.response_message)
                    answer = json_response[ANSWER_KEY]
                    explanation = json_response[EXPLANATION_KEY]
                    # Format as PerioperativePrediction
                    pp = PerioperativePrediction(
                        **pp_kwargs,
                        summary=None,
                        answer=answer,
                        explanation=explanation,
                        response_message=bundle.response_message,
                        response_model_name=bundle.model,
                        response_id=bundle.id,
                        system_fingerprint=bundle.system_fingerprint,
                        created_time=bundle.created_time,
                        finish_reason=bundle.finish_reason,
                        response_dict=bundle.response_dict,
                    )
                    # Insert into Database
                    self.db.insert(pp, dry_run=False, expire_on_commit=False)
                    logger.debug(f"Generated and inserted into database: {pp}")
                    return pp

            # Generate LLM Output
            if self.num_concurrent == 1:
                response_bundles = self.chat_model.chat_completions(
                    messages_list=messages_list_to_generate,
                    num_retries=self.num_retries,
                    output_format="bundle",
                    validation_callback=self.qa_validation_callback,
                    callback=bundle_to_pp_callback,
                    description=description,
                )
            else:
                response_bundles = sidethread_event_loop_async_runner(
                    async_function=self.chat_model.async_chat_completions(
                        messages_list=messages_list_to_generate,
                        num_concurrent=self.num_concurrent,
                        num_retries=self.num_retries,
                        output_format="bundle",
                        validation_callback=self.qa_validation_callback,
                        callback=bundle_to_pp_callback,
                        description=description,
                    )
                )
            # Get PerioperativePredictions from LLM generations
            generated_pp_list = [b.callback_response for b in response_bundles]
            # Replace `None` output values in case bundles with PerioperativePredictions
            for case_bundle, pp in zip(cases_needing_generation, generated_pp_list):
                case_bundle.output = pp
            # Return All PerioperativePredictions (Retrieved from DB + LLM generated)
            return case_bundles

    # Zeroshot Q&A from Notes Summary
    def generate_zeroshot_qa_from_notes_summary_for_inference_dataset(
        self, force: bool = False
    ) -> list[CaseBundle]:
        "Generate Zeroshot Q&A from notes summary for inference dataset."
        zeroshot_qa_from_notes_summary_cb = self.generate_zeroshot_qa_from_notes_summary(
            cases=self.inference_dataset, force=force
        )
        zeroshot_qa_from_notes_summary_inputs = pd.DataFrame(
            cb.input for cb in zeroshot_qa_from_notes_summary_cb
        )
        zeroshot_qa_from_notes_summary_outputs = pd.DataFrame(
            cb.output for cb in zeroshot_qa_from_notes_summary_cb
        )
        self.inputs["zeroshot_qa_from_notes_summary"] = zeroshot_qa_from_notes_summary_inputs
        self.outputs["zeroshot_qa_from_notes_summary"] = zeroshot_qa_from_notes_summary_outputs
        # Add zeroshot answer & explanation to dataset
        df = zeroshot_qa_from_notes_summary_outputs
        self.inference_dataset = self.inference_dataset.assign(
            Label=df.label.apply(self.format_answer),
            LabelType=df.label_type,
            ZeroshotFromNotesSummaryAnswer=df.answer.apply(self.format_answer),
            ZeroshotFromNotesSummaryExplanation=df.explanation,
        )
        return self.inference_dataset

    def generate_zeroshot_qa_from_notes_summary(
        self,
        cases: pd.Series | pd.DataFrame,
        force: bool = False,
        description: str = "Zeroshot QA from Summaries",
    ) -> list[CaseBundle]:
        """Generate Zeroshot Q&A from notes summary context.
        If previously generated for input message + model, will instead retrieve from database.

        Args:
            cases (pd.Series | pd.DataFrame): Dataframe with each row as a case.
            force (bool, optional): Force LLM generation for all messages even if
                previously cached response exists. Defaults to False.
            description (str, optional): Description for progress bar.

        Returns:
            list[CaseBundle]: List of CaseBundle items which package input Message and output
                PerioperativePredictions objects together.
        """
        if isinstance(cases, pd.Series):
            cases = cases.to_frame()
        if "Summary" not in cases.columns:
            raise ValueError("Argument `cases` must contain column `Summary`.")

        # Create Prompts for all cases
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        messages_list = [
            pc.zeroshot_qa_with_summary_context(case=case) for _, case in cases.iterrows()
        ]
        # Check to see if prior LLM generation for message exists in database
        result_list = [
            self.retrieve_from_db(messages=messages, force=force) for messages in messages_list
        ]
        # Package Message & PerioperativePrediction together to jointly iterate
        case_bundles = [
            CaseBundle(index=idx, input=message, output=pp)
            for idx, message, pp in zip(range(len(messages_list)), messages_list, result_list)
        ]
        # Identify cases where we didn't retrieve a PerioperativePrediction from database
        cases_needing_generation = [x for x in case_bundles if x.output is None]
        messages_list_to_generate = [x.input for x in cases_needing_generation]
        if not bool(messages_list_to_generate):
            return case_bundles
        else:
            # Unpack Response Bundles into PerioperativePrediction objects
            def bundle_to_pp_callback(
                messages: Message, response: ChatCompletionResponseType
            ) -> PerioperativePrediction:
                "Unpack response bundle, format as PerioperativePrediction, save in Database."
                # Unpack Message
                msgs = messages.messages
                system_message = [m for m in msgs if m["role"] == "system"][0]["content"]
                user_message = [m for m in msgs if m["role"] == "user"][0]["content"]
                # Unpack Metadata from Message
                metadata = messages.metadata
                proc_id = metadata["case"]["ProcID"]
                label = metadata["label"]
                label_type = metadata["label_type"]
                fewshot_cases = metadata["fewshot_cases"]
                num_fewshot = len(fewshot_cases) if fewshot_cases is not None else None
                pp_kwargs = {
                    "experiment_name": self.experiment_name,
                    "task_name": self.task,
                    "task_strategy": "qa",
                    "shot_kind": "zeroshot",
                    "num_fewshot": num_fewshot,
                    "note_kind": self.note_kind,
                    "note_summary": True,
                    "dataset_name": self.inference_dataset_name,
                    "model_name": self.chat_model.model,
                    "proc_id": proc_id,
                    "label": label,
                    "label_type": label_type,
                    "metadata_dict": metadata,
                    "system_message": system_message,
                    "user_message": user_message,
                }
                bundle = response
                if bundle is None or bundle.num_retries == 0:
                    # LLM Generation/Validation failed across all retries.
                    # Return PerioperativePrediction with no answers
                    # and do not insert into database.  Raw response_dict still returned.
                    if bundle is not None:
                        pp_kwargs |= {"response_dict": bundle.response_dict}
                    pp = PerioperativePrediction(**pp_kwargs)
                    return pp
                else:
                    # Unpack Response
                    json_response = json.loads(bundle.response_message)
                    answer = json_response[ANSWER_KEY]
                    explanation = json_response[EXPLANATION_KEY]
                    # Format as PerioperativePrediction
                    pp = PerioperativePrediction(
                        **pp_kwargs,
                        summary=None,
                        answer=answer,
                        explanation=explanation,
                        response_message=bundle.response_message,
                        response_model_name=bundle.model,
                        response_id=bundle.id,
                        system_fingerprint=bundle.system_fingerprint,
                        created_time=bundle.created_time,
                        finish_reason=bundle.finish_reason,
                        response_dict=bundle.response_dict,
                    )
                    # Insert into Database
                    self.db.insert(pp, dry_run=False, expire_on_commit=False)
                    logger.debug(f"Generated and inserted into database: {pp}")
                    return pp

            # Generate LLM Output
            if self.num_concurrent == 1:
                response_bundles = self.chat_model.chat_completions(
                    messages_list=messages_list_to_generate,
                    num_retries=self.num_retries,
                    output_format="bundle",
                    validation_callback=self.qa_validation_callback,
                    callback=bundle_to_pp_callback,
                    description=description,
                )
            else:
                response_bundles = sidethread_event_loop_async_runner(
                    async_function=self.chat_model.async_chat_completions(
                        messages_list=messages_list_to_generate,
                        num_concurrent=self.num_concurrent,
                        num_retries=self.num_retries,
                        output_format="bundle",
                        validation_callback=self.qa_validation_callback,
                        callback=bundle_to_pp_callback,
                        description=description,
                    )
                )
            # Get PerioperativePredictions from LLM generations
            generated_pp_list = [b.callback_response for b in response_bundles]
            # Replace `None` output values in case bundles with PerioperativePredictions
            for case_bundle, pp in zip(cases_needing_generation, generated_pp_list):
                case_bundle.output = pp
            # Return All PerioperativePredictions (Retrieved from DB + LLM generated)
            return case_bundles

    # Zeroshot CoT Q&A from Notes
    def generate_zeroshot_cot_qa_from_notes_for_inference_dataset(
        self, force: bool = False
    ) -> list[CaseBundle]:
        "Generate Zeroshot CoT Q&A from notes for inference dataset."
        zeroshot_cot_qa_from_notes_cb = self.generate_zeroshot_cot_qa_from_notes(
            cases=self.inference_dataset, force=force
        )
        zeroshot_cot_qa_from_notes_inputs = pd.DataFrame(
            cb.input for cb in zeroshot_cot_qa_from_notes_cb
        )
        zeroshot_cot_qa_from_notes_outputs = pd.DataFrame(
            cb.output for cb in zeroshot_cot_qa_from_notes_cb
        )
        self.inputs["zeroshot_cot_qa_from_notes"] = zeroshot_cot_qa_from_notes_inputs
        self.outputs["zeroshot_cot_qa_from_notes"] = zeroshot_cot_qa_from_notes_outputs
        # Add zeroshot CoT answer, explanation & label to dataset
        df = zeroshot_cot_qa_from_notes_outputs
        self.inference_dataset = self.inference_dataset.assign(
            Label=df.label.apply(self.format_answer),
            LabelType=df.label_type,
            ZeroshotCoTFromNotesAnswer=df.answer.apply(self.format_answer),
            ZeroshotCoTFromNotesExplanation=df.explanation,
        )
        return self.inference_dataset

    def generate_zeroshot_cot_qa_from_notes(
        self,
        cases: pd.Series | pd.DataFrame,
        force: bool = False,
        description: str = "Zeroshot CoT QA from Notes",
    ) -> list[CaseBundle]:
        """Generate Zeroshot CoT Q&A from notes context.
        If previously generated for input message + model, will instead retrieve from database.

        Args:
            cases (pd.Series | pd.DataFrame): Dataframe with each row as a case.
            force (bool, optional): Force LLM generation for all messages even if
                previously cached response exists. Defaults to False.
            description (str, optional): Description for progress bar.

        Returns:
            list[CaseBundle]: List of CaseBundle items which package input Message and output
                PerioperativePredictions objects together.
        """
        if isinstance(cases, pd.Series):
            cases = cases.to_frame()

        # Create Prompts for all cases
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        messages_list = [
            pc.zeroshot_cot_qa_with_notes_context(case=case) for _, case in cases.iterrows()
        ]
        # Check to see if prior LLM generation for message exists in database
        result_list = [
            self.retrieve_from_db(messages=messages, force=force) for messages in messages_list
        ]
        # Package Message & PerioperativePrediction together to jointly iterate
        case_bundles = [
            CaseBundle(index=idx, input=message, output=pp)
            for idx, message, pp in zip(range(len(messages_list)), messages_list, result_list)
        ]
        # Identify cases where we didn't retrieve a PerioperativePrediction from database
        cases_needing_generation = [x for x in case_bundles if x.output is None]
        messages_list_to_generate = [x.input for x in cases_needing_generation]
        if not bool(messages_list_to_generate):
            return case_bundles
        else:
            # Unpack Response Bundles into PerioperativePrediction objects
            def bundle_to_pp_callback(
                messages: Message, response: ChatCompletionResponseType
            ) -> PerioperativePrediction:
                "Unpack response bundle, format as PerioperativePrediction, save in Database."
                # Unpack Message
                msgs = messages.messages
                system_message = [m for m in msgs if m["role"] == "system"][0]["content"]
                user_message = [m for m in msgs if m["role"] == "user"][0]["content"]
                # Unpack Metadata from Message
                metadata = messages.metadata
                proc_id = metadata["case"]["ProcID"]
                label = metadata["label"]
                label_type = metadata["label_type"]
                fewshot_cases = metadata["fewshot_cases"]
                num_fewshot = len(fewshot_cases) if fewshot_cases is not None else None
                pp_kwargs = {
                    "experiment_name": self.experiment_name,
                    "task_name": self.task,
                    "task_strategy": "cot_qa",
                    "shot_kind": "zeroshot",
                    "num_fewshot": num_fewshot,
                    "note_kind": self.note_kind,
                    "note_summary": False,
                    "dataset_name": self.inference_dataset_name,
                    "model_name": self.chat_model.model,
                    "proc_id": proc_id,
                    "label": label,
                    "label_type": label_type,
                    "metadata_dict": metadata,
                    "system_message": system_message,
                    "user_message": user_message,
                }
                bundle = response
                if bundle is None or bundle.num_retries == 0:
                    # LLM Generation/Validation failed across all retries.
                    # Return PerioperativePrediction with no answers
                    # and do not insert into database.  Raw response_dict still returned.
                    if bundle is not None:
                        pp_kwargs |= {"response_dict": bundle.response_dict}
                    pp = PerioperativePrediction(**pp_kwargs)
                    return pp
                else:
                    # Unpack Response
                    json_response = json.loads(bundle.response_message)
                    answer = json_response[ANSWER_KEY]
                    explanation = json_response[EXPLANATION_COT_KEY]
                    # Format as PerioperativePrediction
                    pp = PerioperativePrediction(
                        **pp_kwargs,
                        summary=None,
                        answer=answer,
                        explanation=explanation,
                        response_message=bundle.response_message,
                        response_model_name=bundle.model,
                        response_id=bundle.id,
                        system_fingerprint=bundle.system_fingerprint,
                        created_time=bundle.created_time,
                        finish_reason=bundle.finish_reason,
                        response_dict=bundle.response_dict,
                    )
                    # Insert into Database
                    self.db.insert(pp, dry_run=False, expire_on_commit=False)
                    logger.debug(f"Generated and inserted into database: {pp}")
                    return pp

            # Generate LLM Output
            if self.num_concurrent == 1:
                response_bundles = self.chat_model.chat_completions(
                    messages_list=messages_list_to_generate,
                    num_retries=self.num_retries,
                    output_format="bundle",
                    validation_callback=self.qa_validation_callback,
                    callback=bundle_to_pp_callback,
                    description=description,
                )
            else:
                response_bundles = sidethread_event_loop_async_runner(
                    async_function=self.chat_model.async_chat_completions(
                        messages_list=messages_list_to_generate,
                        num_concurrent=self.num_concurrent,
                        num_retries=self.num_retries,
                        output_format="bundle",
                        validation_callback=self.qa_validation_callback,
                        callback=bundle_to_pp_callback,
                        description=description,
                    )
                )
            # Get PerioperativePredictions from LLM generations
            generated_pp_list = [b.callback_response for b in response_bundles]
            # Replace `None` output values in case bundles with PerioperativePredictions
            for case_bundle, pp in zip(cases_needing_generation, generated_pp_list):
                case_bundle.output = pp
            # Return All PerioperativePredictions (Retrieved from DB + LLM generated)
            return case_bundles

    # Zeroshot CoT Q&A from Notes Summary
    def generate_zeroshot_cot_qa_from_notes_summary_for_inference_dataset(
        self, force: bool = False
    ) -> list[CaseBundle]:
        "Generate Zeroshot CoT Q&A from notes summary for inference dataset."
        zeroshot_cot_qa_from_notes_summary_cb = self.generate_zeroshot_cot_qa_from_notes_summary(
            cases=self.inference_dataset, force=force
        )
        zeroshot_cot_qa_from_notes_summary_inputs = pd.DataFrame(
            cb.input for cb in zeroshot_cot_qa_from_notes_summary_cb
        )
        zeroshot_cot_qa_from_notes_summary_outputs = pd.DataFrame(
            cb.output for cb in zeroshot_cot_qa_from_notes_summary_cb
        )
        self.inputs[
            "zeroshot_cot_qa_from_notes_summary"
        ] = zeroshot_cot_qa_from_notes_summary_inputs
        self.outputs[
            "zeroshot_cot_qa_from_notes_summary"
        ] = zeroshot_cot_qa_from_notes_summary_outputs
        # Add zeroshot CoT answer, explanation & label to dataset
        df = zeroshot_cot_qa_from_notes_summary_outputs
        self.inference_dataset = self.inference_dataset.assign(
            Label=df.label.apply(self.format_answer),
            LabelType=df.label_type,
            ZeroshotCoTFromNotesSummaryAnswer=df.answer.apply(self.format_answer),
            ZeroshotCoTFromNotesSummaryExplanation=df.explanation,
        )
        return self.inference_dataset

    def generate_zeroshot_cot_qa_from_notes_summary(
        self,
        cases: pd.Series | pd.DataFrame,
        force: bool = False,
        description: str = "Zeroshot CoT QA from Summaries",
    ) -> list[CaseBundle]:
        """Generate Zeroshot CoT Q&A from notes summary context.
        If previously generated for input message + model, will instead retrieve from database.

        Args:
            cases (pd.Series | pd.DataFrame): Dataframe with each row as a case.
            force (bool, optional): Force LLM generation for all messages even if
                previously cached response exists. Defaults to False.
            description (str, optional): Description for progress bar.

        Returns:
            list[CaseBundle]: List of CaseBundle items which package input Message and output
                PerioperativePredictions objects together.
        """
        if isinstance(cases, pd.Series):
            cases = cases.to_frame()
        if "Summary" not in cases.columns:
            raise ValueError("Argument `cases` must contain column `Summary`.")

        # Create Prompts for all cases
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        messages_list = [
            pc.zeroshot_cot_qa_with_summary_context(case=case) for _, case in cases.iterrows()
        ]
        # Check to see if prior LLM generation for message exists in database
        result_list = [
            self.retrieve_from_db(messages=messages, force=force) for messages in messages_list
        ]
        # Package Message & PerioperativePrediction together to jointly iterate
        case_bundles = [
            CaseBundle(index=idx, input=message, output=pp)
            for idx, message, pp in zip(range(len(messages_list)), messages_list, result_list)
        ]
        # Identify cases where we didn't retrieve a PerioperativePrediction from database
        cases_needing_generation = [x for x in case_bundles if x.output is None]
        messages_list_to_generate = [x.input for x in cases_needing_generation]
        if not bool(messages_list_to_generate):
            return case_bundles
        else:
            # Unpack Response Bundles into PerioperativePrediction objects
            def bundle_to_pp_callback(
                messages: Message, response: ChatCompletionResponseType
            ) -> PerioperativePrediction:
                "Unpack response bundle, format as PerioperativePrediction, save in Database."
                # Unpack Message
                msgs = messages.messages
                system_message = [m for m in msgs if m["role"] == "system"][0]["content"]
                user_message = [m for m in msgs if m["role"] == "user"][0]["content"]
                # Unpack Metadata from Message
                metadata = messages.metadata
                proc_id = metadata["case"]["ProcID"]
                label = metadata["label"]
                label_type = metadata["label_type"]
                fewshot_cases = metadata["fewshot_cases"]
                num_fewshot = len(fewshot_cases) if fewshot_cases is not None else None
                pp_kwargs = {
                    "experiment_name": self.experiment_name,
                    "task_name": self.task,
                    "task_strategy": "cot_qa",
                    "shot_kind": "zeroshot",
                    "num_fewshot": num_fewshot,
                    "note_kind": self.note_kind,
                    "note_summary": True,
                    "dataset_name": self.inference_dataset_name,
                    "model_name": self.chat_model.model,
                    "proc_id": proc_id,
                    "label": label,
                    "label_type": label_type,
                    "metadata_dict": metadata,
                    "system_message": system_message,
                    "user_message": user_message,
                }
                bundle = response
                if bundle is None or bundle.num_retries == 0:
                    # LLM Generation/Validation failed across all retries.
                    # Return PerioperativePrediction with no answers
                    # and do not insert into database.  Raw response_dict still returned.
                    if bundle is not None:
                        pp_kwargs |= {"response_dict": bundle.response_dict}
                    pp = PerioperativePrediction(**pp_kwargs)
                    return pp
                else:
                    # Unpack Response
                    json_response = json.loads(bundle.response_message)
                    answer = json_response[ANSWER_KEY]
                    explanation = json_response[EXPLANATION_COT_KEY]
                    # Format as PerioperativePrediction
                    pp = PerioperativePrediction(
                        **pp_kwargs,
                        summary=None,
                        answer=answer,
                        explanation=explanation,
                        response_message=bundle.response_message,
                        response_model_name=bundle.model,
                        response_id=bundle.id,
                        system_fingerprint=bundle.system_fingerprint,
                        created_time=bundle.created_time,
                        finish_reason=bundle.finish_reason,
                        response_dict=bundle.response_dict,
                    )
                    # Insert into Database
                    self.db.insert(pp, dry_run=False, expire_on_commit=False)
                    logger.debug(f"Generated and inserted into database: {pp}")
                    return pp

            # Generate LLM Output
            if self.num_concurrent == 1:
                response_bundles = self.chat_model.chat_completions(
                    messages_list=messages_list_to_generate,
                    num_retries=self.num_retries,
                    output_format="bundle",
                    validation_callback=self.qa_validation_callback,
                    callback=bundle_to_pp_callback,
                    description=description,
                )
            else:
                response_bundles = sidethread_event_loop_async_runner(
                    async_function=self.chat_model.async_chat_completions(
                        messages_list=messages_list_to_generate,
                        num_concurrent=self.num_concurrent,
                        num_retries=self.num_retries,
                        output_format="bundle",
                        validation_callback=self.qa_validation_callback,
                        callback=bundle_to_pp_callback,
                        description=description,
                    )
                )
            # Get PerioperativePredictions from LLM generations
            generated_pp_list = [b.callback_response for b in response_bundles]
            # Replace `None` output values in case bundles with PerioperativePredictions
            for case_bundle, pp in zip(cases_needing_generation, generated_pp_list):
                case_bundle.output = pp
            # Return All PerioperativePredictions (Retrieved from DB + LLM generated)
            return case_bundles

    # Fewshot Q&A from Notes Summary
    def generate_fewshot_qa_from_notes_summary_for_inference_dataset(
        self,
        num_fewshot: int = 5,
        force: bool = False,
        seed: int = 42,
        dynamic_seed: bool = True,
        balance_fewshot_classes: bool = True,
        num_bins: int = 4,
    ) -> list[CaseBundle]:
        "Generate Fewshot Q&A from note summary for inference dataset."
        fewshot_qa_from_notes_summary_cb = self.generate_fewshot_qa_from_notes_summary(
            cases=self.inference_dataset,
            fewshot_cases=self.fewshot_dataset,
            num_fewshot=num_fewshot,
            force=force,
            seed=seed,
            dynamic_seed=dynamic_seed,
            balance_fewshot_classes=balance_fewshot_classes,
            num_bins=num_bins,
        )
        fewshot_qa_from_notes_summary_inputs = pd.DataFrame(
            cb.input for cb in fewshot_qa_from_notes_summary_cb
        )
        fewshot_qa_from_notes_summary_outputs = pd.DataFrame(
            cb.output for cb in fewshot_qa_from_notes_summary_cb
        )
        self.inputs["fewshot_qa_from_notes_summary"] = fewshot_qa_from_notes_summary_inputs
        self.outputs["fewshot_qa_from_notes_summary"] = fewshot_qa_from_notes_summary_outputs
        # Add fewshot answer, explanation & label to dataset
        df = fewshot_qa_from_notes_summary_outputs
        self.inference_dataset = self.inference_dataset.assign(
            **{
                "Label": df.label.apply(self.format_answer),
                "LabelType": df.label_type,
                f"{num_fewshot}shotFromNotesSummaryAnswer": df.answer.apply(self.format_answer),
                f"{num_fewshot}shotFromNotesSummaryExplanation": df.explanation,
            }
        )
        return self.inference_dataset

    def generate_fewshot_qa_from_notes_summary(
        self,
        cases: pd.Series | pd.DataFrame,
        fewshot_cases: pd.DataFrame,
        num_fewshot: int = 5,
        seed: int = 42,
        dynamic_seed: bool = True,
        balance_fewshot_classes: bool = True,
        num_bins: int = 4,
        force: bool = False,
        description: str | None = None,
    ) -> list[CaseBundle]:
        """Generate Fewshot Q&A from notes summary context.
        If previously generated for input message + model, will instead retrieve from database.

        Args:
            cases (pd.Series | pd.DataFrame): Dataframe with each row as a case.
            fewshot_cases (pd.DataFrame): Dataframe with each row as a case. Fewshot examples
                are drawn from this dataframe.  This dataframe must contain columns `Summary`
                and `Rationale`.
            num_fewshot (int, optional): Number of fewshot examples to insert into prompt.
                Defaults to 5.
            seed (int, optional): Seed used for shuffling and selecting fewshot examples.
                Defaults to 42.
            dynamic_seed (bool, optional): If True, seed for each case will be different and
                set to `seed` + `case.ProcID`. If False, seed for each case will be the same
                and set to `seed`.  The effect of setting this argument True is that different
                fewshot examples will be used for each case's prompt; otherwise, the same
                fewshot examples will be used across all cases' prompts. Defaults to True.
            balance_fewshot_classes (bool, optional): If True, will balance the representation
                of labels in the fewshot examples. If the label is boolean or categorical,
                this is achieved by downsampling examples with common labels and upsampling
                rare labels without replacement. If the label is numeric or continuous, the
                fewshot dataset is binned into groups and then the same downsampling/upsampling
                procedure is performed on group membership. Defaults to True.
            num_bins (bool, optional): This value is ignored if the label is boolean or
                categorical. If the label is numeric or continuous, this value controls the
                number of groups to bin the fewshot dataset when balancing fewshot classes.
                This argument has no effect if `balance_fewshot_classes=False`.
            force (bool, optional): Force LLM generation for all messages even if
                previously cached response exists. Defaults to False.
            description (str, optional): Description for progress bar.

        Returns:
            list[CaseBundle]: List of CaseBundle items which package input Message and output
                PerioperativePredictions objects together.
        """
        if isinstance(cases, pd.Series):
            cases = cases.to_frame()
        if "Summary" not in cases.columns:
            raise ValueError("Argument `cases` must contain column `Summary`.")
        if "Summary" not in fewshot_cases.columns:
            raise ValueError("Argument `fewshot_cases` must contain column `Summary`.")
        if description is None:
            description = (
                "Fewshot QA from Summaries"
                if num_fewshot is None
                else f"{num_fewshot}-shot QA from Summaries"
            )

        # Create Prompts for all cases
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        messages_list = [
            pc.fewshot_qa_with_summary_context(
                case=case,
                fewshot_cases=fewshot_cases,
                num_fewshot=num_fewshot,
                seed=seed,
                dynamic_seed=dynamic_seed,
                balance_fewshot_classes=balance_fewshot_classes,
                num_bins=num_bins,
            )
            for _, case in cases.iterrows()
        ]
        # Check to see if prior LLM generation for message exists in database
        result_list = [
            self.retrieve_from_db(messages=messages, force=force) for messages in messages_list
        ]
        # Package Message & PerioperativePrediction together to jointly iterate
        case_bundles = [
            CaseBundle(index=idx, input=message, output=pp)
            for idx, message, pp in zip(range(len(messages_list)), messages_list, result_list)
        ]
        # Identify cases where we didn't retrieve a PerioperativePrediction from database
        cases_needing_generation = [x for x in case_bundles if x.output is None]
        messages_list_to_generate = [x.input for x in cases_needing_generation]
        if not bool(messages_list_to_generate):
            return case_bundles
        else:
            # Unpack Response Bundles into PerioperativePrediction objects
            def bundle_to_pp_callback(
                messages: Message, response: ChatCompletionResponseType
            ) -> PerioperativePrediction:
                "Unpack response bundle, format as PerioperativePrediction, save in Database."
                # Unpack Message
                msgs = messages.messages
                system_message = [m for m in msgs if m["role"] == "system"][0]["content"]
                user_message = [m for m in msgs if m["role"] == "user"][0]["content"]
                # Unpack Metadata from Message
                metadata = messages.metadata
                proc_id = metadata["case"]["ProcID"]
                label = metadata["label"]
                label_type = metadata["label_type"]
                fewshot_cases = metadata["fewshot_cases"]
                num_fewshot = len(fewshot_cases) if fewshot_cases is not None else None
                pp_kwargs = {
                    "experiment_name": self.experiment_name,
                    "task_name": self.task,
                    "task_strategy": "qa",
                    "shot_kind": "fewshot",
                    "num_fewshot": num_fewshot,
                    "note_kind": self.note_kind,
                    "note_summary": True,
                    "dataset_name": self.inference_dataset_name,
                    "model_name": self.chat_model.model,
                    "proc_id": proc_id,
                    "label": label,
                    "label_type": label_type,
                    "metadata_dict": metadata,
                    "system_message": system_message,
                    "user_message": user_message,
                }
                bundle = response
                if bundle is None or bundle.num_retries == 0:
                    # LLM Generation/Validation failed across all retries.
                    # Return PerioperativePrediction with no answers
                    # and do not insert into database.  Raw response_dict still returned.
                    if bundle is not None:
                        pp_kwargs |= {"response_dict": bundle.response_dict}
                    pp = PerioperativePrediction(**pp_kwargs)
                    return pp
                else:
                    # Unpack Response
                    json_response = json.loads(bundle.response_message)
                    answer = json_response[ANSWER_KEY]
                    explanation = json_response[EXPLANATION_KEY]
                    # Format as PerioperativePrediction
                    pp = PerioperativePrediction(
                        **pp_kwargs,
                        summary=None,
                        answer=answer,
                        explanation=explanation,
                        response_message=bundle.response_message,
                        response_model_name=bundle.model,
                        response_id=bundle.id,
                        system_fingerprint=bundle.system_fingerprint,
                        created_time=bundle.created_time,
                        finish_reason=bundle.finish_reason,
                        response_dict=bundle.response_dict,
                    )
                    # Insert into Database
                    self.db.insert(pp, dry_run=False, expire_on_commit=False)
                    logger.debug(f"Generated and inserted into database: {pp}")
                    return pp

            # Generate LLM Output
            if self.num_concurrent == 1:
                response_bundles = self.chat_model.chat_completions(
                    messages_list=messages_list_to_generate,
                    num_retries=self.num_retries,
                    output_format="bundle",
                    validation_callback=self.qa_validation_callback,
                    callback=bundle_to_pp_callback,
                    description=description,
                )
            else:
                response_bundles = sidethread_event_loop_async_runner(
                    async_function=self.chat_model.async_chat_completions(
                        messages_list=messages_list_to_generate,
                        num_concurrent=self.num_concurrent,
                        num_retries=self.num_retries,
                        output_format="bundle",
                        validation_callback=self.qa_validation_callback,
                        callback=bundle_to_pp_callback,
                        description=description,
                    )
                )
            # Get PerioperativePredictions from LLM generations
            generated_pp_list = [b.callback_response for b in response_bundles]
            # Replace `None` output values in case bundles with PerioperativePredictions
            for case_bundle, pp in zip(cases_needing_generation, generated_pp_list):
                case_bundle.output = pp
            # Return All PerioperativePredictions (Retrieved from DB + LLM generated)
            return case_bundles

    # Fewshot Q&A from Notes Summary
    def generate_fewshot_cot_qa_from_notes_summary_for_inference_dataset(
        self,
        num_fewshot: int = 5,
        force: bool = False,
        seed: int = 42,
        dynamic_seed: bool = True,
        balance_fewshot_classes: bool = True,
        num_bins: int = 4,
    ) -> list[CaseBundle]:
        "Generate Fewshot CoT Q&A from note summary for inference dataset."
        fewshot_cot_qa_from_notes_summary_cb = self.generate_fewshot_cot_qa_from_notes_summary(
            cases=self.inference_dataset,
            fewshot_cases=self.fewshot_dataset,
            num_fewshot=num_fewshot,
            force=force,
            seed=seed,
            dynamic_seed=dynamic_seed,
            balance_fewshot_classes=balance_fewshot_classes,
            num_bins=num_bins,
        )
        fewshot_cot_qa_from_notes_summary_inputs = pd.DataFrame(
            cb.input for cb in fewshot_cot_qa_from_notes_summary_cb
        )
        fewshot_cot_qa_from_notes_summary_outputs = pd.DataFrame(
            cb.output for cb in fewshot_cot_qa_from_notes_summary_cb
        )
        self.inputs["fewshot_cot_qa_from_notes_summary"] = fewshot_cot_qa_from_notes_summary_inputs
        self.outputs[
            "fewshot_cot_qa_from_notes_summary"
        ] = fewshot_cot_qa_from_notes_summary_outputs
        # Add fewshot answer, explanation & label to dataset
        df = fewshot_cot_qa_from_notes_summary_outputs
        self.inference_dataset = self.inference_dataset.assign(
            **{
                "Label": df.label.apply(self.format_answer),
                "LabelType": df.label_type,
                f"{num_fewshot}shotCoTFromNotesSummaryAnswer": df.answer.apply(self.format_answer),
                f"{num_fewshot}shotCoTFromNotesSummaryExplanation": df.explanation,
            }
        )
        return self.inference_dataset

    def generate_fewshot_cot_qa_from_notes_summary(
        self,
        cases: pd.Series | pd.DataFrame,
        fewshot_cases: pd.DataFrame,
        num_fewshot: int = 5,
        seed: int = 42,
        dynamic_seed: bool = True,
        balance_fewshot_classes: bool = True,
        num_bins: int = 4,
        force: bool = False,
        description: str | None = None,
    ) -> list[CaseBundle]:
        """Generate Fewshot CoT Q&A from notes summary context.
        If previously generated for input message + model, will instead retrieve from database.

        Args:
            cases (pd.Series | pd.DataFrame): Dataframe with each row as a case.
            fewshot_cases (pd.DataFrame): Dataframe with each row as a case. Fewshot examples
                are drawn from this dataframe.  This dataframe must contain columns `Summary`
                and `Rationale`.
            num_fewshot (int, optional): Number of fewshot examples to insert into prompt.
                Defaults to 5.
            seed (int, optional): Seed used for shuffling and selecting fewshot examples.
                Defaults to 42.
            dynamic_seed (bool, optional): If True, seed for each case will be different and
                set to `seed` + `case.ProcID`. If False, seed for each case will be the same
                and set to `seed`.  The effect of setting this argument True is that different
                fewshot examples will be used for each case's prompt; otherwise, the same
                fewshot examples will be used across all cases' prompts. Defaults to True.
            balance_fewshot_classes (bool, optional): If True, will balance the representation
                of labels in the fewshot examples. If the label is boolean or categorical,
                this is achieved by downsampling examples with common labels and upsampling
                rare labels without replacement. If the label is numeric or continuous, the
                fewshot dataset is binned into groups and then the same downsampling/upsampling
                procedure is performed on group membership. Defaults to True.
            num_bins (bool, optional): This value is ignored if the label is boolean or
                categorical. If the label is numeric or continuous, this value controls the
                number of groups to bin the fewshot dataset when balancing fewshot classes.
                This argument has no effect if `balance_fewshot_classes=False`.
            force (bool, optional): Force LLM generation for all messages even if
                previously cached response exists. Defaults to False.
            description (str, optional): Description for progress bar.

        Returns:
            list[CaseBundle]: List of CaseBundle items which package input Message and output
                PerioperativePredictions objects together.
        """
        if isinstance(cases, pd.Series):
            cases = cases.to_frame()
        if "Summary" not in cases.columns:
            raise ValueError("Argument `cases` must contain column `Summary`.")
        if "Summary" not in fewshot_cases.columns:
            raise ValueError("Argument `fewshot_cases` must contain column `Summary`.")
        if "Rationale" not in fewshot_cases.columns:
            raise ValueError("Argument `fewshot_cases` must contain column `Rationale`.")
        if description is None:
            description = (
                "Fewshot CoT QA from Summaries"
                if num_fewshot is None
                else f"{num_fewshot}-shot CoT QA from Summaries"
            )

        # Create Prompts for all cases
        pc = PromptComposer(task=self.task, token_limit=self.token_limit)
        messages_list = [
            pc.fewshot_cot_qa_with_summary_context(
                case=case,
                fewshot_cases=fewshot_cases,
                num_fewshot=num_fewshot,
                seed=seed,
                dynamic_seed=dynamic_seed,
                balance_fewshot_classes=balance_fewshot_classes,
                num_bins=num_bins,
            )
            for _, case in cases.iterrows()
        ]
        # Check to see if prior LLM generation for message exists in database
        result_list = [
            self.retrieve_from_db(messages=messages, force=force) for messages in messages_list
        ]
        # Package Message & PerioperativePrediction together to jointly iterate
        case_bundles = [
            CaseBundle(index=idx, input=message, output=pp)
            for idx, message, pp in zip(range(len(messages_list)), messages_list, result_list)
        ]
        # Identify cases where we didn't retrieve a PerioperativePrediction from database
        cases_needing_generation = [x for x in case_bundles if x.output is None]
        messages_list_to_generate = [x.input for x in cases_needing_generation]
        if not bool(messages_list_to_generate):
            return case_bundles
        else:
            # Unpack Response Bundles into PerioperativePrediction objects
            def bundle_to_pp_callback(
                messages: Message, response: ChatCompletionResponseType
            ) -> PerioperativePrediction:
                "Unpack response bundle, format as PerioperativePrediction, save in Database."
                # Unpack Message
                msgs = messages.messages
                system_message = [m for m in msgs if m["role"] == "system"][0]["content"]
                user_message = [m for m in msgs if m["role"] == "user"][0]["content"]
                # Unpack Metadata from Message
                metadata = messages.metadata
                proc_id = metadata["case"]["ProcID"]
                label = metadata["label"]
                label_type = metadata["label_type"]
                fewshot_cases = metadata["fewshot_cases"]
                num_fewshot = len(fewshot_cases) if fewshot_cases is not None else None
                pp_kwargs = {
                    "experiment_name": self.experiment_name,
                    "task_name": self.task,
                    "task_strategy": "cot_qa",
                    "shot_kind": "fewshot",
                    "num_fewshot": num_fewshot,
                    "note_kind": self.note_kind,
                    "note_summary": True,
                    "dataset_name": self.inference_dataset_name,
                    "model_name": self.chat_model.model,
                    "proc_id": proc_id,
                    "label": label,
                    "label_type": label_type,
                    "metadata_dict": metadata,
                    "system_message": system_message,
                    "user_message": user_message,
                }
                bundle = response
                if bundle is None or bundle.num_retries == 0:
                    # LLM Generation/Validation failed across all retries.
                    # Return PerioperativePrediction with no answers
                    # and do not insert into database.  Raw response_dict still returned.
                    if bundle is not None:
                        pp_kwargs |= {"response_dict": bundle.response_dict}
                    pp = PerioperativePrediction(**pp_kwargs)
                    return pp
                else:
                    # Unpack Response
                    json_response = json.loads(bundle.response_message)
                    answer = json_response[ANSWER_KEY]
                    explanation = json_response[EXPLANATION_COT_KEY]
                    # Format as PerioperativePrediction
                    pp = PerioperativePrediction(
                        **pp_kwargs,
                        summary=None,
                        answer=answer,
                        explanation=explanation,
                        response_message=bundle.response_message,
                        response_model_name=bundle.model,
                        response_id=bundle.id,
                        system_fingerprint=bundle.system_fingerprint,
                        created_time=bundle.created_time,
                        finish_reason=bundle.finish_reason,
                        response_dict=bundle.response_dict,
                    )
                    # Insert into Database
                    self.db.insert(pp, dry_run=False, expire_on_commit=False)
                    logger.debug(f"Generated and inserted into database: {pp}")
                    return pp

            # Generate LLM Output
            if self.num_concurrent == 1:
                response_bundles = self.chat_model.chat_completions(
                    messages_list=messages_list_to_generate,
                    num_retries=self.num_retries,
                    output_format="bundle",
                    validation_callback=self.qa_validation_callback,
                    callback=bundle_to_pp_callback,
                    description=description,
                )
            else:
                response_bundles = sidethread_event_loop_async_runner(
                    async_function=self.chat_model.async_chat_completions(
                        messages_list=messages_list_to_generate,
                        num_concurrent=self.num_concurrent,
                        num_retries=self.num_retries,
                        output_format="bundle",
                        validation_callback=self.qa_validation_callback,
                        callback=bundle_to_pp_callback,
                        description=description,
                    )
                )
            # Get PerioperativePredictions from LLM generations
            generated_pp_list = [b.callback_response for b in response_bundles]
            # Replace `None` output values in case bundles with PerioperativePredictions
            for case_bundle, pp in zip(cases_needing_generation, generated_pp_list):
                case_bundle.output = pp
            # Return All PerioperativePredictions (Retrieved from DB + LLM generated)
            return case_bundles
