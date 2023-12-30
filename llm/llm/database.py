from typing import Any, Optional

import config
import pandas as pd
import sqlalchemy
from llm_utils import create_hash
from sqlalchemy import JSON, Boolean, Integer, String, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, Session, mapped_column

from .format import format_asa, resolve_boolean, resolve_numeric


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}
    pass


class PerioperativePrediction(MappedAsDataclass, Base, kw_only=True):
    """This is a dataclass and SQL Alchemy ORM which stores
    experiment settings, input and output messages to LLM and defines a ORM schema using
    SQL Alchemy to map each object to a database table row. This"""

    __tablename__ = "PerioperativePrediction"

    """id: Primary Key (auto-assigned by Database, not part of dataclass init)"""
    id: Mapped[int] = mapped_column(Integer, init=False, primary_key=True)

    # Experiment, Task, Prompt Settings
    """experiment_name: user-defined name for experiment"""
    experiment_name: Mapped[str] = mapped_column(String, default="default")
    """task_name: {"asa", "phase1_duration", "hospital_duration", "hospital_admission", 
    "icu_duration", "icu_admission", "unplanned_admit", "hospital_mortality"}"""
    task_name: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """task_strategy: {"summarize", "cot_generation", "qa", "cot_qa"}"""
    task_strategy: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """shot_kind: {"zeroshot", "fewshot"}"""
    shot_kind: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """num_fewshot: int | None"""
    num_fewshot: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)
    """note_kind: {"last10", "preanes", "last10_summary", "preanes_summary"}"""
    note_kind: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """note_summary: {True, False}"""
    note_summary: Mapped[Optional[bool]] = mapped_column(Boolean, default=None, nullable=True)
    """dataset_name: {task_name}-{note_kind}-{inference/fewshot}"""
    dataset_name: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)

    """proc_id: unique identifier for case"""
    proc_id: Mapped[int] = mapped_column(Integer)

    """label: ground-truth label from dataset"""
    label: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """label_type: data type for the ground-truth label"""
    label_type: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)

    # Parsed JSON Fields from response_message
    """summary: Generated summary response (only if summary task; otherwise `None`)"""
    summary: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """answer: Generated answer response (only if Q&A task; otherwise `None`)"""
    answer: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """explanation: Generated explanation/CoT rationale (if Q&A task or CoT generation)"""
    explanation: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)

    # Messages
    """system_message: Input system message to LLM"""
    system_message: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """user_message: Input user message to LLM"""
    user_message: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """response_message: Output assistant response message from LLM"""
    response_message: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)

    # API Call/Response Metadata
    """model_name: Deployment model name"""
    model_name: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """response_model_name: Model name from API response"""
    response_model_name: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """response_id: ID generated from API response"""
    response_id: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """system_fingerprint: System fingerprint from API response"""
    system_fingerprint: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """created_time: Unix epoch timestamp from API response"""
    created_time: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)
    """finish_reason: From API response, either "stopped" (normal) or "length" (hit max tokens)"""
    finish_reason: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """response_dict: The entire ChatCompletion API response object converted into dict"""
    response_dict: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, default=None, nullable=True
    )
    # Additional Dataset Metadata
    """metadata_dict: Metadata associated messages from dataset"""
    metadata_dict: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, default=None, nullable=True
    )
    # Hashes to indentify unique model + input +/- response
    """input_hash: unique hash for each model_name + input message"""
    input_hash: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    """output_hash: unique for each model_name + input message + response"""
    response_hash: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)

    def __post_init__(self) -> None:
        # Type Coercion to int (PostgreSQL cannot handle numpy.int64)
        self.proc_id = int(self.proc_id)
        # Auto generate input_hash and response_hash
        if all(x is not None for x in (self.model_name, self.system_message, self.user_message)):
            self.input_hash = self.compute_input_hash(
                self.model_name, self.system_message, self.user_message
            )
        if all(
            x is not None
            for x in (
                self.model_name,
                self.system_message,
                self.user_message,
                self.response_message,
            )
        ):
            self.response_hash = self.compute_response_hash(
                self.model_name, self.system_message, self.user_message, self.response_message
            )

    @staticmethod
    def compute_input_hash(model_name: str, system_message: str, user_message: str) -> str:
        return create_hash(
            f"{model_name}|{str(system_message)}|{str(user_message)}",
            digest_size=8,
        )

    @staticmethod
    def compute_response_hash(
        model_name: str, system_message: str, user_message: str, response_message: str
    ) -> str:
        return create_hash(
            f"{model_name}"
            f"|{str(system_message)}"
            f"|{str(user_message)}"
            f"|{str(response_message)}",
            digest_size=8,
        )

    def __repr__(self) -> str:
        return (
            f"PerioperativePrediction<id:{self.id}, "
            f"experiment:{self.experiment_name}, "
            f"proc_id:{self.proc_id}, "
            f"prompt: {self.task_name}-{self.task_strategy}-{self.shot_kind}-{self.note_kind}, "
            f"dataset:{self.dataset_name}, "
            f"model:{self.model_name}, "
            f"input_hash:{self.input_hash}, response_hash:{self.response_hash}>"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Converts dataclass to a pandas table with a single row and format the
        `answer` field into appropriate data type."""
        pp = self.copy()
        pp.answer = self.format_text(pp.answer, task_name=self.task_name)
        return pd.DataFrame(data=[pp])

    @staticmethod
    def format_text(text: str | Any, task_name: str) -> int | float | bool | None:
        "Format text based on task_name.  Used to convert string answer to correct data type."
        match task_name:
            case "asa":
                return format_asa(text)
            case (
                "phase1_duration"
                | "phase2_duration"
                | "pacu_duration"
                | "icu_duration"
                | "hospital_duration"
            ):
                return resolve_numeric(text)
            case (
                "hospital_admission" | "icu_admission" | "unplanned_admit" | "hospital_mortality"
            ):
                return resolve_boolean(text)
            case _:
                raise ValueError(f"Invalid `task_name`: {task_name}")


class Database:
    engine: sqlalchemy.engine.Engine
    metadata: sqlalchemy.MetaData
    dry_run: bool = True
    expire_on_commit: bool = False

    def __init__(self, **kwargs) -> None:
        self.create_database_engine(**kwargs)
        self.create_database_tables()

    def create_database_engine(self, echo=True, **kwargs) -> None:
        url = sqlalchemy.engine.URL.create(
            drivername=kwargs["drivername"] if "drivername" in kwargs else "postgresql",
            username=kwargs["username"] if "username" in kwargs else config.postgres_user,
            password=kwargs["password"] if "password" in kwargs else config.postgres_password,
            host=kwargs["host"] if "host" in kwargs else config.postgres_host,
            database=kwargs["database"] if "database" in kwargs else config.postgres_database,
            port=kwargs["port"] if "port" in kwargs else config.postgres_port,
        )
        self.engine = sqlalchemy.create_engine(url, echo=echo)

    def create_database_tables(self, metadata: sqlalchemy.MetaData = None) -> None:
        self.metadata = Base.metadata if metadata is None else metadata
        self.metadata.create_all(bind=self.engine)

    def drop_table(self, dry_run: bool | None = None) -> None:
        "Drops the PerioperativePrediction Database Table. This is irreversible."
        if not dry_run:
            PerioperativePrediction.__table__.drop(self.engine)

    def insert(
        self,
        perioperative_predictions: PerioperativePrediction | list[PerioperativePrediction],
        dry_run: bool | None = None,
        expire_on_commit: bool | None = None,
    ) -> None:
        "Insert entries in PerioperativePrediction Database Table."
        dry_run = self.dry_run if dry_run is None else dry_run
        expire_on_commit = self.expire_on_commit if expire_on_commit is None else expire_on_commit
        if not isinstance(perioperative_predictions, list):
            perioperative_predictions = [perioperative_predictions]
        with Session(self.engine, expire_on_commit=expire_on_commit) as session:
            session.add_all(perioperative_predictions)
            if not dry_run:
                session.commit()

    def delete(
        self,
        dry_run: bool | None = None,
        expire_on_commit: bool | None = None,
    ) -> int:
        "Deletes all entries in PerioperativePrediction Database Table, returning num rows deleted"
        dry_run = self.dry_run if dry_run is None else dry_run
        expire_on_commit = self.expire_on_commit if expire_on_commit is None else expire_on_commit
        sql = delete(PerioperativePrediction)
        with Session(self.engine, expire_on_commit=expire_on_commit) as session:
            result = session.execute(sql)
            if not dry_run:
                session.commit()
        return result.rowcount

    def delete_experiment(
        self,
        experiment_name: str,
        dry_run: bool | None = None,
        expire_on_commit: bool | None = None,
    ) -> int:
        """Deletes entries in PerioperativePrediction Database Table matching
        experiment_name column, returning num rows deleted."""
        dry_run = self.dry_run if dry_run is None else dry_run
        expire_on_commit = self.expire_on_commit if expire_on_commit is None else expire_on_commit
        sql = delete(PerioperativePrediction).where(
            PerioperativePrediction.experiment_name == experiment_name
        )
        with Session(self.engine, expire_on_commit=expire_on_commit) as session:
            result = session.execute(sql)
            if not dry_run:
                session.commit()
        return result.rowcount

    def delete_where_equals(
        self, dry_run: bool | None = None, expire_on_commit: bool | None = None, **kwargs
    ) -> int:
        """Deletes entries in PerioperativePrediction Database Table matching specific
        property/column values provided in kwargs, returning num rows deleted."""
        dry_run = self.dry_run if dry_run is None else dry_run
        expire_on_commit = self.expire_on_commit if expire_on_commit is None else expire_on_commit
        sql = delete(PerioperativePrediction)
        if kwargs:
            for k, v in kwargs.items():
                sql = sql.where(PerioperativePrediction.__dict__[k] == v)
        with Session(self.engine, expire_on_commit=expire_on_commit) as session:
            result = session.execute(sql)
            if not dry_run:
                session.commit()
        return result.rowcount

    def select(
        self,
        dry_run: bool | None = None,
        expire_on_commit: bool | None = None,
    ) -> list[PerioperativePrediction]:
        "Select all entries in PerioperativePrediction Database Table, returning selected rows."
        dry_run = self.dry_run if dry_run is None else dry_run
        expire_on_commit = self.expire_on_commit if expire_on_commit is None else expire_on_commit
        sql = select(PerioperativePrediction)
        with Session(self.engine, expire_on_commit=expire_on_commit) as session:
            result = session.execute(sql)
            result_list = [row[0] for row in result]
            if not dry_run:
                session.commit()
        return result_list

    def select_experiment(
        self,
        experiment_name: str,
        dry_run: bool | None = None,
        expire_on_commit: bool | None = None,
    ) -> list[PerioperativePrediction]:
        "Select all entries in PerioperativePrediction Database Table, returning selected rows."
        dry_run = self.dry_run if dry_run is None else dry_run
        sql = select(PerioperativePrediction).where(
            PerioperativePrediction.experiment_name == experiment_name
        )
        with Session(self.engine, expire_on_commit=expire_on_commit) as session:
            result = session.execute(sql)
            result_list = [row[0] for row in result]
            if not dry_run:
                session.commit()
        return result_list

    def select_where_equals(
        self, dry_run: bool | None = None, expire_on_commit: bool | None = None, **kwargs
    ) -> list[PerioperativePrediction]:
        """Select entries in PerioperativePrediction Database Table matching specific
        property/column values provided in kwargs, returning selected rows."""
        dry_run = self.dry_run if dry_run is None else dry_run
        expire_on_commit = self.expire_on_commit if expire_on_commit is None else expire_on_commit
        sql = select(PerioperativePrediction)
        if kwargs:
            for k, v in kwargs.items():
                sql = sql.where(PerioperativePrediction.__dict__[k] == v)
        with Session(self.engine, expire_on_commit=expire_on_commit) as session:
            result = session.execute(sql)
            result_list = [row[0] for row in result]
            if not dry_run:
                session.commit()
        return result_list

    def select_distinct_experiment_names(
        self,
        dry_run: bool | None = None,
        expire_on_commit: bool | None = None,
    ) -> list[str]:
        dry_run = self.dry_run if dry_run is None else dry_run
        expire_on_commit = self.expire_on_commit if expire_on_commit is None else expire_on_commit
        sql = select(PerioperativePrediction.experiment_name).distinct()
        with Session(self.engine, expire_on_commit=expire_on_commit) as session:
            result = session.execute(sql)
            experiment_names = [row[0] for row in result]
            if not dry_run:
                session.commit()
        return experiment_names
