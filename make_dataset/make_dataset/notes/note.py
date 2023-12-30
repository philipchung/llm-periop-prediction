import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from llm_utils import (
    DataPaths,
    function_with_cache,
    num_tokens_from_string,
    parallel_process,
    read_pandas,
    string_to_datetime_fmt2,
    truncate_string_tail,
)
from tqdm.auto import tqdm

from make_dataset.pipeline import PipelineMixin, PipelineStep

logger = logging.getLogger(__file__)


@dataclass(kw_only=True)
class Note(PipelineMixin):
    """Data object to clean and process notes.
    This is a base class that provides basic processing.
    Inherit from this class to customize processing and transformations.

    `df`: Dataframe from most recent step of pipeline.
    `raw_df`: The unmodified dataframe. Index is NoteID.
    `processed_df`: Cleaned dataframe. Index is NoteID.
    `collated_df`: All notes from a ProcID compressed into lists in single row. Index is ProcID.
    `concatenated_df`: Collated NoteText list concatenated into a single string. Index is ProcID.
        The concatenation step truncates NoteText string based on `max_token_length`, which
        may produce multiple cached `data_path` files depending on the value of this argument.
    """

    paths: DataPaths
    note_type: str = "notes"
    execute_pipeline: bool = True
    force: bool = False
    # Fields Populated After Running Pipeline
    df: pd.DataFrame | None = None
    raw_df: pd.DataFrame | None = None
    processed_df: pd.DataFrame | None = None
    collated_df: pd.DataFrame | None = None
    concatenated_df: pd.DataFrame | None = None
    # Settings for Note Concatenation
    concatenate_notes: bool = False
    header: str | None = None
    delim_token: str = "\n\n"
    encoding_name: str = "cl100k_base"
    max_token_length: int | None = None

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
                data_path=self.paths[f"{self.note_type}_raw"],
                method=self._load_data,
            ),
            PipelineStep(
                num=1,
                name="clean_data",
                data_path=self.paths.register(
                    self.paths.interim / f"{self.note_type}_cleaned.feather"
                ),
                method=self._clean_data,
            ),
            PipelineStep(
                num=2,
                name="process_data",
                data_path=self.paths.register(
                    self.paths.interim / f"{self.note_type}_processed.feather"
                ),
                method=self._process_data,
            ),
            PipelineStep(
                num=3,
                name="collate_notes",
                data_path=self.paths.register(
                    self.paths.interim / f"{self.note_type}_collated.feather"
                ),
                method=self._collate_notes,
            ),
        ]
        if self.concatenate_notes:
            self.pipeline += [
                PipelineStep(
                    num=4,
                    name="concatenate_notes",
                    data_path=self.paths.register(
                        self.paths.interim
                        / (
                            f"{self.note_type}_concatenated_"
                            f"{self.encoding_name}_"
                            f"maxlen{self.max_token_length}.feather"
                        )
                    ),
                    method=self._concatenate_notes,
                )
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
        _df = read_pandas(Path(data_path)).set_index("NoteID")
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
        """Clean Notes data.

        For custom data cleaning, overload the method `clean_data_logic`.

        Args:
            df (pd.DataFrame, optional): Table of input data.
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df if (df is None or df.empty) else df,
            function=self.clean_data_logic,
            cache_path=data_path,
            set_index="NoteID",
            force=force,
            **kwargs,
        )
        self.df = _df
        return _df

    def _process_data(
        self,
        df: pd.DataFrame | None = None,
        data_path: str | Path | None = None,
        force: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Process Notes data. The default behavior is to generate token counts for
        each note.

        For custom processing, overload the method `process_data_logic`.

        Args:
            df (pd.DataFrame, optional): Table of input data.
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df if (df is None or df.empty) else df,
            function=self.process_data_logic,
            cache_path=data_path,
            set_index="NoteID",
            force=force,
            **kwargs,
        )
        self.df = _df
        self.processed_df = _df
        return _df

    def _collate_notes(
        self,
        df: pd.DataFrame | None = None,
        data_path: str | Path | None = None,
        force: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Transform Notes data.  By default this generates a 1:1 mapping between
        a procedure ProcID and text.  Multiple notes may be written relative to
        a single procedure.  The default behavior of this method is to concatenate
        each of the notes in order of the date of service of the notes.  The result
        of this transformation is a dataframe has index of ProcID rather than NoteID.

        For custom transformation, overload the method `collate_notes_logic`.

        Args:
            df (pd.DataFrame, optional): Table of input data.
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df if (df is None or df.empty) else df,
            function=self.collate_notes_logic,
            cache_path=data_path,
            set_index="ProcID",
            force=force,
            **kwargs,
        )
        self.df = _df
        self.collated_df = _df
        return _df

    def _concatenate_notes(
        self,
        df: pd.DataFrame | None = None,
        data_path: str | Path | None = None,
        force: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Concatenate Collated Notes data.  Transform Notes data step collates
        the notes into a list for each ProcID.  This method concatenates the list
        of notes into a single string text.  The dataframe has index of ProcID.

        For custom transformation, overload the method `concatenate_notes_logic`.

        Args:
            df (pd.DataFrame, optional): Table of input data.
            data_path (str | Path, optional): Path to resulting table on disk.
            force (bool, optional): If true, force re-run rather than load from disk.
        """
        _df = function_with_cache(
            input_data=self.df if (df is None or df.empty) else df,
            function=self.concatenate_notes_logic,
            cache_path=data_path,
            set_index="ProcID",
            force=force,
            header=self.header,
            delim_token=self.delim_token,
            encoding_name=self.encoding_name,
            max_token_length=self.max_token_length,
            **kwargs,
        )
        self.df = _df
        self.concatenated_df = _df
        return _df

    def clean_data_logic(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        _df = notes_df.copy()
        # Format Data Types (index is NoteID)
        _df.index = _df.index.astype(int)
        _df.ProcID = _df.ProcID.astype(int)

        # Format Dates as Pandas Timestamp
        for col in [
            "NoteServiceDate",
            "ContactEnteredDate",
        ]:
            if col in _df.columns:
                # Skip if column is already Timestamp
                if _df[col].dtype != np.dtype("datetime64[ns]"):
                    tqdm.pandas(desc=f"Format {col} DateTime")
                    _df[col] = _df[col].progress_apply(string_to_datetime_fmt2)

        # Ensure Only Signed or Addendum Notes
        if "NoteStatus" in _df.columns:
            _df = _df.loc[_df.NoteStatus.isin(["Signed", "Addendum"])]
        return _df

    def process_data_logic(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        _df = notes_df.copy()
        # Get Token Counts
        _df = self.compute_note_token_length(_df)
        return _df

    def collate_notes_logic(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        _df = notes_df.copy()
        # Compress multiple notes in table into a single row of lists
        tqdm.pandas(desc="Collating Notes")
        _df = (
            _df.reset_index()
            .groupby("ProcID", group_keys=True, as_index=False)
            .progress_apply(collate_multiple_notes_per_proc_id)
            .set_index("ProcID")
        )
        return _df

    def concatenate_notes_logic(
        self,
        notes_df: pd.DataFrame,
        header: str = None,
        delim_token: str = "\n\n",
        encoding_name: str = "cl100k_base",
        max_token_length: int | None = None,
    ) -> pd.DataFrame:
        _df = notes_df.copy()
        # Concatenate list of text to yield single text for each ProcID
        concat_fn = partial(
            concatenate_collated_notes_with_header,
            header=header,
            delim_token=delim_token,
            encoding_name=encoding_name,
            max_token_length=max_token_length,
        )
        if "Suffix" in _df.columns:
            it = ({"series": row, "suffix": row.Suffix} for idx, row in _df.iterrows())
        else:
            it = ({"series": row} for idx, row in _df.iterrows())
        output = parallel_process(
            iterable=it,
            function=concat_fn,
            use_kwargs=True,
            desc="Concatenate Notes",
        )
        _df = pd.DataFrame(output).rename_axis(index="ProcID")
        return _df

    def compute_note_token_length(
        self, notes_df: pd.DataFrame, encoding_name: str = "cl100k_base"
    ) -> pd.DataFrame:
        _df = notes_df.copy()
        fn = partial(num_tokens_from_string, encoding_name=encoding_name)
        output = parallel_process(
            iterable=_df.NoteText.tolist(),
            function=fn,
            desc="Token Length of Notes",
        )
        _df = _df.assign(NoteTextTokenLength=output)
        return _df


def collate_multiple_notes_per_proc_id(notes_per_proc_id: pd.DataFrame) -> pd.Series:
    """For cases that have multiple notes, collate those notes together, resulting
    in one Series (row) per ProcID."""
    _df = notes_per_proc_id.copy()
    cols = _df.columns.tolist()
    _df = _df.sort_values(by=["NoteServiceDate", "ContactEnteredDate"], ascending=True)
    # For each ProcID, compress all columns such that a list contains column contents
    _df = _df.unstack().reset_index(level=1, drop=True)
    _df = _df.groupby(_df.index).apply(list)
    # Unpack ProcID from list
    if "ProcID" in _df.index:
        _df.ProcID = _df.ProcID[0]
    # Rearrange in original column order
    return _df.loc[cols]


def concatenate_collated_notes_with_header(
    series: pd.Series,
    header: str = None,
    delim_token: str = "\n\n",
    encoding_name: str = "cl100k_base",
    max_token_length: int | None = None,
    prefix: str = "",
    suffix: str = "",
) -> pd.Series:
    """Given a series of lists (result of collating notes per ProcID),
    this function loops through each `NoteText` in list and concatenates the
    text with an optional header.  `NoteTextTokenLength` is recomputed for the final text,
    accounting for the concatenated `NoteText` along with the added header and whitespace.
    If `NoteText` exceeds `max_token_length` argument, then only a subset of note
    are included.  The k most recent notes are included, dropping notes with
    `NoteServiceDate` that are oldest. If k=1 and `NoteText` still exceeds
    `max_token_length`, then we will truncate the end of this note, preserving
    as much of the note while maintaing left-to-right integrity.

    Args:
        series (pd.Series): Series with each field containing a list which is the
            result of collating notes per ProcID. Internally, this function will
            expand this series into a DataFrame by exploding the lists.
        header (str, optional): Header used to prefix every note text.  Set to empty string
            "" to disable the header.  If `None` uses a default header.
        delim_token (str, optional): Delimiter token used to join notes. Defaults to "\n\n".
        encoding_name (str, optional): Tiktoken tokenizer encoding used to compute
            new note text length. Defaults to "cl100k_base".
        max_token_length (int | None): Maximum token length imposed on the output NoteText.
            If `None`, then no restriction is imposed on token length.
        prefix (str): Text string that will always be added to beginning of the output text.
            Unlike header, this is added only once, not per note.
        suffix (str): Text string that will always be added to end of the output text.
            This is added only once, not per note.

    Returns:
        pd.Series: Series with all the same fields as the original series passed into
            this function, but with NoteText concatenated into a string and NoteTextTokenLength
            recomputed to reflect the new NoteText.
    """
    # Expand series of lists into a DataFrame. Same schema as notes_df. Sort Chronological.
    _df = pd.DataFrame(series.to_dict()).sort_values(
        by=["NoteServiceDate", "ContactEnteredDate"]
    )
    proc_id = series.name
    delim_token_length = num_tokens_from_string(
        delim_token, encoding_name=encoding_name
    )
    num_notes = _df.shape[0]

    # Get Original Token Length of All Notes for ProcID without Any Processing
    original_notes_list_concat = delim_token.join(_df.NoteText)
    original_notes_token_length = num_tokens_from_string(
        original_notes_list_concat, encoding_name=encoding_name
    )

    # Format Notes with Headers, Get Token Length of Formatted Notes
    formatted_notes = []
    for row in _df.itertuples(index=False):
        if header is None:
            header = (
                f"{row.NoteName} written by {row.AuthorProviderType} at {row.NoteServiceDate}:"
                "\n"
            )
        note_text = row.NoteText
        formatted_note = f"{header}{note_text}"
        formatted_notes += [formatted_note]
    # Formatted Note Token Lengths (list for each note)
    formatted_notes_text_token_length_list = [
        num_tokens_from_string(formatted_note, encoding_name=encoding_name)
        for formatted_note in formatted_notes
    ]
    # Get Combined Length of all Formatted Notes for Example
    formatted_notes_text_token_length = num_tokens_from_string(
        delim_token.join(formatted_notes), encoding_name=encoding_name
    )
    # Prefix & Suffix Note Token Lengths
    prefix_token_length = num_tokens_from_string(prefix, encoding_name=encoding_name)
    suffix_token_length = num_tokens_from_string(suffix, encoding_name=encoding_name)
    full_formatted_token_length = (
        formatted_notes_text_token_length + prefix_token_length + suffix_token_length
    )
    # Add Derived Columns
    _df = _df.assign(
        FormattedNoteText=formatted_notes,
        FormattedNoteTextTokenLength=formatted_notes_text_token_length_list,
        Prefix=prefix,
        PrefixTokenLength=prefix_token_length,
        Suffix=suffix,
        SuffixtokenLengths=suffix_token_length,
    )
    if max_token_length is None:
        ## Concatenate Notes without any Max Token Length Limit
        _valid_df = _df.copy()
    else:
        ## Concatenate Notes adhering to Max Token Length Limit
        # Max Token Length after accounting for Prefix & Suffix
        full_max_token_length = (
            max_token_length - prefix_token_length - suffix_token_length
        )
        # Cumulative Token Sum in Reverse Chronological Order
        _reversed_df = _df.iloc[::-1]
        token_cum_length = _reversed_df.FormattedNoteTextTokenLength.cumsum()
        # Add Delimiter Count into Cumulative Sum
        token_cum_length = token_cum_length + [
            0,
            *([delim_token_length] * (num_notes - 1)),
        ]
        _reversed_df = _reversed_df.assign(TokenCumulativeLength=token_cum_length)
        # Apply Max Token Length Cut-off
        valid_notes_mask = token_cum_length < full_max_token_length
        # Check to see if at least 1 note fits within max_token_length
        if valid_notes_mask.head(n=1).item() is False:
            # No notes fit because last note exceeds max token length, truncate note tail so it fits
            last_note = _reversed_df.iloc[0]
            last_note_text = last_note.FormattedNoteText
            truncated_note_text = truncate_string_tail(
                last_note_text,
                encoding_name=encoding_name,
                max_token_length=full_max_token_length,
            )
            valid_notes = last_note.to_frame().T.assign(
                FormattedNoteText=truncated_note_text,
                FormattedNoteTextTokenLength=full_max_token_length,
            )
        else:
            # At least 1 note can be included.  Drop Excess Notes.
            valid_notes = _reversed_df.loc[valid_notes_mask]
        # Make Chronological Order Again
        _valid_df = valid_notes.iloc[::-1]

    # Make Final Concatenated Notes
    final_body_text = delim_token.join(_valid_df.FormattedNoteText)
    final_text = f"{prefix}{final_body_text}{suffix}"
    final_text_token_length = num_tokens_from_string(
        final_text, encoding_name=encoding_name
    )

    # Collate Dataframe back into Series, Concatenating NoteText and TokenLength
    return pd.Series(
        {
            "NoteID": _valid_df.NoteID.tolist(),
            "NoteServiceDate": _valid_df.NoteServiceDate.tolist(),
            "ContactEnteredDate": _valid_df.ContactEnteredDate.tolist(),
            "NoteName": _valid_df.NoteName.tolist(),
            "NoteStatus": _valid_df.NoteStatus.tolist(),
            "AuthorProviderType": _valid_df.AuthorProviderType.tolist(),
            "NoteText": final_text,
            "NoteTextTokenLength": final_text_token_length,
            "OriginalNoteTextTokenLength": original_notes_token_length,
            "FormattedNoteTextTokenLength": formatted_notes_text_token_length,
            "PrefixTokenLength": prefix_token_length,
            "SuffixTokenLength": suffix_token_length,
            "FullFormattedNoteTextTokenLength": full_formatted_token_length,
        },
        name=proc_id,
    )
