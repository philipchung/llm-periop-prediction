import logging
import re
from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

from make_dataset.notes.note import Note

logger = logging.getLogger(__file__)


@dataclass(kw_only=True)
class PreAnesNote(Note):
    """Data object to clean and transform pre-anesthesia notes.

    `df`: Final dataframe.  Either `collated_df` or `concatenated_df`.
    `raw_df`: The unmodified dataframe. Index is NoteID.
    `processed_df`: Cleaned dataframe. Index is NoteID.
    `collated_df`: All notes from a ProcID compressed into lists in single row. Index is ProcID.
    `concatenated_df`: Collated NoteText list concatenated into a single string. Index is ProcID.
    """

    note_type: str = "preanes_notes"

    def process_data_logic(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        _df = notes_df.copy()
        # Remove Anesthesia Plan (which contains ASA-PS)
        _df = self.remove_anesthesia_plan_from_notes(_df)
        # Get Token Counts
        _df = self.compute_note_token_length(_df)
        return _df

    def remove_anesthesia_plan_from_notes(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        _df = notes_df.copy()
        tqdm.pandas(desc="Splitting Anesthesia Plan from Note")
        split_df = pd.DataFrame(
            data=_df.NoteText.apply(split_anesthesia_plan).tolist(),
            index=_df.index,
        )
        # Replace NoteText with the Split Version without Anesthesia Plan
        # If Split is unsuccessful (~0.06% of notes), the note is following a
        # non-standard template and we will toss these notes.
        successfully_split_note_ids = split_df.loc[
            split_df.SplitSuccessful
        ].index.tolist()
        successfully_split_notes = split_df.loc[successfully_split_note_ids].NoteText
        _df = _df.loc[_df.index.isin(successfully_split_note_ids)]
        _df = _df.assign(NoteText=successfully_split_notes)
        return _df


def split_anesthesia_plan(text: str) -> dict[str, str]:
    "Uses Regular Expressions to identify `ANESTHESIA PLAN` at end of preanesthesia note."
    pattern = r"(?P<anes_plan>(?:ANESTHESIA\W+PLAN)|(?:Anesthesia\W+Plan))"
    match = re.search(pattern=pattern, string=text)
    if match is None:
        return {"NoteText": None, "AnesthesiaPlan": None, "SplitSuccessful": False}
    else:
        matched_span = match.span()
        matched_start = matched_span[0]
        return {
            "NoteText": text[:matched_start],
            "AnesthesiaPlan": text[matched_start:],
            "SplitSuccessful": True,
        }
