import logging
from dataclasses import dataclass

from make_dataset.notes.note import Note

logger = logging.getLogger(__file__)


@dataclass(kw_only=True)
class LastTenNote(Note):
    """Data object to clean and transform last 10 notes before procedure.

    `df`: Final dataframe.  Either `collated_df` or `concatenated_df`.
    `raw_df`: The unmodified dataframe. Index is NoteID.
    `processed_df`: Cleaned dataframe. Index is NoteID.
    `collated_df`: All notes from a ProcID compressed into lists in single row. Index is ProcID.
    `concatenated_df`: Collated NoteText list concatenated into a single string. Index is ProcID.
    """

    note_type: str = "last10_notes"
