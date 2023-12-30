import pandas as pd

from make_dataset.note import (
    collate_multiple_notes_per_proc_id,
    concatenate_collated_notes_with_header,
)


class TestNote:
    notes_df: pd.DataFrame = pd.DataFrame(
        [
            {
                "ProcID": 1234,
                "NoteID": "A97",
                "NoteServiceDate": pd.Timestamp("2020-01-14 08:15:00"),
                "ContactEnteredDate": pd.Timestamp("2020-01-14 08:30:00"),
                "NoteName": "Example Note 1",
                "NoteStatus": "Signed",
                "AuthorProviderType": "Physician",
                "NoteText": (
                    "Lorem ipsum dolor sit amet. Ut unde ullam rem rerum quia "
                    "non enim sint eos harum eius et vero iste At voluptatem "
                    "unde et cupiditate omnis? Id quaerat deserunt eum dolorem "
                    "nisi est suscipit cupiditate sed rerum autem qui consequuntur "
                    "voluptas ut iure placeat ex accusamus sint."
                ),
                "TokenLength": 73,
            },
            {
                "ProcID": 1234,
                "NoteID": "B98",
                "NoteServiceDate": pd.Timestamp("2020-01-15 08:33:00"),
                "ContactEnteredDate": pd.Timestamp("2020-01-15 09:11:00"),
                "NoteName": "Example Note 2",
                "NoteStatus": "Signed",
                "AuthorProviderType": "Physician Assistant",
                "NoteText": (
                    "The rose is red, the violet's blue\n"
                    "The honey's sweet, and so are you.\n"
                    "Thou are my love and I am thine;\n"
                    "I drew thee to my Valentine:\n"
                    "The lot was cast and then I drew,\n"
                    "And Fortune said it should be you.\n"
                ),
                "TokenLength": 55,
            },
            {
                "ProcID": 4567,
                "NoteID": "C99",
                "NoteServiceDate": pd.Timestamp("2020-01-15 15:34:00"),
                "ContactEnteredDate": pd.Timestamp("2020-01-15 09:36:00"),
                "NoteName": "Example Note 3",
                "NoteStatus": "Addendum",
                "AuthorProviderType": "Resident",
                "NoteText": (
                    "We hold these truths to be self-evident, that all men are "
                    "created equal, that they are endowed by their Creator with "
                    "certain unalienable Rights, that among these are Life, Liberty "
                    "and the pursuit of Happiness."
                ),
                "TokenLength": 45,
            },
        ]
    )
    collated_notes: pd.DataFrame = pd.DataFrame(
        [
            {
                "NoteID": ["A97", "B98"],
                "NoteServiceDate": [
                    pd.Timestamp("2020-01-14 08:15:00"),
                    pd.Timestamp("2020-01-15 08:33:00"),
                ],
                "ContactEnteredDate": [
                    pd.Timestamp("2020-01-14 08:30:00"),
                    pd.Timestamp("2020-01-15 09:11:00"),
                ],
                "NoteName": ["Example Note 1", "Example Note 2"],
                "NoteStatus": ["Signed", "Signed"],
                "AuthorProviderType": ["Physician", "Physician Assistant"],
                "NoteText": [
                    (
                        "Lorem ipsum dolor sit amet. Ut unde ullam rem rerum quia "
                        "non enim sint eos harum eius et vero iste At voluptatem "
                        "unde et cupiditate omnis? Id quaerat deserunt eum dolorem "
                        "nisi est suscipit cupiditate sed rerum autem qui consequuntur "
                        "voluptas ut iure placeat ex accusamus sint."
                    ),
                    (
                        "The rose is red, the violet's blue\n"
                        "The honey's sweet, and so are you.\n"
                        "Thou are my love and I am thine;\n"
                        "I drew thee to my Valentine:\n"
                        "The lot was cast and then I drew,\n"
                        "And Fortune said it should be you.\n"
                    ),
                ],
                "TokenLength": [73, 55],
            },
            {
                "NoteID": ["C99"],
                "NoteServiceDate": [pd.Timestamp("2020-01-15 15:34:00")],
                "ContactEnteredDate": [pd.Timestamp("2020-01-15 09:36:00")],
                "NoteName": ["Example Note 3"],
                "NoteStatus": ["Addendum"],
                "AuthorProviderType": ["Resident"],
                "NoteText": [
                    (
                        "We hold these truths to be self-evident, that all men are "
                        "created equal, that they are endowed by their Creator with "
                        "certain unalienable Rights, that among these are Life, Liberty "
                        "and the pursuit of Happiness."
                    ),
                ],
                "TokenLength": [45],
            },
        ],
        index=pd.Series([1234, 4567], name="ProcID"),
    )

    concatenated_notes: list[str] = [
        (
            "Example Note 1 written by Physician at 2020-01-14 08:15:00:"
            "\n"
            "Lorem ipsum dolor sit amet. Ut unde ullam rem rerum quia "
            "non enim sint eos harum eius et vero iste At voluptatem "
            "unde et cupiditate omnis? Id quaerat deserunt eum dolorem "
            "nisi est suscipit cupiditate sed rerum autem qui consequuntur "
            "voluptas ut iure placeat ex accusamus sint."
            "\n\n"
            "Example Note 2 written by Physician Assistant at 2020-01-15 08:33:00:"
            "\n"
            "The rose is red, the violet's blue\n"
            "The honey's sweet, and so are you.\n"
            "Thou are my love and I am thine;\n"
            "I drew thee to my Valentine:\n"
            "The lot was cast and then I drew,\n"
            "And Fortune said it should be you.\n"
        ),
        (
            "Example Note 3 written by Resident at 2020-01-15 15:34:00:\n"
            "We hold these truths to be self-evident, that all men are "
            "created equal, that they are endowed by their Creator with "
            "certain unalienable Rights, that among these are Life, Liberty "
            "and the pursuit of Happiness."
        ),
    ]
    # NOTE: Note collections are truncated in 2 ways:
    #   - drop earliest notes (keep most recent)
    #   - if only one note fits in context window, truncate the tail of that note
    concatenated_notes_short: list[str] = [
        (
            "Example Note 2 written by Physician Assistant at 2020-01-15 08:33:00:"
            "\n"
            "The rose is red, the violet's blue\n"
            "The honey's sweet, and so are you.\n"
            "Thou are my love and I"
        ),
        (
            "Example Note 3 written by Resident at 2020-01-15 15:34:00:"
            "\n"
            "We hold these truths to be self-evident, that all men are "
            "created equal, that they are endowed by their Creator with certain un"
        ),
    ]

    def test_collate_multiple_notes_per_proc_id(self) -> None:
        result = (
            self.notes_df.groupby("ProcID", group_keys=True, as_index=False)
            .apply(collate_multiple_notes_per_proc_id)
            .set_index("ProcID")
        )
        pd.testing.assert_frame_equal(result, self.collated_notes)

    def test_concatenate_notes_full(self) -> None:
        concatenated_df = pd.DataFrame(
            [
                concatenate_collated_notes_with_header(series, max_token_length=4000)
                for idx, series in self.collated_notes.iterrows()
            ]
        )
        result = concatenated_df.NoteText.tolist()
        assert result == self.concatenated_notes

    def test_concatenate_notes_short(self) -> None:
        concatenated_df = pd.DataFrame(
            [
                concatenate_collated_notes_with_header(series, max_token_length=50)
                for idx, series in self.collated_notes.iterrows()
            ]
        )
        result = concatenated_df.NoteText.tolist()
        assert result == self.concatenated_notes_short
