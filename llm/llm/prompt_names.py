# Names for Prompts & Answer Columns

import pandas as pd

num_fewshot = [5, 10, 20, 50]
qa_prompt_names = [
    "ZeroshotFromNotes",
    "ZeroshotFromNotesSummary",
    *[f"{x}shotFromNotesSummary" for x in num_fewshot],
]
cot_qa_prompt_names = [
    "ZeroshotCoTFromNotes",
    "ZeroshotCoTFromNotesSummary",
    *[f"{x}shotCoTFromNotesSummary" for x in num_fewshot],
]
prompt_names = qa_prompt_names + cot_qa_prompt_names
answer_columns = [f"{x}Answer" for x in prompt_names]


def rename_prompt_labels(name: str) -> str:
    if "-" in name:
        name = name.replace("-", " | ")
    if "NotesSummary" in name:
        name = name.replace("NotesSummary", "Summary")
    if "Zeroshot" in name:
        name = name.replace("Zeroshot", "0-Shot")
    if "5shot" in name:
        name = name.replace("5shot", "5-Shot")
    if "10shot" in name:
        name = name.replace("10shot", "10-Shot")
    if "20shot" in name:
        name = name.replace("20shot", "20-Shot")
    if "50shot" in name:
        name = name.replace("50shot", "50-Shot")
    if "CoT" in name:
        name = name.replace("CoT", " CoT")
    return name


def format_prompt_name(s: str) -> str:
    """Converts name format:
    'ZeroshotFromNotesAnswer' -> '0-Shot | Notes'
    '20shotCoTFromNotesSummary' -> '20-Shot CoT | NotesSummary'
    """
    prompt_name = s.removesuffix("Answer")
    prompt_name = "-".join(prompt_name.split(sep="From"))
    return rename_prompt_labels(prompt_name)


formatted_prompt_names = [format_prompt_name(x) for x in answer_columns]
qa_formatted_prompt_names = [format_prompt_name(f"{x}Answer") for x in qa_prompt_names]
cot_qa_formatted_prompt_names = [format_prompt_name(f"{x}Answer") for x in cot_qa_prompt_names]

## DataFrame Column Descriptions:
# Original = original prompt name
# Answer = original prompt name + "Answer"; answer column in predictions dataframe
# Formatted = prompt name formatted for figures
# Type = "Non-CoT" or "CoT"
prompt_name_df = pd.DataFrame(
    {
        "Original": {
            0: "ZeroshotFromNotes",
            1: "ZeroshotFromNotesSummary",
            2: "5shotFromNotesSummary",
            3: "10shotFromNotesSummary",
            4: "20shotFromNotesSummary",
            5: "50shotFromNotesSummary",
            6: "ZeroshotCoTFromNotes",
            7: "ZeroshotCoTFromNotesSummary",
            8: "5shotCoTFromNotesSummary",
            9: "10shotCoTFromNotesSummary",
            10: "20shotCoTFromNotesSummary",
            11: "50shotCoTFromNotesSummary",
        },
        "Answer": {
            0: "ZeroshotFromNotesAnswer",
            1: "ZeroshotFromNotesSummaryAnswer",
            2: "5shotFromNotesSummaryAnswer",
            3: "10shotFromNotesSummaryAnswer",
            4: "20shotFromNotesSummaryAnswer",
            5: "50shotFromNotesSummaryAnswer",
            6: "ZeroshotCoTFromNotesAnswer",
            7: "ZeroshotCoTFromNotesSummaryAnswer",
            8: "5shotCoTFromNotesSummaryAnswer",
            9: "10shotCoTFromNotesSummaryAnswer",
            10: "20shotCoTFromNotesSummaryAnswer",
            11: "50shotCoTFromNotesSummaryAnswer",
        },
        "Formatted": {
            0: "0-Shot | Notes",
            1: "0-Shot | Summary",
            2: "5-Shot | Summary",
            3: "10-Shot | Summary",
            4: "20-Shot | Summary",
            5: "50-Shot | Summary",
            6: "0-Shot CoT | Notes",
            7: "0-Shot CoT | Summary",
            8: "5-Shot CoT | Summary",
            9: "10-Shot CoT | Summary",
            10: "20-Shot CoT | Summary",
            11: "50-Shot CoT | Summary",
        },
        "Type": {
            0: "Non-CoT",
            1: "Non-CoT",
            2: "Non-CoT",
            3: "Non-CoT",
            4: "Non-CoT",
            5: "Non-CoT",
            6: "CoT",
            7: "CoT",
            8: "CoT",
            9: "CoT",
            10: "CoT",
            11: "CoT",
        },
    }
)
