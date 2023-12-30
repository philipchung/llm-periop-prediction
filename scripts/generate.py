import pandas as pd
import typer
from typing_extensions import Annotated

from llm.chat_model import ChatModel
from llm.experiment import Experiment

pd.options.display.max_columns = None
pd.options.display.max_rows = 100

app = typer.Typer()


@app.command()
def run_experiment(
    experiment_name: Annotated[str, typer.Option(help="Unique name for experiment run")],
    task: Annotated[
        str,
        typer.Option(
            help="""Specifies dataset, prompt, and task type
            {"asa", "phase1_duration", "hospital_duration", "hospital_admission",
            "icu_duration", "icu_admission", "unplanned_admit", "hospital_mortality"}"""
        ),
    ],
    note_kind: Annotated[
        str,
        typer.Option(help="""Type of note to use in dataset {"last10", "preanes"}"""),
    ] = "last10",
    num_fewshot: Annotated[
        list[int],
        typer.Option(
            help="""Number of fewshot examples to include in fewshot prompts. If a list
            is provided, we run the fewshot generation once for each value in the list.
            If int provided, will run fewshot generation once.""",
        ),
    ] = [1, 5, 10, 20],
    num_inference_examples: Annotated[
        int,
        typer.Option(
            help="""Limit number of examples in inference dataset. 
            `None` means to use entire dataset.""",
        ),
    ] = None,
    num_fewshot_examples: Annotated[
        int,
        typer.Option(
            help="""Limit number of examples in inference dataset. 
            `None` means to use entire dataset.""",
        ),
    ] = None,
    model: Annotated[str, typer.Option(help="""Name of LLM""")] = "gpt-35-turbo-1106",
    num_concurrent: Annotated[int, typer.Option(help="""Number of concurrent LLM API calls""")] = 8,
    num_retries: Annotated[int, typer.Option(help="""Number of retries for LLM API calls""")] = 3,
) -> None:
    """Create and run experiment.  All LLM generations are persisted in database.

    For each `experiment_name`, LLM generations are performed once for each example
    and each combination of `model_name`, input message prompts
    (`system_message` + `user_message`).  The `user_message` is different for
    each of the LLM generation tasks. After each LLM generation, results are stored
    in database. If a new generation is requested with the same model + input message
    combination, then the existing result will be retrieved from the database instead.
    """
    exp = Experiment(
        experiment_name=experiment_name,
        task=task,
        note_kind=note_kind,
        chat_model=ChatModel(model=model),
        # Optional config to constrain inference & fewshot dataset size
        num_inference_examples=num_inference_examples,
        num_fewshot_examples=num_fewshot_examples,
        # Concurrency & Retries
        num_concurrent=num_concurrent,
        num_retries=num_retries,
    )

    # Inference Summaries
    print("Generate Inference Summaries")
    exp.generate_notes_summary_for_inference_dataset()

    # Zeroshot Q&A
    print("Generate Zeroshot Q&A from Notes")
    exp.generate_zeroshot_qa_from_notes_for_inference_dataset()
    print("Generate Zeroshot Q&A from Summaries")
    exp.generate_zeroshot_qa_from_notes_summary_for_inference_dataset()

    # Zeroshot CoT Q&A
    print("Generate Zeroshot CoT Q&A from Notes")
    exp.generate_zeroshot_cot_qa_from_notes_for_inference_dataset()
    print("Generate Zeroshot CoT Q&A from Summaries")
    exp.generate_zeroshot_cot_qa_from_notes_summary_for_inference_dataset()

    # Fewshot Summaries & CoT Rationale Generation
    print("Generate Fewshot Summaries")
    exp.generate_notes_summary_for_fewshot_dataset()
    print("Generate Fewshot CoT Rationales")
    exp.generate_cot_rationale_for_fewshot_dataset()

    # Fewshot & Fewshot CoT Q&A
    if isinstance(num_fewshot, int):
        num_fewshot = [num_fewshot]
    for n in num_fewshot:
        print(f"Generate {n}-shot Q&A from Summaries")
        exp.generate_fewshot_qa_from_notes_summary_for_inference_dataset(num_fewshot=n)
        print(f"Generate {n}-shot CoT Q&A from Summaries")
        exp.generate_fewshot_cot_qa_from_notes_summary_for_inference_dataset(num_fewshot=n)


if __name__ == "__main__":
    app()
