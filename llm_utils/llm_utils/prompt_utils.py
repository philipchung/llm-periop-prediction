from collections import namedtuple

prompt_name_T = namedtuple("PromptType", ["task_name", "task_strategy", "shot_kind"])


def parse_prompt_name(prompt_name: str | None = None) -> tuple[str, str, str]:
    """Parse prompt_name pattern into subcomponents: task_name, task_strategy, shot_kind

    Pattern: {task_name}-{task_strategy}-{shot_kind}

    task_name: {label_name, "summarize"}
    task_strategy: {"summarize", "coalesce", "qa", "cot", "generate_cot"}
    shot_kind: {"zeroshot", "fewshot"}

    Args:
        prompt_name (str, optional): prompt_name string

    Returns:
        tuple[str, str, str]: subcomponents of the prompt_name string as a namedtuple
    """
    if prompt_name is None:
        return prompt_name_T(task_name="", task_strategy="", shot_kind="")
    else:
        task_name, task_strategy, shot_kind = prompt_name.split("-")
        return prompt_name_T(
            task_name=task_name, task_strategy=task_strategy, shot_kind=shot_kind
        )
