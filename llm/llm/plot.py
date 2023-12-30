import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metric_for_prompts(
    data: pd.DataFrame,
    qa_formatted_prompt_names: list[str] = [
        "0-Shot | Notes",
        "0-Shot | Summary",
        "5-Shot | Summary",
        "10-Shot | Summary",
        "20-Shot | Summary",
        "50-Shot | Summary",
    ],
    cot_qa_formatted_prompt_names: list[str] = [
        "0-Shot CoT | Notes",
        "0-Shot CoT | Summary",
        "5-Shot CoT | Summary",
        "10-Shot CoT | Summary",
        "20-Shot CoT | Summary",
        "50-Shot CoT | Summary",
    ],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    baseline_value: float | None = None,
    baseline_label: str = "Baseline",
) -> tuple[plt.Figure, plt.Axes]:
    """Plots data in `data` as 2 line plots on the same axes.

    Args:
        data (pd.DataFrame): Dataframe in long format.
            Requires columns `prompt`, `length_group`, `value`
        qa_formatted_prompt_names (list[str]): list of string names of prompts to
            plot in first line plot group
        cot_qa_formatted_prompt_names (list[str]): list of string names of prompts to
            plot in second line plot group
        title (str, optional): Plot title. Defaults to "".
        xlabel (str, optional): X-axis label. Defaults to "".
        ylabel (str, optional): Y-axis label. Defaults to "".
        baseline_value (float, optional): Baseline comparison value. Defaults to None.
        baseline_label (str, optional): Legend label for baseline value.

    Returns:
        tuple[plt.Figure, plt.Axes]: tuple of figure and axes.
    """

    # Plot separate Q&A and CoT Q&A lines on the same axes
    def nullify_cot_qa_values(row: pd.Series) -> float | None:
        if row.prompt in qa_formatted_prompt_names:
            return row.value
        else:
            return None

    def nullify_qa_values(row: pd.Series) -> float | None:
        if row.prompt in cot_qa_formatted_prompt_names:
            return row.value
        else:
            return None

    qa_data = data.copy()
    qa_data = qa_data.assign(value=qa_data.apply(nullify_cot_qa_values, axis=1))
    cot_qa_data = data.copy()
    cot_qa_data = cot_qa_data.assign(value=cot_qa_data.apply(nullify_qa_values, axis=1))

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    sns.pointplot(
        data=qa_data,
        x="prompt",
        y="value",
        hue="length_group",
        dodge=True,
        linestyles=[":", ":", ":", "-"],
        ax=ax,
    )
    sns.pointplot(
        data=cot_qa_data,
        x="prompt",
        y="value",
        hue="length_group",
        dodge=True,
        linestyles=[":", ":", ":", "-"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_axisbelow(True)
    ax.grid(axis="both")
    # Set line opacity
    for line in ax.lines:
        line.set_alpha(0.6)
    # Optional Baseline Value
    if baseline_value:
        ax.axhline(y=baseline_value, color="tab:gray", linestyle="--")
    # Legend
    gray_dash = mlines.Line2D(
        [],
        [],
        color="tab:gray",
        linestyle="--",
    )
    spacer = mlines.Line2D(
        [],
        [],
        color="white",
        linestyle=None,
    )
    handles = ax.get_legend_handles_labels()[0][:4]
    labels = ax.get_legend_handles_labels()[1][:4]
    if baseline_value:
        handles = handles + [spacer, spacer, gray_dash]
        labels = labels + ["", "", f"{baseline_label}"]
    ax.legend(
        title="Notes Length Group",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        handles=handles,
        labels=labels,
        frameon=False,
    )
    return fig, ax


def plot_metric_for_prompts_on_axes(
    data: pd.DataFrame,
    ax: plt.Axes,
    qa_formatted_prompt_names: list[str] = [
        "0-Shot | Notes",
        "0-Shot | Summary",
        "5-Shot | Summary",
        "10-Shot | Summary",
        "20-Shot | Summary",
        "50-Shot | Summary",
    ],
    cot_qa_formatted_prompt_names: list[str] = [
        "0-Shot CoT | Notes",
        "0-Shot CoT | Summary",
        "5-Shot CoT | Summary",
        "10-Shot CoT | Summary",
        "20-Shot CoT | Summary",
        "50-Shot CoT | Summary",
    ],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    baseline_value: float | None = None,
    legend_on: bool = True,
) -> plt.Axes:
    """Plots data in `data` as 2 line plots on the same axes.

    This method requires axes to be provided. This method does not draw legends.

    Args:
        data (pd.DataFrame): Dataframe in long format.
            Requires columns `prompt`, `length_group`, `value`
        ax (plt.Axes): Matplotlib axes object on which to create plot.
        qa_formatted_prompt_names (list[str]): list of string names of prompts to
            plot in first line plot group
        cot_qa_formatted_prompt_names (list[str]): list of string names of prompts to
            plot in second line plot group
        title (str, optional): Plot title. Defaults to "".
        xlabel (str, optional): X-axis label. Defaults to "".
        ylabel (str, optional): Y-axis label. Defaults to "".
        baseline_value (float, optional): Baseline comparison value. Defaults to None.
        legend_on (bool, optional): Whether to show legend. Defaults to True.

    Returns:
        plt.Axes: The axes object
    """

    # Plot separate Q&A and CoT Q&A lines on the same axes
    def nullify_cot_qa_values(row: pd.Series) -> float | None:
        if row.prompt in qa_formatted_prompt_names:
            return row.value
        else:
            return None

    def nullify_qa_values(row: pd.Series) -> float | None:
        if row.prompt in cot_qa_formatted_prompt_names:
            return row.value
        else:
            return None

    qa_data = data.copy()
    qa_data = qa_data.assign(value=qa_data.apply(nullify_cot_qa_values, axis=1))
    cot_qa_data = data.copy()
    cot_qa_data = cot_qa_data.assign(value=cot_qa_data.apply(nullify_qa_values, axis=1))

    # fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    sns.pointplot(
        data=qa_data,
        x="prompt",
        y="value",
        hue="length_group",
        dodge=True,
        linestyles=[":", ":", ":", "-"],
        ax=ax,
    )
    sns.pointplot(
        data=cot_qa_data,
        x="prompt",
        y="value",
        hue="length_group",
        dodge=True,
        linestyles=[":", ":", ":", "-"],
        ax=ax,
    )
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_axisbelow(True)
    ax.grid(axis="both")
    # Set line opacity
    for line in ax.lines:
        line.set_alpha(0.6)
    # Optional Baseline Value
    if baseline_value:
        ax.axhline(y=baseline_value, color="tab:gray", linestyle="--")
    ax.get_legend().set_visible(legend_on)
    return ax
