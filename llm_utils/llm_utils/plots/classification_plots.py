from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from llm_utils import drop_na_examples


def make_confusion_matrix_figure_for_outcome(
    preds: dict[str, pd.DataFrame],
    targets: pd.DataFrame | dict[str, pd.DataFrame],
    outcome_var: str,
    outcome_name: str,
    figure_kwargs: dict[str, Any] = {"figsize": (7, 6), "layout": "constrained"},
) -> plt.Figure:
    fig = plt.figure(**figure_kwargs)
    gs = fig.add_gridspec(nrows=2, ncols=2)
    subfig0 = fig.add_subfigure(gs[0, 0])
    subfig1 = fig.add_subfigure(gs[0, 1])
    subfig2 = fig.add_subfigure(gs[1, 0])
    subfig3 = fig.add_subfigure(gs[1, 1])

    for key, value in preds.items():
        outcome_dataset, model_name, note_type = key
        preds = value
        if isinstance(targets, dict):
            target = targets[key]
        elif isinstance(targets, pd.DataFrame):
            target = targets
        else:
            raise ValueError("Unknown input type for `targets`: {targets}")

        # Choose subfigure to generate plot
        match (model_name, note_type):
            case ("gpt35turbo", "preanes"):
                subfig = subfig0
            case ("gpt4", "preanes"):
                subfig = subfig1
            case ("gpt35turbo", "last10"):
                subfig = subfig2
            case ("gpt4", "last10"):
                subfig = subfig3
        # Generate subfigure plot
        generate_confusion_matrix_figures(
            prediction_df=preds,
            label_df=target,
            variables=[outcome_var],
            figure=[subfig],
        )
        # Remove Axes Title (redundant with Figure suptitle)
        subfig.axes[0].set(title="")
        # Add SubFigure Titles & Figure Title
        subfig.suptitle(f"Model: {model_name}, Note Type: {note_type}")
        fig.suptitle(outcome_name, fontsize=16)
    return fig


def make_composite_confusion_matrix_figure(
    prediction_df: pd.DataFrame,
    label_df: pd.DataFrame,
    figure_kwargs: dict = {"figsize": (10, 5), "layout": "constrained"},
    title: str = "",
) -> tuple[plt.Figure, dict]:
    """Generate confusion matrix for variables "asa", "unplanned_admission", "mortality"
    and place all 3 confusion matricies into a single figure that is composed of
    subfigures.

    Args:
        prediction_df (pd.DataFrame): Table of predictions with columns "asa",
            "unplanned_admission", "mortality".
        label_df (pd.DataFrame): Table of labels with columns "asa",
            "unplanned_admission", "mortality".
        figure_kwargs (dict): Keyword arguments for figure constructor.
        title (str): Overall title of figure.

    Returns:
        plt.Figure: Resultant figure with gridspec layout and subfigures.
    """
    # Create Figure & Grid
    fig = plt.figure(**figure_kwargs)
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[2, 1], height_ratios=[1, 1])
    subfig0 = fig.add_subfigure(gs[:, 0])
    subfig1 = fig.add_subfigure(gs[0, 1])
    subfig2 = fig.add_subfigure(gs[1, 1])

    # Generate Subfigures
    output_dict = generate_confusion_matrix_figures(
        prediction_df=prediction_df,
        label_df=label_df,
        variables=[
            "asa",
            "unplanned_admission",
            "mortality",
        ],
        figure=[subfig0, subfig1, subfig2],
    )
    fig.suptitle(title, fontsize=14)
    return fig, output_dict


def generate_confusion_matrix_figures(
    prediction_df: pd.DataFrame,
    label_df: pd.DataFrame,
    variables: list[str] = [
        "asa",
        "unplanned_admission",
        "mortality",
    ],
    figure: list[plt.Figure | None] = [None, None, None],
    figure_kwargs: list[dict | None] = [None, None, None],
    drop_na: bool = True,
) -> dict[str, plt.Figure]:
    """Generate multiple confusion matrix figures, one for each `variables`.

    Args:
        prediction_df (pd.DataFrame): Table of predictions with `variables` columns.
        label_df (pd.DataFrame): Table of labels with `variables` columns.
        variables (list[str], optional): List of column names in `prediction_df` and `label_df`.
        figure (list[plt.Figure  |  None], optional): Existing figure handles to use for
            each set of data specified in `variables`.  This should be provided in same
            order as `variables`.  If list item is `None`, a new figure handle is generated.
        figure_kwargs (list[dict], optional): Figure keyword arguments to use for
            each set of data specified in `variables`.  This should be provided in same
            order as `variables`. If list item is `None`, no keyword arguments are applied.
        drop_na (bool): Whether to drop null data points (in either `prediction_df` or
            `label_df`).

    Returns:
        dict[str, plt.Figure]: Dictionary with `variables` as keys and generated figures
            as values.
    """
    figures_dict = {}
    for idx, var_name in enumerate(variables):
        # Format prediction and target variables
        preds = prediction_df[var_name]
        targets = label_df[var_name]
        if drop_na:
            preds, targets, num_dropped = drop_na_examples(preds, targets)
        else:
            num_dropped = 0

        match var_name:
            case "asa":
                class_labels = [1, 2, 3, 4, 5, 6]
                conf_mat = confusion_matrix(
                    y_true=targets, y_pred=preds, labels=class_labels
                )
                fig, ax = confusion_matrix_figure(
                    data=conf_mat,
                    title="ASA Physical Status Classification",
                    xlabel="Model Prediction",
                    ylabel="Actual",
                    class_labels=class_labels,
                    figure=figure[idx],
                    figure_kwargs=figure_kwargs[idx],
                )

            case "unplanned_admission":
                class_labels = [False, True]
                conf_mat = confusion_matrix(
                    y_true=targets, y_pred=preds, labels=class_labels
                )
                fig, ax = confusion_matrix_figure(
                    data=conf_mat,
                    title="Unplanned Post-operative Admission",
                    xlabel="Model Prediction",
                    ylabel="Actual",
                    class_labels=class_labels,
                    figure=figure[idx],
                    figure_kwargs=figure_kwargs[idx],
                )

            case "mortality":
                class_labels = [False, True]
                conf_mat = confusion_matrix(
                    y_true=targets, y_pred=preds, labels=class_labels
                )
                fig, ax = confusion_matrix_figure(
                    data=conf_mat,
                    title="Hospital Mortality",
                    xlabel="Model Prediction",
                    ylabel="Actual",
                    class_labels=class_labels,
                    figure=figure[idx],
                    figure_kwargs=figure_kwargs[idx],
                )

        # Add figure to figures dict
        figures_dict[var_name] = fig
        figures_dict[f"{var_name}_dropped"] = num_dropped
    return figures_dict


def confusion_matrix_figure(
    data: np.ndarray = None,
    title: str = "",
    xlabel: str = "Model Prediction",
    ylabel: str = "Actual",
    cmap: str = "Blues",
    class_labels: list = None,
    figure: plt.Figure | None = None,
    figure_kwargs: dict | None = {},
) -> tuple[plt.Figure, plt.Axes]:
    if data is None:
        raise ValueError("Argument `data` must be provided.")
    if class_labels is None:
        raise ValueError("Argument `class_labels` must be provided.")
    if figure_kwargs is None:
        figure_kwargs = {}
    else:
        figure_kwargs = {
            "layout": "constrained",
            "figsize": (6.4, 4.8),
            **figure_kwargs,
        }

    # Make Figure
    figure = plt.figure(**figure_kwargs) if figure is None else figure
    ax = figure.subplots(nrows=1, ncols=1)

    sns.heatmap(
        data=data,
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot=True,
        cmap=cmap,
        cbar_kws={"label": "Number of Cases"},
        ax=ax,
    )
    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    return figure, ax


def confusion_matrix_plot(
    preds: Sequence,
    targets: Sequence,
    class_labels: list,
    title: str = "",
    xlabel: str = "Model Prediction",
    ylabel: str = "Actual",
    figure: plt.Figure | None = None,
    figure_kwargs: dict = {
        "figsize": (4, 3),
        "layout": "constrained",
    },
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    conf_mat = confusion_matrix(y_true=targets, y_pred=preds, labels=class_labels).T
    conf_mat = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels).drop(
        columns="None"
    )
    # Plot on Heatmap
    kwargs = {"annot": True, "cmap": "Blues", "fmt": "d"} | kwargs
    # fig, ax = plt.subplots(**figure_kwargs)
    figure = plt.figure(**figure_kwargs) if figure is None else figure
    ax = figure.subplots(nrows=1, ncols=1)
    sns.heatmap(data=conf_mat, ax=ax, **kwargs)
    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    return figure, ax
