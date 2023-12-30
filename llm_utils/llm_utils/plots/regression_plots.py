from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from llm_utils import drop_na_examples


def make_regression_residual_figure_for_outcome(
    preds: dict[str, pd.DataFrame],
    targets: pd.DataFrame | dict[str, pd.DataFrame],
    outcome_var: str,
    outcome_name: str,
    figure_kwargs: dict[str, Any] = {"figsize": (8, 12), "layout": "constrained"},
) -> plt.Figure:
    fig = plt.figure(**figure_kwargs)
    gs = fig.add_gridspec(nrows=4, ncols=1)
    subfig0 = fig.add_subfigure(gs[0])
    subfig1 = fig.add_subfigure(gs[1])
    subfig2 = fig.add_subfigure(gs[2])
    subfig3 = fig.add_subfigure(gs[3])

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
        generate_regression_residual_figures(
            prediction_df=preds,
            label_df=target,
            variables=[outcome_var],
            figure=[subfig],
        )
        # Add SubFigure Titles & Figure Title
        subfig.suptitle(f"Model: {model_name}, Note Type: {note_type}")
        fig.suptitle(outcome_name, fontsize=16)
    return fig


def make_composite_regression_residual_figure(
    prediction_df: pd.DataFrame,
    label_df: pd.DataFrame,
    figure_kwargs: dict = {"figsize": (10, 22), "layout": "constrained"},
    title: str = "",
) -> tuple[plt.Figure, dict]:
    """Generate regression & residual figure for variables "phase1_duration",
    "phase2_duration", "pacu_duration", "icu_duration", "hospital_duration"
    and place all into a single figure that is composed of subfigures.

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
    gs = fig.add_gridspec(nrows=5, ncols=1)
    subfig0 = fig.add_subfigure(gs[0])
    subfig1 = fig.add_subfigure(gs[1])
    subfig2 = fig.add_subfigure(gs[2])
    subfig3 = fig.add_subfigure(gs[3])
    subfig4 = fig.add_subfigure(gs[4])

    # Generate Subfigures
    output_dict = generate_regression_residual_figures(
        prediction_df=prediction_df,
        label_df=label_df,
        variables=[
            "phase1_duration",
            "phase2_duration",
            "pacu_duration",
            "icu_duration",
            "hospital_duration",
        ],
        figure=[subfig0, subfig1, subfig2, subfig3, subfig4],
        figure_kwargs=[None] * 5,
    )
    fig.suptitle(title, fontsize=14)
    return fig, output_dict


def generate_regression_residual_figures(
    prediction_df: pd.DataFrame,
    label_df: pd.DataFrame,
    variables: list[str] = [
        "phase1_duration",
        "phase2_duration",
        "pacu_duration",
        "icu_duration",
        "hospital_duration",
    ],
    regplot_kwargs: list[dict | None] = [{}, {}, {}, {}, {}],
    residplot_kwargs: list[dict | None] = [{}, {}, {}, {}, {}],
    figure: list[plt.Figure | None] = [None, None, None, None, None],
    figure_kwargs: list[dict | None] = [None, None, None, None, None],
    drop_na: bool = True,
) -> dict[str, plt.Figure]:
    """Generate multiple regression & residual figures, one for each `variables`.

    Args:
        prediction_df (pd.DataFrame): Table of predictions with `variables` columns.
        label_df (pd.DataFrame): Table of labels with `variables` columns.
        variables (list[str], optional): List of column names in `prediction_df` and `label_df`.
        regplot_kwargs (list[dict | None]): Keyword arguments to use for regression plot
            for each set of data specified in `variables`.  This should be provided in same
            order as `variables`. If list item is `None`, no keyword arguments are applied.
        residplot_kwargs (list[dict | None]): Keyword arguments to use for residual plot
            for each set of data specified in `variables`.  This should be provided in same
            order as `variables`. If list item is `None`, no keyword arguments are applied.
        figure (list[plt.Figure  |  None]): Existing figure handles to use for
            each set of data specified in `variables`.  This should be provided in same
            order as `variables`.  If list item is `None`, a new figure handle is generated.
        figure_kwargs (list[dict | None]): Figure keyword arguments to use for
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
            case "phase1_duration":
                fig, ax = regression_residual_figure(
                    preds=preds,
                    targets=targets,
                    title="Phase 1 Duration",
                    xlabel="Model Prediction",
                    ylabel1="Actual",
                    ylabel2="Residual",
                    units_label="minutes",
                    regplot_kwargs=regplot_kwargs[idx],
                    residplot_kwargs=residplot_kwargs[idx],
                    figure=figure[idx],
                    figure_kwargs=figure_kwargs[idx],
                )

            case "phase2_duration":
                fig, ax = regression_residual_figure(
                    preds=preds,
                    targets=targets,
                    title="Phase 2 Duration",
                    xlabel="Model Prediction",
                    ylabel1="Actual",
                    ylabel2="Residual",
                    units_label="minutes",
                    regplot_kwargs=regplot_kwargs[idx],
                    residplot_kwargs=residplot_kwargs[idx],
                    figure=figure[idx],
                    figure_kwargs=figure_kwargs[idx],
                )

            case "pacu_duration":
                fig, ax = regression_residual_figure(
                    preds=preds,
                    targets=targets,
                    title="PACU Duration",
                    xlabel="Model Prediction",
                    ylabel1="Actual",
                    ylabel2="Residual",
                    units_label="minutes",
                    regplot_kwargs=regplot_kwargs[idx],
                    residplot_kwargs=residplot_kwargs[idx],
                    figure=figure[idx],
                    figure_kwargs=figure_kwargs[idx],
                )

            case "icu_duration":
                fig, ax = regression_residual_figure(
                    preds=preds,
                    targets=targets,
                    title="ICU Duration",
                    xlabel="Model Prediction",
                    ylabel1="Actual",
                    ylabel2="Residual",
                    units_label="days",
                    regplot_kwargs=regplot_kwargs[idx],
                    residplot_kwargs=residplot_kwargs[idx],
                    figure=figure[idx],
                    figure_kwargs=figure_kwargs[idx],
                )

            case "hospital_duration":
                fig, ax = regression_residual_figure(
                    preds=preds,
                    targets=targets,
                    title="Hospital Duration",
                    xlabel="Model Prediction",
                    ylabel1="Actual",
                    ylabel2="Residual",
                    units_label="days",
                    regplot_kwargs=regplot_kwargs[idx],
                    residplot_kwargs=residplot_kwargs[idx],
                    figure=figure[idx],
                    figure_kwargs=figure_kwargs[idx],
                )

        # Add figure to figures dict
        figures_dict[var_name] = fig
        figures_dict[f"{var_name}_dropped"] = num_dropped
    return figures_dict


def regression_residual_figure(
    preds: Sequence,
    targets: Sequence,
    title: str = "",
    xlabel: str = "Model Prediction",
    ylabel1: str = "Actual",
    ylabel2: str = "Residual",
    units_label: str = "",
    regplot_kwargs: dict | None = {},
    residplot_kwargs: dict | None = {},
    figure: plt.Figure | None = None,
    figure_kwargs: dict | None = {},
) -> tuple[plt.Figure, plt.Axes]:
    if regplot_kwargs is None:
        regplot_kwargs = {}
    else:
        regplot_kwargs = {
            "fit_reg": True,
            "seed": 42,
            "n_boot": 1000,
            "line_kws": {"color": "black", "linestyle": "--", "linewidth": "1"},
            **regplot_kwargs,
        }
    if residplot_kwargs is None:
        residplot_kwargs = {}
    else:
        residplot_kwargs = {
            "lowess": True,
            "line_kws": {"color": "red", "linestyle": "--", "linewidth": "1"},
            **residplot_kwargs,
        }
    if figure_kwargs is None:
        figure_kwargs = {}
    else:
        figure_kwargs = {"layout": "constrained", "figsize": (10, 5), **figure_kwargs}

    units_label = f" ({units_label})" if bool(units_label) else ""

    # Make Figure
    figure = plt.figure(**figure_kwargs) if figure is None else figure
    ax = figure.subplots(nrows=1, ncols=2)

    # Linear Regression Line Plot w/ 95% CI
    sns.regplot(
        x=preds,
        y=targets,
        ax=ax[0],
        **regplot_kwargs,
    )
    ax[0].set(
        title="Actual vs. Predicted Values",
        xlabel=xlabel + units_label,
        ylabel=ylabel1 + units_label,
    )
    ax[0].grid(visible=True, which="major", axis="both")
    # Residuals Plot
    residuals = targets - preds
    sns.residplot(
        x=preds,
        y=residuals,
        ax=ax[1],
        **residplot_kwargs,
    )
    ax[1].set(
        title="Residuals vs. Predicted Values",
        xlabel=xlabel + units_label,
        ylabel=ylabel2 + units_label,
    )
    ax[1].grid(visible=True, which="major", axis="both")
    figure.suptitle(title)
    return figure, ax
