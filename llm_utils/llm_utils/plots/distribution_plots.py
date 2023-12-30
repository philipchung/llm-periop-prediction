import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype


def make_histogram_plot_figure(
    prediction_df: pd.DataFrame,
    labels: pd.Series | dict | None = None,
    actual_asa: str | None = None,
    actual_phase1_duration: int | None = None,
    actual_phase2_duration: int | None = None,
    actual_pacu_duration: int | None = None,
    actual_icu_duration: float | None = None,
    actual_hospital_duration: float | None = None,
    actual_unplanned_admission: bool | None = None,
    actual_mortality: bool | None = None,
    proc_id: int | None = None,
    figure_kwargs: dict = {"figsize": (12, 12), "layout": "constrained"},
    title: str | None = None,
) -> plt.Figure:
    _df = prediction_df.copy()
    num_outputs = _df.shape[0]

    # Convert labels
    if labels is not None:
        if isinstance(labels, dict):
            labels = pd.Series(labels, name=proc_id)
        actual_asa = labels.asa
        actual_phase1_duration = labels.phase1_duration
        actual_phase2_duration = labels.phase2_duration
        actual_pacu_duration = labels.pacu_duration
        actual_icu_duration = labels.icu_duration
        actual_hospital_duration = labels.hospital_duration
        actual_unplanned_admission = labels.unplanned_admission
        actual_mortality = labels.mortality

    # Create Figure with Subfigures
    fig = plt.figure(**figure_kwargs)
    subfigs = fig.subfigures(nrows=3, ncols=3)

    # Row 1 Subfigures
    make_asa_figure(
        data=_df,
        x="asa",
        actual=actual_asa,
        title="ASA Physical Status",
        xlabel="ASA Physical Status",
        figure=subfigs[0, 0],
    )
    make_duration_figure(
        data=_df,
        x="phase1_duration",
        actual=actual_phase1_duration,
        title="PACU Phase 1 Duration",
        xlabel="PACU Phase 1 Duration (minutes)",
        figure=subfigs[0, 1],
    )
    make_duration_figure(
        data=_df,
        x="phase2_duration",
        actual=actual_phase2_duration,
        title="PACU Phase 2 Duration",
        xlabel="PACU Phase 2 Duration (minutes)",
        figure=subfigs[0, 2],
    )
    # Row 2 Subfigures
    make_duration_figure(
        data=_df,
        x="pacu_duration",
        actual=actual_pacu_duration,
        title="PACU Total Duration",
        xlabel="PACU Total Duration (minutes)",
        figure=subfigs[1, 0],
    )
    make_duration_figure(
        data=_df,
        x="icu_duration",
        actual=actual_icu_duration,
        title="ICU Duration",
        xlabel="ICU Duration (days)",
        figure=subfigs[1, 1],
    )
    make_duration_figure(
        data=_df,
        x="hospital_duration",
        actual=actual_hospital_duration,
        title="Hospital Duration",
        xlabel="Hospital Duration (days)",
        figure=subfigs[1, 2],
    )
    # Row 3 Subfigures
    make_boolean_figure(
        data=_df,
        x="unplanned_admission",
        actual=actual_unplanned_admission,
        title="Unplanned Hospital Admission",
        xlabel="Unplanned Hospital Admission",
        figure=subfigs[2, 0],
    )
    make_boolean_figure(
        data=_df,
        x="mortality",
        actual=actual_mortality,
        title="In-Hospital Mortality",
        xlabel="Died During Hospital Encounter",
        figure=subfigs[2, 1],
    )

    # Legend
    green_patch = mlines.Line2D(
        [], [], color="green", linestyle="--", linewidth="2", label="Actual Value"
    )
    fig.legend(
        handles=[green_patch],
        bbox_to_anchor=(0.5, 0.97),
        loc="upper center",
    )
    # Overall Figure Title
    if title is None:
        title = f"Distribution of Model Predictions for Example #{proc_id} (n={num_outputs})\n\n"
    fig.suptitle(title, fontsize=14)
    return fig


def make_boolean_figure(
    data: pd.DataFrame,
    actual: int | str,
    x: str = "asa",
    xlabel: str | None = None,
    title: str | None = None,
    plot_null_count: bool = True,
    boolean_plot_kwargs: dict = {},
    null_count_plot_kwargs: dict = {},
    figure: plt.Figure | None = None,
    figure_kwargs: dict | None = {},
) -> tuple[plt.Figure, plt.Axes]:
    """Create figure with Boolean Count Plot or Subplots.

    Args:
        data (pd.DataFrame): table of duration predictions
        actual (int | float): actual duration value
        x (str, optional): name of column that contains durations in `data`
        xlabel (str | None, optional): X-axis label. Defaults to None.
        title (str | None, optional): Figure title. Defaults to None.
        plot_null_count (bool, optional): Whether to plot number of `None` in `data`
            as an auxillary subplot. Defaults to True.
        boolean_plot_kwargs (dict): Keyword arguments for duration plot.
        null_count_plot_kwargs (dict): Keyword arguments for null count plot.
        figure (plt.Figure | None, optional): Figure object that polt axes should use.
            If `None`, then a new figure is created. Defaults to None.
        figure_kwargs (dict | None): Keyword arguments for figure.  This argument
            has no effect if existing Figure object is passed into `figure` argument and
            is only used for initialization of new figures.

    Returns:
        tuple[plt.Figure, plt.Axes]: Tuple of figure and plot axes.
    """
    boolean_plot_kwargs = {
        "element": "step",
        "kde": False,
        "binrange": (0, 1),
        **boolean_plot_kwargs,
    }
    null_count_plot_kwargs = {"element": "step", **null_count_plot_kwargs}
    if figure_kwargs is None:
        figure_kwargs = {}
    else:
        figure_kwargs = {"layout": "constrained", "figsize": (8, 6), **figure_kwargs}

    _df = data.copy()

    # Make Figure
    figure = plt.figure(**figure_kwargs) if figure is None else figure

    if plot_null_count:
        # Create Axes
        ax = figure.subplots(nrows=1, ncols=2, sharey=True, width_ratios=[10, 1])

        # ASA Plot
        make_boolean_plot(
            data=data,
            actual=actual,
            x=x,
            xlabel="",
            title="",
            ax=ax[0],
            **boolean_plot_kwargs,
        )
        ax[0].grid(visible=True, which="major", axis="y")
        # Null Count Plot
        data_col = _df[x]
        null_rows = data_col.loc[data_col.isnull()].apply(lambda x: "None")
        sns.histplot(x=null_rows, ax=ax[1], **null_count_plot_kwargs)
        ax[1].set(xlabel=None)
        ax[1].grid(visible=True, which="major", axis="y")
        ax[1].tick_params(axis="y", which="both", left=False)
        ax[1].set_xticks([0], labels=["None"])
        # Shared Title
        figure.suptitle(title)
        figure.supxlabel(xlabel)
    else:
        ax = figure.subplots(nrows=1, ncols=1)
        make_boolean_plot(
            data=data,
            actual=actual,
            x=x,
            xlabel=xlabel,
            title=title,
            ax=ax,
            **boolean_plot_kwargs,
        )
    return figure, ax


def make_asa_figure(
    data: pd.DataFrame,
    actual: int | str,
    x: str = "asa",
    xlabel: str | None = "ASA Physical Status",
    title: str | None = "ASA Physical Status",
    plot_null_count: bool = True,
    asa_plot_kwargs: dict = {},
    null_count_plot_kwargs: dict = {},
    figure: plt.Figure | None = None,
    figure_kwargs: dict = {},
) -> tuple[plt.Figure, plt.Axes]:
    """Create figure with ASA Count Plot or Subplots.

    Args:
        data (pd.DataFrame): table of duration predictions
        actual (int | float): actual duration value
        x (str, optional): name of column that contains durations in `data`
        xlabel (str | None, optional): X-axis label. Defaults to None.
        title (str | None, optional): Figure title. Defaults to None.
        plot_null_count (bool, optional): Whether to plot number of `None` in `data`
            as an auxillary subplot. Defaults to True.
        asa_plot_kwargs (dict): Keyword arguments for duration plot.
        null_count_plot_kwargs (dict): Keyword arguments for null count plot.
        figure (plt.Figure | None, optional): Figure object that plot axes should use.
            If `None`, then a new figure is created. Defaults to None.
        figure_kwargs (dict): Keyword arguments for figure.  This argument
            has no effect if existing Figure object is passed into `figure` argument and
            is only used for initialization of new figures.
            Defaults to {"layout": "constrained", "figsize": (8, 6)}.

    Returns:
        tuple[plt.Figure, plt.Axes]: Tuple of figure and plot axes.
    """
    asa_plot_kwargs = {
        "element": "step",
        "kde": True,
        "binrange": (0, 5),
        **asa_plot_kwargs,
    }
    null_count_plot_kwargs = {"element": "step", **null_count_plot_kwargs}
    figure_kwargs = {"layout": "constrained", "figsize": (8, 6), **figure_kwargs}
    _df = data.copy()

    # Make Figure
    figure = plt.figure(**figure_kwargs) if figure is None else figure

    if plot_null_count:
        # Create Axes
        ax = figure.subplots(nrows=1, ncols=2, sharey=True, width_ratios=[10, 1])

        # ASA Plot
        make_asa_plot(
            data=data,
            actual=actual,
            x=x,
            xlabel="",
            title="",
            ax=ax[0],
            **asa_plot_kwargs,
        )
        ax[0].grid(visible=True, which="major", axis="y")
        # Null Count Plot
        data_col = _df[x]
        null_rows = data_col.loc[data_col.isnull()].apply(lambda x: "None")
        sns.histplot(x=null_rows, ax=ax[1], **null_count_plot_kwargs)
        ax[1].set(xlabel=None)
        ax[1].grid(visible=True, which="major", axis="y")
        ax[1].tick_params(axis="y", which="both", left=False)
        ax[1].set_xticks([0], labels=["None"])
        # Shared Title
        figure.suptitle(title)
        figure.supxlabel(xlabel)
    else:
        ax = figure.subplots(nrows=1, ncols=1)
        make_asa_plot(
            data=data,
            actual=actual,
            x=x,
            xlabel=xlabel,
            title=title,
            ax=ax,
            **asa_plot_kwargs,
        )
    return figure, ax


def make_duration_figure(
    data: pd.DataFrame,
    actual: int | float,
    x: str,
    xlabel: str | None = None,
    title: str | None = None,
    plot_null_count: bool = True,
    duration_plot_kwargs: dict = {},
    null_count_plot_kwargs: dict = {},
    figure: plt.Figure | None = None,
    figure_kwargs: dict = {},
) -> tuple[plt.Figure, plt.Axes]:
    """Create figure with Duration Count Plot or Subplots.

    Args:
        data (pd.DataFrame): table of duration predictions
        actual (int | float): actual duration value
        x (str, optional): name of column that contains durations in `data`
        xlabel (str | None, optional): X-axis label. Defaults to None.
        title (str | None, optional): Figure title. Defaults to None.
        plot_null_count (bool, optional): Whether to plot number of `None` in `data`
            as an auxillary subplot. Defaults to True.
        duration_plot_kwargs (_type_, optional): Keyword arguments for duration plot.
        null_count_plot_kwargs (_type_, optional): Keyword arguments for null count plot.
        figure (plt.Figure | None, optional): Figure object that plot axes should use.
            If `None`, then a new figure is created. Defaults to None.
        figure_kwargs (_type_, optional): Keyword arguments for figure.  This argument
            has no effect if existing Figure object is passed into `figure` argument and
            is only used for initialization of new figures.

    Returns:
        tuple[plt.Figure, plt.Axes]: Tuple of figure and plot axes.
    """
    duration_plot_kwargs = {
        "element": "step",
        "kde": False,
        **duration_plot_kwargs,
    }
    null_count_plot_kwargs = {"element": "step", **null_count_plot_kwargs}
    figure_kwargs = {"layout": "constrained", "figsize": (8, 6), **figure_kwargs}
    _df = data.copy()

    # Make Figure
    figure = plt.figure(**figure_kwargs) if figure is None else figure

    if plot_null_count:
        # Create Axes
        ax = figure.subplots(nrows=1, ncols=2, sharey=True, width_ratios=[10, 1])

        # Duration Plot
        make_duration_plot(
            data=_df,
            x=x,
            actual=actual,
            ax=ax[0],
            title="",
            xlabel="",
            **duration_plot_kwargs,
        )
        ax[0].grid(visible=True, which="major", axis="y")
        # Null Count Plot
        data_col = _df[x]
        null_rows = data_col.loc[data_col.isnull()].apply(lambda x: "None")
        sns.histplot(x=null_rows, ax=ax[1], **null_count_plot_kwargs)
        ax[1].set(xlabel=None)
        ax[1].grid(visible=True, which="major", axis="y")
        ax[1].tick_params(axis="y", which="both", left=False)
        ax[1].set_xticks([0], labels=["None"])
        # Shared Title
        figure.suptitle(title)
        figure.supxlabel(xlabel)
    else:
        ax = figure.subplots(nrows=1, ncols=1)
        make_duration_plot(
            data=data,
            actual=actual,
            x=x,
            xlabel=xlabel,
            title=title,
            ax=ax,
            **duration_plot_kwargs,
        )
    return figure, ax


def make_boolean_plot_with_null(
    data: pd.DataFrame,
    actual: int | str,
    x: str = "mortality",
    xlabel: str | None = None,
    title: str | None = None,
    **kwargs,
) -> plt.Axes:
    "Boolean Plot with `None` values ignored and not plotted."
    _df = data.copy()
    # Convert Boolean column into categorical
    cat_dtype = CategoricalDtype(categories=["True", "False", "None"], ordered=True)
    cat_mapping = dict(zip(cat_dtype.categories, range(0, 3)))
    cat_col = (
        _df[x]
        .apply(lambda x: str(bool(x)) if pd.notna(x) else "None")
        .astype(cat_dtype)
    )
    _df = _df.copy().assign(**{f"{x}": cat_col})

    # Num Examples
    num_outputs = _df.shape[0]

    default_kwargs = {
        "binrange": (0, 2),
        "element": "step",
        "kde": True,
        "ax": None,
    }
    kwargs = {**default_kwargs, **kwargs}
    # Plot
    p = sns.histplot(data=_df, x=x, **kwargs)
    p.set(
        title=title if title is not None else x,
        xlabel=xlabel if xlabel is not None else x,
        ylim=(0, num_outputs),
    )
    # Vertical Line for Ground Truth Value
    actual_str = str(actual)
    actual_int = cat_mapping[actual_str]
    p.axvline(
        x=actual_str,
        ymax=0.7,
        color="green",
        linestyle="--",
        linewidth="2",
        marker="D",
    )
    p_ymin, p_ymax = p.get_ylim()
    p.annotate(
        text=actual_str,
        color="green",
        xy=(actual_int, 0.8 * p_ymax),
        xycoords="data",
        ha="center",
    )
    return p


def make_boolean_plot(
    data: pd.DataFrame,
    actual: int | str,
    x: str = "mortality",
    xlabel: str | None = None,
    title: str | None = None,
    **kwargs,
) -> plt.Axes:
    "Boolean Plot with `None` values ignored and not plotted."
    _df = data.copy()
    # Convert Boolean column into categorical
    cat_dtype = CategoricalDtype(categories=["True", "False"], ordered=True)
    cat_mapping = dict(zip(cat_dtype.categories, range(0, 2)))
    cat_col = (
        _df[x].apply(lambda x: str(bool(x)) if pd.notna(x) else None).astype(cat_dtype)
    )
    _df = _df.copy().assign(**{f"{x}": cat_col})

    # Num Examples
    num_outputs = _df.shape[0]

    default_kwargs = {
        "binrange": (0, 1),
        "element": "step",
        "kde": True,
        "ax": None,
    }
    kwargs = {**default_kwargs, **kwargs}
    # Plot
    p = sns.histplot(data=_df, x=x, **kwargs)
    p.set(
        title=title if title is not None else x,
        xlabel=xlabel if xlabel is not None else x,
        ylim=(0, num_outputs),
    )
    # Vertical Line for Ground Truth Value
    actual_str = str(actual)
    actual_int = cat_mapping[actual_str]
    p.axvline(
        x=actual_str,
        ymax=0.7,
        color="green",
        linestyle="--",
        linewidth="2",
        marker="D",
    )
    p_ymin, p_ymax = p.get_ylim()
    p.annotate(
        text=actual_str,
        color="green",
        xy=(actual_int, 0.8 * p_ymax),
        xycoords="data",
        ha="center",
    )
    return p


def make_asa_plot_with_null(
    data: pd.DataFrame,
    actual: int | str,
    x: str = "asa",
    xlabel: str | None = "ASA Physical Status",
    title: str | None = "ASA Physical Status",
    **kwargs,
) -> plt.Axes:
    "ASA Plot with `None` value treated as a categorical."
    _df = data.copy()
    # Convert ASA column into categorical
    cat_dtype = CategoricalDtype(
        categories=["1", "2", "3", "4", "5", "6", "None"], ordered=True
    )
    cat_mapping = dict(zip(cat_dtype.categories, range(1, 8)))
    cat_col = (
        _df[x].apply(lambda x: str(int(x)) if pd.notna(x) else "None").astype(cat_dtype)
    )
    _df = _df.copy().assign(**{f"{x}": cat_col})

    # Num Examples
    num_outputs = _df.shape[0]

    # Optionally override default kwargs
    default_kwargs = {
        "binrange": (0, 6),
        "element": "step",
        "kde": True,
        "ax": None,
    }
    kwargs = {**default_kwargs, **kwargs}
    # Plot
    p = sns.histplot(data=_df, x=x, **kwargs)
    p.set(
        title=title if title is not None else x,
        xlabel=xlabel if xlabel is not None else x,
        ylim=(0, num_outputs),
    )
    # Vertical Line for Ground Truth Value
    actual_str = str(actual)
    actual_int = cat_mapping[actual_str]
    p.axvline(
        x=actual_str,
        ymax=0.7,
        color="green",
        linestyle="--",
        linewidth="2",
        marker="D",
    )
    p_ymin, p_ymax = p.get_ylim()
    p.annotate(
        text=actual_str,
        color="green",
        xy=(actual_int - 1, 0.8 * p_ymax),
        xycoords="data",
        ha="center",
    )
    return p


def make_asa_plot(
    data: pd.DataFrame,
    actual: int | str,
    x: str = "asa",
    xlabel: str | None = "ASA Physical Status",
    title: str | None = "ASA Physical Status",
    **kwargs,
) -> plt.Axes:
    "ASA Plot with `None` value ignored and not plotted."
    _df = data.copy()
    # Convert ASA column into categorical
    cat_dtype = CategoricalDtype(
        categories=["1", "2", "3", "4", "5", "6"], ordered=True
    )
    cat_mapping = dict(zip(cat_dtype.categories, range(1, 7)))
    cat_col = (
        _df[x].apply(lambda x: str(int(x)) if pd.notna(x) else None).astype(cat_dtype)
    )
    _df = _df.copy().assign(**{f"{x}": cat_col})

    # Num Examples
    num_outputs = _df.shape[0]

    # Optionally override default kwargs
    default_kwargs = {
        "binrange": (0, 5),
        "element": "step",
        "kde": True,
        "ax": None,
    }
    kwargs = {**default_kwargs, **kwargs}
    # Plot
    p = sns.histplot(data=_df, x=x, **kwargs)
    p.set(
        title=title if title is not None else x,
        xlabel=xlabel if xlabel is not None else x,
        ylim=(0, num_outputs),
    )
    # Vertical Line for Ground Truth Value
    actual_str = str(actual)
    actual_int = cat_mapping[actual_str]
    p.axvline(
        x=actual_str,
        ymax=0.7,
        color="green",
        linestyle="--",
        linewidth="2",
        marker="D",
    )
    p_ymin, p_ymax = p.get_ylim()
    p.annotate(
        text=actual_str,
        color="green",
        xy=(actual_int - 1, 0.8 * p_ymax),
        xycoords="data",
        ha="center",
    )
    return p


def make_duration_plot(
    data: pd.DataFrame,
    actual: int | float,
    x: str = "pacu_duration",
    xlabel: str | None = None,
    title: str | None = None,
    **kwargs,
) -> plt.Axes:
    "Duration Plot with `None` values ignored and not plotted."
    _df = data.copy()
    # Optionally override default kwargs
    default_kwargs = {
        "element": "step",
        "kde": True,
        "ax": None,
    }
    kwargs = {**default_kwargs, **kwargs}

    # Plot
    p = sns.histplot(data=_df, x=x, **kwargs)
    p.set(
        title=title if title is not None else x,
        xlabel=xlabel if xlabel is not None else x,
        xlim=(0, max(_df[x].max(), actual)),
    )
    # Set x-margin to 5%
    p_xmin, p_xmax, p_ymin, p_ymax = p.axis()
    p_xrange = p_xmax - p_xmin
    margin = 0.05  # Percent Margin
    p.axis((p_xmin - margin * p_xrange, p_xmax + margin * p_xrange, p_ymin, p_ymax))
    # Add Vertical Line Annotations
    p = add_vertical_line_labels(
        p=p, data=_df, x=x, actual=actual, p_ymax=p_ymax, **kwargs
    )
    return p


def add_vertical_line_labels(
    p: plt.Axes,
    data: pd.DataFrame,
    x: str,
    actual: float,
    p_ymax: float,
    show_label: bool = True,
    show_mean: bool = True,
    show_mode: bool = True,
    **kwargs,
) -> plt.Axes:
    if show_label:
        # Vertical Line for Ground Truth Value
        p.axvline(
            x=actual,
            ymax=0.7,
            color="green",
            linestyle="--",
            linewidth="2",
            marker="D",
        )
        p.annotate(
            text=f"{actual:.1f}",
            color="green",
            xy=(actual, 0.8 * p_ymax),
            xycoords="data",
            ha="center",
        )
    if show_mean:
        # Mean of non-null values
        ans = data.loc[:, x]
        ans = ans.loc[ans.notna()]
        mean_value = ans.mean()
        p.axvline(
            x=mean_value,
            ymax=0.6,
            color="blue",
            linestyle="--",
            linewidth="2",
            marker="D",
        )
        p.annotate(
            text=f"{mean_value:.1f}",
            color="blue",
            xy=(mean_value, 0.7 * p_ymax),
            xycoords="data",
            ha="center",
        )
    if show_mode:
        # Mode (Majority Vote) of non-null values
        ans = data.loc[:, x]
        ans = ans.loc[ans.notna()]
        mode_values = ans.mode()
        # Possible to have 2 or more mode values if tied
        for idx, mode_value in mode_values.items():
            p.axvline(
                x=mode_value,
                ymax=0.5,
                color="orange",
                linestyle="--",
                linewidth="2",
                marker="D",
            )
            p.annotate(
                text=f"{mode_value:.1f}",
                color="orange",
                xy=(mode_value, 0.6 * p_ymax),
                xycoords="data",
                ha="center",
            )
    return p
