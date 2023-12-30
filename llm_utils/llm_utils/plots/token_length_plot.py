import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_token_length_plot(
    data: pd.DataFrame,
    x: str,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    # Make Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8) if figsize is None else figsize)
    ax = sns.kdeplot(data=data, x=x)
    # Vertical Line
    l1 = ax.axvline(x=4000, ymin=0, ymax=1, linestyle=":", color="orange")
    l2 = ax.axvline(x=8192, ymin=0, ymax=1, linestyle=":", color="green")
    l3 = ax.axvline(x=32768, ymin=0, ymax=1, linestyle=":", color="red")
    # Text Labels
    l1_text_coord = (l1.get_xdata()[0], ax.get_ylim()[1] * 0.5)
    l2_text_coord = (l2.get_xdata()[0], ax.get_ylim()[1] * 0.5)
    l3_text_coord = (l3.get_xdata()[0], ax.get_ylim()[1] * 0.5)
    ax.annotate(
        text="gpt35turbo",
        xy=l1_text_coord,
        xytext=l1_text_coord,
        rotation="vertical",
        horizontalalignment="center",
    )
    ax.annotate(
        text="gpt4",
        xy=l2_text_coord,
        xytext=l2_text_coord,
        rotation="vertical",
        horizontalalignment="center",
    )
    ax.annotate(
        text="gpt4_32k",
        xy=l3_text_coord,
        xytext=l3_text_coord,
        rotation="vertical",
        horizontalalignment="center",
    )
    # Title
    ax.set(
        title="Num Tokens in Notes vs. Input Context Length" if title is None else title
    )
    return fig, ax
