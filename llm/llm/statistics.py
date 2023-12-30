from itertools import combinations

import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests


def pairwise_mannwhitneyu(
    boot_metric_values: pd.DataFrame, correction_method: str = "bonferroni"
) -> pd.DataFrame:
    """Compute Mann-Whitney U rank test (Wilcoxon rank-sum) & corrected p-values
    for all pairs of experimental conditions.  Each experimental condition
    is a column in `boot_metric_values`. This test is appropriate if samples
    from each experimental condition are independent from one another.
    If samples are the same (paired) then use `pairwise_wilcoxon`.

    Args:
        boot_metric_values (pd.DataFrame): Dataframe of bootstrapped metric
            values for each experimental condition. Columns are each
            experimental condition. Rows represent metric computed from
            a single bootstrap sample iteration for each experimental condition.
        correction_method (str): Method for multiple hypothesis testing correction.

    Returns:
        pd.DataFrame: Dataframe with each row representing a pairwise
            comparison between experimental conditions along with the
            corrected p-value and whether to reject the null hypothesis.
    """
    pairwise_comparisons = list(combinations(boot_metric_values.columns, r=2))
    # Mann-Whitney U rank test for each pairwise comparison
    T = []
    for x, y in pairwise_comparisons:
        T += [mannwhitneyu(x=boot_metric_values[x].tolist(), y=boot_metric_values[y].tolist())]
    T = pd.DataFrame(T)
    # Multiple hypothesis testing correction
    reject_H0, p_vals_corrected, alpha_corrected_sidak, alpha_corrected_bonferroni = multipletests(
        pvals=T.pvalue,
        alpha=0.05,
        method=correction_method,
    )
    # Return all pairwise comparisons along with corrected p-values
    df = pd.concat(
        [
            pd.DataFrame(pairwise_comparisons, columns=["Prompt1", "Prompt2"]),
            T,
            pd.DataFrame(
                {
                    "corrected_pvalue": p_vals_corrected,
                    "reject_H0": reject_H0,
                    "corrected_alpha_sidak": alpha_corrected_sidak,
                    "corrected_alpha_bonferroni": alpha_corrected_bonferroni,
                }
            ),
        ],
        axis="columns",
    )
    return df


def pairwise_wilcoxon(
    boot_metric_values: pd.DataFrame, correction_method: str = "bonferroni"
) -> pd.DataFrame:
    """Compute Wilcoxon signed-rank test & corrected p-values
    for all pairs of experimental conditions.  Each experimental condition
    is a column in `boot_metric_values`. This test is appropriate if samples
    from each experimental condition are same (paired).  If the samples
    from each experimental condition are independent, use `pairwise_mannwhitneyu`.

    Args:
        boot_metric_values (pd.DataFrame): Dataframe of bootstrapped metric
            values for each experimental condition. Columns are each
            experimental condition. Rows represent metric computed from
            a single bootstrap sample iteration for each experimental condition.
        correction_method (str): Method for multiple hypothesis testing correction.

    Returns:
        pd.DataFrame: Dataframe with each row representing a pairwise
            comparison between experimental conditions along with the
            corrected p-value and whether to reject the null hypothesis.
    """
    pairwise_comparisons = list(combinations(boot_metric_values.columns, r=2))
    # Wilcoxon signed-rank test for each pairwise comparison
    T = []
    for x, y in pairwise_comparisons:
        T += [wilcoxon(x=boot_metric_values[x].tolist(), y=boot_metric_values[y].tolist())]
    T = pd.DataFrame(T)
    # Multiple hypothesis testing correction
    reject_H0, p_vals_corrected, alpha_corrected_sidak, alpha_corrected_bonferroni = multipletests(
        pvals=T.pvalue,
        alpha=0.05,
        method=correction_method,
    )
    # Return all pairwise comparisons along with corrected p-values
    df = pd.concat(
        [
            pd.DataFrame(pairwise_comparisons, columns=["Prompt1", "Prompt2"]),
            T,
            pd.DataFrame(
                {
                    "corrected_pvalue": p_vals_corrected,
                    "reject_H0": reject_H0,
                    "corrected_alpha_sidak": alpha_corrected_sidak,
                    "corrected_alpha_bonferroni": alpha_corrected_bonferroni,
                }
            ),
        ],
        axis="columns",
    )
    return df
