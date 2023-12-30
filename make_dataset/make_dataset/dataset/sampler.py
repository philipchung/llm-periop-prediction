from functools import partial

import pandas as pd
from llm_utils import parallel_process


class DatasetSampler:
    """Class that implements methods to sample data. This class can be used as a mixin."""

    def select_cases_per_patient(
        self,
        df: pd.DataFrame,
        patient_id_var: str = "PAT_ID",
        max_cases: int = 1,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Given a table of cases, randomly sample up to `max_cases` number of cases
        per patient.

        Args:
            df (pd.DataFrame): Table of cases.
            pat_id_var (str): Column name in dataframe to uniquely identify a patient.
            max_cases (int): Max number of cases to sample for each patient.  Setting
                this to 1 will ensure each patient has only 1 case in the resultant table.
            seed (int): Random seed

        Returns:
            pd.DataFrame: Table of cases
        """
        _df = df.copy()
        fn = partial(subsample_cases_by_patient, max_cases=max_cases, seed=seed)
        output = parallel_process(
            iterable=(t[1] for t in _df.reset_index().groupby(patient_id_var)),
            function=fn,
            desc=f"Select up to {max_cases} cases per Patient",
        )
        return pd.concat(output, axis=0)

    def sample_inverse_frequency(
        self,
        df: pd.DataFrame,
        col: str | None = "ASA",
        n: int = 100,
        replace: bool = False,
        seed: int = 42,
        keep_weights: bool = False,
    ) -> pd.DataFrame:
        """Given a table of cases, uses column `col` to perform inverse frequency sampling.

        Args:
            df (pd.DataFrame): Table of cases.
            col (str | None): Column name in dataframe to use for inverse frequency sampling.
                Values in this column should be a categorical variable. If `None`, then
                samples will be drawn from `df` with uniform weights.
            n (int): Number of cases to sample
            replace (bool): Whether to sample with replacement.  Does not sample with
                replacement by default.
            seed (int): Random seed
            keep_weights (bool): If true, a new column called "InvFreqWeight" which
                contains the weights used to sample each category in "col" is added to
                the returned dataframe.

        Returns:
            pd.DataFrame: Table of cases.
        """
        _df = df.copy()
        # Sample with Equal Weights
        if col is None:
            sample = _df.sample(n=n, replace=replace, weights=None, random_state=seed)
        # Inverse Frequency Sampling based on `col` variable
        else:
            col_for_weighting = _df.loc[:, col]
            frequency = col_for_weighting.value_counts(normalize=True).sort_index(
                ascending=True
            )
            inv_frequency = 1 / frequency
            inv_frequency_dict = inv_frequency.to_dict()
            inv_freq_weights = col_for_weighting.apply(lambda x: inv_frequency_dict[x])
            _df = _df.assign(InvFreqWeight=inv_freq_weights)

            sample = _df.sample(
                n=n, replace=replace, weights="InvFreqWeight", random_state=seed
            )
            if not keep_weights:
                sample = sample.drop(columns="InvFreqWeight")
        return sample

    def sample(
        self, df: pd.DataFrame, n: int = 100, replace: bool = False, seed: int = 42
    ):
        "Sample up to `n` rows from `df`."
        num_rows = df.shape[0]
        return df.sample(n=min(n, num_rows), replace=replace, random_state=seed)


def subsample_cases_by_patient(
    cases_for_patient: pd.DataFrame, max_cases: int = 1, seed: int = 42
) -> pd.DataFrame:
    num_cases = cases_for_patient.shape[0]
    if num_cases == 1:
        return cases_for_patient
    elif max_cases < num_cases:
        return cases_for_patient.sample(n=max_cases, replace=False, random_state=seed)
    else:
        return cases_for_patient
