import pandas as pd


def drop_na_examples(
    preds: pd.Series, targets: pd.Series
) -> tuple[pd.Series, pd.Series, int]:
    _preds, _targets = preds.copy(), targets.copy()
    is_na = _preds.isna() | _targets.isna()
    _preds = _preds.loc[~is_na]
    _targets = _targets.loc[~is_na]
    num_dropped = is_na.sum().item()
    return _preds, _targets, num_dropped
