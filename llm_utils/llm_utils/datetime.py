import datetime
from typing import Optional

import pandas as pd


def string_to_datetime(
    text: str, fmt: Optional[str] = None
) -> Optional[datetime.datetime]:
    "Converts string date to datetime object.  If `NaN`, returns `pd.NaT`."
    if fmt is None:
        fmt = r"%Y-%m-%d %H:%M:%S"
    if pd.isna(text):
        return pd.NaT
    else:
        return datetime.datetime.strptime(text, fmt)


def string_to_datetime_fmt1(text: str) -> Optional[datetime.datetime]:
    fmt = r"%m/%d/%y %H:%M"
    return string_to_datetime(text, fmt)


def string_to_datetime_fmt2(text: str) -> Optional[datetime.datetime]:
    fmt = r"%Y-%m-%d %H:%M:%S"
    return string_to_datetime(text, fmt)


def timedelta2minutes(td: pd.Timedelta | datetime.timedelta) -> int:
    return int(td.total_seconds() / 60)


def timedelta2hours(td: pd.Timedelta | datetime.timedelta) -> int:
    return int(td.total_seconds() / (60 * 60))


def timedelta2days(td: pd.Timedelta | datetime.timedelta) -> float:
    return float(td.total_seconds() / (60 * 60 * 24))
