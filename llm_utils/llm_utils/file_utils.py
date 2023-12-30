import json
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(name="FileLogger")


def pickle_save(
    object: Any, path: str | Path, protocol: int = pickle.HIGHEST_PROTOCOL
) -> None:
    with open(path, "wb") as handle:
        pickle.dump(object, handle, protocol=protocol)
        logger.info(f"Saved: {path}")


def pickle_load(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        output = pickle.load(handle)
        logger.info(f"Loaded: {path}")
        return output


def read_pandas(
    path: str | Path, set_index: str | None = None, **kwargs: Any
) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        _df = pd.read_csv(path, low_memory=False, **kwargs)
    elif path.suffix == ".feather":
        _df = pd.read_feather(path, **kwargs)
    elif path.suffix == ".parquet":
        _df = pd.read_parquet(path, **kwargs)
    else:
        raise ValueError("read_pandas cannot read file extension.")

    logger.info(f"Loaded: {path}")

    # Optionally set a column as pandas dataframe index
    _df = _df.set_index(set_index) if set_index is not None else _df
    return _df


def save_pandas(
    df: pd.DataFrame,
    path: str | Path,
    format: str | None = None,
    unset_index: bool = False,
    **kwargs: Any,
) -> None:
    path = Path(path)
    # Make Parent Directory if Not Exist
    path.parent.mkdir(parents=True, exist_ok=True)
    # Optionally convert pandas index to a column in dataframe
    # (feather format does not support pandas indices)
    if unset_index or format == "feather" or path.suffix == ".feather":
        df = df.reset_index(drop=False)
    if format == "csv" or path.suffix == ".csv":
        df.to_csv(path_or_buf=path, **kwargs)
    elif format == "feather" or path.suffix == ".feather":
        df.to_feather(path=path, **kwargs)
    elif format == "parquet" or path.suffix == ".parquet":
        df.to_parquet(path=path, **kwargs)
    else:
        raise ValueError("Must specify file format or suffix for save_pandas.")

    logger.info(f"Saved: {path}")


def read_json(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r") as json_file:
        dictionary = json.load(json_file)
        logger.info(f"Loaded: {path}")
    return dictionary


def save_json(dictionary: dict, path: str | Path) -> None:
    path = Path(path)
    with open(path, "w") as json_file:
        json.dump(dictionary, json_file)
        logger.info(f"Saved: {path}")


def function_with_cache(
    input_data: Any = None,
    function: callable = None,
    cache_path: str | Path = None,
    set_index: str = None,
    force: bool = False,
    data_kind: str = "pandas",
    **kwargs,
) -> pd.DataFrame:
    """Applies `function` to `input_data`.
    Tries to load cached computation to avoid recomputing.

    Args:
        input_data (Any): Data, used as argument to `function`.  If `None`,
            function is called with only `**kwargs`.
        function (callable): Function to apply to `input_data`
        cache_path (str | Path): Location cached data is stored.
        set_index (str): Column to use as index once cached data is loaded.  Only used
            if `data_kind` is `pandas`, ignored for `json`.
        force (bool, optional): If true, force re-run rather than load from disk.
        data_kind (str): Type of data persisted on disk (`pandas`, `json`).
        **kwargs: Any additional keyword arguements are passed directly to `function`.

    Returns:
        pd.DataFrame: Table with function applied
    """
    if (function is None) or (cache_path is None):
        raise ValueError("Must provide arguments `function` and `cache_path`.")

    match data_kind:
        case "pandas":
            return function_with_pandas_cache(
                input_data=input_data,
                function=function,
                cache_path=cache_path,
                set_index=set_index,
                force=force,
                **kwargs,
            )
        case "json":
            return function_with_json_cache(
                input_data=input_data,
                function=function,
                cache_path=cache_path,
                force=force,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unknown `data_kind` {data_kind}.")


def function_with_pandas_cache(
    input_data: Any = None,
    function: callable = None,
    cache_path: str | Path = None,
    set_index: str = None,
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Applies `function` to `input_data`.
    Tries to load cached computation to avoid recomputing.

    Args:
        input_data (Any): Data, used as argument to `function`.  If `None`,
            function is called with only `**kwargs`.
        function (callable): Function to apply to `input_data`
        cache_path (str | Path): Location cached data is stored.
        set_index (str): Column to use as index once cached data is loaded.
        force (bool, optional): If true, force re-run rather than load from disk.
        **kwargs: Any additional keyword arguements are passed directly to `function`.

    Returns:
        pd.DataFrame: Table with function applied
    """
    if (function is None) or (cache_path is None):
        raise ValueError("Must provide arguments `function` and `cache_path`.")

    def fn() -> Any:
        if input_data is not None:
            return function(input_data, **kwargs)
        else:
            return function(**kwargs)

    if not force:
        try:
            _df = read_pandas(cache_path, set_index=set_index)
        except FileNotFoundError:
            _df = fn()
            save_pandas(df=_df, path=cache_path, compression="zstd")
    else:
        _df = fn()
        save_pandas(df=_df, path=cache_path, compression="zstd")
    return _df


def function_with_json_cache(
    input_data: Any = None,
    function: callable = None,
    cache_path: str | Path = None,
    force: bool = False,
    **kwargs,
) -> dict:
    """Applies `function` to `input_data`.
    Tries to load cached computation to avoid recomputing.

    Args:
        input_data (Any): Data, used as argument to `function`.  If `None`,
            function is called with only `**kwargs`.
        function (callable): Function to apply to `input_data`
        cache_path (str | Path): Location cached data is stored.
        force (bool, optional): If true, force re-run rather than load from disk.
        **kwargs: Any additional keyword arguements are passed directly to `function`.

    Returns:
        dict: JSON object represented as python dictionary.
    """
    if (function is None) or (cache_path is None):
        raise ValueError("Must provide arguments `function` and `cache_path`.")

    def fn() -> Any:
        if input_data is not None:
            return function(input_data, **kwargs)
        else:
            return function(**kwargs)

    if not force:
        try:
            _dictionary = read_json(cache_path)
            try:
                _dictionary = json_dict_to_python_dict(_dictionary)
            except Exception:
                logger.warning(
                    "Could not convert JSON to python dict for {_dictionary}"
                )
        except FileNotFoundError:
            _dictionary = fn()
            _dictionary = python_dict_to_json_dict(_dictionary)
            save_json(dictionary=_dictionary, path=cache_path)
    else:
        _dictionary = fn()
        _dictionary = python_dict_to_json_dict(_dictionary)
        save_json(dictionary=_dictionary, path=cache_path)
    return _dictionary


def python_dict_to_json_dict(dictionary: dict) -> dict:
    "Converts non-primative python objects to string form for JSON compatibility."
    json_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, Path):
            json_dict[k] = v.as_posix()
        elif isinstance(v, datetime):
            json_dict[k] = datetime.isoformat(v)
        else:
            json_dict[k] = v
    return json_dict


def json_dict_to_python_dict(dictionary: dict) -> dict:
    "Converts stringified JSON values in dictionary to python objects."
    python_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, bool):
            python_dict[k] = v
        if match_filepath(str(v)):
            python_dict[k] = Path(v)
        elif match_isoformat_date(str(v)):
            python_dict[k] = datetime.fromisoformat(v)
        else:
            python_dict[k] = v
    return python_dict


def match_isoformat_date(text: str) -> bool:
    iso8601_pattern = re.compile(
        r"^(?P<full>((?P<year>\d{4})([/-]?(?P<mon>(0[1-9])|(1[012]))([/-]?(?P<mday>(0[1-9])|([12]\d)|(3[01])))?)?(?:T(?P<hour>([01][0-9])|(?:2[0123]))(\:?(?P<min>[0-5][0-9])(\:?(?P<sec>[0-5][0-9]([\,\.]\d{1,10})?))?)?(?:Z|([\-+](?:([01][0-9])|(?:2[0123]))(\:?(?:[0-5][0-9]))?))?)?))$"  # noqa: E501
    )
    m = iso8601_pattern.match(text)
    return True if m is not None else False


def match_filepath(text: str) -> bool:
    filepath_pattern = re.compile(r"(\/.*?\.[\w:]+)")
    m = filepath_pattern.match(text)
    return True if m is not None else False
