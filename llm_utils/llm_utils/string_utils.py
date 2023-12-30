import hashlib


def snakecase_to_camelcase(snake_string: str) -> str:
    return snake_string.title().replace("_", "")


def camelcase_to_snake(camel_string: str) -> str:
    return "".join(
        ["_" + c.lower() if c.isupper() else c for c in camel_string]
    ).lstrip("_")


def create_hash(data: str, digest_size: int = 8) -> str:
    """Creates unique hash value using BLAKE2b algorithm.

    Args:
        data (str): input string used to generate hash
        digest_size (int): The hash as a hexadecimal string of double length of `digest_size`.

    Returns:
        str: Hash value.
    """
    data = data.encode("UTF-8")
    digest = hashlib.blake2b(data, digest_size=digest_size).hexdigest()
    return digest


def create_uuid(data: str, output_format: str = "T-SQL") -> str:
    """Creates unique UUID using BLAKE2b algorithm.

    Args:
        data (str): input data used to generate UUID
        output_format (str): Output format.
            `raw` results in raw 32-char digest being returned as UUID.
            `T-SQL` results in 36-char UUID string (32 hex values, 4 dashes)
                delimited in the same style as `uniqueidentifier` in T-SQL databases.

    Returns:
        Formatted UUID.
    """
    data = data.encode("UTF-8")  # type: ignore
    digest = hashlib.blake2b(data, digest_size=16).hexdigest()  # type: ignore
    if output_format == "raw":
        return digest
    elif output_format == "T-SQL":
        uuid = (
            f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:]}"
        )
        return uuid
    else:
        raise ValueError(
            f"Unknown argument {output_format} specified for `return_format`."
        )
