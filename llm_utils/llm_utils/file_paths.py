import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class DataPaths(collections.abc.MutableMapping):
    project_dir: str | Path | None = None
    raw_adt: str | Path = field(init=False)
    raw_case: str | Path = field(init=False)
    data_dir: str | Path | None = None
    data_version: int = 3

    def __post_init__(self) -> None:
        "Called upon object instance creation."
        # Set Default Data Directories and Paths
        if self.project_dir is None and self.data_dir is None:
            raise ValueError("Must supply `data_dir` or `project_dir`.")

        if self.project_dir is None:
            self.project_dir = Path(__file__).parent.parent
        else:
            self.project_dir = Path(self.project_dir)
        if self.data_dir is None:
            self.data_dir = self.project_dir / "data" / f"v{self.data_version}"
        else:
            self.data_dir = Path(self.data_dir)

        # Raw Directory
        self.raw = self.data_dir / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        # Interim Directory
        self.interim = self.data_dir / "interim"
        self.interim.mkdir(parents=True, exist_ok=True)
        # Processed Directory
        self.processed = self.data_dir / "processed"
        self.processed.mkdir(parents=True, exist_ok=True)
        # Chat Completions Directory
        self.chat_completions = self.data_dir / "chat_completions"
        self.chat_completions.mkdir(parents=True, exist_ok=True)

    def register(self, path: str | Path, name: str | None = None) -> Path:
        "Register the path as an attribute to the DataPaths object and return path."
        path = Path(path)
        if name is None:
            name = path.stem
        setattr(self, name, path)
        return path

    def __setitem__(self, key, value) -> None:
        self.__dict__[key] = value

    def __getitem__(self, key) -> Any:
        return self.__dict__[key]

    def __delitem__(self, key) -> None:
        del self.__dict__[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self) -> str:
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)

    def __repr__(self) -> str:
        """echoes class, id, & reproducible representation in the REPL"""
        return "{}, DataPaths({})".format(
            super(DataPaths, self).__repr__(), self.__dict__
        )
