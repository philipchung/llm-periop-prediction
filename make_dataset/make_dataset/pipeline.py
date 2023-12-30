import collections
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from tqdm.auto import tqdm

log = logging.getLogger(name="PipelineLogger")


@dataclass(kw_only=True)
class PipelineStep:
    # Num should be monotonically increasing in pipeline
    num: int
    # String name for step
    name: str
    # Path to location on disk to persist results from `method`
    data_path: Path | None = None
    # Flag that describes whether `data_path` exists on disk
    data_exists: bool | None = None
    # Any function with signature that contains **kwargs and returns the result to be
    # persisted at `data_path`
    method: Callable
    # Dictionary of arguments to be passed into `method`
    arguments: dict[str, Any] = field(default_factory=lambda: {})
    # Result returned by `method` and persisted on disk at `data_path`
    result: Any | None = None
    # Flag that describes whether this pipeline step has been executed
    executed: bool = False

    def __post_init__(self) -> None:
        if self.data_path is not None:
            self.data_exists = self.data_path.exists()
            # If data_path does not exist, we still set it as argument
            # so functions can use it to save data to the path location.
            self.arguments |= {"data_path": self.data_path}


class PipelineMixin(collections.abc.MutableMapping):
    """
    Implements logic for creating and running a pipeline with ability to cache
    results on disk.
    """

    pipeline: list[PipelineStep] = field(init=False)

    def create_pipeline(self, *args, **kwargs) -> Any | None:
        "Overwrite this method with a pipeline that is a list of PipelineStep."
        self.pipeline = []

    def check_if_data_path_exists(self, *args, **kwargs) -> list[PipelineStep]:
        "Check to see if `data_path` exists for each pipeline step."
        for step in self.pipeline:
            step.data_exists = (
                step.data_path.exists() if step.data_path is not None else False
            )
        return self.pipeline

    def resolve_steps(
        self,
        steps: int | list[int] | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
    ) -> list[int]:
        """Resolve different way of specifying pipeline steps.
        Steps can be specified either with `steps` argument or `start_step` & `end_step`.
        If no steps are specified, all pipeline steps are selected.

        Args:
            steps (int | list[int] | tuple[int, int]): Steps should be provided as a list
                of integer indices that specifies which steps.  The indicies do not
                have to be continuous.
            start_step (int | None): Start step (inclusive).
            end_step (int | None): End step (exclusive)

        Returns:
            list[int]: List of integer indices for steps.
        """
        if steps is not None:
            if not isinstance(steps, list):
                steps = [int(steps)]
            step_indices = steps
            return step_indices
        elif start_step is not None and end_step is not None:
            if end_step == -1:
                end_step = len(self.pipeline)
            step_indices = list(range(start_step, end_step))
        else:
            # If no steps specified, add arguments to all pipeline steps
            step_indices = list(range(0, len(self.pipeline)))
        return step_indices

    def add_argument_to_pipeline_steps(
        self,
        steps: int | list[int] | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
        arguments: dict[str, Any] = {},
    ) -> None:
        """Add keyword argument(s) to pipeline steps.

        Args:
            steps (list[int] | tuple[int, int]): Steps should be provided as a list
                of integer indices that specifies which steps.  The indicies do not
                have to be continuous.
            start_step (int | None): Start step (inclusive).
            end_step (int | None): End step (exclusive)
            arguments (dict[str, Any], optional): Arguments dict to add to pipeline step.
                Defaults to {}.

        Raises:
            ValueError: _description_
        """
        step_indices = self.resolve_steps(steps, start_step, end_step)
        # Add Arguments for each step
        for i in step_indices:
            step = self.pipeline[i]
            step.arguments |= arguments

    def run_pipeline(
        self, force: bool = False, run_all: bool = False, **kwargs
    ) -> None:
        """Run the pipeline.
        Overwrite this method with specifics on how to run the pipeline for the class.
        By default, this method will execute `run_pipeline_from_existing` if argument
        `force` is False or `run_pipeline_steps` if `force` is True.

        Args:
            force (bool): If True, forces computation with execution of PipelineStep.method
                rather than loading data from PipelineStep.data_path.
            run_all (bool): If True, forces each step to be run rather than starting from
                last step with existing data.
        """
        if force:
            self.add_argument_to_pipeline_steps(arguments={"force": force, **kwargs})
        if run_all or force:
            self.run_pipeline_steps(**kwargs)
        else:
            self.run_pipeline_last_existing(**kwargs)

    def run_pipeline_last_existing(
        self,
        steps: int | list[int] | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
        **kwargs,
    ) -> None:
        """Run pipeline steps starting from most recent existing `data_path`.
        This skips all pipeline steps prior to the most recent existing."""
        self.check_if_data_path_exists()
        # Get Step indices for each step
        step_indices = self.resolve_steps(steps, start_step, end_step)
        step_indices = sorted(step_indices)
        # Check if data exists for each step
        last_data_existing_step_idx = 0
        for idx, i in enumerate(step_indices):
            step = self.pipeline[i]
            if step.data_exists:
                last_data_existing_step_idx = idx
        # Run pipeline starting from last data existing step
        new_step_indices = step_indices[last_data_existing_step_idx:]
        self.run_pipeline_steps(steps=new_step_indices)

    def run_pipeline_steps(
        self,
        steps: int | list[int] | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
        **kwargs,
    ) -> None:
        "Run pipeline steps."
        step_indices = self.resolve_steps(steps, start_step, end_step)
        print(
            f"{self.__class__.__name__} Pipeline: "
            + " > ".join([f"{step.num}:{step.name}" for step in self.pipeline])
            + f"\nRunning steps: {str(step_indices)}"
        )
        # Execute each step
        for i in (pbar := tqdm(step_indices)):
            step = self.pipeline[i]
            pbar.set_description(
                desc=f"{self.__class__.__name__}: {step.num}:{step.name}"
            )
            self.execute_step(step)
        # Execute when done with all steps
        self.on_run_pipeline_finish()

    def execute_step(self, step: PipelineStep) -> Any:
        try:
            step.result = step.method(**step.arguments)
            step.executed = True
            log.info(f"Executed step num: {step.num}, name: {step.name}")
        except Exception as e:
            raise Exception(
                f"Failed to execute step num: {step.num}, name: {step.name}.\n"
                f"Exception: {e}"
            )
        return step.result

    def on_run_pipeline_finish(self, *args, **kwargs) -> None:
        "Executes immediately after pipeline is finished running."
        pass

    def executed_steps(self) -> list:
        return [step for step in self.pipeline if step.executed]

    def executed_step_names(self) -> list:
        return [step.name for step in self.pipeline if step.executed]

    def get_pipeline_step(self, name: str) -> PipelineStep:
        for step in self.pipeline:
            if step.name == name:
                return step

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
        return "{}, PipelineMixin({})".format(
            super(PipelineMixin, self).__repr__(), self.__dict__
        )
