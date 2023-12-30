from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressBar(Progress):
    """Progress Bar for tasks with clearly defined endpoint.
    Example Usage:
    ```python
    with ProgressBar() as p:
        for i in p.track(range(5)):
            print(i)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.columns = (
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )


class IndeterminateProgressBar(Progress):
    """Progress Bar for tasks with no clearly defined endpoint.
    Example Usage:
    ```python
    with IndeterminateProgressBar() as p:
        for i in p.track(range(5)):
            print(i)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.columns = (
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            TimeElapsedColumn(),
        )


class MultiProgresBar(Progress):
    """Multiple progress bars with mixing of determinate and indeterminate task bars.

    Example Usage:

    ```python
    import time

    with MultiProgressBar() as p:
        # Add first task progress bar
        taskone = p.add_task("task1", kind="determinate")
        for i in p.track(range(5), task_id=taskone):
            # Add nested task spinner
            tasktwo = p.add_task("task2", kind="indeterminate")
            iterations = 0
            while iterations < 5:
                time.sleep(0.5)
                iterations += 1
            # Remove nested task spinner
            p.remove_task(tasktwo)
    ```

    """

    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("kind") in ("determinate", None):
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                )
            if task.fields.get("kind") == "indeterminate":
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    SpinnerColumn(),
                    TimeElapsedColumn(),
                )
            yield self.make_tasks_table([task])
