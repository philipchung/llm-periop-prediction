from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from typing_extensions import Self


class Base(object):
    def __post_init__(self) -> None:
        # just intercept the __post_init__ calls so they
        # aren't relayed to `object`
        # https://stackoverflow.com/questions/59986413/achieving-multiple-inheritance-using-python-dataclasses
        pass


@dataclass(kw_only=True)
class MeanCI(Base):
    """Wrapper for metric results.

    If no bootstrap, the metric value is in `mean`.
    If bootstrapped, the mean metric value is in `mean` and confidence
        interval is in `lower_ci` and `upper_ci`.
    """

    boot_values: Sequence | None = None
    ci_percent: float = 0.95
    # Computed Fields
    mean_value: float | None = None
    lower_ci: float | None = None
    upper_ci: float | None = None
    lower_ci_percentile: float | None = None
    upper_ci_percentile: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # Get CI Percentiles
        self.lower_ci_percentile = (1 - self.ci_percent) / 2
        self.upper_ci_percentile = self.lower_ci_percentile + self.ci_percent
        # Only compute mean + CI if array of boot_values provided
        if self.boot_values is not None:
            boot_values = pd.Series(self.boot_values)
            self.mean_value = boot_values.mean()
            self.lower_ci = boot_values.quantile(q=self.lower_ci_percentile)
            self.upper_ci = boot_values.quantile(q=self.upper_ci_percentile)


@dataclass(kw_only=True)
class BootstrapMetrics(Base):
    """Base class for bootstrap metrics."""

    """List-like sequence of predictions."""
    preds: Sequence
    """List-like sequence of labels."""
    targets: Sequence
    """Name of experimental condition. Output dataframe column axis is given this name."""
    name: str | None = None
    """When `num_bootstrap_samples` is None, each bootstrap iteration will create a
    bootstrap sample size equal to the size of `preds` and `targets`."""
    num_bootstrap_samples: int | None = None
    """Number of bootstrap iterations"""
    num_bootstrap_iterations: int | None = 2500
    """Computed value on whether to bootstrap."""
    bootstrap: bool = field(init=False)
    """Controls randomness in bootstrap sampling and iterations."""
    seed: int = 42
    """Number of worker processes for multiprocessing."""
    num_workers: int = cpu_count()
    metrics: dict = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__post_init__()
        self.bootstrap = (
            False
            if (self.num_bootstrap_iterations is None or self.num_bootstrap_iterations == 0)
            else True
        )

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.metrics, orient="index").rename_axis(
            index="Metric", columns=f"{self.name}"
        )

    def bootstrap_sample(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        "Bootstrap sample (with replacement) `preds` and `targets`"
        rng = np.random.default_rng(self.seed if seed is None else seed)
        # Number of bootstrap samples for a single iteration
        num_samples = (
            len(self.preds) if self.num_bootstrap_samples is None else self.num_bootstrap_samples
        )
        # Create sample index (with replacement)
        indices = list(range(len(self.preds)))
        boot_indices = rng.choice(indices, size=num_samples, replace=True)
        # Get bootstrap sample
        boot_preds = np.array(self.preds)[boot_indices]
        boot_targets = np.array(self.targets)[boot_indices]
        return boot_preds, boot_targets

    def bootstrap_fn(self, metric_fn: Callable) -> list:
        "Calls `metric_fn` multiple times with different bootstrap sampled"
        rng = np.random.default_rng(self.seed)
        bootstrap_seeds = rng.permutation(self.num_bootstrap_iterations)
        boot_pred_target_tuples = [self.bootstrap_sample(s) for s in bootstrap_seeds]
        boot_preds, boot_targets = list(zip(*boot_pred_target_tuples))

        results = []
        for boot_pred, boot_target in zip(boot_preds, boot_targets):
            results += [metric_fn(boot_pred, boot_target)]
        # with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
        #     results = list(executor.map(metric_fn, boot_preds, boot_targets))
        return np.array(results)

    def compute(self) -> pd.DataFrame:
        # Implement custom metric computation logic here, storing all metrics
        # as a dict {"metric_name": value} in the property self.metrics
        return self.to_pandas()


@dataclass(kw_only=True)
class ClassificationMetrics(BootstrapMetrics):
    """Bootstrap classification metrics computation."""

    """List-like sequence of all unique class labels."""
    class_labels: list | None = None
    """Subset of metrics to compute. {"MCC", "Accuracy", "F1/Micro", "F1/Macro", 
    "F1", "Precision", "Recall", "Support"}"""
    include: Sequence[str] | None = ("MCC", "Accuracy", "F1/Micro", "F1")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.class_labels is None:
            self.class_labels = sorted(np.unique(self.preds).tolist())

    def mcc(self, preds: Sequence | None = None, targets: Sequence | None = None) -> float:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        preds = [int(x) if isinstance(x, bool) else x for x in preds]
        targets = [int(x) if isinstance(x, bool) else x for x in targets]
        return matthews_corrcoef(y_true=targets, y_pred=preds)

    def accuracy(self, preds: Sequence | None = None, targets: Sequence | None = None) -> float:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        preds = [int(x) if isinstance(x, bool) else x for x in preds]
        targets = [int(x) if isinstance(x, bool) else x for x in targets]
        return accuracy_score(y_true=targets, y_pred=preds)

    def classwise_precision_recall_f1(
        self, preds: Sequence | None = None, targets: Sequence | None = None
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        preds = [int(x) if isinstance(x, bool) else x for x in preds]
        targets = [int(x) if isinstance(x, bool) else x for x in targets]
        labels = [int(x) if isinstance(x, bool) else x for x in self.class_labels]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true=targets,
            y_pred=preds,
            labels=labels,
            zero_division=0.0,
        )
        return precision, recall, f1, support

    def micro_precision_recall_f1(
        self, preds: Sequence | None = None, targets: Sequence | None = None
    ) -> tuple[float, float, float, float]:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        preds = [int(x) if isinstance(x, bool) else x for x in preds]
        targets = [int(x) if isinstance(x, bool) else x for x in targets]
        labels = [int(x) if isinstance(x, bool) else x for x in self.class_labels]
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true=targets,
            y_pred=preds,
            labels=labels,
            zero_division=0.0,
            average="micro",
        )
        return precision_micro, recall_micro, f1_micro

    def macro_precision_recall_f1(
        self, preds: Sequence | None = None, targets: Sequence | None = None
    ) -> tuple[float, float, float, float]:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        preds = [int(x) if isinstance(x, bool) else x for x in preds]
        targets = [int(x) if isinstance(x, bool) else x for x in targets]
        labels = [int(x) if isinstance(x, bool) else x for x in self.class_labels]
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true=targets,
            y_pred=preds,
            labels=labels,
            zero_division=0.0,
            average="macro",
        )
        return precision_macro, recall_macro, f1_macro

    def compute(self) -> pd.DataFrame:
        # Class-Aggregate Metrics
        if "MCC" in self.include:
            self.metrics |= {
                "Aggregate/MCC/MCC": MeanCI(boot_values=self.bootstrap_fn(self.mcc))
                if self.bootstrap
                else MeanCI(mean_value=self.mcc())
            }
        if "Accuracy" in self.include:
            self.metrics |= {
                "Aggregate/Accuracy/Accuracy": MeanCI(boot_values=self.bootstrap_fn(self.accuracy))
                if self.bootstrap
                else MeanCI(mean_value=self.accuracy())
            }
        if any(x in self.include for x in ["Precision/Micro", "Recall/Micro", "F1/Micro"]):
            if self.bootstrap:
                prf1_micro = self.bootstrap_fn(self.micro_precision_recall_f1)
                precision_micro, recall_micro, f1_micro = [
                    np.squeeze(x) for x in np.hsplit(prf1_micro, 3)
                ]
            else:
                precision_micro, recall_micro, f1_micro = self.micro_precision_recall_f1()
            if "Precision/Micro" in self.include:
                self.metrics |= {
                    "Aggregate/Precision/Micro": MeanCI(boot_values=precision_micro)
                    if self.bootstrap
                    else MeanCI(mean_value=precision_micro)
                }
            if "Recall/Micro" in self.include:
                self.metrics |= {
                    "Aggregate/Recall/Micro": MeanCI(boot_values=recall_micro)
                    if self.bootstrap
                    else MeanCI(mean_value=recall_micro)
                }
            if "F1/Micro" in self.include:
                self.metrics |= {
                    "Aggregate/F1/Micro": MeanCI(boot_values=f1_micro)
                    if self.bootstrap
                    else MeanCI(mean_value=f1_micro)
                }
        if any(x in self.include for x in ["Precision/Macro", "Recall/Macro", "F1/Macro"]):
            if self.bootstrap:
                prf1_macro = self.bootstrap_fn(self.macro_precision_recall_f1)
                precision_macro, recall_macro, f1_macro = [
                    np.squeeze(x) for x in np.hsplit(prf1_macro, 3)
                ]
            else:
                precision_macro, recall_macro, f1_macro = self.macro_precision_recall_f1()

            if "Precision/Macro" in self.include:
                self.metrics |= {
                    "Aggregate/Precision/Macro": MeanCI(boot_values=precision_macro)
                    if self.bootstrap
                    else MeanCI(mean_value=precision_macro)
                }
            if "Recall/Macro" in self.include:
                self.metrics |= {
                    "Aggregate/Recall/Macro": MeanCI(boot_values=recall_macro)
                    if self.bootstrap
                    else MeanCI(mean_value=recall_macro)
                }
            if "F1/Macro" in self.include:
                self.metrics |= {
                    "Aggregate/F1/Macro": MeanCI(boot_values=f1_macro)
                    if self.bootstrap
                    else MeanCI(mean_value=f1_macro)
                }

        # Class-Specific Metrics
        if any(x in self.include for x in ["Precision", "Recall", "F1"]):
            if self.bootstrap:
                classwise_prf1s = self.bootstrap_fn(self.classwise_precision_recall_f1)
                precision, recall, f1, _ = [np.squeeze(x) for x in np.hsplit(classwise_prf1s, 4)]
                _, _, _, support = self.classwise_precision_recall_f1()
            else:
                precision, recall, f1, support = self.classwise_precision_recall_f1()
            for idx, class_name in enumerate(self.class_labels):
                if "Precision" in self.include:
                    self.metrics |= {
                        f"Classwise/Precision/{class_name}": MeanCI(boot_values=precision[idx])
                        if self.bootstrap
                        else MeanCI(mean_value=precision[idx])
                    }
                if "Recall" in self.include:
                    self.metrics |= {
                        f"Classwise/Recall/{class_name}": MeanCI(boot_values=recall[idx])
                        if self.bootstrap
                        else MeanCI(mean_value=recall[idx])
                    }
                if "F1" in self.include:
                    self.metrics |= {
                        f"Classwise/F1/{class_name}": MeanCI(boot_values=f1[idx])
                        if self.bootstrap
                        else MeanCI(mean_value=f1[idx])
                    }
                if "Support" in self.include:
                    self.metrics |= {
                        f"Classwise/Support/{class_name}": MeanCI(
                            mean_value=support[idx],
                            lower_ci=support[idx],
                            upper_ci=support[idx],
                        )
                        if self.bootstrap
                        else MeanCI(mean_value=support[idx])
                    }
        return self.to_pandas()


@dataclass(kw_only=True)
class RegressionMetrics(BootstrapMetrics):
    """Bootstrap regression metrics computation."""

    """Subset of metrics to compute. {"MAE", "MAPE", "MaxError", "RMSE", "MSE"}"""
    include: Sequence[str] | None = ("MAE", "MaxError")

    def mae(self, preds: Sequence | None = None, targets: Sequence | None = None) -> float:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        return mean_absolute_error(y_true=targets, y_pred=preds)

    def mape(self, preds: Sequence | None = None, targets: Sequence | None = None) -> float:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        return mean_absolute_percentage_error(y_true=targets, y_pred=preds)

    def rmse(self, preds: Sequence | None = None, targets: Sequence | None = None) -> float:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        return mean_squared_error(y_true=targets, y_pred=preds, squared=False)

    def mse(self, preds: Sequence | None = None, targets: Sequence | None = None) -> float:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        return mean_squared_error(y_true=targets, y_pred=preds, squared=True)

    def max_error(self, preds: Sequence | None = None, targets: Sequence | None = None) -> float:
        preds = self.preds if preds is None else preds
        targets = self.targets if targets is None else targets
        return max_error(y_true=targets, y_pred=preds)

    def compute(self) -> pd.DataFrame:
        if "MAE" in self.include:
            self.metrics |= {
                "Aggregate/MAE/MAE": MeanCI(boot_values=self.bootstrap_fn(self.mae))
                if self.bootstrap
                else MeanCI(mean_value=self.mae())
            }
        if "MAPE" in self.include:
            self.metrics |= {
                "Aggregate/MAE/MAPE": MeanCI(boot_values=self.bootstrap_fn(self.mape))
                if self.bootstrap
                else MeanCI(mean_value=self.mape())
            }
        if "MaxError" in self.include:
            self.metrics |= {
                "Aggregate/MAE/MaxError": MeanCI(boot_values=self.bootstrap_fn(self.max_error))
                if self.bootstrap
                else MeanCI(mean_value=self.max_error())
            }
        if "RMSE" in self.include:
            self.metrics |= {
                "Aggregate/MSE/RMSE": MeanCI(boot_values=self.bootstrap_fn(self.rmse))
                if self.bootstrap
                else MeanCI(mean_value=self.rmse())
            }
        if "MSE" in self.include:
            self.metrics |= {
                "Aggregate/MSE/MSE": MeanCI(boot_values=self.bootstrap_fn(self.mse))
                if self.bootstrap
                else MeanCI(mean_value=self.mse())
            }
        return self.to_pandas()


def dummy_classifier_metrics(
    targets: Sequence,
    class_labels: Sequence,
    strategy: str = "uniform",
    seed: int = 42,
    name: str = "Baseline",
    num_bootstrap_iterations: int = 2500
) -> pd.DataFrame:
    """Construct dummy classifier that does not use features to make predictions
    and computes metrics. Useful as a baseline representing random predictions.

    Args:
        targets (Sequence): list-like sequence of target labels
        class_labels (Sequence): list of class labels
        strategy (str): See documentation for sklearn.dummy.DummyClassifier.
            "uniform" = randomly predict each class with equal probability
            "stratified" = random predict each class based on its frequency
                in `targets`.
        seed (int, optional): Random seed. Defaults to 42.
        name (str, optional): Name for output dataframe column axis. Defaults to "Baseline".
        num_bootstrap_iterations (int, optional): Number of bootstrap iterations.
            Defaults to 2500.

    Returns:
        pd.DataFrame: Dataframe of bootstrapped metrics.
    """
    # Construct fake inputs which are not actually used by model
    y = targets
    X = np.zeros(shape=y.shape)
    model = DummyClassifier(strategy=strategy, random_state=seed).fit(X, y)
    preds = model.predict(X)
    # Compute metrics
    boot_metrics = ClassificationMetrics(
        preds=preds,
        targets=targets,
        class_labels=class_labels,
        name=name,
        seed=seed,
        num_bootstrap_iterations=num_bootstrap_iterations
    ).compute()
    return pd.DataFrame(boot_metrics)


def dummy_classifier_confusion_matrix_metrics(
    targets: Sequence,
    class_labels: Sequence,
    strategy: str = "uniform",
    seed: int = 42,
    name: str = "Baseline",
    num_bootstrap_iterations: int = 2500
) -> pd.DataFrame:
    """Construct dummy classifier that does not use features to make predictions
    and computes metrics derived from confusion matrix.
    Useful as a baseline representing random predictions.

    Args:
        targets (Sequence): list-like sequence of target labels
        class_labels (Sequence): list of class labels
        strategy (str): See documentation for sklearn.dummy.DummyClassifier.
            "uniform" = randomly predict each class with equal probability
            "stratified" = random predict each class based on its frequency
                in `targets`.
        seed (int, optional): Random seed. Defaults to 42.
        name (str, optional): Name for output dataframe column axis. Defaults to "Baseline".
        num_bootstrap_iterations (int, optional): Number of bootstrap iterations.
            Defaults to 2500.

    Returns:
        pd.DataFrame: Dataframe of bootstrapped metrics.
    """
    # Construct fake inputs which are not actually used by model
    y = targets
    X = np.zeros(shape=y.shape)
    model = DummyClassifier(strategy=strategy, random_state=seed).fit(X, y)
    preds = model.predict(X)
    # Compute metrics from confusion matrix
    boot_metrics = ConfusionMatrixMetrics(
        preds=preds,
        targets=targets,
        class_labels=class_labels,
        name=name,
        seed=seed,
        num_bootstrap_iterations=num_bootstrap_iterations
    ).compute()
    return pd.DataFrame(boot_metrics)


def dummy_regressor_metrics(
    targets: Sequence, strategy: str = "mean", seed: int = 42, name: str = "Baseline"
) -> pd.DataFrame:
    """Construct dummy regressor that does not use features to make predictions
    and computes metrics. Useful as a baseline representing random predictions.

    Args:
        targets (Sequence): list-like sequence of target labels
        strategy (str): See documentation for sklearn.dummy.DummyRegressor.
            "mean" = always predict mean of targets
            "median" = always predict median of targets
        seed (int, optional): Random seed. Defaults to 42.
        name (str, optional): Name for output dataframe column axis. Defaults to "Baseline".

    Returns:
        pd.DataFrame: Dataframe of bootstrapped metrics.
    """
    # Construct fake inputs which are not actually used by model
    y = targets
    X = np.zeros(shape=y.shape)
    model = DummyRegressor(strategy=strategy).fit(X, y)
    preds = model.predict(X)
    # Compute metrics
    boot_metrics = RegressionMetrics(
        preds=preds,
        targets=targets,
        name=name,
        seed=seed,
    ).compute()
    return pd.DataFrame(boot_metrics)


ETA = 1e-12


@dataclass(kw_only=True)
class ConfusionMatrixMetrics(BootstrapMetrics):
    """Confusion matrix & metrics derived from confusion matrix."""

    """List-like sequence of predictions."""
    preds: Sequence
    """List-like sequence of labels."""
    targets: Sequence
    """Class Labels"""
    class_labels: Sequence
    """Name of experimental condition. Output dataframe column axis is given this name."""
    name: str | None = None
    """Computed confusion matrix w/ Predictions on y-axis and Actual on x-axis"""
    confusion_matrix: pd.DataFrame | None = None
    """All Metrics:
    - True Positive Rate (TPR), Recall, Sensitivity
    - True Negative Rate (TNR), Specificity
    - Positive Predictive Value (PPV), Precision
    - Negative Predictive Value (NPV)
    - True Positive (TP)
    - True Negative (TN)
    - False Positive (FP)
    - False Negative (FN)
    """
    metrics: dict = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__post_init__()

    def to_pandas(self) -> pd.DataFrame:
        "Convert metrics to Pandas Dataframe."
        return pd.DataFrame.from_dict(self.metrics, orient="index").rename_axis(
            index="Metric", columns=f"{self.name}"
        )

    def confmat(
        self, preds: Sequence | None = None, targets: Sequence | None = None
    ) -> pd.DataFrame:
        conf_mat = pd.DataFrame(
            data=confusion_matrix(y_true=targets, y_pred=preds, labels=self.class_labels).T,
            index=self.class_labels,
            columns=self.class_labels,
        ).rename_axis(index="Predicted", columns="Actual")
        return conf_mat

    def confmat_metrics(
        self,
        preds: Sequence | None = None,
        targets: Sequence | None = None,
        conf_mat: pd.DataFrame | None = None,
    ) -> dict[str, list]:
        """Compute metrics from confusion matrix.
        Appropriate for both 2x2 and multiclass.
        """
        # Compute Confusion Matrix
        if conf_mat is not None:
            cm = conf_mat
        else:
            cm = self.confmat(preds=preds, targets=targets)
        # Compute metrics derived from confusion matrix
        m = {}
        for class_label in self.class_labels:
            # Get regions of confusion matrix
            class_predictions = cm.loc[class_label, :]
            class_actuals = cm.loc[:, class_label]
            not_class_actual_or_preds = cm.loc[cm.index != class_label, cm.columns != class_label]
            # Compute TP, FN, FP, FN
            class_tp = cm.loc[class_label, class_label]
            class_fn = class_predictions.sum() - class_tp
            class_fp = class_actuals.sum() - class_tp
            class_tn = not_class_actual_or_preds.values.sum()
            # Compute Sensitivity (TPR), Specificity (TNR), PPV, NPV
            class_tpr = class_tp / (class_tp + class_fn + ETA)
            class_tnr = class_tn / (class_tn + class_fp + ETA)
            class_ppv = class_tp / (class_tn + class_fn + ETA)
            class_npv = class_tn / (class_tn + class_fn + ETA)
            # Add to metrics to accumulate
            m |= {
                f"TP/{class_label}": class_tp,
                f"FN/{class_label}": class_fn,
                f"FP/{class_label}": class_fp,
                f"TN/{class_label}": class_tn,
                f"TPR/{class_label}": class_tpr,
                f"TNR/{class_label}": class_tnr,
                f"PPV/{class_label}": class_ppv,
                f"NPV/{class_label}": class_npv,
            }
        return m

    def compute(self) -> Self:
        if self.bootstrap:
            # Compute Confusion Matrix with original preds & targets
            # (note, we don't store the bootstrapped confusion matrices
            # since it is too cumbersome to visualize all of them)
            self.confusion_matrix = self.confmat(preds=self.preds, targets=self.targets)

            # Compute metrics on different bootstrap samples,
            # yielding array of dict of list.
            bootstrap_results = self.bootstrap_fn(self.confmat_metrics)
            # Convert array of dict into dict of list
            bootstrap_results = {
                k: [dic[k] for dic in bootstrap_results] for k in bootstrap_results[0]
            }
            # Compute Mean and CI for each metric
            self.metrics = {k: MeanCI(boot_values=v) for k, v in bootstrap_results.items()}
        else:
            # Compute Confusion Matrix & Metrics without bootstrap
            self.confusion_matrix = self.confmat(preds=self.preds, targets=self.targets)
            self.metrics = self.confmat_metrics(conf_mat=self.confusion_matrix)

        return self.to_pandas()

    def plot(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        annot: bool = True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        square=True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """If existing axes is passed in, will generate plot on that axes.
        Otherwise will generate new figure and axes with plot."""
        if fig is None and ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        ax = sns.heatmap(
            data=self.confusion_matrix,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            cbar=cbar,
            square=square,
            ax=ax,
            **kwargs,
        )
        ax.set_title(self.name if title is None else title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return fig, ax
