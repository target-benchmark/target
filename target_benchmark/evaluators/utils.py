import importlib
import inspect
import pkgutil
from typing import Dict, List, Type

import target_benchmark.tasks
from target_benchmark.dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from target_benchmark.dataset_loaders.NeedleInHaystackDataLoader import (
    NeedleInHaystackDataLoader,
)
from target_benchmark.tasks.AbsTask import AbsTask


def construct_dataset_name_for_eval(non_nih_dataset: AbsDatasetLoader, nih_datasets: List[NeedleInHaystackDataLoader]) -> str:
    """

    Construct a unique identifier for the corpus with nih datasets added.
    Uses the name of the NIH datasets and the number of tables included from the NIH datasets.

    """

    dataset_name = non_nih_dataset.dataset_name
    for nih_dataset in nih_datasets:
        dataset_name += f"_{nih_dataset.dataset_name}"
        if nih_dataset.num_tables:
            dataset_name += f"_{nih_dataset.num_tables}"
        else:
            dataset_name += "_all"
    return dataset_name


def corpus_gen(dataloaders: List[AbsDatasetLoader], corpus_format: str, batch_size: int):
    for dataloader in dataloaders:
        yield from dataloader.get_corpus_iter(corpus_format, batch_size)


def find_subclasses(package, cls) -> Dict[str, Type[AbsTask]]:
    subclasses = {}
    # Traverse through all modules in the given package
    for finder, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(name)
            # Check all classes defined in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if this class is a subclass of the given class and not the class itself
                if issubclass(obj, cls) and obj is not cls:
                    subclasses[obj.get_default_task_name()] = obj
        except ImportError:
            # Handle cases where a module may not be importable
            continue
    return subclasses


def find_tasks() -> Dict[str, Type[AbsTask]]:
    return find_subclasses(target_benchmark.tasks, AbsTask)


def get_task_names() -> list[str]:
    return list(find_tasks().keys())
