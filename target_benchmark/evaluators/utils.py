import importlib
import inspect
import pkgutil
from typing import Dict, Type

import target_benchmark.tasks
from target_benchmark.tasks.AbsTask import AbsTask


def find_subclasses(package, cls) -> Dict[str, Type[AbsTask]]:
    subclasses = {}
    # Traverse through all modules in the given package
    for finder, name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
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
