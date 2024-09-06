---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: target
    language: python
    name: python3
---

```python
from target_benchmark.retrievers import LlamaIndexRetriever
```

```python
from target_benchmark.evaluators import TARGET, get_task_names
```

```python
get_task_names()
```

```python
from target_benchmark.tasks import QuestionAnsweringTask
QuestionAnsweringTask._get_default_dataset_config()
```

```python
target = TARGET(("Table Question Answering Task", "fetaqa"))
llamaindex_retriever = LlamaIndexRetriever()
```

```python
results = target.run(llamaindex_retriever, "validation")
```

```python
results
```

```python

```
