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
target_fetaqa = TARGET(("Table Question Answering Task", "fetaqa"))
target_ottqa = TARGET(("Table Question Answering Task", "ottqa"))
target_tabfact = TARGET(("Fact Verification Task", "tabfact"))
```

# Llamaindex

```python
from target_benchmark.retrievers import LlamaIndexRetriever
llamaindex_retriever = LlamaIndexRetriever()
```

## Test with validation sets

```python
results = target_fetaqa.run(llamaindex_retriever, "validation", top_k = 10)
```

```python
results
```

```python
results
```

## Fetaqa Test

```python
results_test = target_fetaqa.run(llamaindex_retriever, "test", top_k = 10)
```

```python
results_test
```

## OTTQA Validation Set

```python
results_llama_ottqa_val = target_ottqa.run(llamaindex_retriever, "validation", top_k=10)
```

## Tabfact Test Set

```python
results_llama_tabfact_test = target_tabfact.run(llamaindex_retriever, "test", top_k=10)
```

```python
results_llama_tabfact_test
```

# Default Naive OAI

```python
from target_benchmark.retrievers import OpenAIEmbedder
oai_embedder = OpenAIEmbedder()
```

## Fetaqa Test

```python
results_oai_fetaqa_test = target_fetaqa.run(oai_embedder, "test", top_k = 10)
```

```python
results_oai_fetaqa_test
```

## OTTQA Val

```python
results_oai_ottqa_val = target_ottqa.run(oai_embedder, "validation", top_k = 10)
```

```python
results_oai_ottqa_val
```

## Tabfact Test

```python
results_oai_tabfact_test = target_tabfact.run(oai_embedder, "test", top_k=10)
```

```python
results_oai_tabfact_test
```

# HNSW OAI

```python
from target_benchmark.retrievers import HNSWOpenAIEmbeddingRetriever
hnsw_oai = HNSWOpenAIEmbeddingRetriever()

```

```python
results_hnsw_oai_fetaqa_test = target_fetaqa.run(hnsw_oai, "test", top_k = 10)
```

```python
results_hnsw_oai_fetaqa_test
```

# OTTQA

```python
from target_benchmark.retrievers import OTTQARetriever
```

```python
ottqa_tfidf = OTTQARetriever(encoding="bm25")
results_ottqa_tfidf_fetaqa_test = target_fetaqa.run(ottqa_tfidf, "test", top_k = 10)
```

```python

```
