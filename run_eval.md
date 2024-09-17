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
get_task_names()
```

```python
from target_benchmark.tasks import TableRetrievalTask
TableRetrievalTask._get_default_dataset_config()
```

```python
target_fetaqa = TARGET(("Table Retrieval Task", "fetaqa"))
target_ottqa = TARGET(("Table Retrieval Task", "ottqa"))
target_tabfact = TARGET(("Table Retrieval Task", "tabfact"))
target_spider = TARGET(("Table Retrieval Task", "spider-test"))
target_infiagentda = TARGET(("Table Retrieval Task", "infiagentda"))
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
results_oai_fetaqa_test = target_fetaqa.run(oai_embedder, "test", top_k = 10, batch_size=100, retrieval_results_file="./oai_fetaqa_test_retrieval_results.jsonl")
```

```python
results_oai_fetaqa_test
```

## OTTQA Val

```python
results_oai_ottqa_val = target_ottqa.run(oai_embedder, "validation", top_k = 10, batch_size=100, retrieval_results_file="./oai_ottqa_val_retrieval_results.jsonl")
```

```python
results_oai_ottqa_val
```

## Tabfact Test

```python
results_oai_tabfact_test = target_tabfact.run(oai_embedder, "test", top_k=10, batch_size=100, retrieval_results_file="oai_tabfact_test_retrieval_results.jsonl")
```

```python
results_oai_tabfact_test
```

## Spider Test

```python
results_oai_spider_test = target_spider.run(oai_embedder, "test", top_k=10, batch_size=100, retrieval_results_file="./oai_spider_test_retrieval_results.jsonl")
```

```python
results_oai_spider_test
```

## Infiagentda Test
DON'T INCLUDE FOR NOW?

```python
results_oai_infiagentda_test = target_infiagentda.run(oai_embedder, "test", top_k=10, batch_size=100, retrieval_results_file="./oai_infiagentda_test_retrieval_results.jsonl")
```

# HNSW OAI

```python
from target_benchmark.retrievers import HNSWOpenAIEmbeddingRetriever
hnsw_oai = HNSWOpenAIEmbeddingRetriever()

```

## Fetaqa Test

```python
results_hnsw_oai_fetaqa_test = target_fetaqa.run(hnsw_oai, "test", top_k = 10, batch_size=100, retrieval_results_file="./hnsw_oai_fetaqa_test_retrieval_results.jsonl")
```

```python
results_hnsw_oai_fetaqa_test
```

## OTTQA Val

```python
results_hnsw_oai_ottqa_test = target_ottqa.run(hnsw_oai, "validation", top_k = 10, batch_size=100, retrieval_results_file="./hnsw_oai_ottqa_val_retrieval_results.jsonl")
```

```python
results_hnsw_oai_ottqa_test
```

## Tabfact Test

```python
results_hnsw_oai_tabfact_test = target_tabfact.run(hnsw_oai, "test", top_k = 10, batch_size=100, retrieval_results_file="./hnsw_oai_tabfact_test_retrieval_results.jsonl")
```

```python
results_hnsw_oai_tabfact_test
```

## Spider Test

```python
results_hnsw_oai_spider_test = target_spider.run(hnsw_oai, "test", top_k = 10, batch_size=100, retrieval_results_file="./hnsw_oai_spider_test_retrieval_results.jsonl")
```

```python
results_hnsw_oai_spider_test
```

# OTTQA

```python
from target_benchmark.retrievers import OTTQARetriever
tfidf_title = OTTQARetriever(encoding="tfidf", withtitle=True)

```

```python
results_tfidf_with_title_fetaqa_test = target_fetaqa.run(tfidf_title, "test", top_k = 10, batch_size=100, retrieval_results_file="./tfidf_fetaqa_test_retrieval_results.jsonl")
```

```python
results_tfidf_with_title_fetaqa_test
```

```python

```
