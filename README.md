# TARGET: Table Retrieval for Generative Tasks Benchmark

## Set Up TARGET

**Install via pip**

```python
pip install target_benchmark
```

**Install from source**

```shell
git clone https://github.com/target-benchmark/target.git

cd target

pip install -e .
```

If you want to use the default generators for generating downstream task answers, you need to add your OpenAI API key as one of the environment variables:

```shell
export OPENAI_API_KEY=<your openai api key>
```

## Features
- run evaluations on TARGET's baseline retrievers
- implement your own custom retrievers and generators
- create your own custom task

## Usage Example: Evaluate Baseline Retriever

Let's see how we can run evaluation on a baseline retriever. We'll use LlamaIndex as an example:

```python
from target_benchmark.evaluators import TARGET, get_task_names
# you can run `get_task_names()` to get all available tasks
from target_benchmark.retrievers import LlamaIndexRetriever

# specify a task and a dataset to run evaluations on.
target_fetaqa = TARGET(("Table Retrieval Task", "fetaqa"))
# create a new retriever object
llamaindex_retriever = LlamaIndexRetriever()
# run the evaluation!
performance = target_fetaqa.run(retriever=llamaindex_retriever, split="test", top_k=10)

# if you'd like, you can also persist the retrieval and downstream generation results
performance = target_fetaqa.run(retriever=llamaindex_retriever, split="test", top_k=10, retrieval_results_file="./retrieval.jsonl", downstream_results_file="./downstream.jsonl")
```

## Create Retrievers

TARGET offers a simple interface for creating custom retrievers. You can either inherit from the `AbsCustomEmbeddingRetriever` class or the `AbsStandardEmbeddingRetriever` class.

### Inheriting from `AbsCustomEmbeddingRetriever` Class

Inherit from this class if your retriever uses a **custom format for embedding tables** (e.g., specific directory structures or file types). The TARGET evaluator assumes that your retriever will manage the persistence of embeddings during evaluation.

**When to Use This Class**

- **Custom Embedding Formats**: Your retriever requires specific storage formats for embeddings.
- **Self-Managed Persistence**: You handle the storage and retrieval of embeddings yourself.

**Implementing the Required Methods**

To use this class, implement the following two methods:

1. **`embed_corpus`**
	- **Parameters**:
     - `dataset_name`: Identifier for the dataset.
     - `corpus`: The dataset to embed, provided as an iterable of dictionaries.

2. **`retrieve`**
   - **Parameters**:
     - `query`: The user's query string.
     - `dataset_name`: Identifier for the dataset.
     - `top_k`: Number of top results to return.
   - **Returns**: A list of tuples, where each tuple contains `(database_id, table_id)` of a retrieved table.

```python
from target_benchmark.retrievers import AbsCustomEmbeddingRetriever
class YourRetriever(AbsCustomEmbeddingRetriever):
    # you can specify a `expected_corpus_format`
    # (ie nested array, dictionary, dataframe, etc.),
    # the corpus tables will be converted to this format
    # before passed into the `embed_corpus` function.
    def __init__(self, expected_corpus_format: str = "nested array", **kwargs):
        super().__init__(expected_corpus_format=expected_corpus_format)

    # returns a list of tuples, each being (database_id, table_id) of the retrieved table
    def retrieve(self, query: str, dataset_name: str, top_k: int) -> List[Tuple]:
        pass

    # returns nothing since the embedding persistence is dealt with within this function.
    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        pass
```


### Inherit from `AbsStandardEmbeddingRetriever` Class
Inherit from this class if your retriever returns a vector embedding for each table and query. It automatically handles vector data storage using an **in-memory Qdrant vector database**, so data is **not persisted across calls to `TARGET.run`**. (support for persistence across evaluation runs will be included in the future)

**Why Inherit from This Class?**

Consider inheriting from this class instead of `AbsCustomEmbeddingRetriever` if:

- **Simple Embedding Output**: Your retriever outputs embeddings as vectors (lists of floats).
- **No Special Storage Needs**: Your retrieval system doesn't require specific persistence formats or folder structures.

**How to Use This Class**

To inherit from this class, you need to implement two methods:

1. **`embed_query`**: Returns an embedding vector for a given query.
   - **Parameters**:
     - `query`: The user's query string.
     - `dataset_name`: Identifier for the dataset.
   - **Returns**: embedding of query in a numpy array

2. **`embed_corpus`**: Returns embedding vectors for each item in the corpus (e.g., tables or documents).
	- **Parameters**:
     - `dataset_name`: Identifier for the dataset.
     - `corpus_entry`: An entry in the corpus dataset.
    - **Returns**: embedding of corpus entry in a numpy array

```python
from target_benchmark.retrievers import AbsStandardEmbeddingRetriever
class YourRetriever(AbsStandardEmbeddingRetriever):
    def __init__(self, expected_corpus_format: str = "nested array", **kwargs):
        super().__init__(expected_corpus_format=expected_corpus_format)

    #return the embeddings for the query as a numpy array
    def embed_query(self, query: str, dataset_name: str,) -> np.ndarray:
        pass

    # returns embedding of the passed in table as a numpy array
    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> np.ndarray:
        pass
```

### Note on `corpus` and `corpus_entry` Formatting

TARGET provides standardized formatting for the corpus datasets. More specifically, each TARGET corpus dataset includes the following columns:
- **database_id (str)**: database that the table belongs to.
- **table_id (str)**: table's identifier.
- **table**: the actual table contents. default format is nested array, but you can specify the expected format to be `dictionary` or `dataframe` in your retriever's constructor. Tables are automatically converted to the expected format before passed into the `embed_corpus` function.
- **context (dict)**: any metadata associated with the table. for example, text-2-sql datasets' context often include primary and foreign key information.


Both retriever classes' `embed_corpus` function takes in corpus information.
- `AbsStandardEmbeddingRetriever`: `corpus_entry` is a single entry within the corpus dataset. for example, it may look like this:
```python
{
    "database_id": "0",
    "table_id": "totto_source/train_json/example-10461.json",
    "table": <table contents in the retriever's expected format>,
    "context": {"table_page_title": "1982 Illinois gubernatorial election",
  "table_section_title": "Results"},
}
```
- `AbsCustomEmbeddingRetriever`: `corpus` is an iterable of dictionaries. Each dictionary contains a batch of corpus entries. For example:
```python
{
    "database_id": ["0", "1"],
    "table_id": ["Serbia_at_the_European_Athletics_Championships_2", "List_of_University_of_Texas_at_Austin_alumni_20"],
    "table": [<table content>, <table content>],
    "context": [{"section_title": "Indoor -- List of Medalists"}, {"section_title": "Literature , writing , and translation"}],
}
```
The length of the lists will correspond to the batch size specified when calling `TARGET.run`.

## Create Custom Generators

Creating your customer generators for downstream tasks is also straightforward. You only need to implement one function,
- **`generate`**
	- **Parameters**:
     - `table_str`: String of the retrieved table contents.
     - `query`: The natural language query.


```python
from target_benchmark.generators import AbsGenerator
class YourCustomGenerator(AbsGenerator):
    # returns the answer to the query
    def generate(self, table_str: str, query: str) -> str:
        pass
```

To use your generators, first create a task object, and pass the generator into the task object:

```python
from target_benchmark.evaluators import TARGET
from target_benchmark.tasks import QuestionAnsweringTask
qa_task = QuestionAnsweringTask(task_generator=YourGenerator())
target_evaluator = TARGET(downstream_tasks=qa_task)
```
Note that here instead of specifying the task by its name, we are passing in a task object instead with the generator set to our created custom generator.
