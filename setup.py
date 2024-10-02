from setuptools import find_packages, setup

setup(
    name="target_benchmark",
    version="0.1.0",
    author="Xingyu Ji, Madelon Hulsebos",
    author_email="jixy2012@berkeley.edu, madelon@berkeley.edu",
    description="Table Retrieval for Generative Tasks Benchmark",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/target-benchmark/target",
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.19.0",
        "evaluate>=0.4.2",
        "func-timeout>=4.3.5",
        "hnswlib>=0.8.0",
        "langchain>=0.1.16",
        "langchain-community>=0.0.34",
        "langchain-core>=0.1.45",
        "langchain-openai>=0.0.8",
        "langchain-text-splitters>=0.0.1",
        "llama-index>=0.10.58",
        "nltk>=3.8.1",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "pexpect>=4.9.0",
        "pydantic>=2.7.4",
        "python-dateutil>=2.9.0",
        "python-dotenv>=1.0.1",
        "qdrant_client>=1.9.1",
        "regex>=2023.10.3",
        "rouge_score>=0.1.2",
        "sacrebleu>=2.4.2",
        "scikit_learn>=1.3.0",
        "scipy>=1.13.0",
        "setuptools>=69.1.1",
        "spacy>=3.7.4",
        "transformers>=4.41.2",
        "tqdm>=4.65.0",
    ],
    tests_require=[
        "unittest",
    ],
    # TODO: lazy importing for extras
    # extras_require={
    #     "llamaindex_retriever": [
    #         "llama-index>=0.10.58"
    #     ],  # Optional dependencies for llama index
    #     "ottqa_retriever": [
    #         "nltk>=3.8.1",
    #         "pexpect>=4.9.0",
    #         "spacy>=3.7.4",
    #     ],
    # },
)
