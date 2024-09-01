# TARGET: Table Retrieval for Generative Tasks Benchmark

## Set Up Target

1. Clone this repository

2. create a python venv with `python -m venv <your venv name>`. Alteratively, create a conda environment.

3. run `pip install -e .` to install dependencies.

4. in your .bashrc file (or other shell configuration files), add the following environment variables:

```
export OPENAI_API_KEY=<your azure openai api key>
```

5. Log into the target benchmark huggingface account with huggingface cli to access the private datasets.

```
huggingface-cli login
```

This will prompt you to input a token. Ask one of the other members on the team for this token!

6. to run one of the tests in the test folder, run the following command in the root directory of the project. I'm using the evaluator_basic_test as an example:

```
python -m tests.evaluator_basic_test
```
