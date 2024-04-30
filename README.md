# TARGET: Table Retrieval for Generative Tasks Benchmark

## Set Up Target

1. Clone this repository

2. create a python venv with `python -m venv <your venv name>`. Alteratively, create a conda environment.

3. run `pip install -r requirements.txt` to install dependencies.

4. in your .bashrc file (or other shell configuration files), add the following environment variables:

```
export AZURE_OPENAI_API_KEY=<your azure openai api key>
export AZURE_OPENAI_ENDPOINT="https://target-openai-canada-east.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="target-gpt4-32k"
```

Our Azure OpenAI resources are located at [this link](https://portal.azure.com/#@epicdatalaboutlook.onmicrosoft.com/resource/subscriptions/6bbef843-a5c2-4407-abae-0eec3d8123ca/resourceGroups/learning-tables/providers/Microsoft.CognitiveServices/accounts/target-openai-canada-east/overview).
