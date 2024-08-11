import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    messages=[
        {
            "role": "system",
            "content": "You are an Xbox customer support agent whose primary goal is to help users with issues they are experiencing with their Xbox devices. You are friendly and concise. You only provide factual answers to queries, and do not provide answers that are not related to Xbox.",
        },
        {"role": "user", "content": "How much is a PS5?"},
    ],
    temperature=0,
    max_tokens=350,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
)

print(response.to_json())
