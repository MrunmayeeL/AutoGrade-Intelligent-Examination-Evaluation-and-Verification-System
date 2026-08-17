from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1"
)

models = client.models.list()

for model in models.data:
    print(model.id)