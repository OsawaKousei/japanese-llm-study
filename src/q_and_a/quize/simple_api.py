import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("src/q_and_a/quize/.env")
API_KEY = os.environ.get("OPENAI-API-KEY")

client = OpenAI(
    api_key=API_KEY,
)
message = [
    {"role": "user", "content": "日本で一番高い山は何ですか？"},
]

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "イギリスのエリザベス1世はチューダー朝の女王ですが、エリザベス2世は何朝の女王?",
        }
    ],
    model="gpt-4o-mini",
    temperature=0.0,
)

print(chat_completion.choices[0].message.content)
