import asyncio
import os
from typing import Awaitable, Callable, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv("src/q_and_a/quize/.env")
API_KEY = os.environ.get("OPENAI-API-KEY")

client = AsyncOpenAI(
    api_key=API_KEY,
)

T = TypeVar("T")


async def retry_on_error(
    openai_call: Callable[[], Awaitable[T]],
    max_num_trials: int = 3,
    first_wait_time: int = 10,
) -> T | None:
    for i in range(max_num_trials):
        try:
            return await openai_call()
        except Exception as e:
            if i == max_num_trials - 1:
                raise
            print(f"Error occurred: {e}")
            wait_time_seconds = first_wait_time * (2**i)
            await asyncio.sleep(wait_time_seconds)

    return None


async def _async_batch_run_chatgpt(
    messages_list: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    stop: str | list[str] | None,
) -> list[str]:
    tasks = [
        retry_on_error(
            openai_call=lambda x=ms: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        )
        for ms in messages_list
    ]

    comletions = await asyncio.gather(*tasks)
    return [c.choices[0].message.content for c in comletions]


def batch_run_chatgpt(
    messages_list: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    stop: str | list[str] | None = None,
) -> list[str]:
    return asyncio.run(
        _async_batch_run_chatgpt(messages_list, temperature, max_tokens, stop)
    )


if __name__ == "__main__":
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of the United States?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    ]

    completions = batch_run_chatgpt(
        messages_list=messages_list,
        temperature=0.7,
        max_tokens=100,
        stop=["\n", "In conclusion,"],
    )

    print(completions)
