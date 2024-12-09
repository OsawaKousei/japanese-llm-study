import tiktoken
from datasets import load_dataset
from prompt_maker import SimplePromptMaker

encording = tiktoken.encoding_for_model("gpt-4o-mini")


def caliculate_prompt_cost(
    prompts: list[str],
    num_output_tokens: int = 40,
    model: str = "gpt-4o-mini",
    usd_per_token: float = 0.150 / 1000000,
) -> float:
    encording = tiktoken.encoding_for_model(model)
    total_num_prompt_tokens = 0

    for prompt in prompts:
        total_num_prompt_tokens += len(encording.encode(prompt))

    avg_num_prompt_tokens = total_num_prompt_tokens / len(prompts)
    print(f"Average number of tokens per prompt: {avg_num_prompt_tokens}")

    total_num_output_tokens = num_output_tokens * len(prompts)

    total_cost = (total_num_prompt_tokens + total_num_output_tokens) * usd_per_token

    print(f"Total cost: ${total_cost:.2f}")

    return total_cost


if __name__ == "__main__":
    quiz_dataset = load_dataset("llm-book/aio", split="validation")

    max_answer_length = 0
    for answers in quiz_dataset["answers"]:
        for answer in answers:
            answer_length = len(encording.encode(answer))
            max_answer_length = max(max_answer_length, answer_length)

    print(max_answer_length)

    questions = quiz_dataset["question"]
    simple_prompt_maker = SimplePromptMaker()
    prompts = simple_prompt_maker.run(questions)
    caliculate_prompt_cost(prompts)
