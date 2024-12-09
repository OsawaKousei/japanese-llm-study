from abc import ABCMeta, abstractmethod

from async_api import batch_run_chatgpt


class PromptMaker(metaclass=ABCMeta):
    @abstractmethod
    def run(self, questions: list[str]) -> list[str]:
        pass


class SimplePromptMaker(PromptMaker):
    def run(self, questions: list[str]) -> list[str]:
        return [
            "あなたは今からクイズに答えてもらいます。"
            "問題を与えますので、その回答のみを簡潔に出力してください。\n"
            f"問題: {q}\n"
            "回答: "
            for q in questions
        ]


class InContextPromptMaker(PromptMaker):

    def __init__(self, examples: list[tuple[str, str]]):
        self.prompt = (
            "あなたは今からクイズに答えてもらいます。"
            "問題を与えますので、その回答のみを簡潔に出力してください。\n"
        )
        for question, answer in examples:
            self.prompt += f"問題: {question}\n回答: {answer}\n"

    def run(self, questions: list[str]) -> list[str]:
        prompts = [self.prompt + f"問題: {q}\n回答: " for q in questions]
        return prompts


if __name__ == "__main__":
    maker = SimplePromptMaker()
    questions = ["1 + 1 = ?", "2 + 2 = ?", "3 + 3 = ?"]
    prompts = maker.run(questions)
    for prompt in prompts:
        print(prompt)

    answers = batch_run_chatgpt(
        [[{"role": "user", "content": p}] for p in maker.run(questions)],
        temperature=0.0,
    )
    for answer in answers:
        print(answer)

    from datasets import load_dataset

    quiz_dataset = load_dataset("llm-book/aio", split="validation")

    num_examples = 3
    in_context_examples = [quiz_dataset[i] for i in range(num_examples)]

    print("In-context examples:")
    for example in in_context_examples:
        print(f"Question: {example['question']}")
        print(f"Answers: {example['answers']}")
        print()

    q_and_a_list = [(e["question"], e["answers"][0]) for e in in_context_examples]

    quiz_prompt_maker = InContextPromptMaker(q_and_a_list)

    prompts = quiz_prompt_maker.run([e["question"] for e in in_context_examples])
    for prompt in prompts:
        print(prompt)
