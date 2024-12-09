from async_api import batch_run_chatgpt
from datasets import Dataset, load_dataset
from prompt_maker import InContextPromptMaker, PromptMaker
from tqdm import tqdm


def get_chatpgt_outputs_for_quiz(
    quiz_prompt_maker: PromptMaker,
    quiz_dataset: Dataset,
    batch_size: int,
) -> list[str]:
    output_answers: list[str] = []

    with tqdm(total=len(quiz_dataset)) as pbar:
        for batch in quiz_dataset.iter(batch_size=batch_size):
            prompts = quiz_prompt_maker.run(batch["question"])
            inputs = [[{"role": "user", "content": p}] for p in prompts]

            answers = batch_run_chatgpt(inputs)

            for question, answer in zip(batch["question"], answers):
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print()

            output_answers += answers
            pbar.update(len(answers))

    return output_answers


def calculate_quiz_accuracy(
    output_answers: list[str],
    correct_answers_list: list[list[str]],
) -> float:
    num_correct = 0

    for output_answer, correct_answers in zip(output_answers, correct_answers_list):
        num_correct += int(any(a in output_answer for a in correct_answers))

    return num_correct / len(output_answers)


if __name__ == "__main__":
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

    output_answers = get_chatpgt_outputs_for_quiz(
        quiz_prompt_maker=quiz_prompt_maker,
        quiz_dataset=quiz_dataset,
        batch_size=4,
    )

    accuracy = calculate_quiz_accuracy(
        output_answers=output_answers,
        correct_answers_list=[item["answers"] for item in quiz_dataset],
    )

    print(f"Quiz accuracy: {accuracy:.2f}")  # 0.62
