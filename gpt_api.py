import os
import openai
import random
import argparse
import numpy as np
from datasets import load_dataset

openai.api_key = os.getenv("OPENAI_API_KEY")


# response = openai.Completion.create(
#   model="davinci-instruct-beta",
#   prompt="Correct this English text: Today I have went to the store to to buys some many bottle of water.\n\nToday I have gone to the store to buy some water bottles.",
#   temperature=0.7,
#   max_tokens=256,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )


def pre_process(num_shots=8):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    # random sampling num_shots / 2 true and num_shots / 2 false prompts
    print(f"Randomly selecting {num_shots} prompts from dataset...")
    true_samples, ct_true = [], 0
    false_samples, ct_false = [], 0
    rand_idx = np.random.randint(0, len(dataset['train']), num_shots * 2)
    for idx in rand_idx:
        sample = dataset['train'][int(idx)]
        if sample['answer'] and ct_true < num_shots // 2:
            true_samples.append(sample)
            ct_true += 1
        elif sample['answer'] is False and ct_false < num_shots // 2:
            false_samples.append(sample)
            ct_false += 1

    # randomly shuffling samples to avoid recency bias
    samples = true_samples + false_samples
    random.shuffle(samples)

    return samples


def create_prompt_str(samples):
    prompt = ""
    for samp in samples:
        prompt += "Passage: "
        prompt += samp['passage']
        prompt += "\n"
        prompt += "Question: "
        prompt += samp['question'] + "?"
        prompt += "\n"
        prompt += "Answer: "
        prompt += "True" if samp['answer'] else "False"
        prompt += "\n\n"
    return prompt


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    # load the data and models
    samples = pre_process(num_shots=8)
    prompt = create_prompt_str(samples)
    print(prompt)

    print(" >>>>>>>>  Sending prompt to GPT-3 API ... ")
