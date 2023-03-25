import os
import openai
import random
import argparse
import numpy as np
from datasets import load_dataset

openai.api_key = os.getenv("OPENAI_API_KEY")


def pre_process(num_shots=8, eval_records=30):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    # random sampling num_shots / 2 true and num_shots / 2 false prompts
    print(f"Randomly selecting {num_shots} prompts from dataset ...")
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

    # random sampling eval_records prompts from the validation data
    print(f"Randomly selecting {eval_records} prompts from validation dataset ...")
    rand_idx = np.random.randint(0, len(dataset['validation']), eval_records)
    test_samples = [dataset['validation'][int(idx)] for idx in rand_idx]

    return samples, test_samples


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
        prompt += "\n"
    return prompt


def evaluate_gpt3(prompt, test_samples):
    correct_preds = 0
    total_preds = len(test_samples)
    for test_sample in test_samples:
        prompt_curr_sample = prompt
        prompt_curr_sample += "Passage: "
        prompt_curr_sample += test_sample['passage']
        prompt_curr_sample += "\n"
        prompt_curr_sample += "Question: "
        prompt_curr_sample += test_sample['question'] + "?"
        prompt_curr_sample += "\n"
        prompt_curr_sample += "Answer:"

        # call the OpenAI GPT API on the prompt along with the current to pred the ans
        response = openai.Completion.create(
            model="davinci",
            prompt=prompt_curr_sample,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        gpt_ans = response['choices'][0]['text']
        # remove extra prompts from the gpt's response - end when \n is encountered
        if '\n' in gpt_ans:
            gpt_ans = gpt_ans[:gpt_ans.index('\n')]

        gpt_pred = 'true' in gpt_ans.lower()
        print(f"Predicted: {gpt_pred}, Actual: {test_sample['answer']}")
        if gpt_pred == test_sample['answer']:
            correct_preds += 1

    print(f'Correct Preds: {correct_preds}, Total Preds: {total_preds}')
    return correct_preds / total_preds * 100


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    # load the data and preprocess for prompts and eval data
    samples, test_samples = pre_process(num_shots=8, eval_records=30)
    prompt = create_prompt_str(samples=samples)

    print(" >>>>>>>>  Sending prompts to GPT-3 API ... ")
    accuracy = evaluate_gpt3(prompt=prompt, test_samples=test_samples)
    print(f'Accuracy: {accuracy}%')
