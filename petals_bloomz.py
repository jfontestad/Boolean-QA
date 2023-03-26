import random
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from petals_bloomz import DistributedBloomForCausalLM


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


def evaluate_petals(prompt, test_samples, model_name='bigscience/bloomz', ptuning_mode='ptune'):
    # init the petals tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistributedBloomForCausalLM.from_pretrained(model_name, tuning_mode=ptuning_mode)

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

        inputs = tokenizer.encode(prompt_curr_sample, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=3)
        result = tokenizer.decode(outputs[0])

        petals_ans = result
        # remove previous prompts from the petals's response - take only last line containing ans when \n is encountered
        if '\n' in petals_ans:
            petals_ans = petals_ans[petals_ans.rindex('\n') + 1:]

        petals_pred = any(true_keyword in petals_ans.lower() for true_keyword in ['true', 'yes'])
        print(f"Predicted: {petals_pred}, Actual: {test_sample['answer']}")
        if petals_pred == test_sample['answer']:
            correct_preds += 1

    print(f'Correct Preds: {correct_preds}, Total Preds: {total_preds}')
    return correct_preds / total_preds * 100


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="bigscience/bloomz-560m")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    # load the data and preprocess for prompts and eval data
    samples, test_samples = pre_process(num_shots=8, eval_records=100)
    prompt = create_prompt_str(samples=samples)

    print(" >>>>>>>>  Sending prompts to Petals model ... ")
    accuracy = evaluate_petals(prompt=prompt, test_samples=test_samples, model_name=args.model,
                               ptuning_mode='ptune')
    print(f'Accuracy: {accuracy}%')
