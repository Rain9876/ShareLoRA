from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer
import transformers
import evaluate
import re
import random

def stratified_sample(dataset, frac=0.1):
    # Group indices by subtask_name
    groups = {}
    for idx, example in enumerate(dataset):
        key = example['subtask_name']
        groups.setdefault(key, []).append(idx)

    selected_indices = []
    # For each subtask, sample 10% (at least one example if the group is non-empty)
    for subtask, indices in groups.items():
        sample_size = max(1, int(len(indices) * frac))
        selected_indices.extend(random.sample(indices, sample_size))

    # Create a new dataset with the selected indices
    return dataset.select(selected_indices)

def extract_final_answer(answer_text):
    """
    Extracts the final answer letter from an answer explanation.
    It searches for a pattern such as "The answer is (B)" or "The answer is B".
    """
    match = re.search(r"The answer is\s*\(?\s*([A-Z])\s*\)?", answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        return None


def format_questions(text):
    """
    Processes a multi-question text where each question starts with "Q:" and its
    answer explanation follows a line with "A:". The function extracts the final answer
    from each explanation and returns the formatted text with the original question
    text and the answer (e.g., "A: B").
    """
    # Split the text by lines that start with "Q:" (using a lookahead for newlines)
    question_blocks = re.split(r'\n(?=Q:)', text.strip())
    formatted_blocks = []

    for block in question_blocks:
        # Split each block into question and answer parts at the first occurrence of "A:"
        parts = block.split("A:")
        if len(parts) < 2:
            # If there is no "A:" marker, leave the block unchanged
            formatted_blocks.append(block)
            continue

        question_text = parts[0].rstrip()
        answer_explanation = parts[1].strip()
        final_answer = extract_final_answer(answer_explanation)

        if final_answer:
            formatted_block = question_text + "\n\nA: " + final_answer
        else:
            formatted_block = question_text + "\n\nA: "

        formatted_blocks.append(formatted_block)

    return "\n\n".join(formatted_blocks)



def load_eval_dataset(args, tokenizer):
    eval_dataset = None
    label_idx = None

    if args.mmlu_dataset == "mmlu_pro":
        mmlu_pro = load_dataset("meta-llama/Llama-3.1-8B-evals", "Llama-3.1-8B-evals__mmlu_pro__details")
        eval_dataset = mmlu_pro['latest'].map(lambda x: {
            'input': format_questions(x['input_final_prompts'][0]),
            'output': x['input_correct_responses'][0].strip('\"'),
        })

        label_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
            tokenizer("E", add_special_tokens=False).input_ids[0],
            tokenizer("F", add_special_tokens=False).input_ids[0],
            tokenizer("G", add_special_tokens=False).input_ids[0],
            tokenizer("H", add_special_tokens=False).input_ids[0],
            tokenizer("I", add_special_tokens=False).input_ids[0],
            tokenizer("J", add_special_tokens=False).input_ids[0],
        ]

    elif args.mmlu_dataset == "mmlu":
        mmlu = load_dataset("meta-llama/Llama-3.1-8B-evals", "Llama-3.1-8B-evals__mmlu__details")
        eval_dataset = mmlu['latest'].map(lambda x: {
            'input': x['input_final_prompts'][0][:-1],
            'output':x['input_correct_responses'][0].replace('Answer: ', ''),
        })

        label_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]

    elif args.mmlu_dataset ==  "arc":
        arc = load_dataset("meta-llama/Llama-3.1-8B-evals", "Llama-3.1-8B-evals__arc_challenge__details")
        eval_dataset = arc['latest'].map(lambda x: {
            'input': x['input_final_prompts'][0][:-1],
            'output': x['input_correct_responses'][0].strip('\"')[-1:],
        })

        label_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]

    elif args.mmlu_dataset == "boolq":
        # label = {"yes": 1.0, "no": 0.0}
        boolq = load_dataset("meta-llama/Llama-3.1-8B-evals", 'Llama-3.1-8B-evals__boolq__details')
        eval_dataset = boolq['latest'].map(lambda x: {
            'input': x['input_final_prompts'][0].rstrip('yesno'),
            # 'output': label[x['input_correct_responses'][0].replace('Answer: ', '')],
            'output': x['input_correct_responses'][0].replace('Answer: ', ''),

        })

        label_idx = [
            tokenizer("yes", add_special_tokens=False).input_ids[0],
            tokenizer("no", add_special_tokens=False).input_ids[0],
        ]

    elif args.mmlu_dataset == "winogrande":
        wino = load_dataset("meta-llama/Llama-3.1-8B-evals", 'Llama-3.1-8B-evals__winogrande__details')
        eval_dataset = wino['latest'].map(lambda x: {
            'input': x['input_final_prompts'][0][:-1],
            'output': x['input_correct_responses'][0][-1:],
        })

        label_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
        ]

    elif args.mmlu_dataset == "commonsense":
        commonsense = load_dataset("meta-llama/Llama-3.1-8B-evals", "Llama-3.1-8B-evals__commonsenseqa__details")
        eval_dataset = commonsense['latest'].map(lambda x: {
            'input': x['input_final_prompts'][0][:-1],
            'output': x['input_correct_responses'][0][-1:],
        })

        label_idx = [
            tokenizer("a", add_special_tokens=False).input_ids[0],
            tokenizer("b", add_special_tokens=False).input_ids[0],
            tokenizer("c", add_special_tokens=False).input_ids[0],
            tokenizer("d", add_special_tokens=False).input_ids[0],
            tokenizer("e", add_special_tokens=False).input_ids[0],
        ]

    # if "test" in args.mmlu_split:
    #     eval_dataset = eval_dataset[args.mmlu_split]

    if args.max_mmlu_samples is not None:

        # shuffled_mmlu_test_dataset = mmlu_test_dataset.shuffle(seed=args.seed)

        if args.mmlu_dataset == "mmlu_pro" or args.mmlu_dataset == "mmlu":
            eval_dataset = stratified_sample(eval_dataset, frac=0.1)   # 10% for each categories
        else:
            if args.max_mmlu_samples < len(eval_dataset):
                eval_dataset = eval_dataset.select(range(args.max_mmlu_samples))

    return eval_dataset, label_idx

