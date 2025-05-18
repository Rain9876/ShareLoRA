import re
import numpy as np
from transformers import EvalPrediction
from typing import Dict, List, Optional


import evaluate
from transformers import TrainerCallback

class IncrementalMetricCallback(TrainerCallback):
    def __init__(self):
        # Load the accuracy metric from Hugging Face Evaluate
        self.metric = evaluate.load("accuracy")
    
    def on_prediction_step(self, args, state, control, **kwargs):
        # This method is called on each prediction step during evaluation.
        # Here we assume that `predictions` and `label_ids` are provided in kwargs.
        
        preds = kwargs.get("predictions")
        labels = kwargs.get("label_ids") 
        tokenizer = kwargs.get("tokenizer")  # ensure you pass the tokenizer via kwargs

        if preds is None or labels is None or tokenizer is None:
            return control

        # Decode the predictions and labels (they should be strings)
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Update the metric for each sample in the batch
        for pred, label in zip(pred_texts, label_texts):
            pred_ans = extract_boxed_answer(pred)
            label_ans = extract_boxed_answer(label)
            # Here we try integer conversion if possible; otherwise, string comparison.
            try:
                pred_val = int(pred_ans)
                label_val = int(label_ans)
            except ValueError:
                pred_val, label_val = pred_ans, label_ans
            self.metric.add_batch(predictions=[pred_val], references=[label_val])
        
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        # When evaluation is complete, compute and log the final metric.
        final_metric = self.metric.compute()
        print("Final Incremental Accuracy:", final_metric)
        return control


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer



def extract_boxed_answer(text: str) -> str:
    # Find the starting index of the last "\boxed{"
    start_idx = text.rfind(r'\boxed{')
    if start_idx == -1:
        return text[-10:].strip()

    # Move index to start of content
    start_idx += len(r'\boxed{')
    brace_count = 1
    idx = start_idx
    # Iterate until we balance all opening braces
    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    # The content is between start_idx and idx-1 (where the matching '}' was found)
    return text[start_idx:idx-1].strip()

import numpy as np
from transformers import EvalPrediction
import numpy as np
from transformers import EvalPrediction
import torch

class StreamingMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.all_preds = []
        self.all_labels = []

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = False) -> dict:
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        preds = torch.argmax(preds, axis=-1)
        
        # Move GPU tensors to CPU and convert to numpy arrays if necessary.
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy()
        if hasattr(labels, "cpu"):
            labels = labels.cpu().numpy()
        
        def filter_tokens(token_ids, filter_value=-100):
            return [token for token in token_ids if token != filter_value]

        # Apply filtering to each label sequence.
        labels = np.array([filter_tokens(seq) for seq in labels])
        
        # Decode predictions and labels using the provided tokenizer.
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        if not compute_result:
            # Accumulate the decoded strings.
            self.all_preds.extend(decoded_preds)
            self.all_labels.extend(decoded_labels)
            return {}
        else:
            # Apply the extraction logic (e.g., extracting the content inside \boxed{...})
            #print("===="*10)
            #print(self.all_preds[-1])
            #print("----"*10)
            #print(self.all_labels[-1])
            #print("===="*10)

            pred_answers = [extract_boxed_answer(pred) for pred in self.all_preds]
            label_answers = [extract_boxed_answer(label) for label in self.all_labels]
            
            # Compute accuracy.
            correct = sum(1 for p, l in zip(pred_answers, label_answers) if p == l)
            accuracy = correct / len(self.all_preds) if self.all_preds else 0.0

            # Optionally reset the state.
            self.all_preds = []
            self.all_labels = []
            return {"accuracy": accuracy}

