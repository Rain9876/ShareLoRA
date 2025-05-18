# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
#import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse
import random
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer, Trainer,
    LlamaForCausalLM
)
#from model.modeling_llama import LlamaForCausalLM
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    inject_adapter_in_model,
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    PeftType,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig, PEFT_TYPE_TO_CONFIG_MAPPING, PeftConfig, PeftModelForCausalLM
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import pickle as pkl


from accelerate.utils import get_balanced_memory, infer_auto_device_map
from accelerate import dispatch_model, infer_auto_device_map    
import time
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2000"

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    adapter_order:str =field(
        default=None,
        metadata={"help": "The order of add the PEFT modules, ['lora','prefix', 'prompt', 'bitfit']"}
    )
    combined_configs:str=field(
        default=None,
        metadata={"help": "The order of add the PEFT modules, ['lora','prefix', 'prompt', 'bitfit']"}
    )
    # For Lora
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Lora dropout."}
    )
    # For Prefix
    prefix_virtual_token:Optional[int] = field(
        default=50,
        metadata={"help":"Number of virtual token for prefix tuning"}
    )
    prefix_token_dim:Optional[int]  = field(
        default=None,
        metadata={"help": ""}
    )
    prefix_transformer_submodules:Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    prefix_attention_heads:Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    prefix_attention_heads:Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    prefix_layers:Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    prefix_projection:Optional[bool] = field(
        default=False,
        metadata={"help": ""}
    )

    prompt_text:Optional[str] =field(
        default=None,
        metadata={"help":""}
    )
    prompt_tuning_init:Optional[str] =field(
        default="RANDOM",
        metadata={"help":""}
    )

    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    deepspeed: str = field(default=None, metadata={"help": ""})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

        
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')

        if state.best_model_checkpoint is not None:
            #checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
            checkpoint_folder = state.best_model_checkpoint
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        if args.adapter_order is not None:
            for i in range(len(args.adapter_order)-1, -1, -1): # peel the adapters
                peft_model_path = os.path.join(checkpoint_folder, f"{args.adapter_order[i]}_model")
                print(f"Save {args.adapter_order[i]}_model")
                # Three cases for now:
                # size of adapter order == 1, direct save
                # size of adapter order == 2, if contains bitfit, direct save, else save each level
                # size of adapter order == 3, if contains bitfit

                if args.adapter_order[i] == "bitfit":
                    save_bias_term(args, kwargs['model'], peft_model_path)
                    continue
                if len(args.adapter_order) == 1 or (len(args.adapter_order) == 2 and "bitfit" in args.adapter_order):
                    kwargs["model"].save_pretrained(peft_model_path, save_embedding_layers=False)
                    print(f"Saving checkpoints: ", peft_model_path)
                    print(f"Check adapter_order {len(args.adapter_order)}")

                elif len(args.adapter_order) == 2 or len(args.adapter_order) == 3:
                    if i == len(args.adapter_order)-1:
                        kwargs["model"].peft_config = args.combined_configs[args.adapter_order[i]]
                        kwargs["model"].save_pretrained(peft_model_path)
                    elif i == len(args.adapter_order)-2:
                        model = get_next_peft_model(args, kwargs["model"])
                        model.peft_config = args.combined_configs[args.adapter_order[i]]
                        model.save_pretrained(peft_model_path)
                        kwargs["model"].peft_config = args.combined_configs[args.adapter_order[i+1]]
                        print(model.peft_config)
                        print(kwargs["model"].peft_config)

                else:
                    print("NUM OF ADAPTER ERROR")
            print_weight_config(args, kwargs["model"])

        else:
            print(kwargs["model"].peft_config)
            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)
        pytorch_model_path = os.path.join(checkpoint_folder, "model.safetensors")
        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_next_peft_model(args, model):
    if len(args.adapter_order) < 2:
        return model

    if isinstance(model, PeftModelForCausalLM): # Skip the first PEFT Model, and find the next one
        model = model.base_model # PeftModelForCausalLM has attr base_model

    while not isinstance(model, PeftModelForCausalLM):
        try:
            # Check if model has a 'model' attribute that is a PeftModelForCausalLM
            if hasattr(model, 'model') and isinstance(model.model, PeftModelForCausalLM):
                model = model.model
                break  # Exit loop if the desired model is found

            # Check if model has a 'base_model' attribute that is a PeftModelForCausalLM
            if hasattr(model, 'base_model') and isinstance(model.base_model, PeftModelForCausalLM):
                model = model.base_model
                break  # Exit loop if the desired model is found

        except Exception as e:
            # Print the exception and mention that the other models are built on top of PEFT
            print(f"Encountered an error: {e}")
            print("Other models are built on top of PEFT")
            break
    return model

def set_peft_config(args, model):
    if len(args.adapter_order) == 2:
        print("SET PEFT CONFIG")
        print(get_next_peft_model(args, model).peft_config is model.peft_config)
        get_next_peft_model(args, model).peft_config = args.combined_configs[args.adapter_order[0]]
        print(get_next_peft_model(args,model).peft_config)
        print(model.peft_config)
        print(model.word_embeddings.weight)
        print(model.base_model.base_model.model.model.embed_tokens.weight)


def print_weight_config(args, model):
    # Print Weights
    if len(args.adapter_order) == 2:
        if "lora" in args.adapter_order[1]:
            # Prefix,Lora
            #print(model.base_model.model.base_model.model.layers[0].mlp.gate_proj.lora_A.default.weight)
            print(model.base_model.model.prompt_encoder.default.embedding.weight)
        else:
            # Lora,Prefix
            #print(model.base_model.base_model.model.model.layers[0].mlp.gate_proj.lora_A.default.weight)
            print(model.prompt_encoder.default.embedding.weight)
    else:
        if "lora" in args.adapter_order[0]:
            print(model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight)

        else:
            print(model.prompt_encoder.default.embedding.weight)

def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    if args.deepspeed is not None:
        device_map = None

    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))


    if args.adapter_order is not None and "bitfit" in args.adapter_order and 'llama' in args.model_name_or_path:   
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
            cache_dir=args.cache_dir,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map='auto',
            max_memory=max_memory,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
        )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('=' * 80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('=' * 80)

    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="left",
        use_fast=False,  # Fast tokenizer giving issues.
        #tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,  # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(0),
            "pad_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id
                # model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })


    #if not args.full_finetune:
    #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            if args.adapter_order is not None:
                for i in args.adapter_order: # peel the adapters
                    # model = PeftModel.from_pretrained(model, "timdettmers/guanaco-13b", is_trainable=False)
                    model = PeftModel.from_pretrained(model, join(checkpoint_dir, f"{i}_model"), is_trainable=False)
                    model.peft_config = {"default": PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig._get_peft_type(join(checkpoint_dir, f"{i}_model"))].from_pretrained(join(checkpoint_dir, f"{i}_model"))}
                    args.combined_configs[i] = {"default": model.peft_config["default"]}
                print("Load Checkpoint only support the Inference, does not support adaptive tuning again")
                set_trainable_parameters(args, model, False)
                print_weight_config(args, model)
            else:
                model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
                
        else:
            print("Initalize adapters from ...")

            if args.adapter_order is not None:
                for i in args.adapter_order:
                    if i == "bitfit":
                        continue
                    model = add_peft_modules(args, model, i)
                set_trainable_parameters(args, model, True)
            else:
                model = add_peft_modules(args, model, 'lora')

    ################################## ##################################
    ################## Share LoRA Modules
    ################################## ##################################

    shared_lora_A_value = None
    shared_lora_A_key = None
    shared_lora_A_query = None

    shared_lora_B_value = None
    shared_lora_B_key = None
    shared_lora_B_query = None

    shareA = True
    shareB = False

    for name, module in model.named_modules():

        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        
            print("***************************inital lora")

            lora_a = module.lora_A['default']
            lora_b = module.lora_B['default']

            if  "q_proj" in name and "bias" not in name:
                print(f"{'query' in name}")

                if shareA:
                    if shared_lora_A_query is None:
                        shared_lora_A_query = lora_a.weight
                    else:
                        print(f"query share {name}")
                        device = lora_a.weight.device
                        lora_a.weight = shared_lora_A_query.to(device)

                if shareB:
                    if shared_lora_B_query is None:
                        shared_lora_B_query = lora_b.weight
                    else:
                        print(f"query share {name}")
                        device = lora_b.weight.device
                        lora_b.weight = shared_lora_B_query.to(device)

            # if  "k_proj" in name and "bias" not in name:
            #     print(f"{'key' in name}")
            #     if shareA:
            #         if shared_lora_A_key is None:
            #             shared_lora_A_key = lora_a.weight
            #         else:
            #             print(f"key share {name}")
            #             device = lora_a.weight.device
            #             lora_a.weight = shared_lora_A_key.to(device)
            #
            #     if shareB:
            #         if shared_lora_B_key is None:
            #             shared_lora_B_key = lora_b.weight
            #         else:
            #             print(f"key share {name}")
            #             device = lora_b.weight.device
            #             lora_b.weight = shared_lora_B_key.to(device)
            #
            
            if  "v_proj" in name and "bias" not in name:
                print(f"{'value' in name}")

                if shareA:
                    if shared_lora_A_value is None:
                        shared_lora_A_value = lora_a.weight
                    else:
                        print(f"value share {name}")
                        device = lora_a.weight.device
                        lora_a.weight = shared_lora_A_value.to(device)

                if shareB:
                    if shared_lora_B_value is None:
                        shared_lora_B_value = lora_b.weight
                    else:
                        print(f"value share {name}")
                        device = lora_b.weight.device
                        lora_b.weight = shared_lora_B_value.to(device)


            #if isinstance(lora_a, torch.nn.Linear) and args.do_train:
            #   # torch.nn.init.xavier_normal_(module.weight)
            #    lora_a.weight.data.normal_(mean=0.0, std=0.02)
            #    if lora_a.bias is not None:
            #        lora_a.bias.data.zero_()
            #if isinstance(lora_b, torch.nn.Linear) and args.do_train:
            #   # torch.nn.init.xavier_normal_(module.weight)
            #    lora_b.weight.data.normal_(mean=0.0, std=0.02)
            #    if lora_b.bias is not None:
            #        lora_b.bias.data.zero_()

        if 'prompt_encoder' in name:
            if args.bf16 and hasattr(module, 'weight'):
                module = module.to(torch.bfloat16)

            if isinstance(module, torch.nn.Linear) and args.do_train:
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, torch.nn.Embedding) and args.do_train:
                # print("******Prefix Embedding initialized")
                # torch.nn.init.xavier_normal_(module.weight)
                module.weight.data.normal_(mean=0.0, std=0.02)
        if 'norm' in name:
            module = module.to(torch.bfloat16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    seen = set()
    print("Check Shared Data")
    for name, param in model.named_parameters():
        # Check if any parameter's memory location is already seen
        if "lora" in name:
            if id(param) in seen:
                print(f"Weights shared: {name}")
            else:
                seen.add(id(param))

    print(model)
    return model, tokenizer

def save_bias_term(args, model, checkpoint_dir):
    bias_val = {}
    for name, param in model.named_parameters():
        if "bias" not in name:
            continue
        if "lora" in name or "prompt" in name:
            continue
        bias_val[name] = param.detach().cpu().data
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(join(checkpoint_dir, 'bias.pkl'), 'wb') as f:
        pkl.dump(bias_val, f)
    
def load_bias_term(args, model, checkpoint_dir, is_trainable=True):
    bias_val_load = {}
    with open(join(checkpoint_dir, 'bias.pkl'), 'rb') as f:
        bias_val_load = pkl.load(f)
    for name, param in model.named_parameters():
        if "bias" not in name:
            continue
        if "lora" in name or "prompt" in name:
            continue
        if name in bias_val_load:
            param.data = bias_val_load[name]
            if is_trainable:
                param.requires_grad = True  
    return model

def add_peft_modules(args, model, peft_name):
    config = None
    if peft_name == "lora":
        print(f'adding LoRA modules...')
        modules = find_all_linear_names(args, model)
        modules = ["q_proj","v_proj"]
        print(modules)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        #model = inject_adapter_in_model(config, model)

        args.combined_configs[peft_name] = {"default": config}
        model = get_peft_model(model, config,"default")
        model.peft_config = {"default": config}
        
        # args.combined_configs[peft_name] = {f"{peft_name}_adapter": config}
        # model = get_peft_model(model, config, f"{peft_name}_adapter")
    elif peft_name == "prefix":
        print("adding Prefix modules ...")
        config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING",
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.prefix_virtual_token,
            inference_mode=False,
            token_dim=args.prefix_token_dim,
            num_transformer_submodules=args.prefix_transformer_submodules,
            num_attention_heads=args.prefix_attention_heads,
            num_layers=args.prefix_layers,
            prefix_projection=True,
            encoder_hidden_size=768
            )
        args.combined_configs[peft_name] = {"default": config}
        model = get_peft_model(model, config)
        model.peft_config = {"default": config}
        # args.combined_configs[peft_name] = {f
        # "{peft_name}_adapter": config}
        # model = get_peft_model(model, config, f"{peft_name}_adapter")
    elif peft_name == "prompt":
        print("adding Prompt modules ...")
        ## Cannot embed both Prompt and Prefix, only one is allowed
        # args.prompt_text = ALPACA_PROMPT_DICT["prompt_input"]
        config = PromptTuningConfig(
            peft_type="PROMPT_TUNING",
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.prefix_virtual_token,
            token_dim=args.prefix_token_dim,
            num_transformer_submodules=args.prefix_transformer_submodules,
            num_attention_heads=args.prefix_attention_heads,
            num_layers=args.prefix_layers,
            prompt_tuning_init=args.prompt_tuning_init,
            prompt_tuning_init_text= args.prompt_text if args.prompt_tuning_init == "TEXT" else None,
            tokenizer_name_or_path= args.model_name_or_path if args.prompt_tuning_init == "TEXT" else None,
            tokenizer_kwargs= {
                    "cache_dir":args.cache_dir,
                    "padding_side":"right",
                    "use_fast":False, # Fast tokenizer giving issues.
                    "tokenizer_type":'llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
                    "trust_remote_code":args.trust_remote_code} if args.prompt_tuning_init == "TEXT" else None
        )
        args.combined_configs[peft_name] = {"default": config}
        model = get_peft_model(model, config)
        model.peft_config = {"default": config}
        # args.combined_configs[peft_name] = {f"{peft_name}_adapter": config}
        # model = get_peft_model(model, config, f"{peft_name}_adapter")
    elif peft_name == "p_tuning":
        print("adding p_tuning modules ...")
        ## Cannot embed both Prompt, P_tuning and Prefix, only one is allowed
        config = PromptEncoderConfig(
            peft_type=PeftType.P_TUNING,
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.prefix_virtual_token,
            token_dim=args.prefix_token_dim,
            num_transformer_submodules=args.prefix_transformer_submodules,
            num_attention_heads=args.prefix_attention_heads,
            num_layers=args.prefix_layers,
            encoder_reparameterization_type="MLP",
            encoder_dropout=0.1,
            encoder_num_layers=2,
            encoder_hidden_size = 768,
        )
        args.combined_configs[peft_name] = {"default": config}
        model = get_peft_model(model, config)
        model.peft_config = {"default": config}
        # args.combined_configs[peft_name] = {f"{peft_name}_adapter": config}
        # model = get_peft_model(model, config, f"{peft_name}_adapter")

    return model 

def set_trainable_parameters(args, model, require_grad=True):
    """
    Set parametera trainable.
    """

    kw = args.adapter_order
    if "prefix" in args.adapter_order:
        kw = list(map(lambda x: x.replace('prefix', 'prompt'), args.adapter_order))
    if "p_tuning" in args.adapter_order:
        kw = list(map(lambda x: x.replace('p_tuning', 'prompt'), args.adapter_order))
    if "bitfit" in args.adapter_order:
        kw = list(map(lambda x: x.replace('bitfit', 'bias'), kw))

    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        for k in kw:
            if k in name:
                param.requires_grad = require_grad
                    # Set original to be False

        if param.requires_grad:
            # print(f"Trainable parameter {name:<20} | {param.numel():>5} parameters")
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        # if ".layers." not in name or ".layers.0" in name and param.requires_grad:
        if param.requires_grad:
            print(f"Trainable parameter {name:<20} | {param.numel():>5} parameters | {param.dtype} | {param.device}")
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
                labels.append(torch.tensor(copy.deepcopy(tokenized_target)))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side='left')
        #labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else torch.stack(labels)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side='left')

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels

        if self.predict_with_generate: ## Extra Columns info save for only predictions
            extra_columns = [{key: value for key, value in instance.items() if key not in ['input', 'output']} for instance in instances]
            data_dict['info'] = extra_columns

        return data_dict


def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}


def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """

    def is_level5(example):
        # Adjust the condition based on the dataset schema.
        return example.get('level') == 'Level 5'

    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'math':
            return load_dataset("xDAN2099/lighteval-MATH", "default")
        elif dataset_name == 'math_hard':
            return load_dataset("xDAN2099/lighteval-MATH", "default").filter(is_level5)
        elif dataset_name == 'gsm8k':
            return load_dataset("openai/gsm8k", "main")
        elif dataset_name == 'codealpaca':
            return load_dataset("sahil2801/CodeAlpaca-20k", "default")
        elif dataset_name == 'hf_codealpaca':
            return load_dataset("HuggingFaceH4/CodeAlpaca_20K", "default")
        elif dataset_name == 'humaneval':
            return load_dataset("openai/openai_humaneval", 'openai_humaneval')
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    import re
    from utils.math_utils import normalize_final_answer, remove_boxed, last_boxed_only_string
    def find_answer(text: str) -> str:
        return normalize_final_answer(remove_boxed(last_boxed_only_string(text)))

    def chat_template(text, answer):
        messages = [
            {"role": "user", "content": f"{text}"},
            {"role": "assistant", "content": f"{answer}\nFinal Answer: The final answer is ${find_answer(answer)}$. I hope it is correct."}]
        return messages

    def apply_chat_template(example):
        messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": f"{example}"}]
        return messages

    def format_dataset(dataset, dataset_format):
        if (
                dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
                (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        ## Eval Template from lm eval harness
        elif dataset_format == "MATH":
            from few_shot import MATH_4cot_json
            dataset = dataset.map(lambda x: {
                'input': "Problem:\n" + x["problem"] + "\n\nSolution:\n",
                'output': x['solution'] + r"\nFinal Answer: The final answer is $" + str(find_answer(x['solution'])) +"$. I hope it is correct.",
            })
        ## Eval Template from
        elif dataset_format == "GSM8K":
            #from few_shot import GSM8K_8cot
            from few_shot import GSM8K_8cot_json
            chats = '\n\n'.join([f"Question: {t['problem']}\n\nAnswer: {t['solution']}"  for t in GSM8K_8cot_json])
            print(chats)
            dataset = dataset.map(lambda x: {
                'input': chats + "\n\nQuestion: " + x["question"] + "\n\nAnswer: ",
                #'input': "Question: " + x["question"] + "\n\nAnswer: ",
                'output': re.sub(r'\n####', ' The final answer is', x['answer'])
            })
        ## Eval Template from
        elif dataset_format == "HF_CODE":
            dataset = dataset.map(lambda x: {
                'input': x["prompt"],
                'output': x['completion'],
            })

        elif dataset_format == "HE":
            dataset = dataset.map(lambda x: {
                'input': x["prompt"],
                'output': x['canonical_solution'],
            })

        elif dataset_format == 'input-output':
            # leave as is
            pass

        # Remove unused columns.
        if 'train' in dataset.column_names:
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
            )

        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'test' in dataset:
            eval_dataset = dataset['test']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    
    training_args.combined_configs = {} 
    if training_args.adapter_order is not None:
        training_args.adapter_order = training_args.adapter_order.split(",")
    
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)
    set_seed(args.seed)

    if args.do_train:
        checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
        print(checkpoint_dir)
        if completed_training:
            print('Detected that training was already completed!')
    else:
        checkpoint_dir = args.output_dir
        print(f"Evaluate on the checkpoint {args.output_dir}")

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    from eval_metric import StreamingMetrics
    training_args.batch_eval_metrics=True

    streaming_metrics = StreamingMetrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=streaming_metrics,
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                #'eval': 'data/mmlu/five_shot_mmlu_val.json',                
                'subtest': 'data/mmlu/five_shot_mmlu_test_selective_large.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
                #'test': 'data/MMLU-PRO-Test-fiveshot_no_cot.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        #mmlu_eval_dataset = mmlu_dataset["eval"]
        
        mmlu_test_dataset = None
        if "test" in args.mmlu_split:
            mmlu_test_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            # shuffled_mmlu_test_dataset = mmlu_test_dataset.shuffle(seed=args.seed)
            mmlu_test_dataset = mmlu_test_dataset.select(range(args.max_mmlu_samples))
        
        #abcd_idx = [
        #    tokenizer("A", add_special_tokens=False).input_ids[0],
        #    tokenizer("B", add_special_tokens=False).input_ids[0],
        #    tokenizer("C", add_special_tokens=False).input_ids[0],
        #    tokenizer("D", add_special_tokens=False).input_ids[0],
        #    tokenizer("E", add_special_tokens=False).input_ids[0],
        #    tokenizer("F", add_special_tokens=False).input_ids[0],
        #    tokenizer("G", add_special_tokens=False).input_ids[0],
        #    tokenizer("H", add_special_tokens=False).input_ids[0],
        #    tokenizer("I", add_special_tokens=False).input_ids[0],
        #    tokenizer("J", add_special_tokens=False).input_ids[0],
        #]

        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]

        accuracy = evaluate.load("accuracy")

        class MMLUPROEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.args.group_by_length = False
                if mmlu_test_dataset is not None:
                    data_loader = trainer.get_eval_dataloader(mmlu_test_dataset)
                    trainer.model.eval()
                    preds, refs = [], []
                    loss_mmlu = 0
                    for batch in tqdm(data_loader, total=len(data_loader)):
                        (loss, logits, labels) = trainer.prediction_step(trainer.model, batch, prediction_loss_only=False, )
                        # There are two tokens, the output, and eos token.
                        for i, logit in enumerate(logits):
                            label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                            logit_abcd = logit[label_non_zero_id - 1][abcd_idx]
                            preds.append(torch.argmax(logit_abcd).item())
                        labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
                        refs += [abcd_idx.index(label) for label in labels.tolist()]
                        loss_mmlu += loss.item()
                    # Extract results by subject.
                    results = {'mmlu_loss': loss_mmlu / len(data_loader)}
                    subject = mmlu_test_dataset['subject']
                    subjects = {s: {'refs': [], 'preds': []} for s in set(subject)}
                    for s, p, r in zip(subject, preds, refs):
                        subjects[s]['preds'].append(p)
                        subjects[s]['refs'].append(r)
                    subject_scores = []
                    for subject in subjects:
                        subject_score = accuracy.compute(
                            references=subjects[subject]['refs'],
                            predictions=subjects[subject]['preds']
                        )['accuracy']
                        results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                        subject_scores.append(subject_score)
                    results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                    if not args.do_train:
                        print(results)
                    trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len
                trainer.args.group_by_length = False 
        trainer.add_callback(MMLUPROEvalCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}
    set_peft_config(args, model)
    
    print("===="*10)
    print(model.peft_config)
    print("===="*10)

    #print_weight_config(args, model)

    from datetime import datetime
    from pathlib import Path

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
        
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")

        print(training_args.generation_config)

        results = []
        start = time.time()

        trainer.model.eval()

        terminators = [
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Open the output file before starting the loop
        output_file_path = os.path.join(args.output_dir, 'predictions.jsonl')
        predict_loader = trainer.get_test_dataloader(data_module['predict_dataset'])
        print("===== Predict loader size", len(predict_loader))
        with open(output_file_path, 'w') as fout:
            # Process each batch in the dataloader
            for idx, batch in enumerate(predict_loader):
                if idx % 10 == 0:
                    print(f"Processing batch {idx}")

                # Generate predictions using the model
                #if idx >= 100:  # Stop after 2 batches
                #    break

                outputs = trainer.model.generate(
                    input_ids= batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels = batch["labels"],
                    generation_config = training_args.generation_config,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    )

                # Move outputs to CPU and convert to numpy for decoding
                predictions = outputs.cpu().numpy()

                # Replace -100 with pad_token_id before decoding
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

                # Decode predictions
                decoded_predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                # Decode input texts and labels from the batch
                input_texts = tokenizer.batch_decode(
                    batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                labels = np.where(batch['labels'].cpu().numpy() != -100, batch['labels'].cpu().numpy(), tokenizer.pad_token_id)

                label_texts = tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                ) if 'labels' in batch else [''] * len(input_texts)

                # Write each decoded prediction with the corresponding input and label to the file immediately
                for i, prediction in enumerate(decoded_predictions):
                    example = {}
                    #example['ids'] = predictions[i].tolist()
                    if "info" in example:
                        example['info'] = batch['info'][i]
                    example['input'] = input_texts[i]
                    example['label'] = label_texts[i]
                    example['prediction_with_input'] = prediction.strip()
                    # Remove the input text from the prediction if needed
                    example['prediction'] = prediction.replace(input_texts[i], '')
                    fout.write(json.dumps(example) + '\n')

        print(f"Predictions written to {output_file_path}")
        print((time.time()-start) / 60)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
