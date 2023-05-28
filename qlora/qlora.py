# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
from os.path import exists, isdir, join
from typing import Dict, List

import bitsandbytes as bnb
import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import (LoraConfig, PeftModel, get_peft_model,
                  get_peft_model_state_dict, prepare_model_for_int8_training,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from rich.logging import RichHandler
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Seq2SeqTrainer, set_seed)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .callbacks import SavePeftModelCallback
from .config import (DataArguments, GenerationArguments, ModelArguments,
                     TrainingArguments)
from .dataset import (DEFAULT_PAD_TOKEN, IGNORE_INDEX, make_data_module,
                      make_text_data_module)

torch.backends.cuda.matmul.allow_tf32 = True

log = logging.getLogger(__name__)


def find_all_linear_names(model, bits: int) -> List[str]:
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_accelerate_model(
    checkpoint_dir,
    model_name_or_path: str,
    max_memory_mb: int,
    full_finetune: bool,
    bits: int,
    fp16: bool,
    bf16: bool,
    double_quant,
    quant_type,
    gradient_checkpointing: bool,
    trust_remote_code: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{max_memory_mb}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}

    log.info("Loading base model %s...", model_name_or_path)
    compute_dtype = (
        torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type,  # {'fp4', 'nf4'}
        ),
        torch_dtype=(
            torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
        ),
        trust_remote_code=trust_remote_code,
    )
    if compute_dtype == torch.float16 and bits == 4:
        major, _minor = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)
    modules = find_all_linear_names(model, bits=bits)

    model.config.torch_dtype = (
        torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
    )

    if not full_finetune:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if not full_finetune:
        if checkpoint_dir is not None:
            log.info("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(
                model, join(checkpoint_dir, "adapter_model")
            )
            for name, p in model.named_parameters():
                if "lora" in name:
                    log.info(name, p.sum())
        else:
            log.info(f"adding LoRA modules...")
            model = get_peft_model(model, config)

    if gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def print_trainable_parameters(model, bits: int):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if bits == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}"
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
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        "input": [],
        "output": [],
    }
    for example_instances in examples["instances"]:
        for instance in example_instances:
            out["input"].append(instance["instruction_with_input"])
            out["output"].append(instance["output"])
    if extract_reformulations:
        for example_reformulations in examples["reformulations"]:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out["input"].append(instance["instruction_with_input"])
                    out["output"].append(instance["output"])
    return out


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith(
                "checkpoint"
            ):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        log.info(f"Found a previous checkpoint at: %s", checkpoint_dir)
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        generation_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        log.info("Detected that training was already completed!")

    if args.full_finetune:
        assert args.bits in [16, 32]
    model = get_accelerate_model(
        checkpoint_dir,
        max_memory_mb=args.max_memory_MB,
        full_finetune=args.full_finetune,
        model_name_or_path=args.model_name_or_path,
        bits=args.bits,
        fp16=args.fp16,
        bf16=args.bf16,
        double_quant=args.double_quant,
        quant_type=args.quant_type,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    training_args.skip_loading_checkpoint_weights = True

    model.config.use_cache = False
    print_trainable_parameters(model, bits=args.bits)
    log.info("loaded model")
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if any(
        key in args.model_name_or_path for key in ["llama", "7B", "13B", "30B", "65B"]
    ):
        # LLaMA tokenizer does not have special tokens set.
        # Add them to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(model.config.pad_token_id),
            }
        )

    if args.dataset_type == "instruct":
        data_module = make_data_module(tokenizer=tokenizer, args=args)
    elif args.dataset_type == "text":
        data_module = make_text_data_module(tokenizer=tokenizer, args=args)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        log.info("%s, %s, %.2f", k, v, v / total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
