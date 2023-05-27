import copy
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

PROMPT_DICT = {
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
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [example["input"] for example in instances]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
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
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = PROMPT_DICT["prompt_input"]
    else:
        prompt_format = PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


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

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - oa-rlhf (OpenAssistant) primary message tree only, 9209 examples
        - oa-rlhf-assistant (OpenAssistant) all assistant  replies with ranking
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available

    Not Available:
        - vicuna, not released at the moment.
    """
    # Load dataset.
    # Alpaca
    if args.dataset == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca")
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=["instruction"])
    # Alpaca clean
    elif args.dataset == "alpaca-clean":
        dataset = load_dataset("yahma/alpaca-cleaned")
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=["instruction"])
    # Chip2
    elif args.dataset == "chip2":
        dataset = load_dataset("laion/OIG", data_files="unified_chip2.jsonl")
        dataset = dataset.map(
            lambda x: {
                "input": x["text"].split("\n<bot>: ")[0].replace("<human>: ", ""),
                "output": x["text"].split("\n<bot>: ")[1],
            },
            remove_columns=["text", "metadata"],
        )
    # Self Instruct
    elif args.dataset == "self-instruct":
        dataset = load_dataset("yizhongw/self_instruct", name="self_instruct")
        for old, new in [["prompt", "input"], ["completion", "output"]]:
            dataset = dataset.rename_column(old, new)
    # Anthropic rlhf
    elif args.dataset == "hh-rlhf":
        dataset = load_dataset("Anthropic/hh-rlhf")
        dataset = dataset.map(
            lambda x: {"input": "", "output": x["chosen"]},
            remove_columns=["chosen", "rejected"],
        )
    # LongForm
    elif args.dataset == "longform":
        dataset = load_dataset("akoksal/LongForm")
    elif args.dataset == "vicuna":
        raise NotImplementedError("Vicuna data was not released.")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")
    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )

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
        data_collator=data_collator,
    )
