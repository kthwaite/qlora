import copy
from dataclasses import dataclass
from typing import Dict, Generator, Sequence

import torch
import transformers
from datasets import Dataset, load_dataset
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
class DataCollatorForCausalLM:
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


class InstructDatasetMeta(type):
    _REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        if hasattr(new_cls, "_NAME"):
            cls._REGISTRY[new_cls._NAME] = new_cls
        return new_cls

    @classmethod
    def get(cls, name: str):
        return cls._REGISTRY[name]


class InstructDataset(metaclass=InstructDatasetMeta):
    @classmethod
    def load(cls):
        raise NotImplementedError(...)

    @classmethod
    def construct(
        cls,
        tokenizer,
        do_train,
        do_eval,
        do_predict,
        source_max_len,
        target_max_len,
        train_on_source,
        predict_with_generate,
        eval_dataset_size,
        max_eval_samples,
        group_by_length,
        max_train_samples,
    ):
        train_dataset = None
        eval_dataset = None
        dataset = cls.load()
        if do_train:
            train_dataset = dataset["train"]
            if max_train_samples is not None and len(train_dataset) > max_train_samples:
                train_dataset = train_dataset.select(range(max_train_samples))
            if group_by_length:
                train_dataset = train_dataset.map(
                    lambda x: {"length": len(x["input"]) + len(x["output"])}
                )
        if do_eval or do_predict:
            if "eval" in dataset:
                eval_dataset = dataset["eval"]
            else:
                print(
                    "Splitting train dataset in train and validation according to `eval_dataset_size`"
                )
                dataset = dataset["train"].train_test_split(
                    test_size=eval_dataset_size, shuffle=True, seed=42
                )
                eval_dataset = dataset["test"]
            if max_eval_samples is not None and len(eval_dataset) > max_eval_samples:
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            if group_by_length:
                eval_dataset = eval_dataset.map(
                    lambda x: {"length": len(x["input"]) + len(x["output"])}
                )

        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=source_max_len,
            target_max_len=target_max_len,
            train_on_source=train_on_source,
            predict_with_generate=predict_with_generate,
        )
        return dict(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            predict_dataset=eval_dataset,
            data_collator=data_collator,
        )


class Alpaca(InstructDataset):
    _NAME = "alpaca"

    @classmethod
    def load(cls):
        dataset = load_dataset("tatsu-lab/alpaca")
        return dataset.map(extract_alpaca_dataset, remove_columns=["instruction"])


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
    if args.dataset == "alpaca-clean":
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


def split_chunks(arr: str, step: int) -> Generator[str, None, None]:
    for i in range(0, len(arr), step):
        yield arr[i : i + step]


def cut_chunk_for_newline(chunk: str, max_length: int) -> str:
    if "\n" not in chunk:
        return chunk

    first_newline = chunk.index("\n")
    if first_newline < max_length:
        chunk = chunk[first_newline + 1 :]

    if "\n" not in chunk:
        return chunk

    last_newline = chunk.rindex("\n")
    if len(chunk) - last_newline < max_length:
        chunk = chunk[:last_newline]

    return chunk


def make_text_data_module(
    tokenizer,
    args,
    hard_cut_string: str = "\n\n",
    cutoff_len: int = 256,
    overlap_len: int = 128,
    newline_favor_len: int = 128,
) -> dict:
    with open(args.dataset, "r", encoding="utf-8") as file:
        raw_text = file.read().replace("\r", "")

    cut_string = hard_cut_string.replace("\\n", "\n")
    out_tokens = []
    for text_part in raw_text.split(cut_string):
        if text_part.strip() == "":
            continue

        tokens = tokenizer.encode(text_part)
        step = cutoff_len - overlap_len
        if step <= 0:
            raise ValueError(
                f"Error: overlap_len ({overlap_len}) cannot be greater than or equal "
                f"to cutoff_len ({cutoff_len})"
            )

        tokens = list(split_chunks(tokens, step))
        for i in range(1, len(tokens)):
            tokens[i] = tokens[i - 1][-overlap_len:] + tokens[i]

        out_tokens.extend(tokens)
        del tokens
    del raw_text  # Note: could be a gig for a large dataset, so delete redundant data as we go to be safe on RAM
    text_chunks = [tokenizer.decode(x) for x in out_tokens]
    del out_tokens

    if newline_favor_len > 0:
        text_chunks = [cut_chunk_for_newline(x, newline_favor_len) for x in text_chunks]

    train_dataset = Dataset.from_list(
        [tokenize(tokenizer, x, cutoff_len) for x in text_chunks]
    )
    del text_chunks
    eval_dataset = None

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "predict_dataset": eval_dataset,
        "data_collator": transformers.DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
        ),
    }


def encode(tokenizer, text: str, add_bos_token: bool, cutoff_len: int):
    result = tokenizer.encode(text, truncation=True, max_length=cutoff_len)
    if not add_bos_token and result[0] == tokenizer.bos_token_id:
        result = result[1:]
    return result


def tokenize(tokenizer, prompt: str, cutoff_len: int):
    input_ids = encode(tokenizer, prompt, add_bos_token=True, cutoff_len=cutoff_len)
    input_ids = [tokenizer.pad_token_id] * (cutoff_len - len(input_ids)) + input_ids
    labels = [1] * len(input_ids)

    input_ids = torch.tensor(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }
