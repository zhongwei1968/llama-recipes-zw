# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

from datasets import load_dataset
import datasets
from .utils import Concatenator

from datasets import load_dataset
from pathlib import Path
from torch.utils.data import Dataset
from .utils import ConcatDataset


class TextCorpus(Dataset):
    def __init__(
            self,
            dataset_config,
            tokenizer,
            split=None,
    ):
        self.tokenizer = tokenizer

        try:
            # load and process the data set
            dataset = load_dataset("text", data_files={"train": dataset_config.data_path},
                                   download_mode="force_redownload")

            dataset = dataset.map(self.add_eos, batched=True, batch_size=1000, num_proc=1, remove_columns=["text"])

            self.dataset = dataset.map(
                lambda sample: tokenizer(sample["text2"]),
                batched=True,
                remove_columns=["text2"],
            )

        except Exception as e:
            print(
                "Loading of text corpus dataset failed!")
            raise e

    def add_eos(self, examples):
        # there is two or more empty lines between each article. return list of article as examples.
        # add new line '\n' to each line.
        # If there are at least two consecutive empty lines, add the EOS token

        articles = []
        lines = []
        eos_token = self.tokenizer.eos_token
        consecutive_empty_lines = 0
        for line in examples["text"]:
            stripped_line = line.rstrip()
            if stripped_line:
                # If there are at least two consecutive empty lines, add the EOS token
                if consecutive_empty_lines >= 2:
                    lines.append(eos_token)
                    articles.append('\n'.join(lines))
                    lines = []
                consecutive_empty_lines = 0
                lines.append(stripped_line)
            else:
                consecutive_empty_lines += 1

        # If the last line is empty, append the EOS token
        if consecutive_empty_lines >= 2 or (consecutive_empty_lines == 1 and lines[-1] != eos_token):
            # not add eos_token
            lines.append(eos_token)
            articles.append('\n'.join(lines))

        return {'text2': articles}

    def __len__(self):
        return self.dataset["train"].shape[0]

    def __getitem__(self, index):
        sample = self.dataset["train"][index]
        source_ids = sample["input_ids"]

        src_mask = sample["attention_mask"]

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": source_ids.copy(),
        }


def get_dataset(
        dataset_config, tokenizer, split=None
):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    dataset = TextCorpus(
        dataset_config,
        tokenizer=tokenizer,
        split=split,
    )

    return ConcatDataset(dataset, chunk_size=dataset_config.input_length)

