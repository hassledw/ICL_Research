import os
import sys
sys.path.insert(0, 'src/')

from datasets import Dataset, load_dataset, DownloadMode, concatenate_datasets
from typing import List, Tuple
from transformers import HfArgumentParser

from config import Arguments
from logger_config import logger
from utils import save_dataset
from evaluation import BaseEval
from model_utils import build_eval_model
from inference.inference_utils import get_prompt_save_path

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]


def main():
    out_path: str = get_prompt_save_path(args=args)
    if os.path.exists(out_path):
        logger.info('Prompt file {} exists. Skip.'.format(out_path))
        return

    # Load the specified datasets
    datasets = {
        "snli": load_dataset("snli", split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD),
        "cosmos_qa": load_dataset("cosmos_qa", split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD),
        "arc_challenge": load_dataset("ai2_arc", "ARC-Challenge", split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD),
        "come": load_dataset("come", split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD),
        "yelp": load_dataset("yelp_polarity", split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
    }

    model: BaseEval = build_eval_model(args=args, corpus=None)  # corpus is not used in one-shot learning

    task_ds_list: List[Dataset] = []
    for task_name, dataset in datasets.items():
        if len(dataset) > args.max_test_samples:
            logger.info('Task: {}, random sample {}/{} for evaluation'.format(
                task_name, args.max_test_samples, len(dataset))
            )
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_test_samples))
        logger.info('Task: {}, {} samples for evaluation'.format(task_name, len(dataset)))

        # For one-shot learning, use only one example as a prompt
        one_shot_example = dataset.shuffle(seed=args.seed).select([0])
        one_shot_prompt = one_shot_example['text'][0]  # assuming 'text' field contains the relevant data

        dataset = dataset.add_column('input_prompt', [one_shot_prompt for _ in range(len(dataset))])
        task_ds_list.append(dataset)

    one_shot_ds: Dataset = concatenate_datasets(task_ds_list)
    save_dataset(one_shot_ds, out_path)
    logger.info('Save {} examples to {}'.format(len(one_shot_ds), out_path))


if __name__ == '__main__':
    main()