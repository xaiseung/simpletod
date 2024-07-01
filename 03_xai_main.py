"""
Fine-tuning pretrained language model (GPT2) on Task-oriented Dialogue
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
# import shutil
from typing import Any, Dict, List, Tuple
from functools import partial

import numpy as np
import torch
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    # get_linear_schedule_with_warmup,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import datasets

# comment this if you want to load gpt2 class from transformers
from models import GPT2LMHeadModel
from models import GPT2Config, GPT2SmallConfig

# uncomment this if you want to load gpt2 class from transformers
# from transformers import GP2Config, GPT2LMHeadModel

from data.dataset.language_model import *
from utils.model import *
from utils.language_model import get_optimizer_scheduler
from utils.gpt2_args_parser import ArgsParser

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('recent.log')
logger.addHandler(file_handler)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-small": (GPT2SmallConfig, GPT2LMHeadModel, GPT2Tokenizer),
}


def get_model_tokenizer(args):
    if args.model_type in MODEL_CLASSES:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    else:
        assert args.model_name_or_path
        config_class, model_class, tokenizer_class = AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if args.model_type == "llama3":
        # TODO: Add args for Quantization
        #bnb_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_use_double_quant=True,
        #    bnb_4bit_quant_type="nf4",
        #    bnb_4bit_compute_dtype=torch.bfloat16
        #)
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            device_map="auto",
            torch_dtype = torch.bfloat16,
            #quantization_config=bnb_config
        )
        tokenizer.model_max_length = args.block_size
        args.block_size = min(args.block_size, tokenizer.model_max_length)
        tokenizer.eos_token = "<|end_of_text|>"
        tokenizer.eos_token_id = tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0]
        model.config.eos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if args.model_name_or_path:
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = model_class(config=config)

        model.to(args.device)

    if args.model_name_or_path == 'openai-gpt':
        tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    elif args.model_name_or_path == 'gpt2':
        pass

    return model, tokenizer, model_class, args

#from LORA Code
def get_num_layers(model):
    numbers = set()
    for name, _ in model.named_parameters():
        for number in re.findall(r'\d+', name):
            numbers.add(int(number))
    return max(numbers)

def get_last_layer_linears(model, n=1):
    if n < 1:
        raise ValueError("num_trained_layers should be >= 1")
    names = []
    
    num_layers = get_num_layers(model)
    tgt_layers = [num_layers - i for i in range(n)]
    for name, module in model.named_modules():
        if any([str(num_tgt) in name for num_tgt in tgt_layers]) and not "encoder" in name:
            if isinstance(module, torch.nn.Linear):
                names.append(name)
    return names

class CustomCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        batch["labels"] = batch["input_ids"]
        return batch

CUSTOM_TRAINER_OUTPUT_DIR = "./"
class CustomTrainer(SFTTrainer):
    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)

        output_eval_file = os.path.join(CUSTOM_TRAINER_OUTPUT_DIR, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(""))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info(f"")
        for l in self.state.log_history:
            logger.info(f"{l}")
        return result


def tokenize_fn(example, tokenizer):
    to_break = lambda s: re.sub(r"\\n", r"\n", s)
    if hasattr(example["text"], "__getitem__"):
        for i in range(len(example["text"])):
            example["text"][i] = to_break(example["text"][i])
    else:
        example["text"] = to_break(example["text"])
    be = tokenizer(example["text"],
                   truncation=True,
                   add_special_tokens=False)
    return be

def train(args, tokenized_datasets, model, tokenizer):
    global CUSTOM_TRAINER_OUTPUT_DIR
    # total iteration and batch size
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(tokenized_datasets["train"]) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(tokenized_datasets["train"]) // args.gradient_accumulation_steps * args.num_train_epochs

    total_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) * args.gradient_accumulation_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = {}".format(len(tokenized_datasets["train"])))
    logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(total_batch_size))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(t_total))

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.train()
    model.zero_grad()

    data_collator = CustomCollator(tokenizer=tokenizer, return_tensors="pt")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        do_eval=args.evaluate_during_training,
        evaluation_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        #bf16=True,
        #bf16_full_eval=True,
        max_seq_length = args.block_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    # TODO: add arg to argparser for peft config
    #  - more option for target_modules outta 'last layer'
    linears_last_layer = get_last_layer_linears(model, n=args.num_trained_layers)
    if len(linears_last_layer) == 0:
        linears_last_layer = None
    
    peft_config= None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=linears_last_layer,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        lora_layers = filter(lambda p: p.requires_grad, model.parameters())


        # Prepare optimizer and schedule (linear warmup and decay)
        #optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)
        optimizer = AdamW(
            lora_layers,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
            )
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )
    CUSTOM_TRAINER_OUTPUT_DIR = args.output_dir
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
        peft_config=peft_config,
    )
    train_output = trainer.train(resume_from_checkpoint=args.should_continue)

    return train_output.global_step, train_output.training_loss


def evaluate(args, model, tokenizer, prefix="", tokenized_datasets=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    #eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    if tokenized_datasets is None:
        raw_datasets = datasets.load_dataset(
            "text",
            data_files={"test":args.eval_data_file})

        tokenized_datasets = raw_datasets.map(partial(tokenize_fn, tokenizer=tokenizer), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")
    
    test_datasets = tokenized_datasets["test"]
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = {}".format(len(test_datasets)))
    logger.info("  Batch size = {}".format(args.per_gpu_eval_batch_size * max(1, args.n_gpu)))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    data_collator = CustomCollator(tokenizer=tokenizer, return_tensors="pt")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        #bf16=True,
        #bf16_full_eval=True,
        max_seq_length = args.block_size,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_datasets,
        tokenizer=tokenizer,
    )
    eval_output = trainer.evaluate()

    result = {"eval_loss": eval_output.eval_loss}

    return result


def main():
    parser = ArgsParser()
    parser.parser.add_argument("--use_lora", action="store_true", help="Training LLM with Low-Rank Adaptation")
    parser.parser.add_argument("--lora_r", type=int, default=8, help="Dimension of adapter in LoRA")
    parser.parser.add_argument("--lora_alpha", type=float, default=8.0, help="Adapter scaler Alpha for LoRA")
    parser.parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate of LoRA Adapter.")
    parser.parser.add_argument("--num_trained_layers", type=int, default=1, help="The number of layer that will be trained")
    
    args = parser.parse()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "--eval_data_file should be specified when do_eval is true"
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("--should_continue is true, but no checkpoint found in --output_dir")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # initialize distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )


    model, tokenizer, model_class, args = get_model_tokenizer(args)
   
    logger.info("Training/evaluation parameters {}".format(args))

    
    tokenized_datasets = None
    # Training
    if args.do_train:
        # TODO: support for datasets `shuffling something`
        #train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        raw_datasets = datasets.load_dataset(
            "text",
            data_files={"train":args.train_data_file,
                        "test":args.eval_data_file}
            )
        rnd_idx = np.arange(len(
            raw_datasets["test"]))
        np.random.shuffle(rnd_idx)
        # TODO: make args for eval set size
        desired_idx = set(rnd_idx[:400])
        raw_datasets["eval"] = raw_datasets["test"].filter(lambda example, idx: idx in desired_idx, with_indices=True)

        tokenized_datasets = raw_datasets.map(partial(tokenize_fn, tokenizer=tokenizer), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")

        global_step, train_loss = train(args, tokenized_datasets, model, tokenizer)
        logger.info(" global_step = {}, average loss = {}".format(global_step, train_loss))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]

        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("models.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        else:
            last = -1
            for ckpt in glob.glob(args.output_dir+"/checkpoint*"):
                if last < os.path.getmtime(ckpt):
                    last = os.path.getmtime(ckpt)
                    checkpoints = [ckpt]
        logger.info("Evaluate the following checkpoints: {}".format(checkpoints))

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, tokenized_datasets=tokenized_datasets)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results

if __name__ == "__main__":
    main()
