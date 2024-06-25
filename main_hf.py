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
        tokenizer.pad_token = tokenizer.eos_token
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


def get_training_info(dataloader, args):
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")
    return global_step, epochs_trained, steps_trained_in_current_epoch



def train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, tr_loss, logging_loss, global_step, steps_trained_in_current_epoch, tb_writer, args):
    """train one epoch"""
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        model.train()
        outputs = model(inputs, labels=labels)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # Log metrics
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if (args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            # save checkpoint
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.evaluate_during_training:
                    save_checkpoint(model, optimizer, scheduler, tokenizer, args, global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    return model, optimizer, scheduler, global_step, tr_loss, logging_loss

def get_num_layers(model):
    numbers = set()
    for name, _ in model.named_parameters():
        for number in re.findall(r'\d+', name):
            numbers.add(int(number))
    return max(numbers)

def get_last_layer_linears(model):
    names = []
    
    num_layers = get_num_layers(model)
    for name, module in model.named_modules():
        if str(num_layers) in name and not "encoder" in name:
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

def train(args, tokenized_datasets, model, tokenizer):
    global CUSTOM_TRAINER_OUTPUT_DIR
    # total iteration and batch size
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(tokenized_datasets["train"]) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(tokenized_datasets["train"]) // args.gradient_accumulation_steps * args.num_train_epochs

    total_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)

    #if args.fp16:
    #    try:
    #        from apex import amp
    #    except ImportError:
    #        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    ## multi-gpu training
    #if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    #    model = torch.nn.DataParallel(model)

    ## Distributed training
    #if args.local_rank != -1:
    #    model = torch.nn.parallel.DistributedDataParallel(
    #        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    #    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = {}".format(len(tokenized_datasets["train"])))
    logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(total_batch_size))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(t_total))

    # is it really unnessasary? I'm scared to delete this, so I make it as comment and leave it.
    #global_step, epochs_trained, steps_trained_in_current_epoch = get_training_info(tokenized_datasets["train"], args)


    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.train()
    model.zero_grad()
    
    # TODO: remove below commented code
    #train_iterator = trange(
    #    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    #)
    #for _ in train_iterator:
    #    model, optimizer, scheduler, global_step, tr_loss, logging_loss = train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, tr_loss, logging_loss, global_step,
    #                              steps_trained_in_current_epoch, tb_writer, args)
    #    if args.max_steps > 0 and global_step > args.max_steps:
    #        train_iterator.close()
    #        break
    #if args.local_rank in [-1, 0]:
    #    tb_writer.close()


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
    linears_last_layer = get_last_layer_linears(model)
    if len(linears_last_layer) == 0:
        linears_last_layer = None
    
    peft_config= None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            #target_modules=linears_last_layer,
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
    train_output = trainer.train()

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
            data_files={"eval":args.eval_data_file})
        def tokenize_fn(example):
            be = tokenizer(example["text"], truncation=True)
            return be

        tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")

    # Prepare dataloader
    #eval_dataloader, args = get_dataloader(eval_dataset, tokenizer, args, split='eval')

    ## multi-gpu evaluate
    #if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    #    model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = {}".format(len(tokenized_datasets["eval"])))
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
    # TODO: add arg to argparser for peft config
    #  - more option for target_modules outta 'last layer'
    linears_last_layer = get_last_layer_linears(model)
    if len(linears_last_layer) == 0:
        linears_last_layer = None 

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        #train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        #optimizers=(optimizer, scheduler),
        #peft_config=peft_config,
    )
    eval_output = trainer.evaluate()


    result = {"eval_loss": eval_output.eval_loss}

    # # TODO: remove it, it was already wrote by CustomTrainer
    #output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    #with open(output_eval_file, "w") as writer:
    #    logger.info("***** Eval results {} *****".format(prefix))
    #    for key in sorted(result.keys()):
    #        logger.info("  %s = %s", key, str(result[key]))
    #        writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = ArgsParser()
    parser.parser.add_argument("--use_lora", action="store_true", help="Training LLM with Low-Rank Adaptation")
    parser.parser.add_argument("--lora_r", type=int, default=8, help="Dimension of adapter in LoRA")
    parser.parser.add_argument("--lora_alpha", type=float, default=8.0, help="Adapter scaler Alpha for LoRA")
    parser.parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate of LoRA Adapter.")
    
    args = parser.parse()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "--eval_data_file should be specified when do_eval is true"
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("--should_continue is true, but no checkpoint found in --output_dir")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # if not the first process, do not load pretrained model & vocab

    model, tokenizer, model_class, args = get_model_tokenizer(args)
   
    if args.local_rank == 0:
        torch.distributed.barrier()  # finish barrier, when first process has loaded pretrained model & vocab

    logger.info("Training/evaluation parameters {}".format(args))

    
    tokenized_datasets = None
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # only first process will preprocess data/caching
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

        def tokenize_fn(example):
            be = tokenizer(example["text"], truncation=True)
            return be

        tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")

        if args.local_rank == 0:
            torch.distributed.barrier() # end of barrier

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
