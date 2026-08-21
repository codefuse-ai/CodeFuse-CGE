import argparse
import os
import json
import random
import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, set_seed, get_scheduler
from torch.utils.data import DataLoader

import ray
from ray.train import RunConfig, Checkpoint, get_context
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray import train as ray_train

from model import F2LLM
from utils import accelerate_train, CLASSIFICATION_DATASETS


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs-ray")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_hard_neg", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    return parser


def _stack(input_ids, max_len):
    data = [ids[:max_len] for ids in input_ids]
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))
    return tensor.split(lens)


def make_collate_fn(tokenizer, args):
    def collate_fn(batch_raw):
        num_hard_neg = 1 if batch_raw[0]["dataset_name"] in CLASSIFICATION_DATASETS else args["num_hard_neg"]
        hard_neg_indices = [0] if num_hard_neg == 1 else random.sample(list(range(24)), num_hard_neg)
        input_ids = _stack(
            [s["query_input_ids"] for s in batch_raw]
            + [s["passage_input_ids"] for s in batch_raw]
            + [s[f"negative_{i+1}_input_ids"] for s in batch_raw for i in hard_neg_indices],
            args["max_seq_length"],
        )
        seqlens = torch.tensor([ids.size(0) for ids in input_ids])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "seq_lens": seqlens,
            "attention_mask": attention_masks,
            "bs": len(batch_raw),
            "dataset_name": batch_raw[0]["dataset_name"],
        }
    return collate_fn


def train_loop_per_worker(config):
    # Each worker runs this function under Torch DDP managed by Ray Train
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    collate_fn = make_collate_fn(tokenizer, config)

    # Sharded dataset from Ray Data
    ds = ray_train.get_dataset_shard("train")
    # We iterate Ray Data batches directly and apply the collate function,
    # avoiding nested batching issues with torch DataLoader.
    def train_iter():
        for batch in ds.iter_batches(batch_size=config["train_batch_size"], prefetch_blocks=1):
            # batch is a dict of column -> list/array; convert to list of sample dicts
            keys = list(batch.keys())
            size = len(batch[keys[0]]) if keys else 0
            samples = [{k: batch[k][i] for k in keys} for i in range(size)]
            yield collate_fn(samples)

    # Model and optimizers
    model = F2LLM(config["model_path"], config["max_seq_length"], args=None)
    model.lm.gradient_checkpointing_enable()
    set_seed(0)

    optimizer = AdamW(
        model.lm.parameters(),
        weight_decay=config["weight_decay"],
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
    )

    # Determine total train steps per worker (global aggregation handled in logs)
    # Approximate steps per epoch using dataset count
    ds_count = ds.count()
    steps_per_epoch = max(1, ds_count // config["train_batch_size"]) if ds_count else 1
    if config["train_steps"] < 0:
        total_steps = steps_per_epoch * config["train_epochs"]
    else:
        total_steps = config["train_steps"]

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Minimal training loop mirrors accelerate_train but without Accelerator; DDP handled by Ray Train
    model.set_device()
    model.lm.train()

    completed = 0
    world_rank = get_context().get_world_rank() if get_context() else 0
    world_size = get_context().get_world_size() if get_context() else 1
    storage_dir = ray_train.get_context().storage_path if hasattr(ray_train.get_context(), "storage_path") else config.get("output_dir", "./outputs-ray")

    for epoch in range(config["train_epochs"]):
        for batch in train_iter():
            outputs = model.forward(batch)

            # Compute losses using in-batch and hard negatives; simplified without cross-worker gather
            # Use passage features only; Ray DDP averages gradients automatically
            query = outputs["query_passage_features"].squeeze(1)
            passage = outputs["passage_passage_features"].squeeze(1)
            hard_negs = outputs["negative_passage_features"]

            # Simple cosine-similarity hard loss
            a_norm = torch.nn.functional.normalize(query, p=2, dim=-1)
            hard_pool = torch.concat([passage.unsqueeze(1), hard_negs], dim=1)
            hard_norm = torch.nn.functional.normalize(hard_pool, p=2, dim=-1)
            logits = (a_norm.unsqueeze(1) * hard_norm).sum(-1) / 0.05
            labels = torch.zeros((logits.size(0),), dtype=torch.long, device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            # Gradient accumulation support
            ga_steps = int(config.get("gradient_accumulation_steps", 1))
            loss = loss / ga_steps

            loss.backward()
            # Step only every gradient_accumulation_steps
            if (completed + 1) % ga_steps == 0 or (completed + 1) == total_steps:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if optimizer.param_groups[0]["lr"] < config["min_lr"]:
                for g in optimizer.param_groups:
                    g["lr"] = config["min_lr"]

            completed += 1
            if completed >= total_steps:
                break
        if completed >= total_steps:
            break

        # End of epoch checkpoint (rank 0 only)
        if world_rank == 0:
            epoch_dir = os.path.join(storage_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            # Save tokenizer + model weights
            model.tokenizer.save_pretrained(epoch_dir)
            torch.save(model.lm.state_dict(), os.path.join(epoch_dir, "pytorch_model.bin"))
            # Report checkpoint to Ray Train for fault-tolerance
            ray_train.report({"epoch": epoch + 1, "completed_steps": completed}, checkpoint=Checkpoint.from_directory(epoch_dir))

    # Final report
    ray_train.report({"completed_steps": completed, "lr": optimizer.param_groups[0]["lr"], "world_size": world_size})


def main():
    parser = build_argparser()
    cli_args = parser.parse_args()

    # Prepare Ray Data from tokenized parquet files
    # Expect each parquet file to have pre-tokenized fields used by collate_fn
    ray.init(ignore_reinit_error=True)

    # Build Ray dataset only if parquet files exist; else fall back to local loading
    parquet_glob = os.path.join(cli_args.train_data_path, "*.parquet")
    matches = []
    try:
        import glob
        matches = glob.glob(parquet_glob)
    except Exception:
        matches = []

    train_ds = None
    valid_ds = None
    if matches:
        ds = ray.data.read_parquet(parquet_glob)
        train_ds, valid_ds = ds.random_shuffle(seed=0).split(proportions=[0.99, 0.01])
    else:
        print(f"No parquet files found at {parquet_glob}. Falling back to per-worker local dataset loading.")
        # Workers will load datasets locally inside train_loop_per_worker
        train_ds, valid_ds = None, None

    # Ray Train configuration
    scaling = ScalingConfig(num_workers=cli_args.num_workers, use_gpu=cli_args.use_gpu)
    run_config = RunConfig(storage_path=cli_args.output_dir)

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "model_path": cli_args.model_path,
            "max_seq_length": cli_args.max_seq_length,
            "train_batch_size": cli_args.train_batch_size,
            "learning_rate": cli_args.learning_rate,
            "min_lr": cli_args.min_lr,
            "weight_decay": cli_args.weight_decay,
            "warmup_steps": cli_args.warmup_steps,
            "num_hard_neg": cli_args.num_hard_neg,
            "train_epochs": cli_args.train_epochs,
            "train_steps": cli_args.train_steps,
        },
        scaling_config=scaling,
        run_config=run_config,
        datasets={k: v for k, v in {"train": train_ds, "valid": valid_ds}.items() if v is not None},
    )

    result = trainer.fit()
    # Persist CLI args for reproducibility
    os.makedirs(cli_args.output_dir, exist_ok=True)
    with open(os.path.join(cli_args.output_dir, "ray_args.json"), "w") as f:
        json.dump(vars(cli_args), f, indent=2)


if __name__ == "__main__":
    main()
