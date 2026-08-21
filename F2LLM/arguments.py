from dataclasses import dataclass, asdict
import argparse, json


@dataclass
class Args:

    model_path: str
    experiment_id: str
    # save dir
    output_dir: str
    tb_dir: str
    cache_dir: str
    # training arguments
    train_data_path: str
    train_batch_size: int = 8
    max_seq_length: int = 2048
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    # MRL
    use_mrl: bool = False
    # knowledge distillation (kd)
    teacher_model_path: str = None
    kd_weight: float = 1.0     # weight for the distillation loss
    # cosine or constant_with_warmup
    scheduler_type: str = "cosine"
    # embedding-related settings
    num_hard_neg: int = 7
    num_hard_neg_clustering: int = 9
    # train steps take precedence over epochs, set to -1 to disable
    train_steps: int = -1
    train_epochs: int = 5
    log_interval: int = 20
    checkpointing_steps: int = 100
    validation_steps: int = 100
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # just placeholder, for logging purpose
    num_processes: int=0

    def dict(self):
        return asdict(self)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arg = parser.parse_args()
    with open(arg.config) as f:
        config = json.load(f)
    args = Args(**config)
    args.output_dir = f"{args.output_dir}/{args.experiment_id}"
    args.tb_dir = f"{args.tb_dir}/{args.experiment_id}"
    assert args.scheduler_type in ["cosine", "constant_with_warmup"], "unsupported scheduler"
    return args