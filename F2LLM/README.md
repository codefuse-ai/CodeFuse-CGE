## F2LLM

F2LLM-v2 is a new family of general-purpose, multilingual embedding models in 8 distinct sizes ranging from 80M to 14B. Trained on a newly curated composite of 60 million publicly available high-quality data, F2LLM-v2 supports more than 200 languages, with a particular emphasis on previously underserved mid- and low-resource languages.

F2LLMs are fully open. Models, data, and intermediate checkpoints are available at:

- Stage-1 model and checkpoints: [🤗F2LLM-v2-0.6B-Preview](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B-Preview), [🤗F2LLM-v2-1.7B-Preview](https://huggingface.co/codefuse-ai/F2LLM-v2-1.7B-Preview), [🤗F2LLM-v2-4B-Preview](https://huggingface.co/codefuse-ai/F2LLM-v2-4B-Preview), [🤗F2LLM-v2-8B-Preview](https://huggingface.co/codefuse-ai/F2LLM-v2-8B-Preview), [🤗F2LLM-v2-14B-Preview](https://huggingface.co/codefuse-ai/F2LLM-v2-14B-Preview)
- Stage-2 model and checkpoints: [🤗F2LLM-v2-80M](https://huggingface.co/codefuse-ai/F2LLM-v2-80M), [🤗F2LLM-v2-160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M), [🤗F2LLM-v2-330M](https://huggingface.co/codefuse-ai/F2LLM-v2-330M), [🤗F2LLM-v2-0.6B](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B), [🤗F2LLM-v2-1.7B](https://huggingface.co/codefuse-ai/F2LLM-v2-1.7B), [🤗F2LLM-v2-4B](https://huggingface.co/codefuse-ai/F2LLM-v2-4B), [🤗F2LLM-v2-8B](https://huggingface.co/codefuse-ai/F2LLM-v2-8B), [🤗F2LLM-v2-14B](https://huggingface.co/codefuse-ai/F2LLM-v2-14B)
- Data: [🤗F2LLM-v2 data](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2)

Models and data of V1 are available at:

- Model: [🤗F2LLM 0.6B](https://huggingface.co/codefuse-ai/F2LLM-0.6B), [🤗F2LLM 1.7B](https://huggingface.co/codefuse-ai/F2LLM-1.7B), [🤗F2LLM 4B](https://huggingface.co/codefuse-ai/F2LLM-4B).
- Data: [🤗F2LLM data](https://huggingface.co/datasets/codefuse-ai/F2LLM).

### Train

🔥🔥 **New features in V2**:

- Add support for Matryoshka Representation Learning (MRL)
- Add support for knowledge distillation (KD)
- Switched to a more disk-efficient data format.

In this repo we provide a streamlined and efficient script for training embedding models. To reproduce the training of F2LLMs, please:

- Setup environment following `requirements.txt`. We note that transformers>=4.51.0 is required for training Qwen3 models.
- Download data and backbone models from Hugging Face (we use Qwen3 models).
- Run `tokenize_data_qwen.py` to tokenize the downloaded data, and then cancatenate all corpus files into a single `corpus.parquet` file.
- Modify model path, data path, and other arguments in `configs/config.json`.
- Start training with `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.json`.
## Ray Distributed Training

- Install Ray: `pip install -r requirements.txt`
- Launch local Ray training:

```bash
python ray_run.py \
  --model_path Qwen/Qwen2.5-7B \
  --output_dir ./outputs-ray \
  --cache_dir ./cache \
  --train_data_path ./training_data/data_tokenized_qwen \
  --max_seq_length 1024 \
  --train_batch_size 2 \
  --train_epochs 1 \
  --train_steps -1 \
  --use_gpu \
  --num_workers 2
```

- Multi-node: start a Ray cluster (see Ray docs) and submit the job via `ray job submit` or run on the head node; the script uses `TorchTrainer` with DDP and reports checkpoints to Ray storage. Checkpoints are saved under `outputs-ray/epoch_*` and can be used for fault-tolerant restarts.

Notes:
- This Ray runner consumes the same tokenized parquet fields as the Accelerate pipeline.
- Cross-worker in-batch retrieval loss is simplified initially; extendable via Ray Train collectives.

Note: we recommend setting `num_processes` to 1 in `configs/accelerate_config.yaml` and launch the training code once to generate cache for training data before starting the actual training.

For multi-node training, run on the main node:

```
accelerate launch --config_file configs/accelerate_config.yaml --num_machines N_NODE --num_processes N_PROCESSES --machine_rank 0 --main_process_ip MASTER_IP --main_process_port MASTER_PORT run.py --config configs/config.json
```

where N_NODE is the number of machines; N_PROCESSES is N_NODE\*8; MASTER_IP is the IP address of your master node, and MASTER_PORT is a port available on your machine (e.g. 6379).

On worker nodes, also run the above commmand but modify `machine_rank` accordingly.

### Citation

If you use the F2LLM models, data, or code, please cite the following technical reports.

```
@article{2026F2LLM-v2,
  title={F2LLM-v2: Inclusive, Performant, and Efficient Embeddings for a Multilingual World}, 
  author={Ziyin Zhang and Zihan Liao and Hang Yu and Peng Di and Rui Wang},
  journal      = {CoRR},
  volume       = {abs/2603.19223},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2603.19223},
  doi          = {10.48550/ARXIV.2603.19223},
  eprinttype    = {arXiv},
  eprint       = {2603.19223}
}
```

```
@article{2025F2LLM,
  title={F2LLM Technical Report: Matching SOTA Embedding Performance with 6 Million Open-Source Data},
  author={Ziyin Zhang and Zihan Liao and Hang Yu and Peng Di and Rui Wang},
  journal      = {CoRR},
  volume       = {abs/2510.02294},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2510.02294},
  doi          = {10.48550/ARXIV.2510.02294},
  eprinttype    = {arXiv},
  eprint       = {2510.02294}
}
```
