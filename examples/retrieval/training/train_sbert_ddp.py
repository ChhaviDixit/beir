import glob
import signal
import os
import json
import pathlib
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import TrainerCallback
from datasets import Dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# ** Initialize Distributed Training **
def setup_ddp():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    print(f"DDP initialized: Running on GPU {local_rank} of {torch.cuda.device_count()}")

# ** Get local rank (assigned by Slurm or torchrun) **
local_rank = int(os.environ.get("LOCAL_RANK", 0))

setup_ddp()

# ** Set device for this process **
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

# ** Define paths **
model_name = "snowflake-arctic-embed-m-v1.5_ED"
save_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name}-hotpotqa-lr1e-5-epochs10-temperature20_full_dev")
trainer_state_path = os.path.join(save_dir, "trainer_state.json")

os.makedirs(save_dir, exist_ok=True)

# ** Load dataset **
dataset = "hotpotqa"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
dev_data = [{"doc_id": doc_id, "text": doc["text"]} for doc_id, doc in dev_corpus.items()]
dev_dataset = Dataset.from_list(dev_data)  # Convert to Hugging Face Dataset


# ** Convert BEIR data into Hugging Face Dataset format **
train_data = [{"query": queries[qid], "positive": corpus[pos_id]["text"]} for qid, pos_doc_ids in qrels.items() for pos_id in pos_doc_ids]
train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})

# ** Create Distributed Sampler **
train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)

# ** Define evaluator (runs after each epoch) **
ir_evaluator = InformationRetrievalEvaluator(
    queries=dev_queries, corpus=dev_corpus, relevant_docs=dev_qrels,
    name="hotpotqa-dev", show_progress_bar=(local_rank == 0)
)

# ** Load model (resume if exists) **
if os.path.exists(os.path.join(save_dir, "config.json")):
    model = SentenceTransformer(save_dir)
else:
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")

model.to(device)

# ** Wrap model in DDP (only for training) **
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# ** Custom Callback to Save Best Model **
class BestModelCallback(TrainerCallback):
    def __init__(self, evaluator, save_path, metric="ndcg_at_10"):
        self.evaluator = evaluator
        self.save_path = save_path
        self.best_score = -float("inf")
        self.metric = metric

    def on_epoch_end(self, args, state, control, **kwargs):
        if local_rank == 0:  # Only rank 0 saves model
            results = self.evaluator.compute_metrics(state.model)
            current_score = results.get(self.metric, -float("inf"))

            if current_score > self.best_score:
                self.best_score = current_score
                state.model.save(self.save_path)
                print(f"New best model saved with {self.metric} = {current_score:.4f}")

# ** Handle SIGUSR1 for Checkpoint Saving **
def handle_sigusr1(signum, frame):
    if local_rank == 0:  # Only master process saves
        print("\nSIGUSR1 received. Saving checkpoint...")
        
        trainer_state = {"epoch": trainer.state.epoch}
        with open(trainer_state_path, "w") as f:
            json.dump(trainer_state, f)

        model.module.save(save_dir) if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.save(save_dir)
        print(f"Checkpoint and state saved at {save_dir}. Exiting...")
    dist.barrier()
    exit(0)

signal.signal(signal.SIGUSR1, handle_sigusr1)

# ** Load Last Epoch If Available **
start_epoch = 0
if os.path.exists(trainer_state_path):
    with open(trainer_state_path, "r") as f:
        state = json.load(f)
        start_epoch = state.get("epoch", 0)
    print(f"Resuming training from epoch {start_epoch}.")

# ** Training Arguments (Distributed) **
training_args = SentenceTransformerTrainingArguments(
    output_dir=save_dir,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=int(len(train_dataset) * 10 / 16 * 0.1),
    logging_steps=1,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_hotpotqa-dev_ndcg@10",
    greater_is_better=True,
    ddp_find_unused_parameters=False  # Optimize for DDP
)

# ** Initialize DDP**
#setup_ddp()

# ** Resume Trainer from Last Epoch **
trainer = SentenceTransformerTrainer(
    model=model.module,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    loss=losses.MultipleNegativesRankingLoss(model=model),
    evaluator=ir_evaluator if local_rank == 0 else None,  # Only rank 0 evaluates
    callbacks=[]
)

# ** Find Latest Checkpoint **
checkpoint_pattern = os.path.join(save_dir, "checkpoint-*")
existing_checkpoints = glob.glob(checkpoint_pattern)
existing_checkpoints.sort(reverse=True)

if existing_checkpoints:
    latest_checkpoint = existing_checkpoints[0]
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("No checkpoint found. Starting training from scratch.")
    trainer.train()

# ** Cleanup DDP after training**
dist.destroy_process_group()

