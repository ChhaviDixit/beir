import os
import pathlib
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# ** Get local rank **
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# ** Initialize DDP **
def setup_ddp():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    print(f"DDP initialized: Running on GPU {local_rank} of {torch.cuda.device_count()}")

setup_ddp()

device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

print(device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.device_count())  # Prints the number of available GPUs
print(torch.cuda.current_device())  # Prints the current device index
print(torch.__version__)
#print(sentence_transformers.__version__)


# ** Paths **
model_name = "snowflake-arctic-embed-m-v1.5_ED"
save_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name}-hotpotqa-lr1e-5-epochs10-temperature20_full_dev")
os.makedirs(save_dir, exist_ok=True)

# ** Load dataset **
dataset = "hotpotqa"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

# Create InputExample list
train_examples = []
for qid, pos_ids in qrels.items():
    query = queries[qid]
    for pos_id in pos_ids:
        if pos_id in corpus:
            pos_text = corpus[pos_id]["text"]
            train_examples.append(InputExample(texts=[query, pos_text]))


# ** Load model **
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
model.to(device)

# ** Wrap in DDP if needed **
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

# Create tokenizing dataset
#train_dataset = SentenceTransformerDataset(train_examples, model.module)  # DDP wraps model
train_data = [{"query": example.texts[0], "text": example.texts[1]} for example in train_examples]
train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})
#train_dataset = train_examples

# ** Create DDP Sampler **
train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)


# ** Training Args **
training_args = SentenceTransformerTrainingArguments(
    output_dir=save_dir,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=int(len(train_dataset) * 10 / 16 * 0.1),
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    save_total_limit=10,
    ddp_find_unused_parameters=False
)

# ** Train **
trainer = SentenceTransformerTrainer(
    model=model.module,
    args=training_args,
    train_dataset=train_dataset,
    loss=losses.MultipleNegativesRankingLoss(model=model.module),
    eval_dataset=None,
    callbacks=[]
)

#torch.autograd.set_detect_anomaly(True)

trainer.train()

# Cleanup
dist.destroy_process_group()

