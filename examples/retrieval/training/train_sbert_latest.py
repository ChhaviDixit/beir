from sentence_transformers import losses, SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from beir import util, LoggingHandler
from transformers import TrainerCallback
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pathlib
import os
import logging
import torch
import sentence_transformers
import shutil
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.device_count())  # Prints the number of available GPUs
print(torch.cuda.current_device())  # Prints the current device index
print(torch.__version__)
print(sentence_transformers.__version__)

#sys.path.append(f'{os.getcwd()}/../../../../sentence_transformers_energydistance')

# Logging setup
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Dataset setup
dataset = "hotpotqa"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# Load training data
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

# Load full dev set
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
# Convert BEIR corpus to Hugging Face Dataset
dev_data = [{"doc_id": doc_id, "text": doc["text"]} for doc_id, doc in dev_corpus.items()]
dev_dataset = Dataset.from_list(dev_data)  # Convert to Hugging Face Dataset


# SentenceTransformer model setup
#model_name = "distilbert-base-uncased_ED"
#model = SentenceTransformer("distilbert-base-uncased")
model_name = "snowflake-arctic-embed-m-v1.5_CosSim"
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
#model_name = "distilbert-base-uncased_CosSim_test"
#model = SentenceTransformer("distilbert-base-uncased")
model.to(device)
print(f"Model device: {next(model.parameters()).device}")

#retriever = TrainRetriever(model=model, batch_size=16)

# Prepare training samples
#train_samples = retriever.load_train(corpus, queries, qrels)
#train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
#train_dataset = train_dataloader.dataset  # Extract dataset from DataLoader

# Convert BEIR training samples into a Hugging Face Dataset
#train_data = [{"query": sample.texts[0], "positive": sample.texts[1]} for sample in train_samples]

# Convert to Dataset format
#train_dataset = Dataset.from_dict({key: [d[key] for d in train_data] for key in train_data[0]})


# Convert BEIR data into Hugging Face Dataset format
train_data = []
for qid, pos_doc_ids in qrels.items():
    query = queries[qid]
    for pos_id in pos_doc_ids:
        train_data.append({"query": query, "positive": corpus[pos_id]["text"]})

train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})

# Print dataset info
print(train_dataset)

# Prepare dev evaluator using the entire dev set
#ir_evaluator = model.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
# Define the evaluator using BEIR's dev set
ir_evaluator = InformationRetrievalEvaluator(
    queries=dev_queries,  # Dictionary of query_id -> query_text
    corpus=dev_corpus,    # Dictionary of doc_id -> {"text": doc_text, "title": doc_title}
    relevant_docs=dev_qrels,  # Dictionary of query_id -> {doc_id: 1, ...} for positive documents
    name="hotpotqa-dev",
    show_progress_bar=True
)



# Hyperparameter setup
learning_rates = [3e-5]
epochs_list = [10]

# Custom callback to print loss during training
class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Step {state.global_step}, Loss: {logs['loss']:.4f}")

# Custom callback to save the best model based on validation performance
class BestModelCallback(TrainerCallback):
    def __init__(self, evaluator, model_save_path, metric="ndcg_at_10"):
        self.evaluator = evaluator
        self.model_save_path = model_save_path
        self.best_score = -float("inf")
        self.metric = metric

    def on_epoch_end(self, args, state, control, **kwargs):
        results = self.evaluator.compute_metrics(state.model)
        current_score = results.get(self.metric, -float("inf"))

        if current_score > self.best_score:
            self.best_score = current_score
            state.model.save(self.model_save_path)
            print(f"New best model saved with {self.metric} = {current_score:.4f}")


# Hyperparameter tuning loop
for lr in learning_rates:
    for num_epochs in epochs_list:
        print(f"Training with learning rate: {lr}, epochs: {num_epochs}")

        # Configure optimizer and training parameters
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        warmup_steps = int(len(train_dataset) * num_epochs / 16 * 0.1)
        model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name}-hotpotqa-lr{lr}-epochs{num_epochs}-temperature20_full_dev")
        os.makedirs(model_save_path, exist_ok=True)

        # Initialize the BestModelCallback
        #best_model_callback = BestModelCallback(evaluator=ir_evaluator, model_save_path=model_save_path, metric="ndcg_at_10")
        
        training_args = SentenceTransformerTrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            learning_rate=lr,  # Specify learning rate here
            warmup_steps=warmup_steps,    # Optional: set warmup steps if needed
            logging_steps=1,  # Log frequently if needed
            save_strategy="epoch",  # Save only after each epoch
            evaluation_strategy="epoch",  # Evaluate after each epoch
            save_total_limit=1,  # Keep only the best model
            load_best_model_at_end=True,  # ✅ Load the best model after training
            metric_for_best_model="eval_hotpotqa-dev_cosine_ndcg@10",  # ✅ Use validation loss to determine the best model
            greater_is_better=True,  # ✅ Higher ndcg is better
        )

        # Initialize SentenceTransformerTrainer
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            loss=train_loss,
            evaluator=ir_evaluator,
            callbacks=[]  # Add both callbacks
        )

        # Train the model using the trainer
        trainer.train()

