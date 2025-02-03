from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from transformers import EarlyStoppingCallback
import pathlib
import os
import sys
import random
import logging
import torch
import sentence_transformers

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.device_count())  # Prints the number of available GPUs
print(torch.cuda.current_device())  # Prints the current device index
print(torch.__version__)
print(sentence_transformers.__version__)


sys.path.append(f'{os.getcwd()}/../../../../sentence_transformers_energydistance')

# Logging setup
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Dataset setup
#dataset = "hotpotqa"
dataset = "fever"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# Load training data
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

# Load full dev set
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

# Subset selection
def create_subset(dev_corpus, dev_queries, dev_qrels, max_docs=20000, random_seed=42):
    # Fix random seed for deterministic behavior
    random.seed(random_seed)

    # Collect all relevant document IDs from qrels
    relevant_docs = set([doc_id for query_id in dev_qrels for doc_id in dev_qrels[query_id]])

    # Add irrelevant documents to reach the max_docs limit
    irrelevant_docs = [doc_id for doc_id in dev_corpus if doc_id not in relevant_docs]
    selected_irrelevant_docs = random.sample(irrelevant_docs, min(max_docs - len(relevant_docs), len(irrelevant_docs)))

    # Final subset corpus includes all relevant docs and a fixed set of irrelevant docs
    subset_corpus = {doc_id: dev_corpus[doc_id] for doc_id in list(relevant_docs) + selected_irrelevant_docs}

    return subset_corpus, dev_queries, dev_qrels

def subset_dev_set(queries, qrels, corpus, subset_fraction=0.2, seed=42):
    # Step 1: Set the random seed for deterministic sampling
    random.seed(seed)
    
    # Step 2: Sample 20% of the queries
    query_ids = list(queries.keys())
    num_subset_queries = int(len(query_ids) * subset_fraction)
    selected_query_ids = random.sample(query_ids, num_subset_queries)

    # Step 3: Filter the relevant documents for the selected queries
    subset_qrels = {}
    subset_queries = {}
    subset_corpus_ids = set()

    for query_id in selected_query_ids:
        if query_id in qrels:
            subset_qrels[query_id] = qrels[query_id]
            subset_queries[query_id] = queries[query_id]

            # Collect all relevant document IDs for the selected queries
            for corpus_id, score in qrels[query_id].items():
                if score >= 1:
                    subset_corpus_ids.add(corpus_id)

    # Step 4: Create a new subset of the corpus with only relevant documents
    subset_corpus = {doc_id: corpus[doc_id] for doc_id in subset_corpus_ids}

    print(f"Deterministic subset contains {len(subset_queries)} queries and {len(subset_corpus)} documents.")
    return subset_queries, subset_qrels, subset_corpus

# Example usage, random sampling
#subset_queries, subset_qrels, subset_corpus = subset_dev_set(dev_queries, dev_qrels, dev_corpus, subset_fraction=0.2, seed=42)

# Create the subset corpus
#subset_corpus, subset_queries, subset_qrels = create_subset(dev_corpus, dev_queries, dev_qrels)

# SentenceTransformer model setup
model_name = "distilbert-base-uncased_ED"
model = SentenceTransformer("distilbert-base-uncased")
model.to('cuda')
print(f"Model device: {next(model.parameters()).device}")

retriever = TrainRetriever(model=model, batch_size=16)

# Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

# Prepare dev evaluator using the subset, instead of using the subset lets use the entire dev set
#ir_evaluator = retriever.load_ir_evaluator(subset_corpus, subset_queries, subset_qrels)
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

# Hyperparameter optimization setup
#learning_rates = [1e-5, 2e-5, 4e-5]
#epochs_list = [1, 3, 5]
learning_rates = [3e-5]
epochs_list = [10]



class EarlyStoppingCallback:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.patience_counter = 0

    def __call__(self, score, epoch, steps):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.patience_counter += 1
            print(f"No improvement in score. Patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                print("Early stopping triggered")
                raise KeyboardInterrupt  # Stop training early
        else:
            self.best_score = score
            self.patience_counter = 0
            print(f"New best score: {self.best_score}")

# Early stopping parameters
patience = 3  # Number of evaluations to wait for improvement

# Hyperparameter tuning loop
for lr in learning_rates:
    for num_epochs in epochs_list:
        print(f"Training with learning rate: {lr}, epochs: {num_epochs}")

        # Configure optimizer and training parameters
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
        warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
        #evaluation_steps = int(len(train_samples) // retriever.batch_size * 0.5)
        model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name}-fever-lr{lr}-epochs{num_epochs}-temperature50_full_dev")
        #model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name}-hotpotqa-lr{lr}-epochs{num_epochs}-temperature200")
        os.makedirs(model_save_path, exist_ok=True)

        # Initialize EarlyStoppingCallback
        early_stopping_callback = EarlyStoppingCallback(patience=3, min_delta=0.0001)

        # Train the model with the given training objective
        retriever.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=ir_evaluator,
            epochs=num_epochs,
            output_path=model_save_path,
            optimizer_params={"lr": lr},
            warmup_steps=warmup_steps,
            #evaluation_steps=evaluation_steps,  # Evaluate twice per epoch
            save_best_model=True,
            use_amp=True
            #callback=early_stopping_callback  # Add EarlyStoppingCallback here
            #callbacks = [early_stopping_callback] #For newer version of sentence-transformers
        )
