# HPML Project: IBM Project 8: Energy Distance in IR Tasks

## Team Information
- **Members**:
  - Chhavi Dixit (CD3496)
  - Chandhru Karthick (CK3255)
  - Elie Gross (EG3346)

---

## Problem Statement
Cosine similarity uses only the [CLS] token for both query and document representation. It specifically focuses on the angle between the two vectors and overlooks statistical distribution of data. This leads to loss of information for long context retrieval while there is no actual “distance” between query and vector. The project is exploring new distance metrics and use-cases to be used alongside cosine similarity. The focus here is on retrieval tasks, as improving retrieval is a huge bump in modern language model pipelines where retrieval errors propagate multiplicatively down the line. If it can generalize well to long queries as expected we would be able to deal with long context IR with better precision. We are exploring three modeuls, exploring hamming distance within Energy distance, exploring JS Diveregence a an alternate distance metric, testing performance on other benchmark datasets. This repository focuses on implementing and testing JS Divergence as distance metric.

---

## Model Description
The model uses embeddings learned from the distilbert model on HotPotQA dataset's train and dev set. In the distance metrics, the JS divergence function is defined in "Sentence-transformers" and the "MTEB" repositories to be used for training and testing respectively. This repository is used for training the embedding models with the new distance metric.

## Final Results
Information Clamping in range epsilon=1e-6
| Metric               | Value       |
|----------------------|-------------|
| ndcg at 1 | 0.0023      |
| ndcg at 3    | 0.00282    |
| ndcg at 5           | 0.00349       |
| ndcg at 10      | 0.00443       |
| ndcg at 100  | 0.00557        |
| ndcg at 1000               | 0.02107 |

Information Clamping in range epsilon=1e-5
| Metric               | Value       |
|----------------------|-------------|
| ndcg at 1 | 0.07549      |
| ndcg at 3    | 0.06435    |
| ndcg at 5           | 0.07129       |
| ndcg at 10      | 0.0821       |
| ndcg at 100  | 0.10972        |
| ndcg at 1000               | 0.13704 |

Hence, reducing the range gives better results.


## Reproducibility Instructions: Energy Distance Project Training and Inference

### Setting up Python Environment and Installing Required Libraries
1. conda create --name myenv39 python=3.9
2. conda activate myenv39
3. pip install --upgrade pip --index-url https://pypi.org/simple
4. pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
5. git clone https://github.com/gnatesan/sentence-transformers-3.4.1.git
6. git clone https://github.com/gnatesan/mteb-1.34.14.git
7. pip install -e /path_to_sentence-transformers/sentence-transformers-3.4.1
8. pip install -e /path_to_mteb/mteb-1.34.14
9. git clone https://github.com/gnatesan/beir.git

### Wandb Dashboard
View training and evaluation metrics here: https://wandb.ai/wisebayes-columbia-university/HPML-Energy?nw=nwuserwisebayes

### Sanity Check
1. conda create --name testenv python=3.9
2. conda activate testenv
3. pip install --upgrade pip --index-url https://pypi.org/simple
4. pip install sentence-transformers==3.4.1
5. pip install mteb==1.34.26
6. sbatch inference_CosSim.sh (Make sure the batch script calls eval_dataset.py and a baseline model is being used. *i.e. model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")*)
7. Cross reference the inference results with what is on the leaderboard. https://huggingface.co/spaces/mteb/leaderboard

### Model Training
1. cd /path_to_beir/beir/examples/retrieval/training
2. Before running training, make sure the model, model_name, and hyperparameters (LR, scale) are correct. 
nano train_sbert_latest_2.py or nano train_sbert_ddp_2.py to change model, model_name, and LR. 
nano sentence-transformers-3.4.1/sentence-transformers/losses/MultipleNegativesRankingLoss.py to change scale. 
3. sbatch train.sh OR sbatch train_ddp.sh if using multiple GPUs
4. Trained model will be saved in /path_to_beir/beir/examples/retrieval/training/output

### Model Evaluation
1. sbatch inference_ED.sh if evaluating an ED trained model (myenv39 conda environment must be setup)
2. sbatch inference_CosSim.sh if evaluating a cosine similarity trained model (testenv conda environment must be setup)
3. Make sure the proper python script in the batch file is being run (if evaluating entire dataset or subset based on query lengths)

### IMPORTANT FILES
1. train.sh - Batch script to run model training on a single GPU.  
2. train_ddp.sh - Batch script to run model training on multiple GPUs. Make sure number of GPUs requested are properly set.
3. inference_ED.sh - Batch script to run inference on an ED trained model. Can run on either entire dataset or subset based on query lengths.
4. inference_CosSim.sh Batch script to run inference on a CosSim trained model. Can run on either entire dataset or subset based on query lengths.
5. train_sbert_latest_2.py - Python script to run model training on a single GPU. Uses ir_evaluator to evaluate on a dev set after each epoch of training and only saves the best model, make sure ir_evaluator is enabled.
6. train_sbert_ddp_2.py - Python script to run model training on multiple GPUs using DDP. Currently does not use an ir_evaluator to evaluate on a dev set after each epoch of training.
7. eval_dataset.py - Python script to run inference on entire BEIR dataset.
8. eval_dataset_subset_length.py - Python script to run inference on subset of BEIR dataset based on query lengths.

### IMPORTANT NOTES
1. All files used for training should be present when you clone the beir repository in beir/examples/retrieval/training folder.
