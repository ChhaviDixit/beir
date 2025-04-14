import os
import glob
import pandas as pd
import torch
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from beir.datasets.data_loader import GenericDataLoader
from beir import util


def evaluate_on_gpu(rank, gpu_id, checkpoints, dataset, save_dir, corpus, queries, qrels):
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] Starting evaluation for {len(checkpoints)} checkpoints")

    results_file = os.path.join(save_dir, f"eval_results_gpu{gpu_id}.csv")
    results = []

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name=f"{dataset}-dev",
        show_progress_bar=False
    )

    for ckpt in checkpoints:
        print(f"[GPU {gpu_id}] Evaluating {ckpt}")
        model = SentenceTransformer(ckpt, device=f"cuda:{gpu_id}")
        scores = evaluator(model)

        row = {"checkpoint": os.path.basename(ckpt)}
        row.update(scores)
        results.append(row)

        pd.DataFrame(results).to_csv(results_file, index=False)

    print(f"[GPU {gpu_id}] Done!")


def main():
    save_dir = "/gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/output/snowflake-arctic-embed-m-v1.5_ED-hotpotqa-lr3e-5-epochs10-temperature20_full_dev/"  # <- Replace with your actual path
    dataset = "hotpotqa" # Make sure the correct dataset is being used

    data_path = util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
        "datasets"
    )
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")

    all_checkpoints = sorted(glob.glob(os.path.join(save_dir, "checkpoint-*")), key=os.path.getmtime)
    num_gpus = 5 # Set this value how many GPUs were requested in batch script
    checkpoints_per_gpu = len(all_checkpoints) // num_gpus

    processes = []
    for i in range(num_gpus):
        start = i * checkpoints_per_gpu
        end = len(all_checkpoints) if i == num_gpus - 1 else (i + 1) * checkpoints_per_gpu
        ckpt_subset = all_checkpoints[start:end]

        p = mp.Process(
            target=evaluate_on_gpu,
            args=(i, i, ckpt_subset, dataset, save_dir, corpus, queries, qrels)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge all GPU-specific result files
    result_files = glob.glob(os.path.join(save_dir, "eval_results_gpu*.csv"))
    all_dfs = [pd.read_csv(f) for f in result_files]
    merged_df = pd.concat(all_dfs, ignore_index=True)

    final_results_file = os.path.join(save_dir, f"Information-Retrieval_evaluation_{dataset}-dev_results.csv")
    merged_df.to_csv(final_results_file, index=False)

    print(f"\n✅ All checkpoints evaluated and results saved to: {final_results_file}")

    # Cleanup temporary GPU result files
    for f in result_files:
        os.remove(f)
    print("🧹 Temporary GPU result files deleted.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

