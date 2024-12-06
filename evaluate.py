import argparse
from functools import partial
from datasets import load_dataset

def map_calc(row, topks):
    gold = set(row['gold_titles'])
    retrieved = {k:set(row['retrieved_titles'][:k]) for k in topks}
    precision = {}
    recall = {}
    
    for k, retrieved_titles in retrieved.items():
        retrieved_set = set(retrieved_titles)
        true_positives = len(gold.intersection(retrieved_set))

        precision[k] = true_positives / k if k > 0 else 0
        recall[k] = true_positives / len(gold) if len(gold) > 0 else 0
    
    row['precision'] = precision
    row['recall'] = recall

    return row

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--result_path', type=str, default='data/hotpotqa_query_ner_relevant.jsonl')
    parser.add_argument('--topks', type=int, nargs='+', default=['2','5','10','20'], help="A list of numbers")
    parser.add_argument("--save_mapped_path", type=str, default="data/hotpotqa_graph_sym.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_result_path", type=str, default="data/hotpotqa_graph_sym.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    args = parser.parse_args()
    
    dataset = load_dataset('json', data_files=args.result_path)["train"]
    dataset = dataset.map(partial(map_calc,topks=args.topks))

    precision_accumulator = {k: 0 for k in args.topks}
    recall_accumulator = {k: 0 for k in args.topks}
    num_rows = len(dataset)
    
    for row in dataset:
        for k in args.topks:
            precision_accumulator[k] += row['precision'][k]
            recall_accumulator[k] += row['recall'][k]
    
    average_precision = {k: precision_accumulator[k] / num_rows for k in args.topks}
    average_recall = {k: recall_accumulator[k] / num_rows for k in args.topks}
    average_f1 = {k: 2*average_precision*average_recall/(average_recall+average_precision) if average_precision+average_recall!=0 else 0 for k in args.topks}
    reformatted = [{'k':k,'precision':average_precision[k], 'recall':average_recall[k], 'f1':average_f1} for k in args.topks]

    dataset.to_json(args.save_mapped_path, orient="records", lines=True)