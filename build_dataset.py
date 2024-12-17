import json
import argparse
from functools import partial
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm

def format_dataset(row,dataset_type):
    if(dataset_type == "musique"):
        return row
    elif(dataset_type == "hotpotqa"):
        row["question"]= row["question_text"]
        row["answer"]=row["answer_text"]
        row["paragraphs"]=[{"idx":tmp["wikipedia_id"],"title":tmp["wikipedia_title"],"is_supporting":tmp["is_supporting"],"paragraph_text":tmp["paragraph_text"]}for tmp in row["contexts"]]
        return row
    elif(dataset_type == "2wikimultihopqa"):
        row["question"]= row["question_text"]
        row["answer"]=row["answer_text"]
        row["paragraphs"]=[{"idx":tmp["wikipedia_id"],"title":tmp["wikipedia_title"],"is_supporting":tmp["is_supporting"],"paragraph_text":tmp["paragraph_text"]}for tmp in row["contexts"]]
        return row
    else:
        assert("dataset type not recognized")
    
def get_dataset(dataset_path,dataset_type):
    if(dataset_type == "musique"):
        dataset = load_dataset('json', data_files=dataset_path)["train"] 
        return dataset.map(partial(format_dataset,dataset_type=dataset_type))
    elif(dataset_type == "hotpotqa"):
        dataset = load_dataset('json', data_files=dataset_path)["train"] 
        return dataset.map(partial(format_dataset,dataset_type=dataset_type))
    elif(dataset_type == "2wikimultihopqa"):
        with open(dataset_path,"r") as f:
            a = list(f)
        x=[json.loads(t) for t in a]
        dataset = Dataset.from_list(x)
        return dataset.map(partial(format_dataset,dataset_type=dataset_type))
    else:
        assert("dataset type not recognized")

def gold(row):
    row["gold_titles"]=[t["title"] for t in row["paragraphs"] if t["is_supporting"]]
    row["gold_paragraphs"]=[t["title"]+"\n"+t["paragraph_text"] for t in row["paragraphs"] if t["is_supporting"]]
    return row

def format_question(dataset, question_format, dataset_type):
    if(question_format=="all"):
        return dataset
    elif(question_format=="1000" and dataset_type=="2wikimultihopqa"):
        a1 = dataset.filter(lambda x:x['other_info']['type']=='comparison').select(range(250))
        a2 = dataset.filter(lambda x:x['other_info']['type']=='compositional').select(range(250))
        a3 = dataset.filter(lambda x:x['other_info']['type']=='inference').select(range(250))
        a4 = dataset.filter(lambda x:x['other_info']['type']=='bridge_comparison').select(range(250))
        new_dataset = concatenate_datasets([a1,a2,a3,a4])
        return new_dataset
    elif(question_format=="1000" and dataset_type=="hotpotqa"):
        a1 = dataset.filter(lambda x:x['type']=='comparison').select(range(500))
        a4 = dataset.filter(lambda x:x['type']=='bridge').select(range(500))
        new_dataset = concatenate_datasets([a1,a4])
        return new_dataset
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Dataset Builder")
    parser.add_argument("--original_data_path", type=str, default="raw_data/musique_dev_20k.jsonl", help="model name for evaluation")
    parser.add_argument("--query_save_path", type=str, default="data/musique_dev_query_1000.jsonl", help="model name for evaluation")
    parser.add_argument("--t2p_save_path", type=str, default="data/musique_dev_t2p.jsonl", help="model name for evaluation")
    parser.add_argument("--dataset_type", default="musique", choices=["2wikimultihopqa","musique","hotpotqa"],  help="model name for evaluation")
    parser.add_argument("--question_format", choices=["all","1000"], default="1000", help="model name for evaluation")
    
    args = parser.parse_args()

    original_dataset = get_dataset(args.original_data_path,args.dataset_type)

    query_dataset = format_question(original_dataset,args.question_format,args.dataset_type)
    query_dataset = query_dataset.map(gold).select_columns(["id",'question','answer','gold_titles'])
    query_dataset.to_json(args.query_save_path)

    data_t2p = {t["title"]: t["title"] + "\n" + t["paragraph_text"] 
                for data in tqdm(original_dataset, desc="building t2p data") 
                for t in data["paragraphs"]}

    data_t2p = [{"title": title, "passage": passage} for title, passage in data_t2p.items()]
    with open(args.t2p_save_path, "w") as f:
        f.writelines(json.dumps(item) + "\n" for item in data_t2p)