import transformers
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import json
from tqdm import tqdm
import os

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def encode_batch_contriever(data, tokenizer, model, batch_size=1024):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
            outputs = model(**inputs)
            batch_embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

def batch_similarity(search_space, queries, batch_size=1024):
    """
    Calculate similarities in batches between the search space and the queries.
    """
    results = []
    with torch.no_grad():
        queries = queries / queries.norm(dim=1, keepdim=True)
        search_space = search_space / search_space.norm(dim=1, keepdim=True)

        for i in tqdm(range(0, len(queries), batch_size), desc="Batching Similarity"):
            query_batch = queries[i:i + batch_size].to('cuda')
            similarities = torch.matmul(search_space.to('cuda'), query_batch.T).T
            ranked_indices = torch.argsort(similarities, dim=1, descending=True)

            mask = similarities > 0.8
            # Filter similarities and indices based on the mask
            similarities = similarities[mask].cpu()
            ranked_indices = ranked_indices[mask].cpu()
            results.append((ranked_indices, similarities))

    # Combine results from all batches
    all_indices = torch.cat([r[0] for r in results], dim=0)
    all_similarities = torch.cat([r[1] for r in results], dim=0)
    return all_indices, all_similarities

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Optimized Full Run")
    parser.add_argument('--dataset_entity_path', type=str, default='data/hipporag/hotpotqa_entity_list.json')
    parser.add_argument("--entity_vector_path", type=str, default="data/hipporag/hotpotqa_entity_vector.pt")
    parser.add_argument('--graph_path', type=str, default='data/hipporag/hotpotqa_graph_nonsym.json')
    parser.add_argument("--graph_save_path", type=str, default="data/hipporag/hotpotqa_graph_sym.json")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever")
    
    args = parser.parse_args()

    with open(args.dataset_entity_path, 'r') as f:
        list_entities = json.loads(f.readline())

    with open(args.graph_path, 'r') as f:
        list_graphs = json.loads(f.readline())
    graph = {}

    entity_to_index = {entity: idx for idx, entity in enumerate(list_entities)}

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    model = AutoModel.from_pretrained(args.encoder_model).to('cuda')
    vec_entities = torch.load(args.entity_vector_path)    

    # Prepare entity embeddings as queries
    entity_embeddings = encode_batch_contriever(list_entities, tokenizer, model)

    # Compute batched similarity
    indices, similarities = batch_similarity(vec_entities, entity_embeddings, batch_size=128)

    # Save top-k results with threshold filtering
    args.topk = min(len(list_entities), args.topk)

    for idx, (ranked_indices, ranked_scores) in tqdm(enumerate(zip(indices, similarities)), total=len(list_entities)):
        for i in range(args.topk):
            if ranked_scores[i] > args.threshold:
                graph[(idx, ranked_indices[i].item())] = ranked_scores[i].item()
            else:
                break

    # Save the graph
    with open(args.graph_save_path, 'w') as f:
        json.dump([[str(k[0]), str(k[1]), str(v)] for k, v in graph.items()], f)
