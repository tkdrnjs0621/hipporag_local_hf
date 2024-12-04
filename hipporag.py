import json
import logging
import os
import _pickle as pickle
from collections import defaultdict
from glob import glob

import igraph as ig
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.lm_wrapper.util import init_embedding_model
from src.named_entity_extraction_parallel import named_entity_recognition
from src.processing import processing_phrases, min_max_normalize

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

COLBERT_CKPT_DIR = "exp/colbertv2.0"


class HippoRAG:
    def __init__(self, graph_data,)
    def __init__(self, corpus_name='hotpotqa', extraction_model='openai', extraction_model_name='gpt-3.5-turbo-1106',
                 graph_creating_retriever_name='facebook/contriever', extraction_type='ner', graph_type='facts_and_sim', sim_threshold=0.8, node_specificity=True,
                 doc_ensemble=False,
                 colbert_config=None, dpr_only=False, graph_alg='ppr', damping=0.1, recognition_threshold=0.9, corpus_path=None,
                 qa_model: str = None, linking_retriever_name=None):
        
        self.corpus_name = corpus_name
        self.extraction_model_name = extraction_model_name
        self.extraction_model_name_processed = extraction_model_name.replace('/', '_')
        # self.client = init_langchain_model(extraction_model, extraction_model_name)

        assert graph_creating_retriever_name
        if linking_retriever_name is None:
            linking_retriever_name = graph_creating_retriever_name
        self.graph_creating_retriever_name = graph_creating_retriever_name  # 'colbertv2', 'facebook/contriever', or other HuggingFace models
        self.graph_creating_retriever_name_processed = graph_creating_retriever_name.replace('/', '_').replace('.', '')
        self.linking_retriever_name = linking_retriever_name
        self.linking_retriever_name_processed = linking_retriever_name.replace('/', '_').replace('.', '')

        self.extraction_type = extraction_type
        self.graph_type = graph_type
        self.phrase_type = 'ents_only_lower_preprocess'
        self.sim_threshold = sim_threshold
        self.node_specificity = node_specificity
        if colbert_config is None:
            self.colbert_config = {'root': f'data/lm_vectors/colbert/{corpus_name}',
                                   'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}
        else:
            self.colbert_config = colbert_config  # a dict, 'root', 'doc_index_name', 'phrase_index_name'

        self.graph_alg = graph_alg
        self.damping = damping
        self.recognition_threshold = recognition_threshold

        self.version = 'v3'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        try:
            self.named_entity_cache = pd.read_csv('output/{}_queries.named_entity_output.tsv'.format(self.corpus_name), sep='\t')
        except Exception as e:
            self.named_entity_cache = pd.DataFrame([], columns=['query', 'triples'])

        if 'query' in self.named_entity_cache:
            self.named_entity_cache = {row['query']: eval(row['triples']) for i, row in
                                       self.named_entity_cache.iterrows()}
        elif 'question' in self.named_entity_cache:
            self.named_entity_cache = {row['question']: eval(row['triples']) for i, row in self.named_entity_cache.iterrows()}

        self.embed_model = init_embedding_model(self.linking_retriever_name)
        self.dpr_only = dpr_only
        self.doc_ensemble = doc_ensemble
        self.corpus_path = corpus_path

        # Loading Important Corpus Files
        if not self.dpr_only:
            self.load_index_files()

            # Construct Graph
            self.build_graph()

            # Loading Node Embeddings
            self.load_node_vectors()
        else:
            self.load_corpus()

        if (doc_ensemble or dpr_only) and self.linking_retriever_name not in ['colbertv2', 'bm25']:
            # Loading Doc Embeddings
            self.get_dpr_doc_embedding()

        self.statistics = {}
        self.ensembling_debug = []
        # if qa_model is None:
        #     qa_model = LangChainModel('openai', 'gpt-3.5-turbo')
        self.qa_model = qa_model#init_langchain_model(qa_model.provider, qa_model.model_name)

    def rank_docs(self, query: str, top_k=10):
        """
        Rank documents based on the query
        @param query: the input phrase
        @param top_k: the number of documents to return
        @return: the ranked document ids and their scores
        """

        assert isinstance(query, str), 'Query must be a string'
        query_ner_list = self.query_ner(query)

        if self.doc_ensemble or self.dpr_only:
            query_embedding = self.embed_model.encode_text(query, return_cpu=True, return_numpy=True, norm=True)
            query_doc_scores = np.dot(self.doc_embedding_mat, query_embedding.T)
            query_doc_scores = query_doc_scores.T[0]

        if len(query_ner_list) > 0:  # if no entities are found, assign uniform probability to documents
            all_phrase_weights, linking_score_map = self.link_node_by_dpr(query_ner_list)

        # Run Personalized PageRank (PPR) or other Graph Algorithm Doc Scores
        if not self.dpr_only:
            if len(query_ner_list) > 0:
                combined_vector = np.max([all_phrase_weights], axis=0)

                if self.graph_alg == 'ppr':
                    ppr_phrase_probs = self.run_pagerank_igraph_chunk([all_phrase_weights])[0]
                else:
                    assert False, f'Graph Algorithm {self.graph_alg} Not Implemented'

                fact_prob = self.facts_to_phrases_mat.dot(ppr_phrase_probs)
                ppr_doc_prob = self.docs_to_facts_mat.dot(fact_prob)
                ppr_doc_prob = min_max_normalize(ppr_doc_prob)
            else:  # dpr_only or no entities found
                ppr_doc_prob = np.ones(len(self.extracted_triples)) / len(self.extracted_triples)

        # Combine Query-Doc and PPR Scores
        if self.doc_ensemble or self.dpr_only:
            # doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
            if len(query_ner_list) == 0:
                doc_prob = query_doc_scores
                self.statistics['doc'] = self.statistics.get('doc', 0) + 1
            elif np.min(list(linking_score_map.values())) > self.recognition_threshold:  # high confidence in named entities
                doc_prob = ppr_doc_prob
                self.statistics['ppr'] = self.statistics.get('ppr', 0) + 1
            else:  # relatively low confidence in named entities, combine the two scores
                # the higher threshold, the higher chance to use the doc ensemble
                doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
                query_doc_scores = min_max_normalize(query_doc_scores)

                top_ppr = np.argsort(ppr_doc_prob)[::-1][:10]
                top_ppr = [(top, ppr_doc_prob[top]) for top in top_ppr]

                top_doc = np.argsort(query_doc_scores)[::-1][:10]
                top_doc = [(top, query_doc_scores[top]) for top in top_doc]

                top_hybrid = np.argsort(doc_prob)[::-1][:10]
                top_hybrid = [(top, doc_prob[top]) for top in top_hybrid]

                self.ensembling_debug.append((top_ppr, top_doc, top_hybrid))
                self.statistics['ppr_doc_ensemble'] = self.statistics.get('ppr_doc_ensemble', 0) + 1
        else:
            doc_prob = ppr_doc_prob

        # Return ranked docs and ranked scores
        sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
        sorted_scores = doc_prob[sorted_doc_ids]

        if not (self.dpr_only) and len(query_ner_list) > 0:
            # logs
            phrase_one_hop_triples = []
            for phrase_id in np.where(all_phrase_weights > 0)[0]:
                # get all the triples that contain the phrase from self.graph_plus
                for t in list(self.kg_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1]])
                for t in list(self.kg_inverse_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1], 'inv'])

            # get top ranked nodes from doc_prob and self.doc_to_phrases_mat
            nodes_in_retrieved_doc = []
            for doc_id in sorted_doc_ids[:5]:
                node_id_in_doc = list(np.where(self.doc_to_phrases_mat[[doc_id], :].toarray()[0] > 0)[0])
                nodes_in_retrieved_doc.append([self.phrases[node_id] for node_id in node_id_in_doc])

            # get top ppr_phrase_probs
            top_pagerank_phrase_ids = np.argsort(ppr_phrase_probs, kind='mergesort')[::-1][:20]

            # get phrases for top_pagerank_phrase_ids
            top_ranked_nodes = [self.phrases[phrase_id] for phrase_id in top_pagerank_phrase_ids]
            logs = {'named_entities': query_ner_list, 'linked_node_scores': [list(k) + [float(v)] for k, v in linking_score_map.items()],
                    '1-hop_graph_for_linked_nodes': phrase_one_hop_triples,
                    'top_ranked_nodes': top_ranked_nodes, 'nodes_in_retrieved_doc': nodes_in_retrieved_doc}
        else:
            logs = {}

        return sorted_doc_ids.tolist()[:top_k], sorted_scores.tolist()[:top_k], logs

    def get_phrases_in_doc_str(self, doc: str):
        # find doc id from self.dataset_df
        try:
            doc_id = self.dataset_df[self.dataset_df.paragraph == doc].index[0]
            phrase_ids = self.doc_to_phrases_mat[[doc_id], :].nonzero()[1].tolist()
            return [self.phrases[phrase_id] for phrase_id in phrase_ids]
        except:
            return []
        
    def query_ner(self, query):
        if self.dpr_only:
            query_ner_list = []
        else:
            # Extract Entities
            try:
                if query in self.named_entity_cache:
                    query_ner_list = self.named_entity_cache[query]['named_entities']
                else:
                    query_ner_json, total_tokens = named_entity_recognition(self.client, query)
                    query_ner_list = eval(query_ner_json)['named_entities']

                query_ner_list = [processing_phrases(p) for p in query_ner_list]
            except:
                self.logger.error('Error in Query NER')
                query_ner_list = []
        return query_ner_list

    def get_neighbors(self, prob_vector, max_depth=1):

        initial_nodes = prob_vector.nonzero()[0]
        min_prob = np.min(prob_vector[initial_nodes])

        for initial_node in initial_nodes:
            all_neighborhood = []

            current_nodes = [initial_node]

            for depth in range(max_depth):
                next_nodes = []

                for node in current_nodes:
                    next_nodes.extend(self.g.neighbors(node))
                    all_neighborhood.extend(self.g.neighbors(node))

                current_nodes = list(set(next_nodes))

            for i in set(all_neighborhood):
                prob_vector[i] += 0.5 * min_prob

        return prob_vector

    def load_corpus(self):
        if self.corpus_path is None:
            self.corpus_path = 'data/{}_corpus.json'.format(self.corpus_name)
        assert os.path.isfile(self.corpus_path), 'Corpus file not found'
        self.corpus = json.load(open(self.corpus_path, 'r'))
        self.dataset_df = pd.DataFrame()
        self.dataset_df['paragraph'] = [p['title'] + '\n' + p['text'] for p in self.corpus]

    def load_index_files(self):
        index_file_pattern = 'output/openie_{}_results_{}_*.json'.format(self.corpus_name, self.extraction_model_name_processed)
        possible_files = glob(index_file_pattern)
        if len(possible_files) == 0:
            self.logger.critical(f'No extraction files found: {index_file_pattern} ; please check if working directory is correct or if the extraction has been done.')
            return
        max_samples = np.max(
            [int(file.split('{}_'.format(self.extraction_model_name_processed))[1].split('.json')[0]) for file in possible_files])
        extracted_file = json.load(open(
            'output/openie_{}_results_{}_{}.json'.format(self.corpus_name, self.extraction_model_name_processed, max_samples),
            'r'))

        self.extracted_triples = extracted_file['docs']

        if self.corpus_name == 'hotpotqa':
            self.dataset_df = pd.DataFrame([p['passage'].split('\n')[0] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        if self.corpus_name == 'hotpotqa_train':
            self.dataset_df = pd.DataFrame([p['passage'].split('\n')[0] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        elif 'musique' in self.corpus_name:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        elif self.corpus_name == '2wikimultihopqa':
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
            self.dataset_df['title'] = [s['title'] for s in self.extracted_triples]
        elif 'case_study' in self.corpus_name:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        else:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]

        if self.extraction_model_name != 'gpt-3.5-turbo-1106':
            self.extraction_type = self.extraction_type + '_' + self.extraction_model_name_processed
        self.kb_node_phrase_to_id = pickle.load(open(
            'output/{}_{}_graph_phrase_dict_{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type), 'rb'))
        # self.lose_fact_dict = pickle.load(open(
        #     'output/{}_{}_graph_fact_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
        #                                                             self.extraction_type, self.version), 'rb'))

        # try:
        #     self.relations_dict = pickle.load(open(
        #         'output/{}_{}_graph_relation_dict_{}_{}_{}.{}.subset.p'.format(
        #             self.corpus_name, self.graph_type, self.phrase_type,
        #             self.extraction_type, self.graph_creating_retriever_name_processed, self.version), 'rb'))
        # except:
        #     pass

        # self.lose_facts = list(self.lose_fact_dict.keys())
        # self.lose_facts = [self.lose_facts[i] for i in np.argsort(list(self.lose_fact_dict.values()))]
        self.phrases = np.array(list(self.kb_node_phrase_to_id.keys()))[np.argsort(list(self.kb_node_phrase_to_id.values()))]

        self.docs_to_facts_mat = pickle.load(
            open(
                'output/{}_{}_graph_doc_to_facts_csr_{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type),
                'rb'))  # (num docs, num facts)
        self.facts_to_phrases_mat = pickle.load(open(
            'output/{}_{}_graph_facts_to_phrases_csr_{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type),
            'rb'))  # (num facts, num phrases)

        self.doc_to_phrases_mat = self.docs_to_facts_mat.dot(self.facts_to_phrases_mat)
        self.doc_to_phrases_mat[self.doc_to_phrases_mat.nonzero()] = 1
        self.phrase_to_num_doc = self.doc_to_phrases_mat.sum(0).T

        graph_file_path = 'output/{}_{}_graph_mean_{}_thresh_{}_{}.subset.p'.format(self.corpus_name, self.graph_type,
                                                                                          str(self.sim_threshold), self.phrase_type,
                                                                                          self.graph_creating_retriever_name_processed)
        if os.path.isfile(graph_file_path):
            self.graph_plus = pickle.load(open(graph_file_path, 'rb'))  # (phrase1 id, phrase2 id) -> the number of occurrences
        else:
            print("GRAPH FILE NOT FOUND!!!!!!!!!!!!")
            self.logger.exception('Graph file not found: ' + graph_file_path)

    def build_graph(self):

        edges = set()

        new_graph_plus = {}
        self.kg_adj_list = defaultdict(dict)
        self.kg_inverse_adj_list = defaultdict(dict)

        for edge, weight in tqdm(self.graph_plus.items(), total=len(self.graph_plus), desc='Building Graph'):
            edge1 = edge[0]
            edge2 = edge[1]

            if (edge1, edge2) not in edges and edge1 != edge2:
                new_graph_plus[(edge1, edge2)] = self.graph_plus[(edge[0], edge[1])]
                edges.add((edge1, edge2))
                self.kg_adj_list[edge1][edge2] = self.graph_plus[(edge[0], edge[1])]
                self.kg_inverse_adj_list[edge2][edge1] = self.graph_plus[(edge[0], edge[1])]

        self.graph_plus = new_graph_plus

        edges = list(edges)

        n_vertices = len(self.kb_node_phrase_to_id)
        self.g = ig.Graph(n_vertices, edges)

        self.g.es['weight'] = [self.graph_plus[(v1, v3)] for v1, v3 in edges]
        self.logger.info(f'Graph built: num vertices: {n_vertices}, num_edges: {len(edges)}')

    def load_node_vectors(self):
        # encoded_string_path = 'data/lm_vectors/{}_mean/encoded_strings.txt'.format(self.linking_retriever_name_processed)
        # if self.linking_retriever_name == 'colbertv2':
        #     return
        kb_node_phrase_embeddings_path = 'output/{}_mean_{}_kb_node_phrase_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name)
        # if os.path.isfile(kb_node_phrase_embeddings_path):
        #     self.kb_node_phrase_embeddings = pickle.load(open(kb_node_phrase_embeddings_path, 'rb'))
        #     if len(self.kb_node_phrase_embeddings.shape) == 3:
        #         self.kb_node_phrase_embeddings = np.squeeze(self.kb_node_phrase_embeddings, axis=1)
        #     self.logger.info('Loaded phrase embeddings from: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))
        # else:
        self.kb_node_phrase_embeddings = self.embed_model.encode_text(self.phrases.tolist(), return_cpu=True, return_numpy=True, norm=True)
        pickle.dump(self.kb_node_phrase_embeddings, open(kb_node_phrase_embeddings_path, 'wb'))
        self.logger.info('Saved phrase embeddings to: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))

    def get_dpr_doc_embedding(self):
        cache_filename = 'output/{}_mean_{}_doc_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name)
        # if os.path.exists(cache_filename):
        #     self.doc_embedding_mat = pickle.load(open(cache_filename, 'rb'))
        #     self.logger.info(f'Loaded doc embeddings from {cache_filename}, shape: {self.doc_embedding_mat.shape}')
        # else:
        self.doc_embeddings = []
        self.doc_embedding_mat = self.embed_model.encode_text(self.dataset_df['paragraph'].tolist(), return_cpu=True, return_numpy=True, norm=True)
        pickle.dump(self.doc_embedding_mat, open(cache_filename, 'wb'))
        self.logger.info(f'Saved doc embeddings to {cache_filename}, shape: {self.doc_embedding_mat.shape}')

    def run_pagerank_igraph_chunk(self, reset_prob_chunk):
        """
        Run pagerank on the graph
        :param reset_prob_chunk:
        :return: PageRank probabilities
        """
        pageranked_probabilities = []

        for reset_prob in tqdm(reset_prob_chunk, desc='pagerank chunk'):
            pageranked_probs = self.g.personalized_pagerank(vertices=range(len(self.kb_node_phrase_to_id)), damping=self.damping, directed=False,
                                                            weights='weight', reset=reset_prob, implementation='prpack')

            pageranked_probabilities.append(np.array(pageranked_probs))

        return np.array(pageranked_probabilities)

    def link_node_by_dpr(self, query_ner_list: list):
        """
        Get the most similar phrases (as vector) in the KG given the named entities
        :param query_ner_list:
        :return:
        """
        query_ner_embeddings = self.embed_model.encode_text(query_ner_list, return_cpu=True, return_numpy=True, norm=True)

        # Get Closest Entity Nodes
        prob_vectors = np.dot(query_ner_embeddings, self.kb_node_phrase_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)

        linked_phrase_ids = []
        max_scores = []

        for prob_vector in prob_vectors:
            phrase_id = np.argmax(prob_vector)  # the phrase with the highest similarity
            linked_phrase_ids.append(phrase_id)
            max_scores.append(prob_vector[phrase_id])

        # create a vector (num_phrase) with 1s at the indices of the linked phrases and 0s elsewhere
        # if node_specificity is True, it's not one-hot but a weight
        all_phrase_weights = np.zeros(len(self.phrases))

        for phrase_id in linked_phrase_ids:
            if self.node_specificity:
                if self.phrase_to_num_doc[phrase_id] == 0:  # just in case the phrase is not recorded in any documents
                    weight = 1
                else:  # the more frequent the phrase, the less weight it gets
                    weight = 1 / self.phrase_to_num_doc[phrase_id]

                all_phrase_weights[phrase_id] = weight
            else:
                all_phrase_weights[phrase_id] = 1.0

        linking_score_map = {(query_phrase, self.phrases[linked_phrase_id]): max_score
                             for linked_phrase_id, max_score, query_phrase in zip(linked_phrase_ids, max_scores, query_ner_list)}
        return all_phrase_weights, linking_score_map
