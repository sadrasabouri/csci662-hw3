import json
from tqdm import tqdm
import argparse
from bm25 import * 
from tfidf import *
from st_se import *

def get_arguments():
    """
    Return the arguments for the evaluator main module.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default="bm25", help="retriever model: name of the retriever model you used.")
    parser.add_argument("-n", help="index name: the name of the index that you created with 'index.py'.")
    
    parser.add_argument("-k", help="top k: the number of documents to return in each retrieval run.")

    parser.add_argument("-q", help="question files")
    parser.add_argument("-o", help="output file name to include hte final statistics")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()

    if args.r == "bm25":
        retriever = BM25(args.n).load_model()
    elif args.r == "tfidf":
        retriever = TFIDF(args.n).load_model()
    elif args.r == "st_se":
        retriever = SentenceTransformersSentenceEmbedding(args.n).load_model()
    else:
        # TODO add at least one more retriever
        retriever = None
    
    with open(args.q, 'r') as f:
        questions = json.load(f)
    scores = {
        'jenson': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    # use the search_batch method if you implemented it for faster evaluation
    retrieved_batches = retriever.search_batch([q['question'] for q in questions], int(args.k), return_scores=True)
    for q, docs in zip(tqdm(questions, desc="Answering questions", unit="question"), retrieved_batches):
        pred_doc_ids = [doc['id'] for doc in docs]
        act_doc_ids = [x['id'] for x in q['documents']]
        pred_doc_ids_set = set(pred_doc_ids)
        act_doc_ids_set = set(act_doc_ids)
        jenson_sim = len(pred_doc_ids_set.intersection(act_doc_ids_set)) / len(pred_doc_ids_set.union(act_doc_ids_set))
        precision_at_k = len(pred_doc_ids_set.intersection(act_doc_ids_set)) / len(pred_doc_ids)
        recall_at_k = len(pred_doc_ids_set.intersection(act_doc_ids_set)) / len(act_doc_ids_set)
        if precision_at_k + recall_at_k == 0:
            f1_at_k = 0
        else:
            f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        scores['jenson'].append(jenson_sim)
        scores['precision'].append(precision_at_k)
        scores['recall'].append(recall_at_k)
        scores['f1'].append(f1_at_k)
        print(jenson_sim, precision_at_k, recall_at_k, f1_at_k)

    with open(args.o, 'w') as f:
        for k, v in scores.items():
            f.write(f"avg-{k}@{args.k}: {sum(v) / len(v)}\n")
            print(f"avg-{k}@{args.k}: {sum(v) / len(v)}\n")
