import argparse
from ollama_generator import *
from huggingface_generator import *
from bm25 import * 
from tfidf import *
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description="Generator")
    parser.add_argument('-r', default="bm25", help="retriever model: name of the retriever model you used.")
    parser.add_argument("-n", help="index name: the name of the index that you created with 'index.py'.")
    parser.add_argument("-k", help="top k: the number of documents to return in each retrieval run.")

    parser.add_argument("-p", default="ollama", help="Platform to use for the generator.")
    parser.add_argument("-m", help="type of model to use to generate: gemma2:2b, etc.")
    
    parser.add_argument("-i", help="input file: path of the input file of questions, where each question is in the form: <text> for each newline")
    parser.add_argument("-o", help="output file: path of the file where the answers should be written") # Respect the naming convention for the model: make sure to name it *.answers.txt in your workplace otherwise the grading script will fail

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if args.r == "bm25":
        retriever = BM25(args.n).load_model()
    elif args.r == "tfidf":
        retriever = TFIDF(args.n).load_model()
    else:
        # TODO add at least one more retriever
        retriever = None

    if args.p == 'ollama':
        generator = OllamaModel(model_name=args.m)
    elif args.p in ['HuggingFace', 'huggingface', 'hf', 'HF']:
        generator = HFModel(args.m)

    else:
        ## TODO Add any other models you wish to train
        generator = None

    answers = []
    questions = open(args.i).read().strip().splitlines()
    for q in tqdm(questions, desc="Answering questions with RAG", unit="question"):
        docs = retriever.search(q, int(args.k))
        answer = generator.query(docs, q)
        answers.append(answer)
    
    with open(args.o, 'w') as f:
        for a in answers:
            # Replace newlines in the answer with a space
            single_line_answer = a.replace('\n', ' ')
            f.write(single_line_answer)
            f.write('\n')






