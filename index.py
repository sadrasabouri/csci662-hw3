import argparse
from bm25 import *


def get_arguments():
    # Please do not change the naming of these command line options or delete them. You may add other options for other hyperparameters but please provide with that the default values you used
    parser = argparse.ArgumentParser(description="Given a model name and text, index the text")
    parser.add_argument("-m", default="bm25", help="retriever model: what retriever to use")
    parser.add_argument("-i",  help="input file: the name/path of texts to index")
    parser.add_argument("-o", help="index name: the name/path to save index to disk")
   

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if "bm25" in args.m:
        model = BM25(model_file=args.o)
    else:
        ## TODO add at least one other retriever
        model = None

    # model.index() is responsible for saving index to disk - we don't need to do it separately    
    index = model.index(args.i)