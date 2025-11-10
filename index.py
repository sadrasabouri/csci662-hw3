import os
import argparse
from bm25 import *
from tfidf import *

def get_arguments():
    # Please do not change the naming of these command line options or delete them. You may add other options for other hyperparameters but please provide with that the default values you used
    parser = argparse.ArgumentParser(description="Given a model name and text, index the text")
    parser.add_argument("-m", default="bm25", help="retriever model: what retriever to use")
    parser.add_argument("-i",  help="input file: the name/path of texts to index")
    parser.add_argument("-o", help="index name: the name/path to save index to disk")
    parser.add_argument("--bm25-b", help="BM25 b hyperparameter", type=float, default=0.75)
    parser.add_argument("--bm25-k", help="BM25 k hyperparameter", type=float, default=1.2)
    parser.add_argument("--min-df", help="minimum document frequency", type=int, default=1)
    parser.add_argument("--tokenizer", help="tokenizer to use", type=str, default='whitespace')
    parser.add_argument("--stemmer", help="stemmer to use", type=str, default='english')
    parser.add_argument("--stopwords", help="stopwords to use", type=str, default='english')
    parser.add_argument("--do-lowercasing", help="whether to do lowercasing", type=bool, default=True)
    parser.add_argument("--do-ampersand-normalization", help="whether to do ampersand normalization", type=bool, default=True)
    parser.add_argument("--do-acronyms-normalization", help="whether to do acronyms normalization", type=bool, default=True)
    parser.add_argument("--do-punctuation-removal", help="whether to do punctuation removal", type=bool, default=True)
    parser.add_argument("--do-special-chars-normalization", help="whether to do special characters normalization", type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if "bm25" in args.m:
        model = BM25(model_file=args.o, b=args.bm25_b, k=args.bm25_k, parameters={
            'min_df': args.min_df,
            'tokenizer': args.tokenizer,
            'stemmer': args.stemmer,
            'stopwords': args.stopwords,
            'do_lowercasing': args.do_lowercasing,
            'do_ampersand_normalization': args.do_ampersand_normalization,
            'do_acronyms_normalization': args.do_acronyms_normalization,
            'do_punctuation_removal': args.do_punctuation_removal,
            'do_special_chars_normalization': args.do_special_chars_normalization
        })
    elif "tfidf" in args.m:
        model = TFIDF(model_file=args.o, parameters={
            'min_df': args.min_df,
            'tokenizer': args.tokenizer,
            'stemmer': args.stemmer,
            'stopwords': args.stopwords,
            'do_lowercasing': args.do_lowercasing,
            'do_ampersand_normalization': args.do_ampersand_normalization,
            'do_acronyms_normalization': args.do_acronyms_normalization,
            'do_punctuation_removal': args.do_punctuation_removal,
            'do_special_chars_normalization': args.do_special_chars_normalization
        })
    else:
        model = None

    # model.index() is responsible for saving index to disk - we don't need to do it separately    
    index = model.index(args.i)
