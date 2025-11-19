"""
 Using Retriv library for TF-IDF
"""
from tqdm import tqdm
import json
from retriv import SearchEngine
from RetrievalModel import *

class TFIDF(RetrievalModel):
    """
    The retrier class which index sentences using Term Frequency Inverse Document Frequency (TF-IDF) using Retriv library.
    """
    def __init__(self, model_file, parameters={}):
        self.model_file = model_file
        self.min_df = parameters.get('min_df', 1)
        self.tokenizer = parameters.get('tokenizer', 'whitespace')
        self.stemmer = parameters.get('stemmer', 'english')
        self.stopwords = parameters.get('stopwords', 'english')
        self.do_lowercasing = parameters.get('do_lowercasing', True)
        self.do_ampersand_normalization = parameters.get('do_ampersand_normalization', True)
        self.do_acronyms_normalization = parameters.get('do_acronyms_normalization', True)
        self.do_punctuation_removal = parameters.get('do_punctuation_removal', True)
        self.do_special_chars_normalization = parameters.get('do_special_chars_normalization', True)
        self.index_file = f"{model_file}_mdf{self.min_df}_tkn{self.tokenizer}_stm{self.stemmer}_sw{self.stopwords}_lc{self.do_lowercasing}_an{self.do_ampersand_normalization}_acn{self.do_acronyms_normalization}_pr{self.do_punctuation_removal}_scn{self.do_special_chars_normalization}"
        super().__init__(model_file)


    def index(self, input_file, use_cache=True):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :param use_cache: wether to use cached model
        """
        ## TODO write your code here to calculate term_doc_freqs and relative_doc_lens, 
        # Sadra: will use the `retriv` instead
        if use_cache:
            try:
                self = self.load_model()
                return
            except FileNotFoundError:
                pass
        with open(input_file, 'r') as f:
            data = json.load(f)
            se = SearchEngine(index_name=self.index_file,
                        model="tf-idf",
                        min_df=self.min_df, tokenizer=self.tokenizer,
                        stemmer=self.stemmer, stopwords=self.stopwords,
                        do_lowercasing=self.do_lowercasing,
                        do_ampersand_normalization=self.do_ampersand_normalization,
                        do_acronyms_normalization=self.do_acronyms_normalization,
                        do_punctuation_removal=self.do_punctuation_removal,
                        do_special_chars_normalization=self.do_special_chars_normalization).index(data)
            se.save()
        # then cache the class object using `self.save_model`
        self.save_model()


    def search(self, query, k, return_scores=False):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param query: query to run against tfidf retrieval index
        :param k: the number of retrieval results
        :return: predictions list
        """
        if k == 0:
            return []
        ## TODO write your code here (and change return)
        # Sadra: will use the `retriv` instead
        retrieved = SearchEngine.load(self.index_file).search(query=query, cutoff=k)
        if return_scores:
            return retrieved
        return [doc['text'] for doc in retrieved]

    def search_batch(self, queries, k, return_scores=False):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param queries: list of queries to run against tfidf retrieval index
        :param k: the number of retrieval results
        :return: predictions list
        """
        results = []
        for query in tqdm(queries, desc="Searching batch"):
            retrieved = self.search(query, k, return_scores)
            results.append(retrieved)
        return results
