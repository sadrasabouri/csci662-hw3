"""
 Refer to: https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus
 for more information

 This is starter code for implementing BM25 from scratch. 
 You are NOT required to do this from scratch
 You are allowed to use libraries like retriv
"""
from RetrievalModel import *

class BM25(RetrievalModel):
    def __init__(self, model_file, b=.75, k=1.2):
        self.b = b
        self.k = k
        self.token2id = {}
        self.doc_ids = {} 
        # TODO add more dicts as needed to store frequencies/scores
        self.avg_num_words_per_doc = None
        super().__init__(model_file)


    def index(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        """
        ## TODO write your code here to calculate term_doc_freqs and relative_doc_lens, 
        
        # then cache the class object using `self.save_model`


    def search(self, query, k):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param query: query to run against bm25 retrieval index
        :param k: the number of retrieval results
        :return: predictions list
        """
        ## TODO write your code here (and change return)





