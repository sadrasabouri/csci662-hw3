"""
 Using sentence-transformers/all-MiniLM-L6-v2 sentence embedding models for retrier
"""
from typing import List, Dict
import re
import unicodedata
import string
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from RetrievalModel import *

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():  # only works on macOS
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"PyTorch version: {torch.__version__} on {DEVICE}")

SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceTransformersSentenceEmbedding(RetrievalModel):
    """
    The retrier class which index sentences based using sentence transformer.
    """

    def __init__(self, model_file, parameters={}):
        self.model_file = model_file
        self.model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)
        self.model.to(DEVICE)
        self.tokenizer = parameters.get('tokenizer', 'whitespace')
        self.stemmer = parameters.get('stemmer', 'english')
        self.stopwords = parameters.get('stopwords', 'english')
        self.do_lowercasing = parameters.get('do_lowercasing', False)
        self.do_ampersand_normalization = parameters.get('do_ampersand_normalization', False)
        self.do_acronyms_normalization = parameters.get('do_acronyms_normalization', False)
        self.do_punctuation_removal = parameters.get('do_punctuation_removal', False)
        self.do_special_chars_normalization = parameters.get('do_special_chars_normalization', False)
        self.id_list = None
        self.text_list = None
        self.embeddings = None
        super().__init__(model_file)
        

    def _clean_text(self, text: str):
        """
        Clean the text according to the given

        :param text: given text string to clean
        """
        if self.do_lowercasing:
            text = text.lower()
        if self.do_ampersand_normalization:
            text = text.replace('&', ' and ')
        if self.do_acronyms_normalization:
            text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '', text)
            text = re.sub(r'(?<=[A-Z])\.$', '', text)
        if self.do_special_chars_normalization:
            text = unicodedata.normalize('NFKD', text)
            text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
            text = text.replace('—', '-').replace('–', '-')
            text = text.encode('ascii', 'ignore').decode('ascii')
        if self.do_punctuation_removal:
            text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        
        if self.tokenizer == 'whitespace':
            tokens = text.split()
        else:
            tokens = word_tokenize(text)
        
        if self.stopwords:
            stop_words = set(stopwords.words(self.stopwords))
            tokens = [t for t in tokens if t.lower() not in stop_words]
        
        if self.stemmer:
            stemmer = PorterStemmer() if self.stemmer == 'english' else SnowballStemmer(self.stemmer)
            tokens = [stemmer.stem(t) for t in tokens]
        
        return ' '.join(tokens)
        

    def _clean_data(self, data: List[Dict[str, str]]):
        """
        Clean the given data as a list to be cleaned.

        :para data: the given data in the format of a list of dictionaries with 
        """
        new_list = []
        for item in tqdm(data, desc="Cleaning data"):
            new_item = item.copy()
            new_item['text'] = self._clean_text(new_item['text'])
            new_list.append(new_item)
        return new_list

    def index(self, input_file, use_cache=True, batch_size=128):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :param use_cache: wether to use cached model
        :param batch_size: batch size used for model inference
        """
        if use_cache:
            try:
                self = self.load_model()
                self.model.to(DEVICE)
                return
            except FileNotFoundError:
                pass
        with open(input_file, 'r') as f:
            data = json.load(f)
            data = self._clean_data(data)
            self.id_list = [x['id'] for x in data]
            self.text_list = [x['text'] for x in data]
            for i in tqdm(range(0, len(self.text_list), batch_size), desc="Generating embeddings"):
                batch_texts = self.text_list[i:i+batch_size]
                batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True, device=DEVICE)
                if i == 0:
                    embs = batch_embeddings
                else:
                    embs = torch.cat((embs, batch_embeddings), dim=0)
            self.embeddings = embs
        # then cache the class object using `self.save_model`
        self.save_model()

    def search(self, query, k, return_scores=False):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param query: query to run against sentence embedding retrieval index
        :param k: the number of retrieval results
        :return: predictions list
        """
        if k == 0:
            return []
        query = self._clean_text(query)
        q_embedding = self.model.encode([query], convert_to_tensor=True, device=DEVICE)
        similarities = self.model.similarity(self.embeddings, q_embedding)
        retrieved = sorted(
            [{'id': self.id_list[i],
              'text': self.text_list[i],
              'score': float(similarities[i])} for i in range(len(self.id_list))],
            key=lambda x: x['score'],
            reverse=True
        )[:k]
        if return_scores:
            return retrieved
        return [doc['text'] for doc in retrieved]
    
    def search_batch(self, queries: List[str], k: int, return_scores=False, batch_size=128) -> List[List[Dict[str, str]]]:
        """
        This method allows searching a batch of queries.

        :param queries: list of query strings
        :param k: number of retrieval results per query
        :param batch_size: batch size used for model inference
        :return: list of lists of retrieved documents per query
        """
        all_retrieved = []
        for i in tqdm(range(0, len(queries), batch_size), desc="Searching batch of queries"):
            batch_queries = queries[i:i+batch_size]
            batch_cleaned = [self._clean_text(q) for q in batch_queries]
            batch_embeddings = self.model.encode(batch_cleaned, convert_to_tensor=True, device=DEVICE)
            for j in tqdm(range(len(batch_queries)), desc="Calculating inner batch similarity"):
                q_embedding = batch_embeddings[j].unsqueeze(0)
                similarities = self.model.similarity(self.embeddings, q_embedding)
                retrieved = sorted(
                    [{'id': self.id_list[m],
                      'text': self.text_list[m],
                      'score': float(similarities[m])} for m in range(len(self.id_list))],
                    key=lambda x: x['score'],
                    reverse=True
                )[:k]
                if return_scores:
                    all_retrieved.append(retrieved)
                else:
                    all_retrieved.append([doc['text'] for doc in retrieved])
        return all_retrieved
