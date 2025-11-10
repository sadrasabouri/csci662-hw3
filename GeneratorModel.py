from abc import ABCMeta, abstractmethod

# TODO define prompt(s)
PROMPT_WO_DOCS = """answer the following question:
<q> {question} </q>
Wrap your final answer in <answer> [ANSWER] </answer> tags.
"""
PROMPT_W_DOCS = """
Based on the following text:
{retrieved_documents}
Use document id (e.g., <doc_0>) to refer to the content to """ + PROMPT_WO_DOCS

class GeneratorModel(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    @abstractmethod
    def query(self, retrieved_documents, question):
        pass