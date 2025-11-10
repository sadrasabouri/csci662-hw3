from abc import ABCMeta, abstractmethod

# TODO define prompt(s)
PROMPT_WO_DOCS = """
Answer the following question:
Q: {question}
"""
PROMPT_W_DOCS = """
Based on the following text:
<text>
{retrieved_documents}
</text>
{PROMPT_WO_DOCS}
"""

class GeneratorModel(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    @abstractmethod
    def query(self, retrieved_documents, question):
        pass