from abc import ABCMeta, abstractmethod

# TODO define prompt(s)
PROMPT = """
<you should define this prompt>
"""

class GeneratorModel(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    @abstractmethod
    def query(self, retrieved_documents, question):
        pass