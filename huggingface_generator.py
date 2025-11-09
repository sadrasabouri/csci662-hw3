# this is generic boilerplate for HuggingFace models. 
# if you choose to implement your language model in some other way

# e.g. VLLM: https://blog.vllm.ai/2023/06/20/vllm.html

# make sure to overwrite the Generator class in the same way.

from transformers import AutoModelForCausalLM, AutoTokenizer
from GeneratorModel import *


class HFModel(GeneratorModel):
	def __init__(self, model_name):
		self.model_name = model_name
		# Load the HuggingFace model and tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForCausalLM.from_pretrained(model_name)

	def query(self, retrieved_documents, question):
		retrieved_documents_str = '\n'.join(retrieved_documents)
		prompt = PROMPT.format(retrieved_documents=retrieved_documents_str, question=question)
		
		# Tokenize the prompt and attend to all tokens
		inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
		attention_mask = inputs["input_ids"].ne(self.tokenizer.pad_token_id).long()

        # Generate a response
		outputs = self.model.generate(
            inputs["input_ids"],
			attention_mask=attention_mask,
            max_new_tokens=100,  # Adjust as needed
            num_return_sequences=1,
            temperature=0.7,  # Adjust temperature for randomness
            top_p=0.9,  # Adjust top-p for nucleus sampling
            do_sample=True
        )

        # Decode the generated response
		response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		return response

