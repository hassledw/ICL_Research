import transformers
import pickle
import torch
# https://huggingface.co/KaiLv
from datasets import load_dataset
import numpy as np
import spacy
import pandas as pd
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
import sys
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

class LlamaSemantic:
    def __init__(self, testname, tokenizer, model):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = tokenizer
        self.model = model
        self.testname = testname

        if testname == "UDR_Yelp":
            self.X_label = "sentence"
            self.y_label = "label"
            self.X_train, self.y_train, self.X_test = self.create_data_arrays("KaiLv/UDR_Yelp", self.X_label, self.y_label)
            self.labels = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}
        elif testname == "UDR_SNLI":
            self.X_label = "sentence"
            self.y_label = "label"
            self.X_train, self.y_train, self.X_test = self.create_data_arrays("KaiLv/UDR_SNLI", self.X_label, self.y_label)
            self.labels = {-1: "none", 0: "entailment", 1: "inconclusive", 2: "contradiction"}

    def create_data_arrays(self, datasetname, X_label, y_label):
        '''
        Creates the X_* and y_* arrays.
        '''
        dataset = load_dataset(datasetname)
        X_train = np.array(dataset["train"][X_label])
        y_train = np.array(dataset["train"][y_label])

        X_test = np.array(dataset["test"][X_label])
        
        return X_train, y_train, X_test
    
    def create_semantic_embeddings(self):
        '''
        Creates the semantic embeddings using SBERT.
        '''
        corpus = self.X_train
        queries = self.X_test

        corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True).to(device)
        corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

        query_embeddings = self.embedder.encode(queries, convert_to_tensor=True).to(device)
        query_embeddings = util.normalize_embeddings(query_embeddings)

        fewshot_examples = util.semantic_search(query_embeddings, corpus_embeddings, top_k=3, score_function=util.dot_score)
        return np.array(fewshot_examples)
    
    def query_model(self, prompt, max_tokens=20):
        '''
        Queries the llama model.
        '''
        inputs = self.tokenizer(
            f"{prompt}",
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to("cuda")

        generation_config = transformers.GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=1,
            repetition_penalty=1.5,
            max_new_tokens=max_tokens,
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                generation_config=generation_config,
            )

        output_text = self.tokenizer.decode(
            generation_output[0].cuda(), skip_special_tokens=True
        ).strip()

        return output_text
    
    def fewshot(self, direction):
        '''
        Runs fewshot k=3 prompts on Llama model, where the fewshot examples are the most semantically similar entries to the query.
        '''
        df = pd.DataFrame(columns=[self.X_label, self.y_label])
        count = 1
        fewshot_examples = self.create_semantic_embeddings()

        for i, entry in enumerate(fewshot_examples):
            query = self.X_test[i]
            train_idx_1 = entry[0]["corpus_id"]
            train_idx_2 = entry[1]["corpus_id"]
            train_idx_3 = entry[2]["corpus_id"]
            
            prompt = f"\nHere are some examples of my task:\n\
1. {self.X_train[train_idx_1][:300]} Response: {self.labels[self.y_train[train_idx_1]]}\n\
2. {self.X_train[train_idx_2][:300]} Response: {self.labels[self.y_train[train_idx_2]]}\n\
3. {self.X_train[train_idx_3][:300]} Response: {self.labels[self.y_train[train_idx_3]]}\n\
{direction}\n\
Review: \"{query}\"\n\
Response: "

            output_text = self.query_model(prompt)

            print(output_text)
            print("\n")

            print(f"Finished {count}/{len(self.X_test)}\n")
            entry = [query, output_text]
            df_entry = pd.DataFrame(entry, index=[self.X_label, self.y_label]).T
            df = pd.concat((df, df_entry))
            count+=1
            torch.cuda.empty_cache()

        df.to_csv(f"/home/grads/hassledw/ICL_Research/{self.testname}_results/{self.testname}-fewshot-llama.csv")

    def zeroshot(self, direction):
        '''
        Runs zeroshot prompt on Llama model.
        '''
        df = pd.DataFrame(columns=[self.X_label, self.y_label])
        count = 1

        for query in self.X_test:
            prompt = f"{direction}\n\
Review: \"{query}\"\n\
Response: "

            output_text = self.query_model(prompt)

            print(output_text)
            print("\n")

            print(f"Finished {count}/{len(self.X_test)}\n")
            entry = [query, output_text]
            df_entry = pd.DataFrame(entry, index=[self.X_label, self.y_label]).T
            df = pd.concat((df, df_entry))
            count+=1
            torch.cuda.empty_cache()

        df.to_csv(f"/home/grads/hassledw/ICL_Research/{self.testname}_results/{self.testname}-zeroshot-llama.csv")
        


if __name__ == "__main__":
    with open("./token.txt") as f:
        token = f.readline()
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token)
    
    ls_yelp = LlamaSemantic("UDR_Yelp", tokenizer, model)
    direction = "rate the sentiment of the below review: \"very negative\", \"negative\", \"neutral\", \"positive\", or \"very positive\"."
    ls_yelp.zeroshot(direction)
    ls_yelp.fewshot(direction)
    
    ls_snli = LlamaSemantic("UDR_SNLI", tokenizer, model)
    direction = ""
    ls_snli.zeroshot(direction)
    ls_snli.fewshot(direction)