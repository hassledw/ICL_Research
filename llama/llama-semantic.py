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
        elif testname == "UDR_ComE":
            self.X_label = "question"
            self.y_label = "label"
            self.X_train, self.y_train, self.X_test = self.create_data_arrays("KaiLv/UDR_ComE", self.X_label, self.y_label)
            self.labels = {'A': 'A', 'B': 'B', 'C': 'C'}
        elif testname == "CosmosQA":
            self.X_label = "prompt"
            self.y_label = "label"
            self.X_train, self.y_train, self.X_test = self.create_data_arrays("cosmos_qa", self.X_label, self.y_label)
            self.labels = {0:'0', 1:'1', 2:'2', 3:'3'}
        elif testname == "ARC-Challenge":
            self.X_label = "prompt"
            self.y_label = "answerKey"
            self.X_train, self.y_train, self.X_test = self.create_data_arrays("ai2_arc", self.X_label, self.y_label)
            self.labels = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}

    def create_data_arrays(self, datasetname, X_label, y_label, test_size=700):
        '''
        Creates the X_* and y_* arrays.
        '''
        X_train = None
        y_train = None
        X_test = None

        if self.testname == "UDR_ComE":
            # if UDR_ComE, just grab the letter of the choice.
            dataset = load_dataset(datasetname)
            X_train = np.array(dataset["train"][X_label])
            y_train = np.array(dataset["train"][y_label])
            X_test = np.array(dataset["test"][X_label])

            X_train = np.array([value[80:] for value in X_train])
            X_train = np.array([value.replace(" Options:", "\nOptions:") for value in X_train])
            X_train = np.array([value.replace(" A.", "\nA.") for value in X_train])
            X_train = np.array([value.replace(" B.", "\nB.") for value in X_train])
            X_train = np.array([value.replace(" C.", "\nC.") for value in X_train])

            y_train = np.array([value[0] for value in y_train])

            X_test = np.array([value[80:] for value in X_test])
            X_test = np.array([value.replace(" Options:", "\nOptions:") for value in X_test])
            X_test = np.array([value.replace(" A.", "\nA.") for value in X_test])
            X_test = np.array([value.replace(" B.", "\nB.") for value in X_test])
            X_test = np.array([value.replace(" C.", "\nC.") for value in X_test])

        elif self.testname == "CosmosQA":
            dataset = load_dataset(datasetname)
            df_X_train = pd.DataFrame(data=dataset["train"])
            df_X_test = pd.DataFrame(data=dataset["validation"])
            df_X_train[X_label] = "Context: " + df_X_train['context'] + "\n" + "Question: " + df_X_train['question'] + "\nOptions:\n0." + df_X_train['answer0'] + "\n1." + df_X_train['answer1'] + "\n2." + df_X_train['answer2'] + "\n3." + df_X_train['answer3'] + "\n"
            df_X_test[X_label] = "Context: " + df_X_test['context'] + "\n" + "Question: " + df_X_test['question'] + "\nOptions:\n0." + df_X_test['answer0'] + "\n1." + df_X_test['answer1'] + "\n2." + df_X_test['answer2'] + "\n3." + df_X_test['answer3'] + "\n"

            X_train = np.array(df_X_train[X_label])
            y_train = np.array(dataset["train"][y_label])
            X_test = np.array(df_X_test[X_label])
            y_test = np.array(dataset["validation"][y_label])

        elif self.testname == "ARC-Challenge":
            dataset = load_dataset(datasetname, 'ARC-Challenge')
            # X_train = np.array(dataset["train"]["question"])
            choices = [choice["text"] for choice in dataset["train"]["choices"]]
            df_X_train = pd.DataFrame(data={"question":dataset["train"]["question"], "choices": choices})
            df_X_train[X_label] = ''
            labels = ['A', 'B', 'C', 'D']

            for index, row in df_X_train.iterrows():
                df_X_train.at[index,X_label] = f"Question: {row['question']}" + "\nChoices:\n"
                for i, (label, choice) in enumerate(zip(labels, row['choices'])):
                    df_X_train.at[index,X_label] += f"{label}. {choice}\n"

            choices = [choice["text"] for choice in dataset["test"]["choices"]]
            df_X_test = pd.DataFrame(data={"question":dataset["test"]["question"], "choices": choices})
            df_X_test[X_label] = ''
            labels = ['A', 'B', 'C', 'D']

            for index, row in df_X_test.iterrows():
                df_X_test.at[index,X_label] = f"Question: {row['question']}" + "\nChoices:\n"
                for i, (label, choice) in enumerate(zip(labels, row['choices'])):
                    df_X_test.at[index,X_label] += f"{label}. {choice}\n"
            

            # print(df_X_train.at[0, 'prompt'])
            # print(df_X_test.at[0, 'prompt'])
            df_y_train = pd.DataFrame(data={"answerKey":dataset["train"]["answerKey"]})
            df_y_test = pd.DataFrame(data={"answerKey":dataset["test"]["answerKey"]})
            df_y_train["answerKey"] = df_y_train["answerKey"].replace({"1": "A", "2": "B", "3": "C", "4": "D"})
            df_y_test["answerKey"] = df_y_test["answerKey"].replace({"1": "A", "2": "B", "3": "C", "4": "D"})

            X_train = np.array(df_X_train[X_label])
            y_train = np.array(df_y_train[y_label])
            X_test = np.array(df_X_test[X_label])
            y_test = np.array(df_y_test[y_label])

        else:
            dataset = load_dataset(datasetname)
            X_train = np.array(dataset["train"][X_label])
            y_train = np.array(dataset["train"][y_label])
            X_test = np.array(dataset["test"][X_label])

        return X_train, y_train, X_test[:test_size]
    
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
    
    def fewshot(self, direction, char_limit=300):
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
            
            prompt = f"\n\
{self.X_train[train_idx_1][:char_limit]}\nResponse: {self.labels[self.y_train[train_idx_1]]}\n\
{self.X_train[train_idx_2][:char_limit]}\nResponse: {self.labels[self.y_train[train_idx_2]]}\n\
{self.X_train[train_idx_3][:char_limit]}\nResponse: {self.labels[self.y_train[train_idx_3]]}\n\
{direction}\n\
{query}\nResponse: "
            

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

            if self.testname == "CosmosQA" or self.testname == "ARC-Challenge":
                prompt = f"{direction}\n{query}\nResponse: "
            else:
                prompt = f"{direction}\nStatement: {query}\nResponse: "

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
    direction = "Rate the following statement \"very negative\", \"negative\", \"neutral\", \"positive\", or \"very positive\"."
    ls_yelp.zeroshot(direction)
    ls_yelp.fewshot(direction)
    
    ls_snli = LlamaSemantic("UDR_SNLI", tokenizer, model)
    direction = ""
    ls_snli.zeroshot(direction)
    ls_snli.fewshot(direction)

    ls_come = LlamaSemantic("UDR_ComE", tokenizer, model)
    direction = "Choose why the following statement is against common sense: \"A\", \"B\", or \"C\""
    ls_come.zeroshot(direction)
    ls_come.fewshot(direction)

    ls_cosmos = LlamaSemantic("CosmosQA", tokenizer, model)
    direction = "Choose the correct response: 0, 1, 2, or 3"
    ls_cosmos.zeroshot(direction)
    ls_cosmos.fewshot(direction, char_limit=2000)

    ls_arc = LlamaSemantic("ARC-Challenge", tokenizer, model)
    direction = "Choose the correct response: \"A\", \"B\", \"C\", or \"D\""
    ls_arc.zeroshot(direction)
    ls_arc.fewshot(direction)
