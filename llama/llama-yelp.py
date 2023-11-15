from transformers import LlamaForCausalLM, LlamaTokenizer
import sys

sys.path.append("..")
from sentencesim import SentenceSimilarity as ss
import transformers
import pickle
import torch
# https://huggingface.co/KaiLv
from datasets import load_dataset
import numpy as np
import spacy
import pandas as pd
import time
from timeout_decorator import timeout
'''
Code to automatically query the llama-7b model
and generate CSV files.

Author: Daniel Hassler
Version: 11/7/2023
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

def create_model():
    '''
    Create the model.
    '''
    with open("./token.txt") as f:
        token = f.readline()
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token)
        # model.to(device)
    return model, tokenizer

def create_data_arrays(datasetname="KaiLv/UDR_Yelp"):
    '''
    Creates the X_* and y_* arrays.
    '''
    dataset = load_dataset(datasetname)
    X_train = np.array(dataset["train"]["sentence"])
    y_train = np.array(dataset["train"]["label"])

    X_test = np.array(dataset["test"]["sentence"])
    # y_test = np.array(dataset["test"]["label"])

    # X_debug = np.array(dataset["debug"]["sentence"])
    # y_debug = np.array(dataset["debug"]["label"])
    
    return X_train, y_train, X_test

def similarity_dict_1(target, X_train, y_train):
    '''
    Generates a sentence similarity dictionary given a target prompt
    and the test data. This is used in the UDR_yelp sentiment classification
    task.
    '''
    sim_model = spacy.load("en_core_web_md")
    similar_sentences = []
    similar_dict = {}
    lookup = {}
    label_dict = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}
    target = sim_model(str(target))

    for sentence, label in zip(X_train, y_train):
        if str(sentence) == str(target):
            continue
        sentence_emb = sim_model(str(sentence))
        similar_sentences.append((sentence, target.similarity(sentence_emb), label))
    
    similar_sentences.sort(key=lambda x: x[1], reverse=True)

    for i, entry in enumerate(similar_sentences):
        sentence, sim, label = entry
        similar_dict[i] = (sentence, label_dict[label])
        lookup[sentence] = i

    return similar_dict, lookup

def similarity_dict_pawar(X_train, y_train, query="I like this item."):
    '''
    Generates an ordered sentence similarity list given test data for entry.
    This is used as the sentiment baseline for k-shot learning task.
    '''
    all_entries = []
    lookup = {}
    sim_obj = ss.SentenceSimilarity()
    
    # metrics
    count = 1
    total = len(X_train)

    for curr_entry_i, (entry, entry_label) in enumerate(zip(X_train, y_train)):
        sim_score = sim_obj.sim(query, entry)
        all_entries.append((entry, entry_label, sim_score))
        print(f"Compared {curr_entry_i + 1}/{total}")

    all_entries = sorted(all_entries, key=lambda x: x[2])
    end_time = time.time()
    count += 1

    with open('semantic_yelp_entries.pkl', 'wb') as file:
        pickle.dump(all_entries, file)

    return all_entries



def query_model(query, prompt, max_tokens=20):
    '''
    Queries the llama model.
    '''
    inputs = tokenizer(
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
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=generation_config,
        )

    output_text = tokenizer.decode(
        generation_output[0].cuda(), skip_special_tokens=True
    ).strip()

    return output_text

def run_zeroshot_yelp_data(name, X_train, y_train, X_test):
    ### RUN ON YELP DATASET
    df = pd.DataFrame(columns=["sentence", "label"])
    count = 1
    similar_dict, lookups = similarity_dict_1("I like this movie", X_train, y_train)
    start_time = time.time()

    for entry in X_test:
        query, y_true = entry
        value = int(lookups[query])

        if "1" in name:
            prompt = f"""
            Please rate the sentiment of the following Review "very negative", "negative", "neutral", "positive", or "very positive".
            #Review:# \"{query}\""
            ###Response:"""
        elif "2" in name:
            prompt = f"""
            #Review#: \"{query}\"
            Please rate the sentiment of the Review: "very negative", "negative", "neutral", "positive", or "very positive"#.
            ###Response: """
        elif "3" in name:
            prompt = f"""
            rate the sentiment of the below review: "very negative", "negative", "neutral", "positive", or "very positive".
            ###\"{query}\" Response: """

        output_text = query_model(query, prompt)

        print(output_text)
        print("\n")

        print(f"Finished {count}/{len(X_test)}\n")
        end_time = time.time()
        entry = [query, output_text]
        df_entry = pd.DataFrame(entry, index=['sentence', 'label']).T
        df = pd.concat((df, df_entry))
        print(f"Time to run sample: {(end_time - start_time) / 60:.2f} minutes")
        print(f"Estimated runtime: {((end_time - start_time) / 60) * (len(X_test) - count):.2f} minutes")
        count+=1
        start_time = end_time
        torch.cuda.empty_cache()

    df.to_csv(f"/home/grads/hassledw/ICL_Research/UDR_yelp_results/{name}.csv")

def run_fewshot_yelp_data(name, X_train, y_train, X_test, query, sim_list):
    sim_obj = ss.SentenceSimilarity()
    df = pd.DataFrame(columns=["sentence", "label"])
    count = 1
    start_time = time.time()

    for entry in X_test:
        query = entry
        val = sim_obj.sim(query, entry)
        nearest_values = min(sim_list, key=lambda x: abs(x[2] - val))[:3]
        
        print(val)
        print(nearest_values)

        if "1" in name:
            # prompt = f"""
            # Here are some demonstration examples for the sentiment classification task:
            # 1. \"{similar_dict[(value + 1) % len(X_test)][0][:200]}...\" = \"{similar_dict[(value + 1) % len(X_test)][1]}\"
            # 2. \"{similar_dict[(value + 2) % len(X_test)][0][:200]}...\" = \"{similar_dict[(value + 2) % len(X_test)][1]}\"
            # Please rate the sentiment of the following text # "very negative", "negative", "neutral", "positive", or "very positive"#.
            # #Review:# \"{query}\""
            # ###Response:"""
            prompt="Bears are cool"
        # elif "2" in name:
        #     prompt = f"""
        #     #Review#: \"{query}\"
        #     #Here are some examples for the task:
        #     1. \"{similar_dict[(value + 1) % len(X_test)][0][:200]}...\" = \"{similar_dict[(value + 1) % len(X_test)][1]}\"
        #     2. \"{similar_dict[(value + 2) % len(X_test)][0][:200]}...\" = \"{similar_dict[(value + 2) % len(X_test)][1]}\"
        #     Please rate the sentiment of the Review: "very negative", "negative", "neutral", "positive", or "very positive"#.
        #     ###Response: """
        # elif "3" in name:
        #     prompt = f"""
        #     #Here are some examples:
        #     1. \"{similar_dict[(value + 1) % len(X_test)][0][:200]}...\" Response: \"{similar_dict[(value + 1) % len(X_test)][1]}\"
        #     2. \"{similar_dict[(value + 2) % len(X_test)][0][:200]}...\" Response: \"{similar_dict[(value + 2) % len(X_test)][1]}\"
        #     3. \"{similar_dict[(value + 3) % len(X_test)][0][:200]}...\" Response: \"{similar_dict[(value + 3) % len(X_test)][1]}\"
        #     rate the sentiment of the below review: "very negative", "negative", "neutral", "positive", or "very positive"#.
        #     4. ###\"{query}\" Response: """

        output_text = query_model(query, prompt)

        print(output_text)
        print("\n")

        print(f"Finished {count}/{len(X_test)}\n")
        end_time = time.time()
        entry = [query, output_text]
        df_entry = pd.DataFrame(entry, index=['sentence', 'label']).T
        df = pd.concat((df, df_entry))
        print(f"Time to run sample: {(end_time - start_time) / 60:.2f} minutes")
        print(f"Estimated runtime: {((end_time - start_time) / 60) * (len(X_test) - count):.2f} minutes")
        count+=1
        start_time = end_time
        torch.cuda.empty_cache()

    df.to_csv(f"/home/grads/hassledw/ICL_Research/UDR_yelp_results/{name}.csv")


if __name__ == "__main__":
    # model, tokenizer = create_model()
    # Run Yelp Test Suite

    X_train, y_train, y_test = create_data_arrays(datasetname="KaiLv/UDR_Yelp")
    sim_list = similarity_dict_pawar(X_train, y_train)
    # run_zeroshot_yelp_data("UDR-yelp-zeroshot-llama-1")
    # run_fewshot_yelp_data("UDR-yelp-fewshot-llama-1")



