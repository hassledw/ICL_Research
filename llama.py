from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
# https://huggingface.co/KaiLv
from datasets import load_dataset
import numpy as np
import spacy
import pandas as pd
import time
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
    y_test = np.array(dataset["test"]["label"])

    X_debug = np.array(dataset["debug"]["sentence"])
    y_debug = np.array(dataset["debug"]["label"])
    
    return X_train, y_train, X_test, y_test, X_debug, y_debug

def similarity_dict(target, X_test, y_test):
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

    for sentence, label in zip(X_test, y_test):
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

def query_model(query, prompt, df):
    '''
    Queries the llama model.
    '''
    inputs = tokenizer(
        f"{prompt}",
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to("cuda")
    torch.cuda.empty_cache()
    generation_config = transformers.GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.75,
        top_k=1,
        repetition_penalty=1.5,
        max_new_tokens=20,
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

    entry = [query, output_text]
    df_entry = pd.DataFrame(entry, index=['sentence', 'label']).T
    df = pd.concat((df, df_entry))
    return output_text

if __name__ == "__main__":
    model, tokenizer = create_model()

    X_train, y_train, X_test, y_test, X_debug, y_debug = create_data_arrays(datasetname="KaiLv/UDR_Yelp")

    df = pd.DataFrame(columns=["sentence", "label"])
    count = 1
    similar_dict, lookups = similarity_dict("I like this movie", X_test, y_test)
    start_time = time.time()

    for entry in zip(X_test, y_test):
        query, y_true = entry
        value = int(lookups[query])
        inputs = tokenizer(
            f"""
            Here are some demonstration examples for the sentiment classification task:
            1. \"{similar_dict[(value + 1) % len(X_test)][0][:200]}...\" = \"{similar_dict[(value + 1) % len(X_test)][1]}\"
            2. \"{similar_dict[(value + 2) % len(X_test)][0][:200]}...\" = \"{similar_dict[(value + 2) % len(X_test)][1]}\"
            Please rate the sentiment of the following text # "very negative", "negative", "neutral", "positive", or "very positive"#.
        ### \"{query}\""
        ### Response:""",
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to("cuda")
        torch.cuda.empty_cache()
        generation_config = transformers.GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=1,
            repetition_penalty=1.5,
            max_new_tokens=20,
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

        entry = [query, output_text]
        df_entry = pd.DataFrame(entry, index=['sentence', 'label']).T
        df = pd.concat((df, df_entry))
        print(output_text)
        print("\n")

        print(f"Finished {count}/{len(X_test)}\n")
        end_time = time.time()
        
        print(f"Time to run sample: {(end_time - start_time) / 60:.2f} minutes")
        print(f"Estimated runtime: {((end_time - start_time) / 60) * (len(X_test) - count):.2f} minutes")
        count+=1
        start_time = end_time
        torch.cuda.empty_cache()

    df.to_csv("/home/grads/hassledw/ICL_Research/UDR-yelp-fewshot-llama")
