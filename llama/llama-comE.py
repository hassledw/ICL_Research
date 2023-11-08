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

    return model, tokenizer

def create_data_arrays(datasetname="KaiLv/UDR_ComE"):
    '''
    Creates the X_* and y_* arrays.
    '''
    dataset = load_dataset(datasetname)
    # X_train = np.array(dataset["train"]["question"])
    # X_train_choices = np.array(dataset["train"]["choices"])
    # y_train = np.array(dataset["train"]["label"])
    def filter_question(text):
        parts = text.split("Select the most corresponding reason why this statement is against common sense.")
        instruction = parts[0]
        parts_split = parts[1].split("Options:")
        question = parts_split[0]
        print(question)
        return question
    df = pd.DataFrame({"question":dataset["test"]["question"]})
    df["question"] = df["question"].apply(filter_question)
    X_test = np.array(df["question"])
    
    X_test_choices = np.array(dataset["test"]["choices"])
    y_test = np.array(dataset["test"]["label"])

    # X_debug = np.array(dataset["debug"]["question"])
    # X_debug_choices = np.array(dataset["debug"]["choices"])
    # y_debug = np.array(dataset["debug"]["label"])
    
    return X_test, X_test_choices, y_test

def query_model(prompt, max_tokens=40):
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

def run_zeroshot_comE(name):
    X_test, X_test_choices, y_test = create_data_arrays(datasetname="KaiLv/UDR_ComE")
    
    df = pd.DataFrame(columns=["statement", "choice"])
    count = 1
    start_time = time.time()

    for entry in zip(X_test, X_test_choices, y_test):
        statement, choices, y_true = entry
        prompt = f"""
        #statement: {statement}\n
        Select the most corresponding reason why this statement is against common sense:
        {choices}
        choice: 
        """
        output_text = query_model(prompt)
        print(output_text)
        print("\n")
        print(f"Finished {count}/{len(X_test)}\n")
        end_time = time.time()
        print(f"Time to run sample: {(end_time - start_time) / 60:.2f} minutes")
        print(f"Estimated runtime: {((end_time - start_time) / 60) * (len(X_test) - count):.2f} minutes")

        entry = [statement, output_text]
        df_entry = pd.DataFrame(entry, index=['statement', 'choice']).T
        df = pd.concat((df, df_entry))
        count+=1
        start_time = end_time
        torch.cuda.empty_cache()

    df.to_csv(f"/home/grads/hassledw/ICL_Research/UDR_comE_results/{name}.csv")


def run_fewshot_comE(label, prompt):
    pass

if __name__ == "__main__":
    model, tokenizer = create_model()
    run_zeroshot_comE("UDR-comE-zeroshot-llama-1")