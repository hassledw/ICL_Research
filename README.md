# A Survey of Improving In-Context Learning for LLMs

## Authors
* Daniel Hassler (hassledw@vt.edu)
* Hoang Anh Just (just@vt.edu)
* Taylor Neller (taylor02@vt.edu)
* Jianan Nie (jianan@vt.edu) 

## Problem Introduction
LLM models such as ChatGPT have the ability to learn in-context (ICL). Given a few demonstrations, these types of LLMs can successfully generate related content in that problem domain for specific tasks. One problem in-context learning exhibits, regarding accurate results, is its dependence on the quality of demonstration examples and prompt structure. In this work, we survey different ICL method approaches and explore results.

## Datasets
**SNLI** 
```
Name: Natural Language Inference
Link: https://huggingface.co/datasets/KaiLv/UDR_SNLI
Description: Natural language inference classification dataset
Labels: 0-2, where 0 is “entailment”, 1 is “inconclusive”, 2 is “contradiction”
```
**CosmosQA** 
```
Name: Commonsense-based Reading Comprehension
Link: https://huggingface.co/datasets/cosmos_qa
Description: reading comprehension QA dataset.
Labels: 0-3, each representing the index to each choice.
```
**ARC Challenge** 
```
Name: AI2 Reasoning Challenge 
Link: https://huggingface.co/datasets/ai2_arc
Description: multiple choice reasoning dataset consisting of “challenging” grade school level questions.
Labels: choices A, B, C, or D.
```
**ComE**
```
Name: Commonsense Explanation
Link: https://huggingface.co/datasets/KaiLv/UDR_ComE
Description: A multi-label classification dataset aimed for commonsense reasoning.
Labels: choices A, B, or C.
```
**Yelp**
```
Name: Yelp
Link: https://huggingface.co/datasets/KaiLv/UDR_Yelp
Description: A multi-label sentiment classification dataset with 38,000 entries of Yelp reviews.
Labels: 0-4, where 0 is “very negative”, 2 is “neutral”, and 4 is “very positive”.
```

## LLM
We chose [LLaMa](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) for our evaluation LLM, as this model was readily available online and small enough to be downloadable in our environment. This model has 7 billion parameters. Although we use a smaller LLM, we speculate that larger variants of LLaMa or different LLMs entirely (GPT-4, GPT-3) would have different/higher level of preformance.

Below, is our high level diagram showing how we incorporated LLaMa, our datasets, and our approaches:
![ICL_Arch](images/ICL_Arch.PNG)

## Methods

### Semantic Embeddings Approach
This approach utilizes semantic embeddings derived from SBERT, a variation of the BERT transformer model that is fine-tuned to derive meaningful semantic embeddings of sentences. SBERT is different from BERT in that it can derive meaningful sentence embeddings for large text corpuses in a fraction of the time compared to traditional methods. The SBERT model uses siamese/triplet networks to fine-tune the BERT model, which aids in semantic similarity by minimizing the difference between positive examples compared to negative examples. The paper referenced for this architecture is [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf)

The idea of using semantic embeddings for k-shot learning in ICL was inspired by the paper [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/pdf/2101.06804.pdf), although this paper is directed towards the GPT-3 model and uses a different embedding/retrieval method. In our approach, we apply the SBERT semantic embeddings into k-shot learning for ICL utilizing 'semantic search' as a retrieval. Below is a diagram depicting the overall method of the semantic embeddings approach:
![ICL_Semantic_Embeddings_Approach](images/SemanticICL.PNG)

The code implementation for SBERT, derived initially from the Sentence-BERT paper, is on Huggingface applied as `SentenceTransformer` and is explained and referenced in greater detail in the `llama` directory of our repo. Look at [similarity.ipynb](https://github.com/hassledw/ICL_Research/blob/main/llama/similarity.ipynb) for more notes.

### Rephrasing Approach

### BM-25 Approach

### Jianan Approach
