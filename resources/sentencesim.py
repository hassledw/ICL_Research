import gensim
from gensim.models import Word2Vec
import numpy as np

class SentenceSimilarity():
    '''
    Class that returns an ordered list of the similarity of the sentences.
    '''
    def __init__(self, corpus):
        self.corpus = corpus # a list of sentences.
        self.tokenized_corpus = [sentence.split() for sentence in corpus]
        self.model = Word2Vec(self.tokenized_corpus, vector_size=1000, window=5, min_count=1, sg=1)

    def create_sentence_embedding(self, sentence):
        '''
        Returns the sentence embedding, the mean of all the word vectors in the sentence.
        '''
        return np.mean([self.model.wv[word] for word in sentence if word in self.model.wv], axis=0)

    def get_sentence_embedding_corpus(self):
        '''
        Compute the sentence embeddings for every entry in the corpus.
        '''
        sentence_embeddings = []
        for sentence in self.tokenized_corpus:
            sentence_embedding = self.create_sentence_embedding(sentence)
            sentence_embeddings.append(sentence_embedding)
        
        return sentence_embeddings

    def similarity_list(self, target_sentence, sentence_embeddings):
        '''
        Returns an ordered list of all the similar options from most similar to least similar.
        '''
        similar_sentences = []
        target_embedding = self.create_sentence_embedding(target_sentence)

        for sentence, embedding in zip(self.corpus, sentence_embeddings):
            similarity = np.dot(target_embedding, embedding) / (np.linalg.norm(target_embedding) * np.linalg.norm(embedding))
            similar_sentences.append((sentence, similarity))

        similar_sentences.sort(key=lambda x: x[1], reverse=True)
        return similar_sentences
    
    def print_top_k(self, k, target_sentence, similar_sentences):
        '''
        Prints the top-k sentences
        '''
        print(f"Top-{k} results for:\n")
        print(f"{target_sentence}\n")

        for i, (sentence, sim) in enumerate(similar_sentences):
            if i >= k:
                break
            print("-" * 20)
            print(f"{i + 1}. {sim}% similarity:\n")
            print(f"{sentence}\n")
            print("-" * 20)
            


if __name__ == "__main__":
    # corpus = [
    #     "this is a sentence about word embeddings",
    #     "word embeddings are powerful for NLP tasks",
    #     "skip-gram and CBOW are popular Word2Vec models",
    #     "Gensim is a popular library for word embeddings",
    # ]

    # sim = SentenceSimilarity(corpus)
    # sentence_embeddings = sim.get_sentence_embedding_corpus()
    # similar_sentences = sim.similarity_list("Bears are cool", sentence_embeddings)
    pass
