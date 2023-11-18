import pandas as pd
'''
Test suite to automatically run metrics on all 5 datasets.

Author: Daniel Hassler
'''

class YelpResults:
    def __init__(self, zero_yelp_df, few_yelp_df):
        self.zero_yelp_df = zero_yelp_df
        self.few_yelp_df = few_yelp_df
        self.truth_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_Yelp_results/UDR-yelp-llama.csv")
    
    def clean_data(self, df):
        '''
        Cleans the data by retrieving the label and dropping None entries. 
        '''
        def get_response_yelp(text):
            '''
            Cleans the text of the label to just get the response
            '''
            valid = ["very negative", "very positive", "negative", "positive", "neutral"]
            valid_dict = {"very negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very positive": 4}
            sentences = text.split("Response:")
            query = sentences[-1].strip("##").strip(" ").lower()
            
            if len(sentences[-1].split(" ")) > 3 or query not in valid:
                for v in valid:
                    if v in query:
                        return valid_dict[v]
                return None
            else:
                return valid_dict[query]
            
        df["label"] = df["label"].apply(get_response_yelp)
        orig_entries = df.shape[0]
        df = df.dropna()
        print(f"Dropped {orig_entries - df.shape[0]} \"None\" entries")
        df["label"] = df["label"].astype(int)
        return df
    
    def gather_yelp_accuracy(self, df):
        '''
        Gets the overall accuracy of df.
        ''' 
        df_results = pd.merge(self.truth_df, df, on=['sentence'], how='inner')[:500]
        accurate_results = df_results[df_results["label_x"] == df_results["label_y"]]
        return len(accurate_results) / len(df_results) * 100
    
    def run_results(self):
        '''
        Runs the results of the Yelp dataset. 
        '''
        self.zero_yelp_df = self.clean_data(self.zero_yelp_df)
        self.few_yelp_df = self.clean_data(self.few_yelp_df)
        print("#### Yelp Results ####")
        print(f"Llama-7b Prediction Accuracy (Zero-shot): {self.gather_yelp_accuracy(self.zero_yelp_df):.2f}%")
        print(f"Llama-7b Prediction Accuracy (Few-shot): {self.gather_yelp_accuracy(self.few_yelp_df):.2f}%")
        print()

class SNLIResults:
    def __init__(self, zero_snli_df, few_snli_df):
        self.zero_snli_df = zero_snli_df
        self.few_snli_df = few_snli_df
        self.truth_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_SNLI_results/UDR-snli-llama.csv")
    
    def clean_data(self, df):
        '''
        Cleans the data by retrieving the label and dropping None entries. 
        '''
        def get_response_snli(text):
            '''
            Cleans the text of the label to just get the response
            '''
            valid = ["entail", "inco", "contra", "in con"]
            valid_dict = {"entailment": 0, "inconclusive": 1, "contradiction": 2}

            sentences = text.split("Response:")
            query = sentences[-1].strip("##").strip(" ").lower()
            
            if query not in valid:
                for i, v in enumerate(valid):
                    if i == 3:
                        return valid_dict[list(valid_dict.keys())[1]]
                    if v in query:
                        return valid_dict[list(valid_dict.keys())[i]]
                print(query)
                return None
            else:
                return valid_dict[query]
            
        df["label"] = df["label"].apply(get_response_snli)
        orig_entries = df.shape[0]
        df = df.dropna()
        print(f"Dropped {orig_entries - df.shape[0]} \"None\" entries")
        df["label"] = df["label"].astype(int)
        return df
    
    def gather_snli_accuracy(self, df):
        '''
        Gets the overall accuracy of df.
        ''' 
        df_results = pd.merge(self.truth_df, df, on=['sentence'], how='inner')[:500]
        accurate_results = df_results[df_results["label_x"] == df_results["label_y"]]
        return len(accurate_results) / len(df_results) * 100
    
    def run_results(self):
        '''
        Runs the results of the SNLI dataset. 
        '''
        self.zero_snli_df = self.clean_data(self.zero_snli_df)
        self.few_snli_df = self.clean_data(self.few_snli_df)
        print("#### SNLI Results ####")
        print(f"Llama-7b Prediction Accuracy (Zero-shot): {self.gather_snli_accuracy(self.zero_snli_df):.2f}%")
        print(f"Llama-7b Prediction Accuracy (Few-shot): {self.gather_snli_accuracy(self.few_snli_df):.2f}%")
        print()

class ComEResults:
    def __init__(self, zero_come_df, few_come_df):
        self.zero_come_df = zero_come_df
        self.few_come_df = few_come_df
        self.truth_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_ComE_results/UDR-ComE-llama.csv")

    def clean_data(self, df):
        '''
        Cleans the data by retrieving the label and dropping None entries. 
        '''
        def get_response_come(text):
            '''
            Cleans the text of the label to just get the response
            '''
            valid = ["A", "B", "C"]

            valid_other = {1: "A", 2: "B", 3: "C"}

            sentences = text.split("Response:")
            query = sentences[-1][:5]

            for char in query:
                if char in valid:
                    return char
                elif char in ["1", "2", "3"]:
                    return valid_other[int(char)]
                
            return None
        
        df["label"] = df["label"].apply(get_response_come)
        orig_entries = df.shape[0]
        df = df.dropna()
        print(f"Dropped {orig_entries - df.shape[0]} \"None\" entries")
        return df
    
    def gather_come_accuracy(self, df):
        '''
        Gets the overall accuracy of df.
        ''' 
        df_results = pd.merge(self.truth_df, df, on=['question'], how='inner')[:500]
        accurate_results = df_results[df_results["label_x"] == df_results["label_y"]]
        return len(accurate_results) / len(df_results) * 100
    
    def run_results(self):
        '''
        Runs the results of the ComE dataset. 
        '''
        # self.zero_come_df = self.clean_data(self.zero_come_df)
        self.zero_come_df = self.clean_data(self.zero_come_df)
        self.few_come_df = self.clean_data(self.few_come_df)
        print("#### ComE Results ####")
        print(f"Llama-7b Prediction Accuracy (Zero-shot): {self.gather_come_accuracy(self.zero_come_df):.2f}%")
        print(f"Llama-7b Prediction Accuracy (Few-shot): {self.gather_come_accuracy(self.few_come_df):.2f}%")
        print()

class CosmosResults:
    def __init__(self, zero_cosmos_df, few_cosmos_df):
        self.zero_cosmos_df = zero_cosmos_df
        self.few_cosmos_df = few_cosmos_df
        self.truth_df = pd.read_csv("/home/grads/hassledw/ICL_Research/CosmosQA_results/CosmosQA-test-llama.csv")

    def clean_data(self, df):
        '''
        Cleans the data by retrieving the label and dropping None entries. 
        '''
        def get_response_cosmos(text):
            '''
            Cleans the text of the label to just get the response
            '''
            valid = ["1", "2", "3"]

            sentences = text.split("Response:")
            query = sentences[-1][:5]

            for char in query:
                if char in valid:
                    return char
                
            return None
        
        df["label"] = df["label"].apply(get_response_cosmos)
        orig_entries = df.shape[0]
        df = df.dropna()
        df["label"] = df["label"].astype(int)
        print(f"Dropped {orig_entries - df.shape[0]} \"None\" entries")
        return df
    
    def gather_cosmos_accuracy(self, df):
        '''
        Gets the overall accuracy of df.
        ''' 
        df_results = pd.merge(self.truth_df, df, on=['prompt'], how='inner')[:500]
        accurate_results = df_results[df_results["label_x"] == df_results["label_y"]]
        return len(accurate_results) / len(df_results) * 100
    
    def run_results(self):
        '''
        Runs the results of the cosmos dataset. 
        '''
        # self.zero_come_df = self.clean_data(self.zero_come_df)
        self.zero_cosmos_df = self.clean_data(self.zero_cosmos_df)
        self.few_cosmos_df = self.clean_data(self.few_cosmos_df)
        print("#### CosmosQA Results ####")
        print(f"Llama-7b Prediction Accuracy (Zero-shot): {self.gather_cosmos_accuracy(self.zero_cosmos_df):.2f}%")
        print(f"Llama-7b Prediction Accuracy (Few-shot): {self.gather_cosmos_accuracy(self.few_cosmos_df):.2f}%")
        print()
        
class ARCResults:
    def __init__(self, zero_arc_df, few_arc_df):
        self.zero_arc_df = zero_arc_df
        self.few_arc_df = few_arc_df
        self.truth_df = pd.read_csv("/home/grads/hassledw/ICL_Research/ARC-Challenge_results/ARC-challenge-llama.csv")

    def clean_data(self, df):
        '''
        Cleans the data by retrieving the label and dropping None entries. 
        '''
        def get_response_arc(text):
            '''
            Cleans the text of the label to just get the response
            '''
            valid = ["A", "B", "C", "D"]
            other_valid = ["(a)", "(b)", "(c)", "(d)"]

            sentences = text.split("Response:")
            query = sentences[-1][:10]

            for char in query:
                if char in valid:
                    return char
                
            for i, other in enumerate(other_valid):
                if other in query:
                    return valid[i]
                
            return None
        
        df["answerKey"] = df["answerKey"].apply(get_response_arc)
        orig_entries = df.shape[0]
        df = df.dropna()
        print(f"Dropped {orig_entries - df.shape[0]} \"None\" entries")
        return df
    
    def gather_arc_accuracy(self, df):
        '''
        Gets the overall accuracy of df.
        ''' 
        df_results = pd.merge(self.truth_df, df, on=['prompt'], how='inner')[:500]
        accurate_results = df_results[df_results["answerKey_x"] == df_results["answerKey_y"]]
        return len(accurate_results) / len(df_results) * 100
    
    def run_results(self):
        '''
        Runs the results of the arc dataset. 
        '''
        # self.zero_come_df = self.clean_data(self.zero_come_df)
        self.zero_arc_df = self.clean_data(self.zero_arc_df)
        self.few_arc_df = self.clean_data(self.few_arc_df)
        print("#### ARC-Challenge Results ####")
        print(f"Llama-7b Prediction Accuracy (Zero-shot): {self.gather_arc_accuracy(self.zero_arc_df):.2f}%")
        print(f"Llama-7b Prediction Accuracy (Few-shot): {self.gather_arc_accuracy(self.few_arc_df):.2f}%")
        print()

zero_arc_df = pd.read_csv("/home/grads/hassledw/ICL_Research/ARC-Challenge_results/ARC-Challenge-zeroshot-llama.csv")
few_arc_df = pd.read_csv("/home/grads/hassledw/ICL_Research/ARC-Challenge_results/ARC-Challenge-fewshot-llama.csv")
arcres = ARCResults(zero_arc_df, few_arc_df)
arcres.run_results()

zero_cosmos_df = pd.read_csv("/home/grads/hassledw/ICL_Research/CosmosQA_results/CosmosQA-zeroshot-llama.csv")
few_cosmos_df = pd.read_csv("/home/grads/hassledw/ICL_Research/CosmosQA_results/CosmosQA-fewshot-llama.csv")
cosmosres = CosmosResults(zero_cosmos_df, few_cosmos_df)
cosmosres.run_results()

zero_come_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_ComE_results/UDR_ComE-zeroshot-llama.csv")
few_come_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_ComE_results/UDR_ComE-fewshot-llama.csv")
comeres = ComEResults(zero_come_df, few_come_df)
comeres.run_results()

zero_snli_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_SNLI_results/UDR_SNLI-zeroshot-llama.csv")
few_snli_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_SNLI_results/UDR_SNLI-fewshot-llama.csv")
yelpres = SNLIResults(zero_snli_df, few_snli_df)
yelpres.run_results()

test_yelp_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_Yelp_results/UDR-yelp-llama.csv")
zero_yelp_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_Yelp_results/UDR_Yelp-zeroshot-llama.csv")
few_yelp_df = pd.read_csv("/home/grads/hassledw/ICL_Research/UDR_Yelp_results/UDR_Yelp-fewshot-llama.csv")
yelpres = YelpResults(zero_yelp_df, few_yelp_df)
yelpres.run_results()
