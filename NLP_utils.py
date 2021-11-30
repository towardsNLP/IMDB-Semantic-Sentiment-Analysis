import numpy as np # linear algebra
import pandas as pd # data processing

import matplotlib.pyplot as plt
import seaborn as sns # data visualization 


import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score


# Libraries and packages for text (pre-)processing 
import string
import re
import nltk 
nltk.download('stopwords')

from gensim.models import Word2Vec

# For type hinting
from typing import List



# ======== Tokenizer Class ========
class Tokenizer: 
    """ After cleaning and denoising steps, in thsi calss the text is broken up into tokens.
    if clean: clean the text from all non-alphanumeric characters,
    if lower: convert the text into lowercase,
    If de_noise: remove HTML and URL components,
    if remove_stop_words: and remove stop-words,
    If keep_neagation: attach the negation tokens to the next token 
     and treat them as a single word before removing the stopwords
     
    Returns:
    List of tokens
    """
    # initialization method to create the default instance constructor for the class
    def __init__(self,
                 clean: bool = True,
                 lower: bool = True, 
                 de_noise: bool = True, 
                 remove_stop_words: bool = True,
                keep_negation: bool = True) -> List[str]:
      
        self.de_noise = de_noise
        self.remove_stop_words = remove_stop_words
        self.clean = clean
        self.lower = lower
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.keep_negation = keep_negation

    # other methods  
    def denoise(self, text: str) -> str:
        """
        Removing html and URL components
        """
        html_pattern = r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"
        url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"

        text = re.sub(html_pattern, " ", text)
        text = re.sub(url_pattern," ",text).strip()
        return text
       
    
    def remove_stopwords(self, tokenized_text: List[str]) -> List[str]:
        text = [word for word in tokenized_text if word not in self.stopwords]
        return text

    
    def keep_negation_sw(self, text: str) -> str:
        """
        A function to save negation words (n't, not, no) from removing as stopwords
        """
        # to replace "n't" with "not"
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"\'t", " not", text)
        # to join not/no into the next word
        text = re.sub("not ", " NOT", text)
        text = re.sub("no ", " NO", text)
        return text
    
    
    def tokenize(self, text: str) -> List[str]:
        """
        A function to tokenize words of the text
        """
        non_alphanumeric_pattern =r"[^a-zA-Z0-9]"
        
        # to substitute multiple whitespace with single whitespace
        text = ' '.join(text.split())

        if self.de_noise:
            text = self.denoise(text)
        if self.lower:
            text = text.lower()
        if self.keep_negation:
            text = self.keep_negation_sw(text)
            
        if self.clean:
            # to remove non-alphanumeric characters
            text = re.sub(non_alphanumeric_pattern," ", text).strip()

        tokenized_text = text.split()

        if self.remove_stop_words:
            tokenized_text = self.remove_stopwords(tokenized_text)

        return tokenized_text

    
    
# ======== Model Evaluation Metrics Function ========
def model_evaluation_metrics (y_true: pd.Series, 
                              y_pred: pd.Series, 
                              report:bool = False,
                              plot: bool = False)-> float:
    """
    A function to calculate F1, Accuracy, Recall, and Percision Score
    If report: it prints classification_report 
    If plot: it prints Confusion Matrix Heatmap
    """
    if report:
        print(classification_report(y_true, 
                            y_pred,
                            digits=4))
    if plot:
        # figure
        fig, ax = plt.subplots(figsize=(4, 4))
        conf_matrix = pd.crosstab(y_true, 
                           y_pred, 
                           rownames=['Actual'], 
                           colnames=['Predicted'])
        sns.heatmap(conf_matrix, 
                    annot=True, fmt=".0f",
                    cmap='RdYlGn', # use orange/red colour map
                    cbar_kws={'fraction' : 0.04}, # shrink colour bar
                    linewidth=0.3, # space between cells
                   ) 
        plt.title('Confusion Matrix', fontsize=14)
        plt.show()
        
    if not report and not plot:
        print('* Accuracy Score: ', "{:.4%}".format(accuracy_score(y_true, y_pred)))
        print('* F1 Score: ', "{:.4%}".format(f1_score(y_true, y_pred )))
        print('* Recall Score: ', "{:.4%}".format(recall_score(y_true , y_pred )))
        print('* Precision Score: ', "{:.4%}".format(precision_score(y_true , y_pred)))
        
    return
    
    
    
    
# ======== Bag-of-Words CountVectorizer Function ========
def bow_vectorizer(doc_tokens: List[str]) -> "text.CountVectorizer":
    """
    Using CountVectorizer, this function converts a collection of tokenized text documents to a matrix of token counts (a Bog-of-Words sparse matrix).
    
    Parameters:
    doc_tokens               : A tokenized document 

    Returns:
    fit_bow_count_vect       : A fit Bog-of-Words model

    """
    # Defining CountVectorizer
    count_vect = CountVectorizer(
        analyzer='word',
        tokenizer=lambda doc:doc,
        preprocessor=lambda doc:doc,
        min_df=5,
        token_pattern=None)
    
    # Create a sparse matrix out of the frequency of vocabulary words in Train Dataset
    fit_bow_count_vect = count_vect.fit(doc_tokens)
    
    return fit_bow_count_vect
    
    
    
    
# ======== Bag-of-Words Sentiment Analysis Model Training Function ========
def bow_sentiment_analysis_model_training(train_data_bow_matrix: "text.CountVectorizer", 
                                          train_data_label: pd.Series) -> "_logistic.LogisticRegressionCV":
    """
    This function builds a LogisticRegressionCV Classifier Model with Bag-of-Words matrix of 
    the Train dataset.
    
    Parameters:
    train_data_bow_matrix    : A Train dataset as Bog-of-Words sparse matrix 
    train_data_label         : Target values of the Train dataset

    Returns:
    bow_logistreg_model      : A fit LogisticRegression model on Bag-of_words vectors
    """
    bow_logistreg_model=LogisticRegressionCV(cv=5,
                          random_state=42,
                          n_jobs=-1,
                          verbose=3,
                          max_iter=300).fit(train_data_bow_matrix, train_data_label)
    
    train_data_predict_label = bow_logistreg_model.predict(train_data_bow_matrix)
    
    print("==> Evaluation Metrics when model applied on Train Data: ")
    print(model_evaluation_metrics(y_true = train_data_label, y_pred = train_data_predict_label))
    return bow_logistreg_model


# ======== Word2Vec Model Trainer Function ========
# Train the word2vec model
def w2v_trainer(doc_tokens: List[str]) -> "keyedvectors.Word2VecKeyedVectors":
    """ 
    Going through a list of lists, where each list within the main list contains a set of tokens from a doc, 
    this function train a Word2Vec model, then creates two objects to store keyed vectors and keyed vocabs   
    Parameters:
    doc_tokens   : A tokenized document 

    Returns:
    keyed_vectors       : A word2vec vocabulary model
    keyed_vocab 
    
    """
    w2v_model = Word2Vec(doc_tokens,
                     iter=10,    # Number of epochs training over corpus
                     workers=3,  # Number of processors (parallelisation)
                     size=300,   # Dimensionality of word embeddings
                     window=5,   # Context window for words during training
                     min_count=2)# Ignore words that appear less than this
    
    # create objects to store keyed vectors and keyed vocabs
    keyed_vectors = w2v_model.wv
    keyed_vocab = keyed_vectors.vocab
    
    return keyed_vectors, keyed_vocab
    
    
    
# ======== Overall Semantic Similarity Score Function ========
def overall_similarity_score(keyed_vectors: "keyedvectors.Word2VecKeyedVectors", 
                     target_tokens: List[str], 
                     doc_tokens: List[str]) -> float:
    """
    Going through a tokenized doc, this function computes vector similarity between 
    doc_tokens and target_tokens as 2 lists by n_similarity(list1, list2) method based on 
    Word2Vec vocabulary (keyed_vectors), 
    then returns the similarity scores.  
    
    Parameters:
    target_tokens  : A set of semantically co-related words  
    doc_tokens     : A tokenized document 
    keyed_vectors  : A word2vec vocabulary model
    
    Returns:
    vector similarity scores between 2 tokenized list doc_tokens and target_tokens  
    """
    
    target_tokens = [token for token in target_tokens if token in keyed_vectors]

    doc_tokens = [token for token in doc_tokens if token in keyed_vectors]
    
    similarity_score = keyed_vectors.n_similarity(target_tokens, doc_tokens)
    
    return similarity_score




# ======== Overall Semantic Sentiment Analysis Function ========
def overall_semantic_sentiment_analysis (keyed_vectors: "keyedvectors.Word2VecKeyedVectors", 
                                         positive_target_tokens: List[str],
                                         negative_target_tokens: List[str],
                                         doc_tokens: List[str], 
                                         doc_is_series: bool = True) -> float:
    """
    A function to calculate the semantic sentiment of the text by calculating its similarity 
    to our negative and positive sets. 
    Using Word2Vec model it builds the document vector by averaging over the vectors building it. 
    So, we will have a vector for every review and two vectors representing our positive and negative sets.
    The positive and negative scores can then be calculated by a simple cosine similarity between the positive and negative vectors.computing vector similarity between 
    doc_tokens and a positive_target_tokens (as positive_score) then computing vector similarity between 
    doc_tokens and a negative_target_tokens (as negative_score), and finally comparing these two scores. 
    
    Parameters:
    keyed_vectors           : A word2vec vocabulary model
    positive_target_tokens  : A list of sentiment or opinion words that indicate positive opinions 
    negative_target_tokens  : A list of sentiment or opinion words that indicate negative opinions  
    doc_tokens              : A tokenized document 
    
    
    Returns:
    positive_score : vector similarity scores between doc_tokens and positive_target_tokens
    negative_score : vector similarity scores between doc_tokens and negative_target_tokens
    
    semantic_sentiment_score  : positive_score - negative_score
    semantic_sentiment_polarity : Overall score: 0 for more negative or 1 for more positive doc
    """
  
    positive_score = doc_tokens.apply(lambda x: overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=positive_target_tokens, 
                                                                 doc_tokens=x))

    negative_score = doc_tokens.apply(lambda x: overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=negative_target_tokens, 
                                                                 doc_tokens=x))

    semantic_sentiment_score = positive_score - negative_score
    
    semantic_sentiment_polarity = semantic_sentiment_score.apply(lambda x: 1 if (x > 0) else 0)
                                          
    return positive_score, negative_score, semantic_sentiment_score, semantic_sentiment_polarity


# ======== List Similarity Computing Function ========
def list_similarity(keyed_vectors: "keyedvectors.Word2VecKeyedVectors", 
                    wordlist1: List[str], 
                    wordlist2: List[str]) -> pd.Series:
    """ A function to calculate vector similarity between 2 lists of tokens"""
    wv1= np.array([keyed_vectors[wd] for wd in wordlist1 if wd in keyed_vectors])
    wv2= np.array([keyed_vectors[wd] for wd in wordlist2 if wd in keyed_vectors])
    wv1 /= np.linalg.norm(wv1, axis=1)[:, np.newaxis]
    wv2 /= np.linalg.norm(wv2, axis=1)[:, np.newaxis]

    return np.dot(wv1, np.transpose(wv2))
    

    
    
# ======== TopN Semantic Similarity Score Function ========
def topn_similarity_score(keyed_vectors: "keyedvectors.Word2VecKeyedVectors", 
                          target_tokens: List[str], 
                          doc_tokens: List[str],
                          topn: int = 10) -> float:
    """ Going through a tokenized doc, this function measures vector similarity between 
    doc_tokens and target_tokens as 2 lists based on Word2Vec vocabulary (keyed_vectors), 
    then returns the similarity scores. 
    To claculate the similarity score it calculates the similarity of every word in the target_tokens set 
    with all the words in the doc_tokens, and keeps the top_n highest scores for each word 
    and then averages over all the kept scores.
    -----
    Parameters:
    target_tokens List[str] : A list of sentiment or opinion words that indicate negative or positive opinions  
    
    doc_tokens List[str]    : A tokenized document 
    
    keyed_vectors           : A word2vec vocabulary model
    
    topn (int)              : An int that indicates the number of
    most similar vectors used to claculate the similarity score.

    
    Returns:
    vector similarity scores between 2 tokenized list doc_tokens and target_tokens  
    """
    topn = min(topn, round(len(doc_tokens)))
    
    target_tokens = [token for token in target_tokens if token in keyed_vectors]

    doc_tokens = [token for token in doc_tokens if token in keyed_vectors]
    
    sim_matrix = list_similarity(keyed_vectors=keyed_vectors, 
                                 wordlist1=target_tokens,
                                 wordlist2=doc_tokens)
    sim_matrix = np.sort(sim_matrix, axis=1)[:, -topn:]
     
    similarity_score = np.mean(sim_matrix)
    
    return similarity_score




# ======== TopN Semantic Sentiment Analysis Function ========
def topn_semantic_sentiment_analysis(keyed_vectors: "keyedvectors.Word2VecKeyedVectors", 
                                      positive_target_tokens: List[str],
                                      negative_target_tokens: List[str],
                                      doc_tokens: List[str],
                                      topn: int = 10) -> float:
    """
    A function to calculate the semantic sentiment of the text by measuring vector similarity between 
    doc_tokens and a positive_target_tokens (as positive_score) then measuring vector similarity between 
    doc_tokens and a negative_target_tokens (as negative_score), and finally comparing these two scores. 
    
    Parameters:
    keyed_vectors           : A word2vec vocabulary model
    positive_target_tokens  : A list of sentiment or opinion words that indicate positive opinions 
    negative_target_tokens  : A list of sentiment or opinion words that indicate negative opinions  
    doc_tokens              : A tokenized document 
    
    
    Returns:
    positive_score            : vector similarity scores between doc_tokens and positive_target_tokens
    negative_score            : vector similarity scores between doc_tokens and negative_target_tokens
    
    semantic_sentiment_score  : positive_score - negative_score
    semantic_sentiment_polarity       : Overall score: 0 for more negative or 1 for more positive doc
    """
  
    positive_score = doc_tokens.apply(lambda x: topn_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=positive_target_tokens, 
                                                                 doc_tokens=x,
                                                                     topn=topn))
                                      
    negative_score = doc_tokens.apply(lambda x: topn_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=negative_target_tokens, 
                                                                 doc_tokens=x,
                                                                     topn=topn))
                                           
    semantic_sentiment_score = positive_score - negative_score
    
    semantic_sentiment_polarity = semantic_sentiment_score.apply(lambda x: 1 if (x > 0) else 0)
                                          
    return positive_score, negative_score, semantic_sentiment_score, semantic_sentiment_polarity


# ======== Text Semantic Sentiment Analysis Function ========
def text_SSA(keyed_vectors: "keyedvectors.Word2VecKeyedVectors",
              tokenizer: "__main__.Tokenizer",
              positive_target_tokens: List[str],
              negative_target_tokens: List[str],
              topn: int = 30) -> float:
    """
    A function to analyze text semantic sentiment.
    """
    repeat = True
    while repeat:
        txt = input("Please insert text here: \n")
        tokenized_txt = tokenizer.tokenize(txt)
        txt_PSS = topn_similarity_score(keyed_vectors=keyed_vectors, 
                                    target_tokens=positive_target_tokens, 
                                    doc_tokens=tokenized_txt,
                                    topn=topn)

        txt_NSS = topn_similarity_score(keyed_vectors=keyed_vectors, 
                                    target_tokens=negative_target_tokens, 
                                    doc_tokens=tokenized_txt,
                                    topn=topn)

        txt_S3 = txt_PSS - txt_NSS
        semantic_sentiment_polarity = "Positive" if (txt_S3 >= 0) else "Negative"
        green = "\033[1;32m"
        red = "\033[1;31m"
        color = green if (txt_S3 >= 0) else red # to print colored text

        print("Tokenized text:\n ", tokenizer.tokenize(txt))
        print("PSS =", txt_PSS)
        print("NSS =", txt_NSS)
        print("S3 =", txt_S3)
        print(color + "Semantic Sentiment =", semantic_sentiment_polarity)
        print("\n")
        repeat = False
        answer = input('Do you want to analyze another text sentiment? (Yes/No): ').upper()
        if answer in ['Y', 'YES', 'YEP', 'YA']:
            repeat = True
            print("-------------------------")

        else:
            print ('\n Thank you! See you later.')
    return
