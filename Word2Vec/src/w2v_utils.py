import numpy as np # linear algebra
import pandas as pd # data processing

# data visualization 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px


import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score


# Libraries and packages for text (pre-)processing 
import string
import re
import nltk 
# nltk.download('stopwords')

from gensim.models import Word2Vec

# For type hinting
from typing import List



class Tokenizer: 
    """ After cleaning and denoising steps, in this class the text is broken up into tokens.
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
        text = re.sub(r"n\'t", " not", text)
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

    
    
def evaluate_model (y_true: pd.Series, 
                              y_pred: pd.Series, 
                              report:bool = False,
                              plot: bool = False)-> float:
    """
    A function to calculate F1, Accuracy, Recall, and Precision Score
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
        
    
    
    
    
def bow_vectorizer(doc_tokens: List[str]):
    """
    Using CountVectorizer, this function converts a list of tokenized text documents to a matrix of token counts (a Bog-of-Words sparse matrix).
    
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
    
    

def train_logistic_regressor(train_data_bow_matrix, 
                             train_data_label: pd.Series):
    """
    This function builds a LogisticRegressionCV Classifier Model with Bag-of-Words matrix of the Train dataset.
    
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
    
    print("==> Evaluation metrics on training data: ")
    evaluate_model(y_true = train_data_label, 
                   y_pred = train_data_predict_label)
    return bow_logistreg_model


def w2v_trainer(doc_tokens: List[str],
                epochs: int = 10,
                workers: int = 3,
                vector_size: int = 300,
                window: int = 5,
                min_count: int = 2):
    """ 
    Going through a list of lists, where each list within the main list contains a set of tokens from a doc, this function trains a Word2Vec model,
    then creates two objects to store keyed vectors and keyed vocabs   
    Parameters:
    doc_tokens   : A tokenized document 
    epochs       : Number of epochs training over the corpus
    workers      : Number of processors (parallelization)
    vector_size  : Dimensionality of word embeddings
    window       : Context window for words during training
    min_count    : Ignore words that appear less than this

    Returns:
    keyed_vectors       : A word2vec vocabulary model
    keyed_vocab 
    
    """
    w2v_model = Word2Vec(doc_tokens,
                         epochs=10,
                         workers=3,
                         vector_size=300,
                         window=5,
                         min_count=2)
    
    # create objects to store keyed vectors and keyed vocabs
    keyed_vectors = w2v_model.wv
    keyed_vocab = keyed_vectors.key_to_index
    
    return keyed_vectors, keyed_vocab
    
    
    
def calculate_overall_similarity_score(keyed_vectors,
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


def overall_semantic_sentiment_analysis (keyed_vectors, 
                                         positive_target_tokens: List[str],
                                         negative_target_tokens: List[str],
                                         doc_tokens: List[str], 
                                         doc_is_series: bool = True) -> float:
    """
    This function calculates the semantic sentiment of the text. 
    It first computes a  vector for the text (average of the  wordvectors building the text document vector)
    and two vectors representing our given positive and negative lists of words 
    and then calculates Positive and Negative Sentiment Scores as cosine similarity 
    between the text vector and the positive and negative vectors respectively.
    
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
  
    positive_score = doc_tokens.apply(lambda x: calculate_overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=positive_target_tokens, 
                                                                 doc_tokens=x))

    negative_score = doc_tokens.apply(lambda x: calculate_overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=negative_target_tokens, 
                                                                 doc_tokens=x))

    semantic_sentiment_score = positive_score - negative_score
    
    semantic_sentiment_polarity = semantic_sentiment_score.apply(lambda x: 1 if (x > 0) else 0)
                                          
    return positive_score, negative_score, semantic_sentiment_score, semantic_sentiment_polarity


def list_similarity(keyed_vectors, 
                    wordlist1: List[str], 
                    wordlist2: List[str]) -> pd.Series:
    """ A function to calculate vector similarity between 2 lists of tokens"""
    wv1= np.array([keyed_vectors[wd] for wd in wordlist1 if wd in keyed_vectors])
    wv2= np.array([keyed_vectors[wd] for wd in wordlist2 if wd in keyed_vectors])
    wv1 /= np.linalg.norm(wv1, axis=1)[:, np.newaxis]
    wv2 /= np.linalg.norm(wv2, axis=1)[:, np.newaxis]

    return np.dot(wv1, np.transpose(wv2))
    

def calculate_topn_similarity_score(keyed_vectors, 
                          target_tokens: List[str], 
                          doc_tokens: List[str],
                          topn: int = 10) -> float:
    """ The function defines the similarity of a single word to a document, 
    as the average of its similarity with the top_n most similar words in that document. 
    To calculate the similarity score it calculates the similarity of every word in the target_tokens set with all the words in the doc_tokens, 
    and keeps the top_n highest scores for each word and then averages over all the kept scores.
    -----
    Parameters:
    target_tokens List[str] : A list of sentiment or opinion words that indicate negative or positive opinions  
    
    doc_tokens List[str]    : A tokenized document 
    
    keyed_vectors           : A word2vec vocabulary model
    
    topn (int)              : An int that indicates the number of
    most similar vectors used to calculate the similarity score.

    
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




def topn_semantic_sentiment_analysis(keyed_vectors, 
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
  
    positive_score = doc_tokens.apply(lambda x: calculate_topn_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=positive_target_tokens, 
                                                                 doc_tokens=x,
                                                                     topn=topn))
                                      
    negative_score = doc_tokens.apply(lambda x: calculate_topn_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=negative_target_tokens, 
                                                                 doc_tokens=x,
                                                                     topn=topn))
                                           
    semantic_sentiment_score = positive_score - negative_score
    
    semantic_sentiment_polarity = semantic_sentiment_score.apply(lambda x: 1 if (x > 0) else 0)
                                          
    return positive_score, negative_score, semantic_sentiment_score, semantic_sentiment_polarity


def text_SSA(keyed_vectors,
              tokenizer,
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
        txt_PSS = calculate_topn_similarity_score(keyed_vectors=keyed_vectors, 
                                    target_tokens=positive_target_tokens, 
                                    doc_tokens=tokenized_txt,
                                    topn=topn)

        txt_NSS = calculate_topn_similarity_score(keyed_vectors=keyed_vectors, 
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

  
def define_complexity_subjectivity_reviews(df_slice,
                                          evaluate: bool = False,
                                          plot: bool = False):
  
      """
      A function to define high sentiment complexity reviews and Low Subjectivity reviews in a slice of dataset. 
      If evaluate: it evaluates the TopSSA model performance in df_slice 
      If plot: It plots distribution of low subjectivity reviews vs. high complexity reviews.
      """
      PSS_mean = df_slice["topn_PSS"].mean()
      PSS_std = df_slice["topn_PSS"].std()
      NSS_mean = df_slice["topn_NSS"].mean()
      NSS_std = df_slice["topn_NSS"].std()

      # High PSS(NSS)
      high_PSS = df_slice["topn_PSS"] > PSS_mean + PSS_std
      high_NSS = df_slice["topn_NSS"] > NSS_mean + NSS_std

      # Low PSS(NSS)
      low_PSS = df_slice["topn_PSS"] < PSS_mean - PSS_std
      low_NSS = df_slice["topn_NSS"] < NSS_mean - NSS_std

      # High sentiment complexity
      high_complexity = high_PSS & high_NSS
      df_slice_high_complexity = df_slice[high_complexity]

      # low subjectivity
      low_subjectivity = low_PSS & low_NSS
      df_slice_low_subjectivity = df_slice[low_subjectivity]

      if evaluate:
          print("\n Number of reviews with high sentiment complexity: ", len(df_slice_high_complexity))
          evaluate_model(df_slice_high_complexity["sentiment"],
                         df_slice_high_complexity["topn_semantic_sentiment_polarity"])


          print("\n Number of reviews with less subjectivity: ", len(df_slice_low_subjectivity))
          evaluate_model(df_slice_low_subjectivity["sentiment"], 
                         df_slice_low_subjectivity["topn_semantic_sentiment_polarity"])
          print("\n")

      if plot:
          # PLOTTING
          # filter positive and negative review based on Target Variable (actual 'y') or 'sentiment' column
          actual_pos_filt = df_slice_high_complexity['sentiment'] == 1
          actual_neg_filt = df_slice_high_complexity['sentiment'] == 0

          actual_pos_low_subjectivity = df_slice_low_subjectivity['sentiment'] == 1
          actual_neg_low_subjectivity = df_slice_low_subjectivity['sentiment'] == 0

          # plotting Semantic Sentiment Score Position of Actual Negative Reviews 
          plt.scatter(df_slice_low_subjectivity['topn_NSS'][actual_neg_low_subjectivity], 
                   df_slice_low_subjectivity['topn_PSS'][actual_neg_low_subjectivity],  
                   label='Actual Negetive Low Subjectivity Reviews',
                     color='deeppink',
                      alpha=0.4 , # set transparency of color
                      s=20 # set size of dots
                     )

          # plotting Semantic Sentiment Score Position of Actual Positive Reviews 
          plt.scatter(df_slice_low_subjectivity['topn_NSS'][actual_pos_low_subjectivity], 
                   df_slice_low_subjectivity['topn_PSS'][actual_pos_low_subjectivity],  
                   label='Actual Positive Low Subjectivity Reviews',
                 color='springgreen',
                      alpha=0.2, # set transparency of color
                      s=20 # set size of dots
                     )

          # plotting Semantic Sentiment Score Position of Actual Negative Reviews 
          plt.scatter(df_slice_high_complexity['topn_NSS'][actual_neg_filt], 
                   df_slice_high_complexity['topn_PSS'][actual_neg_filt],  
                   label='Actual Negetive High Complexity Reviews',
                     color='DarkRed',
                      alpha=0.4 , # set transparency of color
                      s=20 # set size of dots
                     )

          # plotting Semantic Sentiment Score Position of Actual Positive Reviews 
          plt.scatter(df_slice_high_complexity['topn_NSS'][actual_pos_filt], 
                   df_slice_high_complexity['topn_PSS'][actual_pos_filt],  
                   label='Actual Positive High Complexity Reviews',
                 color='DarkGreen',
                      alpha=0.5, # set transparency of color
                      s=20 # set size of dots
                     )
          # naming the x & y axis
          plt.xlabel('Predicted Negative Sentiment Score (NSS)')
          plt.ylabel('Predicted Positive Sentiment Score (PSS)')


          # plotting the bisector
          plt.plot([0,0.4], 
                   [0,0.4], 
                   alpha=0.5,
                   label='Decision Boundry')

          # show a legend on the plot
          plt.legend()

          # giving a title to the graph
          plt.title("""
          Distribution of low subjectivity reviews vs. high complexity reviews
          """)

          # To save the result in the same folder
          plt.savefig('../reports/figures/low_subjectivity_vs_high_complexity_reviews_on_PSS_NSS_plane.png')

          plt.show()

      return df_slice_high_complexity, df_slice_low_subjectivity

  
def explore_high_complexity_reviews(df_slice):  
      """
      A function to plot the distribution of high complexity reviews on the PSS-NSS plane.
      Using plotly, this plot let you explore the reviews by hovering over the datapoints.
      """
      df_slice_high_complexity1 =define_complexity_subjectivity_reviews(df_slice)[0]
      text_high_complexity = df_slice_high_complexity1['review'].str.wrap(100).str.replace("\n", "<br>"),

      fig = px.strip(df_slice_high_complexity1, 
                     x= "topn_NSS", 
                     y="topn_PSS",
                     color= "sentiment",
                     color_discrete_sequence=['red','green'],
                     hover_name = "tokenized_text_len",
                     hover_data=text_high_complexity,
                     height=800,
                     width=800)

      fig.update_layout(
         title = "Distribution of high complexity reviews on the PSS-NSS plane",
         xaxis_title = "Negative Sentiment Score (NSS)",
         yaxis_title = "Positive Sentiment Score (PSS)",
         font = dict(
            family = "Courier New, monospace",
            size = 12,
            color = "#7f7f7f"
         )
      )

      fig.add_trace(px.line(x=[0.2,0.4], y=[0.2,0.4]).data[0])

      fig.show()

      return
    
  
def explore_low_subjectivity_reviews(df_slice):
      """
      A function to plot the distribution of low subjectivity reviews on the PSS-NSS plane.
      Using plotly, this plot let you explore the reviews by hovering over the datapoints.
      """
      df_slice_low_subjectivity = define_complexity_subjectivity_reviews(df_slice)[1]
      text_low_subjectivity = df_slice_low_subjectivity['review'].str.wrap(100).str.replace("\n", "<br>"),


      fig = px.strip(df_slice_low_subjectivity, 
                     x= "topn_NSS", 
                     y="topn_PSS",
                     color= "sentiment",
                     color_discrete_sequence=['green','red'],
                     hover_name = "tokenized_text_len",
                     hover_data=text_low_subjectivity,
                     height=700,
                     width=700)

      fig.update_layout(
         title = "Distribution of low subjectivity reviews on the PSS-NSS plane",
         xaxis_title = "Negative Sentiment Score (NSS)",
         yaxis_title = "Positive Sentiment Score (PSS)",
         font = dict(
            family = "Courier New, monospace",
            size = 12,
            color = "#7f7f7f"
         )
      )

      fig.add_trace(px.line(x=[0,0.2], y=[0,0.2]).data[0])
      fig.show()

      return
