# -*- coding: utf-8 -*-
"""
Created by Tatyana Sasynuik
for KaggleX 2023 Mentorship Progaam

Objective: Explore text summarization space
base from https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
LSA: https://towardsdatascience.com/latent-semantic-analysis-intuition-math-implementation-a194aff870f8


TO DO:
    3/1/2023 use different summarization methods. Maybe do a write up on the pros/cons of each method?


CHANGE LOG:
    3/1/2023 followed the walkthrough exactly, changes some errors that didn't work for me
    3/7/2023 add a similarity score that uses LSA; have output be a dataframe and not a print screen
"""

# Import all necessary libraries
import re
import numpy as np
import networkx as nx
import nltk
import pandas as pd
import plotly as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stopwords = set(stopwords.words("english"))

# Generate clean sentences
def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        # print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        # sentences.pop() #not sure why this exists, seems to empty the list before it is used
    return sentences

# sentences = read_article("msft.txt")

# Sentence Similarity Methods

# Cosine Similarity
def cosine_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

# Cosine Similarity Matrix 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = cosine_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

# matrix = build_similarity_matrix(sentences, stopwords)



# Vectorize sentences (document-term matrix)
def vectorize(text):
    tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
    tfidf = TfidfVectorizer(lowercase=True, 
                           stop_words=stopwords, 
                           tokenizer=tokenizer.tokenize, 
                           max_df=0.2,
                           min_df=0.02
                          )
    tfidf_sparse = tfidf.fit_transform(text)
    tfidf_df = pd.DataFrame(tfidf_sparse.toarray(), 
                            columns=tfidf.get_feature_names())
    return tfidf_df

def lsa_objects_create(text):
    tfidf = vectorize(text)
    lsa_obj = TruncatedSVD(n_components=20, n_iter=100, random_state=42)
    tfidf_lsa_data = lsa_obj.fit_transform(tfidf)
    Sigma = lsa_obj.singular_values_
    V_T = lsa_obj.components_.T
    term_topic_matrix = pd.DataFrame(data=tfidf_lsa_data, 
                                 index = tfidf.columns, 
                                 columns = [f'Latent_concept_{r}' for r in range(0,V_T.shape[1])])
    return term_topic_matrix, Sigma

lsa_objects_create(read_article("msft.txt")).Sigma

sns.barplot(x=list(range(len(Sigma))), y = Sigma)
data = lsa_objects_create[f'Latent_concept_1']
data = data.sort_values(ascending=False)
top_10 = data[:10]
plt.title('Top terms along the axis of Latent concept 1')
fig = sns.barplot(x= top_10.values, y=top_10.index)

# Generate Summary Method
def generate_summary(file_name, top_n=5):
    summarize_text = []
    # Step 1 - Read text and tokenize
    sentences =  read_article(file_name)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stopwords)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))


# Call the summary on some text
generate_summary( "msft.txt",2)
