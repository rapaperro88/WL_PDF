# PDF
from PyPDF2 import PdfFileReader
import pandas as pd
import numpy as np
import os
import re
# CrossRef
import requests
import json
# Cleaning
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import stop_words
import unidecode
import string
# Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Similarity
import gensim
from gensim import models, corpora, similarities
from gensim.models import LdaModel
from indra.literature.pubmed_client import get_metadata_for_ids 
from scipy.stats import entropy

# Frontend
import streamlit as st

# ________________________________________________________________________________________
# ________________________________________________________________________________________
#
#                          METADATA FROM PDF FILE
# ________________________________________________________________________________________
# ________________________________________________________________________________________

def get_full_text(pdfReader):
    '''
    From a PyPDF2.PdfFileReader object, returns string containing 
    the full text extracted from the pdf file.
    '''
    full_text = ""
    for p in range(pdfReader.getNumPages()):
        pageObj = pdfReader.getPage(p) 
        full_text = full_text + " " + pageObj.extractText()
    return full_text

def get_text_from_collection(path):
    '''
    Returns list of strings corresponding to each pdf document in a folder 
    in 'path' argument. Uses get_full_text function.
    '''
    l = []
    pdf_filenames = [pdf for pdf in os.listdir(path) if pdf.endswith(".pdf")]
    for i, file in enumerate(pdf_filenames):
        # open & read pdfs
        pdfFileObj = open(os.path.join(path, file), 'rb') 
        pdfReader = PdfFileReader(pdfFileObj)
        l.append(get_full_text(pdfReader))
    return l   

def compile_vocabulary(list_strings):
    '''
    TODO : Returns 
    '''
    pass       

def get_pdf_data (path):
    '''
    Input: path to folder containing a collection of pdf files
    Output: pandas dataframe with retrieved elements 
    plus a list of filenames where metadata was not found
    '''
    
    pdf_filenames = [pdf for pdf in os.listdir(path) if pdf.endswith(".pdf")]
    
    columns = ["Publi_ID", "Year", "Authors", "Title", "Journal", 
            "DOI", "Keywords", "Total", "words count", "AAV", "Frequency",
                "Publi_ID linked", "Lettre_Chiffre_Lettre"]
    
    df = pd.DataFrame(columns = columns) # df to fill
    encrypted_files = []
    
    for i, file in enumerate(pdf_filenames):
        
        # open & read pdfs
        pdfFileObj = open(os.path.join(path, file), 'rb') 
        pdfReader = PdfFileReader(pdfFileObj) 
        
        # jump encrypted files, keep record though
        if pdfReader.isEncrypted:
            encrypted_files.append(file)
            pass
        
        # Metadata PyPDF2 object
        pdf_info = pdfReader.getDocumentInfo()
        
        full_text = get_full_text(pdfReader)
        
        # retrieve author
        try:  
            author = str(pdf_info["/Author"]).split(", ")
            df.at[i, 'Authors'] = author
        except: 
            df.at[i, 'Authors'] = None
        
        # retrieve keywords
        try: 
            keywords = str(pdf_info["/Keywords"]).split(", ") 
            df.at[i, 'Keywords'] = keywords
        except: 
            df.at[i, 'Keywords'] = None
        
        # retrieve title
        try:   
            title = str(pdf_info["/Title"]).split(", ")[0]
            df.at[i, 'Title'] = title
        except: 
            df.at[i, 'Title'] = None
            
        # retrieve AAV
        try:
            aav = re.findall("(AAV-?[A-Za-z0-9]{1,9}) ", 
                            full_text)
            aav = list(set(aav))
            df.at[i, 'AAV'] = aav
        except:
            df.at[i, 'AAV'] = None
        
        # retrieve doi
        try:  
            doi = str(pdf_info["/doi"]).split(", ")[0]
            df.at[i, 'DOI'] = doi
        except:
            try: # retrieve by most frequent regex match
                doi = re.findall('(10[.][0-9]{4,}(?:[.][0-9]+)*\/(?:(?!["&\'<>])\S)+)', 
                                full_text)
                doi = max(set(doi), key = doi.count)
                df.at[i, 'DOI'] = doi
            except:
                df.at[i, 'DOI'] = None            
            
    # List filenames where no data was retrieved
    mask = df.isna().all(axis=1)
    missing_data = np.array(pdf_filenames)[mask].tolist()
    
    # Drop empty rows
    df = df.dropna( how='all').reset_index(drop=True)
            
    return missing_data, encrypted_files, df


# ________________________________________________________________________________________
# ________________________________________________________________________________________
#
#                          METADATA FROM CROSSREF
# ________________________________________________________________________________________
# ________________________________________________________________________________________

def get_crossref_metadata(df):
    df = df
    for i, doi in enumerate(df["DOI"]):
        url = f"https://api.crossref.org/v1/works/{doi}"
        
        # Request CrossReference API
        try: 
            json_ = json.loads(requests.get(url).content)
        except: continue
        
        # # AUTHOR  
        try:
            author = [(author["given"] + " " + author["family"]) for author in json_["message"]["author"]]
            if len(author) > len(df["Authors"][i]):
                df.at[i,"Authors"] = author
        except: pass

        # TITLE
        if not df["Title"][i]:
            try:
                title = json_["message"]["title"][0]
                df.at[i,"Title"] = title
            except: pass

        # JOURNAL
        try:
            journal = json_["message"]["container-title"][0]
            df.at[i,"Journal"] = journal
        except: pass

        # YEAR
        try:
            year = json_["message"]["created"]["date-parts"][0][0]
            df.at[i,"Year"] = year
        except: pass
            
    return df

# ________________________________________________________________________________________
# ________________________________________________________________________________________
#
#                          CLEANING
# ________________________________________________________________________________________
# ________________________________________________________________________________________


def preprocessing(texte, return_str=False):
    tex = []

    # lower case
    texte = unidecode.unidecode(texte.lower())

    # remove special characters
    texte = re.sub(r'\n', ' ', texte)
    texte = re.sub(r'\d+', '', texte)
    
    # remove punctuation
    texte = texte.translate(str.maketrans('', '', string.punctuation))
    
    # remove whitespaces
    texte = texte.strip()
        
    # tokenization
    tokens = word_tokenize(texte)
        
    # define stop words
    sw_1 = stop_words.get_stop_words('en')
    sw_nltk = set(stopwords.words('english'))
    sw = list(set(sw_1+list(sw_nltk)))
    
    # remove stop words and filster monoletters
    tokens = [i for i in tokens if not i in sw and len(i) > 1]
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    if return_str:
        return (" ").join(tokens)
    
    return tokens

# ________________________________________________________________________________________
# ________________________________________________________________________________________
#
#                          CLUSTERING
# ________________________________________________________________________________________
# ________________________________________________________________________________________


@st.cache(suppress_st_warning=True)
def cluster_abstracts(df, abstracts, labels):
    '''
    df: pandas dataframe
    abstracts: string, name of abstracts column
    labels: string, name of topics column
    '''
    # Preprocess abstract
    df["clean"] = df[abstracts].apply(lambda x: preprocessing(x, return_str=True))
    df["tokens"] = df[abstracts].apply(lambda x: preprocessing(x, return_str=False))

    # Encode topics
    le = LabelEncoder()
    y_true = le.fit_transform(df[labels].values)

    # Embedding of clean string
    print("vectorizing with TF-IDF")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[abstracts])

    # Cluster with kmeans
    print("Clustering using KMeans")
    true_k = len(df[labels].unique()) # TODO: select number of clusters by silhouette score
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    # Perform tSNE to reduce dimensions to 2
    print("Dimension reduction with t-SNE")
    tsne = TSNE(n_components=3, verbose=0, perplexity=100, random_state=0, n_jobs=-1)
    X_embedded = tsne.fit_transform(X)

    # Display clusters
    print("Generating visualisation")
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_embedded[:,0], X_embedded[:,1],  X_embedded[:,2], c=model.labels_, marker=".") # Mapping of articles

    return fig, model.labels_

# ________________________________________________________________________________________
# ________________________________________________________________________________________
#
#                          similarity
# ________________________________________________________________________________________
# ________________________________________________________________________________________

def similarity_train(df, num_topics, abstracts, labels):
    '''
    Implementation of LDA with the inputed data
    '''
    # Preprocess abstract
    df["clean"] = df[abstracts].apply(lambda x: preprocessing(x, return_str=True))
    df["tokens"] = df[abstracts].apply(lambda x: preprocessing(x, return_str=False))

    num_topics = num_topics
    chunksize = 300
    dictionary = corpora.Dictionary(df["tokens"])
    corpus = [dictionary.doc2bow(doc) for doc in df["tokens"].values]

    # low alpha means each document is only represented by a small number of topics, and vice versa
    # low eta means each topic is only represented by a small number of words, and vice versa
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)

    return dictionary,corpus,lda

def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    from: https://www.kaggle.com/ktattan/lda-and-document-similarity
    """
    # lets keep with the p,q notation above
    p = query[None,:].T # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    pp = np.repeat(query[None,:].T, repeats=matrix.shape[0], axis=1)
    return np.sqrt(0.5*(entropy(pp,m) + entropy(q,m)))

def get_most_similar_documents(query,matrix,k):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    from: https://www.kaggle.com/ktattan/lda-and-document-similarity
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[:k] # the top k positional index of the smallest Jensen Shannon distances

def similarity_test(df, text, model, corpus, dictionary, k):

    # Training info
    doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in model[corpus]])

    # text: From tokens to topic distribution
    new_bow = dictionary.doc2bow(preprocessing(text, return_str=False))
    new_doc_distribution = np.array([tup[1] for tup in model.get_document_topics(bow=new_bow)])

    # Get most similar documents ids from df
    most_sim_ids = get_most_similar_documents(new_doc_distribution,doc_topic_dist, k=10)

    # # Get most similar items from df
    # pubmed_id_column_index = 0
    # li = list(df.iloc[list(most_sim_ids), pubmed_id_column_index].values)

    # # query pubmed info:
    # suggest = get_metadata_for_ids(li)
    # title_list = [val["title"] for val in suggest.values()]

    return  most_sim_ids





# ________________________________________________________________________________________
# ________________________________________________________________________________________
#
#                          FRONTEND
# ________________________________________________________________________________________
# ________________________________________________________________________________________


def file_selector(folder_path='./data', text="Select a file", file_ext=".csv"):
        filenames = [n for n in os.listdir(folder_path) if n.endswith(file_ext)]
        selected_filename = st.selectbox(text,filenames)
        return os.path.join(folder_path,selected_filename)



