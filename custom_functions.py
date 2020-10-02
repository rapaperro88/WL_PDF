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

# ________________________________________________________________________________________
# ________________________________________________________________________________________
#
#                          METADATA FROM PDF FILE
# ________________________________________________________________________________________
# ________________________________________________________________________________________

def flask_test(path):
    return path+"test passed !"

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
    Returns 
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