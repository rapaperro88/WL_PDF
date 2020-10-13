import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re
from indra.literature.pubmed_client import get_abstract
from sys import argv
from random import random
from time import sleep

def get_abstracts ():
    '''
    gets a list of pubmed ids for each topic in a .txt file. max_articles is the number of
    abstracts to retrieve.

    usage: python abstract_mining.py <max_articles> <path_to_topics_txt>

    1. For the search topics, a list of urls is generated then each iem is requested 
    via the pubmed api. This generates a list of <max_articles> pubmed IDs.
    2. Then, each ID is used with indra.literature.pubmed_client module to get abstracts
    The results are saved in one scv by topic and one csv with the merged topics.
    '''

    max_articles = int(argv[1])
    path_to_topics_txt = argv[2]

    # Read and list topics
    f = open(path_to_topics_txt, 'r')
    topics = f.read().split(",")
    f.close()

    # List corresponding urls
    urls = []
    for topic in topics:
        sleep(np.random.random()+1)
        try:
            # you may add "&apikey=<your-pubmed-api-key>" to the request
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={topic.replace(' ','+')}&retmax={max_articles}"
            urls.append(url)
        except:
            print(f"ERROR: Could not process {topic}")
    df_list = []
    # 1. Each url is associate with a topic
    for url in urls:
        topic = re.findall(r"&term=(.+)&retmax", url)[0]
        # Request and list pubmed Ids
        xml = requests.get(url).content
        sleep(np.random.random()*0.4+0.01)
        root = ET.fromstring(xml)
        id_list = []
        for i in range(max_articles):
            id_list.append(root[3][i].text)
            
        # 2. Get abstract for each pubmed_id
        record = []
        count = 0
        for pubmed_id in id_list:
            sleep(np.random.random()*0.4+0.01)
            tmp = []
            titled_abstract = None
            abstract = None
            try:
                abstract = get_abstract(pubmed_id, prepend_title=True)
                if abstract:
                    tmp.append(pubmed_id)
                    tmp.append(abstract)
                    tmp.append(topic.replace("+", " "))
                    record.append(tmp)
                    count = count + 1
            except:
                pass
            
        print(f"{count} abstracts retrieved for {url}. Saving to csv.")
        
        # Create and save dataframe to csv   
        cols = ["pubmed_id", "abstract", "topic"]
        df = pd.DataFrame.from_records(record, columns=cols, index=None)
        df_list.append(df)
        df.to_csv(f"{topic}_abstracts.csv", index=False)

    merged = pd.concat(df_list, axis=0, ignore_index=True)
    merged.to_csv("merged.csv", index=False)
if __name__== "__main__" :
    get_abstracts()