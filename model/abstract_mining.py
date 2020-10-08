import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re
from indra.literature.pubmed_client import get_abstract
from sys import argv
import pandas as pd

def get_abstracts ():
    '''
    gets a list of pubmed ids for each topic in a .txt file. max_articles is the number of
    abstracts to retrieve.
    usage: python abstract_mining.py <max_articles> <path_to_topics_txt>
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
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={topic.replace(' ','+')}&retmax={max_articles}"
            urls.append(url)
        except:
            print(f"ERROR: Could not process {topic}")

    # each url is associate with a topic
    for url in urls:
        # Request and list pubmed Ids
        xml = requests.get(url).content
        root = ET.fromstring(xml)
        id_list = []
        for i in range(max_articles):
            id_list.append(root[3][i].text)
            
        # Get title + abstract for each pubmed_id
        record = []
        count = 0
        for pubmed_id in id_list:
            tmp = []
            titled_abstract = None
            abstract = None
            try:
                titled_abstract = get_abstract(pubmed_id, prepend_title=True)
                title = re.split(r'\. ', titled_abstract)[0]
                abstract = re.split(r'\. ', titled_abstract)[1:][0]
                if titled_abstract and abstract:
                    tmp.append(pubmed_id)
                    tmp.append(title)
                    tmp.append(abstract)
                    record.append(tmp)
                    count = count + 1
            except:
                pass
            
        print(f"{count} abstracts retrieved for {url}. Saving to csv.")
        
        # Create and save dataframe to csv   
        cols = ["pubmed_id", "titre", "abstract"]
        df = pd.DataFrame.from_records(record, columns=cols, index=None)
        topic = re.findall(r"&term=(.+)&retmax", url)[0]
        df.to_csv(f"{topic}_abstracts.csv", index=False)

if __name__== "__main__" :
    get_abstracts()