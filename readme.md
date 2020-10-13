# Whitelab pdf project

This project aims to create a neo4j database of genomics peer-reviewed scientific articles. This database could later be used in text similarity search given a certain text or other pdf article.

This project have some the following functionalities :

1. PDF Mining
2. Abstract Mining
3. Abstract Clustering
4. Similarity Suggestion 

# Abstract Mining

1. Clone the repository (`git clone https://github.com/rapaperro88/WL_PDF.git`)

2. Install INDRA for pubmed abstract mining : https://indra.readthedocs.io/en/latest/installation.html

3. Install requirements.txt (example: `pip install -r requirements.txt`)

4. Run **data_mining/abstract_mining.py** file :

   usage: `python abstract_mining.py <max_articles> <path_to_topics_txt>` 

   * `<max_articles>` : int, max number of articles to download per topic
   * `<path_to_topics_txt>`:  relative path to file containing a list of topics (comma separated without a space)



# PDF Mining

If not done :

1. Clone the repository (`git clone https://github.com/rapaperro88/WL_PDF.git`)
2. Install requirements.txt (example: `pip install -r requirements.txt`)

Then :

3. Run **pdf2json.py** file :

   usage: `python pdf2json.py <path-to-folder-containing-pdfs>` 

   * `<path_to_topics_txt>`:  relative path to folder containing pdf files (scientific articles).



# App.py

If not done :

1. Clone the repository (`git clone https://github.com/rapaperro88/WL_PDF.git`)
2. Install requirements.txt (example: `pip install -r requirements.txt`)

Then :

3. Run **app.py** streamlit application (exemple: `streamlit run app.py`)
4. The side menu proposes two functionalities :
   1. **Abstract Clustering**
   2. **Similarity Suggestion** 



## Abstract Clustering

1. Make sure you have retrieved abstracts on topics you are interested in by refering yourself to abstract mining section place your structured csv file in the ./data_mining folder. Alternatively you can use **data_mining/sample.csv**.
2.  In the streamlit interface, select the columns you are interested in for clustering.
3. Run the clustering algorithm.



## Similarity Suggestion

1. Enter a text you want to find similarities for.
2. Display the main keywords and title suggestions.