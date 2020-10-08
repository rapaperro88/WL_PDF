# Whitelab pdf project

This project aims to create a neo4j database of genomics peer-reviewed scientific articles. This database could later be used in text similarity search given a certain text or other pdf article.

## Workflow

In a python based routine : 

1. A structure table (pandas, csv) is set-up by :
   1. extracting the metadata from a collection of pdf files.
   2. The csv will then becompleted using API requested information
2. This can easily be transformed into a neo4j database
3. A NLP model can be implemented to achieve text similarity selection
4. This can be packaged into a user_friendly API (flask)

## Installation

### Prerequisites

* INDRA for pubmed abstract mining : https://indra.readthedocs.io/en/latest/installation.html

### On your local machine

1. Clone the repository (`git clone https://github.com/rapaperro88/WL_PDF.git`)
2. Install the requirements (`pip install -r requirements.txt`)
3. Run pdf_app.py (`python pdf_app.py`)

### From a docker image

1. 

##