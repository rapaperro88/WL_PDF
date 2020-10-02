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

