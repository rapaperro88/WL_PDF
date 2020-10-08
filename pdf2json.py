from custom_functions import *
from sys import argv
from json import loads, dump, dumps

def main():
    '''
    usage: python pdf2json.py <path-to-folder-containing-pdfs>
    '''
    path = argv[1]
    # get metadata from pdf
    missing_data, encrypted_files, df = get_pdf_data(path)
    # complete data from crossref
    df = get_crossref_metadata(df)
    # save to csv
    df.to_csv(path + "/" + "output.csv", index=False)
    # Transform to JSON
    json_string = df.to_json(orient='records')
    json_parsed = loads(json_string)
    with open(path + "/" + "output.json", 'w') as outfile:
        dump(json_parsed, outfile) 
        outfile.close()

if __name__== "__main__" :
    main()
