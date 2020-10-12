import streamlit as st
from custom_functions import *

# Sidebar==clustering:
    # enter csv 
    # display clusters

st.title('Articles')

#############################
########## SIDEBAR ##########
#############################
st.sidebar.title('What task would you like to perform?')

st.sidebar.markdown('''
The two proposed options are:
\n 1. **Cluster: ** enter some abstracts in csv format and visualize their \
    clustering by semantic similarities.
\n 2. **Similarity Search: ** enter a text and get some suggestions on similar content.
''')

task = st.sidebar.selectbox(
    'Select your task',
    ('1. Cluster', '2. Similarity Search')
    )

#############################
########## CLUSTER ##########
#############################
if task=="1. Cluster":
    st.header('Cluster')
    st.markdown('''
    In this section you will load a structured csv. Each abstract is a line on a csv file.  
    The corresponding labels can be also inputed.
    ''')

    # File selector
    csv = file_selector(folder_path='./model', 
        text = "Select a csv file containing all your abstracts:", 
        file_ext=".csv")

    # Read dataset
    df = pd.read_csv(csv)
    cols = list(df.columns)
    st.dataframe(df.head(4))

    # Columns selection
    abstracts = st.selectbox("Select abstract column", cols)
    if st.radio("The data is labeled: ", ["No", "Yes"]) == "Yes":
        labels = st.selectbox("Select labels column", cols)
    else:
        labels = []

    # Run clustering
    img, clusters = cluster_abstracts(df, abstracts, labels)

    # Display image

    # Download clusters as csv

#############################
######## SIMILARITY #########
#############################
if task=="2. Similarity Search":
    st.header('Similarity Search')

    st.markdown('''
    In this section you will input a text or a paragraph. 
    You will receive a suggestion of articles that are similar to this text.
    ''')

    text = st.text_area("Enter text here: ")

    # Run similarity test
    output = similarity_test(text)

    

