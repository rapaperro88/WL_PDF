import streamlit as st
from gensim.models import LdaModel
from gensim.corpora.mmcorpus import MmCorpus
from gensim.test.utils import datapath
from custom_functions import *
from gensim.corpora import Dictionary

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
    csv = file_selector(folder_path='./data_mining', 
        text = "Select a csv file containing all your abstracts:", 
        file_ext=".csv")

    if csv != None:
        # Read dataset
        df = pd.read_csv(csv)
        cols = list(df.columns)
        st.dataframe(df.head(4))

        if cols != None:
            # Columns selection
            abstracts = st.selectbox("Select abstract column", cols)
            labels = st.selectbox("Select labels column", cols)

            if st.button("Run clustering"):
                # Run clustering
                fig, clusters = cluster_abstracts(df, abstracts, labels)

                # Display image
                st.pyplot(fig=fig)

                # Download clusters as csv
                if st.button("Save clusters to txt"):
                    np.savetxt('clusters.txt', clusters, delimiter=" ",  fmt="%s")


#############################
######## SIMILARITY #########
#############################

elif task=="2. Similarity Search":
    st.header('Similarity Search')

    st.markdown('''
    In this section you will input a text or a paragraph. 
    You will receive a suggestion of articles that are similar to this text.
    ''')

    # -------- TRAINING --------
    st.subheader("Training")

    # File selector
    csv = file_selector(folder_path='./data_mining', 
        text = "Select a csv file containing all your abstracts:", 
        file_ext=".csv")

    # Read dataset
    df = pd.read_csv(csv)
    cols = list(df.columns)
    st.dataframe(df.head(4))

    # Columns selection + number of topics input 
    abstracts = st.selectbox("Select abstract column", cols)
    labels = st.selectbox("Select labels column", cols)
    num_topics = st.slider('How many topics do you have in your data ?', 5, 60, step=1, value=8)

    if st.button("Train LDA model"):
        dictionary, corpus, lda = similarity_train(df, num_topics, abstracts, labels)

        # Save dictionary, corpus and lda model        
        lda.save('model/lda.model')
        MmCorpus.save_corpus("model/corpus.mm", corpus)
        dictionary.save_as_text("model/dictionary.txt")

        st.success("Model trained and saved successfully.")

        for n in range(num_topics):            
            st.text(f"topic {n}: {[t[0] for t in lda.show_topic(topicid=n, topn=5)]}") # list of tuples
            


    # -------- TEST --------
    st.subheader("Test")

    text = st.text_area("Enter text here: ")

    if st.button("Suggest Articles"):
        
        # Load pretrained elements
        corpus = MmCorpus('model/corpus.mm')
        lda = LdaModel.load('model/lda.model')
        dictionary = Dictionary.load_from_text("model/dictionary.txt")

        # Run similarity test
        output = similarity_test(df, text, lda, corpus, dictionary,k=5)

        # Print
        st.success("Suggested Articles:")
        for title in output[:5]:
            st.text(title)