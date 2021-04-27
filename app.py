import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import nltk
from dateutil.parser import parse
import streamlit.components.v1 as components
from datetime import datetime
from datetime import date

@st.cache
def load_data():

    wdf = pd.read_csv('data/working.csv')



    def manage_data(df, e = 'es'):

        if e == "es":

            remove_words = [ 'et', 'al', 'et al', 'etal', 'et al .', 'ecology', 'society', 'ecology and society',
                            'ecologyandsociety', 'http', 'www', 'org','wa', 'has','wa ', ' wa', 'fig', 'table', 'was','the',
                            'to', 'of', 'on','for', 'in', 'thus', 'although']

        elif e == "housing":

            remove_words = [ 'bay', 'brevard', 'broward',
                            'charlotte', 'citrus', 'collier', 'dixie', 'escambia',
                             'flagler', 'franklin', 'gulf', 'hernando', 'hillsborough', 'indian river', 'jefferson',
                             'lee', 'levy', 'manatee', 'martin', 'miami-dade', 'monroe', 'nassau', 'okaloosa',
                             'palm beach', 'pasco', 'pinellas', 'st. johns', 'st. lucie', 'santa rosa', 'sarasota',
                             'taylor', 'volusia', 'wakulla', 'walton',
                             'county', 'policy',  'shall',
                            'housing', 'will', ]



        df['text3'] = df['text3'].str.lower()

        pat = '|'.join([r'\b{}\b'.format(w) for w in remove_words])
        #pat2 = '|'.join([r'\b{}\b'.format(w) for w in stop_words])

        df['text4'] = df['text3'].str.replace(pat, '')



        #df['text5'] = df['text4'].apply(lambda x: ' '.join([str(i) for i in x]))


        return df


    wdf = wdf.dropna(subset=['text3'])
    wdf = manage_data(wdf, e='es')
    return wdf

wdf = load_data()




st.sidebar.title("CSP Project")
page = st.sidebar.radio(
     "Pick an option",
     ('Home' , 'Document', 'Word Cloud', 'LDA'),
     )

if page == "Home":
    st.header("Overview")

    st.markdown("This page will serve as an overview of all of the analysis that the other pages consist of.")
    st.markdown("-------")

    st.header("For more information select a particular topic:")
    extra = st.selectbox("Topic", ("How do you clean speeches?", "What can we change in the LDA?", "How does the dynamic LDA work?"))

    if extra =="How do you clean speeches?":
        st.subheader("Cleaning speeches is a key part in the process.")
        st.markdown("Without cleaning up the data, words and symbols that are not informative might emerge like _'.'_ or _'and'_. Primary steps in no particular order include:")
        st.markdown("* Remove stop words")
        st.markdown("* Tag parts of speech and remove non-nouns")
        st.markdown("* Lemmatize words")
        st.markdown("* Remove symbols and punctuation")

    elif extra == "What can we change in the LDA?":
        st.subheader("A few options are given to be changed in the overall LDA")
        st.markdown("* Number of topics")
        st.markdown("This parameter allows the user to adjust the number of topics the unsupervised machine learning fits the data to. More topics can mean more specific words, but at some point the model may underperform.")
        st.markdown("* Bigram Threshold")
        st.markdown("This parameter changes the number of times two words need to appear together for them to form a bigram, for instance _'United States'_ . Lower numbers indicate fewer times.  ")
        st.markdown("* Date")
        st.markdown("The two inputs correspond to the given date range you wish to constrain a topic model to.")
        st.markdown("* Coherence Score")
        st.markdown("This is a measure of how well the LDA model fits the underlying data. Lower scores here are better. It is the Intrinsic UMass Measure.")

    elif extra == "How does the dynamic LDA work?":
        st.subheader("Dynamic Topic Model Splits the Data")

elif page == "Document":

    st.header("This page will show document level information about text")
    jour = st.selectbox("Journal", list(wdf['Source title'].unique())

         )
    paper = st.selectbox("Paper", list(wdf.loc[wdf['Source title']==jour]['Title']))

    st.subheader("Meta Data")
    auth = wdf.loc[wdf['Title']==paper]['Authors'].values
    year = wdf.loc[wdf['Title']==paper]['Year'].values
    file = wdf.loc[wdf['Title']==paper]['filename'].values

    st.markdown("Authors: " + auth)
    st.markdown("Year: " + str(year))
    st.markdown("Filename: " + file)

    st.markdown("")
    st.markdown("")
    st.markdown("___________________")
    st.markdown("")
    st.markdown("")
    st.markdown("Below see the difference between initial OCR'd Text (full), and after it is cut to limit references and abstracts where possible (cut)")

    te = st.radio(
         "Pick text",
         ('Full text', 'Cut Text'),
         )

    if te== "Full text":
        t = wdf.loc[wdf['Title']==paper]['text'].values
    else:
        t = wdf.loc[wdf['Title']==paper]['text3'].values

    st.write(t[0])



elif page== "Word Cloud":

    st.subheader("Word Cloud Page")

    word = st.radio(
         "Pick a subset",
         ('All', 'Journal', 'Document'),
         )

    if word== "All":



        long_string = ','.join(list(wdf['text4'].values))
        save = 'full'

    elif word == "Journal":

        jour = st.selectbox("Journal", list(wdf['Source title'].unique())

             )

        long_string = ','.join(list(wdf.loc[wdf['Source title']==jour]['text4'].values))

        save = jour
    else:

        jour = st.selectbox("Journal", list(wdf['Source title'].unique())

             )
        paper = st.selectbox("Paper", list(wdf.loc[wdf['Source title']==jour]['Title']))

        long_string = ','.join(list(wdf.loc[wdf['Title']==paper]['text4'].values))

        save = paper


    wordcloud = WordCloud(background_color="black",width=800, height=400, max_words=5000,
                      contour_width=3, contour_color='firebrick')

    word = wordcloud.generate(long_string)

    fig, ax = plt.subplots(figsize=(16,10), facecolor='k')
    plt.imshow(word, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(save + '.png',)
    st.image(save +'.png', width=800)




elif page== "LDA":

    st.subheader("LDA Page")

    sub = st.radio(
         "Pick a subset",
         ('All', 'Journal', 'Year'),
         )

    if sub== "All":
        df = wdf
    elif sub == "Journal":

        jour = st.selectbox("Journal", list(wdf['Source title'].unique())
        )
        df = wdf.loc[wdf['Source title']==jour]
    elif sub =="Year":
        year = st.selectbox("Year", list(wdf['Year'].unique()))

        df = wdf.loc[wdf['Year']==year]


    col1, col2 = st.beta_columns(2)
    with col1:

        numtopics = st.number_input('Number topics', min_value=float(3.0), max_value=float(35.0), value=float(7.0), step=float(1))

        bigram = st.number_input('Bigram Throlshold', min_value=float(1.0), max_value=float(100.0), value=float(50.0), step=float(1))

    #first = dt.datetime.strptime(([datetime.date(2019, 7, 6)]))
    #st.write(first)
    with col2:
        st.write("Be aware the full model takes a little while. Year specific takes the least time generally.")
        #s = st.date_input( "When should we start the LDA", date(2020, 3, 1))

        #e = st.date_input( "When should we end the LDA", date(2020, 11, 3))




    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
                result.append(token)
        return result


    def run_lda(df, savename, numtop=numtopics, thresh=bigram):


        doc_processed = df['text4'].map(preprocess)
        dictionary = corpora.Dictionary(doc_processed)
        #doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_processed]

        bigram = gensim.models.Phrases(list(doc_processed), min_count=5, threshold=thresh)
        bigram_mod = gensim.models.phrases.Phraser(bigram)


        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        data_words_bigrams = make_bigrams(doc_processed)
        id2word = corpora.Dictionary(data_words_bigrams)

        texts = data_words_bigrams

        corpus = [id2word.doc2bow(text) for text in texts]


        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=numtop,
                                               random_state=100,
                                               update_every=1,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)


        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        save = savename + ".html"
        pyLDAvis.save_html(vis, save)
        return lda_model, doc_processed, dictionary


    save = 'temp.html'

    lda_model, doc_processed, dictionary = run_lda(df, "temp", numtop=numtopics, thresh=bigram)


    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_processed, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    st.write("The number of topics is: " + str(numtopics))
    st.write('\nCoherence Score (Lower is better): ', coherence_lda)

    HtmlFile = open('temp.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, width=1200, height=1200)
