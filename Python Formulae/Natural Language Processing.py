# Import packages
import pandas as pd
import numpy as np

import gensim
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
import nltk
from nltk.stem.snowball import SnowballStemmer

nltk.downloader.download('punkt')  # obtain resource 'punkt'
nltk.downloader.download('stopwords')  # obtain resource 'stopwords'

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize, MinMaxScaler


def lemmatize_stemming(text):
    """
    Utility function to lemmatize and stem words
    """
    stemmer = SnowballStemmer("english")
    return (stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v')))


def tokenize_lemmatize(text):
    """
    Tokenize lemmatized and stemmed words
    """
    result = []
    for token in gensim.utils.simple_preprocess(text):
        # Tokenize only words that are not stopwords or have a length of greater than 3 characters
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return (result)


def bagofWords(df, colname, keep_num):
    """
    Creates a bag of words based on dataframe and column chosen.

    :param df:  Dataframe to be ingested as part of Bag of Words creation
    :param colname: Column name chosen for Bag of Words creation
    :param keep_num: Top frequent tokens to keep
    :return: dictionary, bow_corpus
    """

    # Creates dictionary containing the number of times a word appears
    preprocessed_docs = df[colname].tolist()
    dictionary = gensim.corpora.Dictionary(preprocessed_docs)

    # Keep only the top keep_n frequent tokens
    dictionary.filter_extremes(keep_n=keep_num)

    # Scale Bag of Words for each ticket with each ticket getting a dictionary reporting
    # how many words and how many times that word appears
    bow_corpus = [dictionary.doc2bow(ticket) for ticket in preprocessed_docs]

    return (dictionary, bow_corpus)


def printTopic(lda_model, topic_num, num_of_topics, output=True):
    """
    Functiont to neatly print the words of each LDA topic

    :param lda_model: LDA model
    :param topic_num: Topic number to be passed through iterator
    :param num_of_topics: Number of topics in LDA model
    :param output:
    :return: Prints a formatted list of num_of_topic terms
    """

    terms = []
    for term, frequency in lda_model.show_topic(topic_num, topn=num_of_topics):
        terms += [term]
        if output:
            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))

    return (terms)


def buildNMF(df, colname, num_topics):
    """
    Creates a non-negative matrix factorization model

    :param df: Dataframe to be ingested as part of NMF creation
    :param colname: Column name chosen for NMF model creation
    :param num_topics: Number of topics to build NMF on
    :return: NMF model
    """

    # Count vectorizer needs string inputs
    preprocessed_docs = df[colname].tolist()

    # Convert list to strings for Count Vectorizer
    docs_str = [' '.join(text) for text in preprocessed_docs]

    # Takes top 5000 best features to contribute to the model
    vectorizer = CountVectorizer(analyzer='word', max_features=5000)
    x_counts = vectorizer.fit_transform(docs_str)

    # Set and use TfidTransformer
    transformer = TfidfTransformer(smooth_idf=False)
    x_tfidf = transformer.fit_transform(x_counts)

    # Normalize tfidf values using unit length of each row
    xtfidf_norm = normalize(x_tfidf, axis=1)

    # Obtain NMF Model
    model = NMF(n_components=num_topics, random_state=0).fit(xtfidf_norm)
    nmf = model.transform(xtfidf_norm)

    return (nmf)


def nmfTopics(nmf, num_topics):
    """
    Use this function to iterate over each topic and obtain the most important scoring words in each cluster and add
    them to a dataframe, df.

    :param nmf: NMF model
    :param num_topics:  Number of topics to build NMF model on
    :return: Pandas dataframe with topics as column headers and words as rows
    """

    # Word IDs need to be reverse-mapped to the words so that we can print the topic names
    feat_names = CountVectorizer.get_feature_names()
    word_dict = {}
    for i in range(num_topics):
        # For each topic, obtain the largest values and add the words they map into the dictionary
        words_ids = nmf.components_[i].argsort()[:, -10 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i + 1)] = words

    return (pd.DataFrame(word_dict))


def runLDA(df, colname, num_topics=5, num_words=25):
    """
    Build LDA model

    :param df: Dataframe of preprocessed text (lemmatize, stemmed, tokenized, etc.)
    :param colname: Column which contains the text
    :param num_topics: Number of topics desired in LDA model
    :param num_words: Number of words chosen per topic
    :return: 3 things: 1) Dataframe containg LDA model and words, 2) Dataframe containing original data plus the LDA vector, topic,
    and correlation figures, and the 3) LDA model
    """

    # Build dictionary and Bag of Words based on text within chosen column
    df_dict, df_bow_corpus = bagofWords(df, colname=colname, keep_num=10000)

    # Run LDA model on Bag of Words
    lda = gensim.models.LdaModel(corpus=df_bow_corpus, num_topics=num_topics, id2word=df_dict, passes=5, alpha='auto',
                                 eta='auto', iterations=300, random_state=0)

    # For each topic within the main corpus LDA, print the words occurring in that topic along with its relative weight
    # and build pandas dataframe
    df_topic = []
    df_words = []
    for idx, topic in lda.print_topics(-1, num_words=num_words):
        df_topic.append(idx)
        df_words.append(topic)
    lda_df = pd.DataFrame({'Topic Number:': df_topic,
                           'Topic Words:': df_words})

    # Append most likely topic suggestion based on inmate messages to the main df
    lda_corpus = lda[df_bow_corpus]
    lda_docs = [doc for doc in lda_corpus]

    # Pull of LDA vector for each document and append to Dataframe
    length = len(lda_corpus)
    lda_vec = np.zeros((length, num_topics))
    for i in range(length - 1):
        for topic in range(num_topics - 1):
            for tmp in range(len(lda_docs[i])):
                lda_vec[i][lda_docs[i][tmp[0]]] = lda_docs[i][tmp][1]
    df['LDA_vector'] = lda_vec.tolist()

    # Extract closest LDA topic by document
    max_topic = []
    for topic in lda_docs:
        max_topic.append(max(topic, key=lambda x: x[1]))


# Add max_topic and percentage to dataframe
df['lda_topic_corr'] = max_topic
df['lda_topic'] = df.lda_topic_corr.str[0]
df['lda_topic_corr'] = df.lda_topic_corr.str[1]

# Add a counts column to the Pandas Dataframe for just the LDA results
top_counts = pd.DataFrame({'Topic Count:': df.lda_topic.value_counts()})
lda_df = lda_df.merge(top_counts, left_on='Topic Number', right_index=True)

return (lda_df, df, lda)
