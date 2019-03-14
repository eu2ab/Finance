# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import codecs, json
import soundfile as sf
import librosa
import os
import sox

from google.protobuf.json_format import MessageToDict
from google.cloud import storage, speech
from google.cloud.speech import enums, types

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.downloader.download('punkt')  # obtain resource 'punkt'
nltk.downloader.download('stopwords')  # obtain resource 'stopwords'
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

nltk.download('wordnet')

from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering


def SpectralClustering(data, row, col, row_calc='sum', cluster_type='bi', n_clusters='auto'):
    """
    Change cluster_type to switch between bi-clustering and co-clustering

    :param data: Dataset
    :param row: Chosen row of checkerboard structure
    :param col: Chosen column of checkerboard structure
    :param cluster_type: Choose between 'bi' or 'co'
    :param row_calc: Choose how rows are treated - 'sum' or 'ave' (average)
    :param n_clusters: Can either be one integer or a pair (default is auto and calculated based on unique number of
    row and column elements)
    :return: Checkerboard structures and model

    spectralClustering(df, 'AccountName', 'MarketOfferingSolution')
    """
    df = data.loc[:, [row, col]].astype(str)  # Filtering to only row and col
    row_entries, I = np.unique(df.iloc[:, 0], return_inverse=True)
    col_entries, J = np.unique(df.iloc[:, 1], return_inverse=True)
    points = np.ones(len(df), int)
    mat = sp.sparse.csr_matrix((point, (I, J)))  # Build out sparse CSR Matrix
    if n_clusters == 'auto':
        n = (len(df[row].unique()), len(df[col].unique()))
    else:
        n = n_clusters

    if cluster_type == 'bi':
        model = SpectralBiclustering(n_clusters=n, random_state=0)
    else:
        model = SpectralCoclustering(n_clusters=n, random_state=0)

    model.fit(mat)  # Fitting model to data
    model.row_labels_ = model.row_labels_ + 1  # Set first levels to 1 instead of 0
    model.column_labels_ = model.column_labels_ + 1  # Set first levels to 1 instead of 0

    dense = mat.todense()  # Converting sparse matrix back to a dense matrix
    dense = pd.DataFrame(dense)  # Set as a pandas dataframe

    if row_calc == 'sum':
        dense[row] = model.row_labels_  # Compute sums by row and column clusters
        dense_ROWSUM = dense.groupby([row]).sum()
        dense_ROWSUM_T = dense_ROWSUM.transpose().copy()
        dense_ROWSUM_T[col] = model.column_labels_
        dense_SUM = dense_ROWSUM_T.groupby([col]).sum()

        # Plotting checkerboard structure
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(dense_SUM.transpose(), linewidths=0.5)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.title("Checkerboard Structure of Clustered Data", fontweight='bold', y=1.08)
        fig.show()
    else:
        dense[row] = model.row_labels_
        dense_ROWMEAN = dense.groupby([row]).mean()
        dense_ROWMEAN_T = dense_ROWMEAN.transpose().copy()
        dense_ROWMEAN_T[col] = model.column_labels_
        dense_MEAN = dense_ROWMEAN_T.groupby([col]).mean()

        # Plotting checkerboard structure
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(dense_MEAN.transpose(), linewidths=0.5)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.title("Checkerboard Structure of Clustered Data", fontweight='bold', y=1.08)
        fig.show()

    clust_df = pd.concat([pd.DataFrame(df.loc[:, [col]].astype(str)),
                          pd.DataFrame(model.row_lables_)], axis=1)
    clust_df.columns = [col, row]
    count_df = clust_df.groupby([col, row]).size().unstack().fillna()

    # Plotting Checkerboard Structure
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(count_df, linewidths=0.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    fig.show()

    return (model)
