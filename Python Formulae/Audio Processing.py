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


def audioFeatures(y, sr, frame_length=2048, hop_length=512):
    """
    Extract audio features from an audio file (167 total)

    :param y: A 1-D (mono) numpy array of the audio file
    :param sr: Sampling rate of audio file
    :param frame_length: Size of the examined window in samples
    :param hop_length: Number of samples skipped for next window
    :return: A numpy array containing features

    DEMO
    file = os.path.join(os.getcwd(), 'filename.wav')
    y, sr = librosa.load(path=file, sr=8000)
    k = audioFeatures(y, sr=sr)
    print(k)
    """

    # If Y is empty, return just zeros
    if len(y) == 0:
        output = np.zeros(152, dtype=float)
        return (output)

    # Get length of audio file
    leng = np.asfarray(len(y) / sr).reshape((1))

    # Calculate MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10, hop_length=hop_length)

    # Calculate MFCC aggregate statistics (over time)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_sd = np.std(mfcc, axis=1)
    mfcc_meds = np.median(mfcc, axis=1)
    mfcc_skews = sp.stats.skew(mfcc, axis=1)
    mfcc_kurts = sp.stats.kurtosis(mfcc, axis=1)

    # Short-time fourier transform
    stft = np.abs(librosa.stft(y))

    # Chromagram from a waveform or power spectrum; takes ~2.5% of file length to complete
    chroma_m = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    chroma_sd = np.std(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    chroma_skew = sp.stats.skew(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    chroma_kurt = sp.stats.kurtosis(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    chroma_med = np.median(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    # Calculate spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)

    # Calculate spectral centroid aggregate statistics
    cent_means = np.mean(cent, axis=1)
    cent_sd = np.std(cent, axis=1)
    cent_meds = np.median(cent, axis=1)
    cent_skews = sp.stats.skew(cent, axis=1)
    cent_kurts = sp.stats.kurtosis(cent, axis=1)

    # Calculate spectral bandwidth
    band = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, p=2)

    # Calculate spectral bandwidth aggregate statistics (over time)
    band_means = np.mean(band, axis=1)
    band_sd = np.std(band, axis=1)
    band_meds = np.median(band, axis=1)
    band_skews = sp.stats.skew(band, axis=1)
    band_kurts = sp.stats.kurtosis(band, axis=1)

    # Calculate spectract contrast
    cont = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, fmin=200.0,
                                             n_bands=3, quantile=0.02)

    # Calculate spectral contrast aggregate statistics
    cont_means = np.mean(cont, axis=1)
    cont_sd = np.std(cont, axis=1)
    cont_meds = np.median(cont, axis=1)
    cont_skews = sp.stats.skew(cont, axis=1)
    cont_kurts = sp.stats.kurtosis(cont, axis=1)

    # Calculate spectral flatness
    flat = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length, amin=1e-10, power=2.0)

    # Calculate spectral flatness aggregate statistics
    flat_means = np.mean(flat, axis=1)
    flat_sd = np.std(flat, axis=1)
    flat_meds = np.median(flat, axis=1)
    flat_skews = sp.stats.skew(flat, axis=1)
    flat_kurts = sp.stats.kurtosis(flat, axis=1)

    # Calculate spectral rolloff
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, roll_percent=0.85)

    # Calculate spectral rolloff aggregate statistics
    roll_means = np.mean(roll, axis=1)
    roll_sd = np.std(roll, axis=1)
    roll_meds = np.median(roll, axis=1)
    roll_skews = sp.stats.skew(roll, axis=1)
    roll_kurts = sp.stats.kurtosis(roll, axis=1)

    # Calculate zero crossing rate
    zero = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)

    # Calculate zero crossing rate aggregate statistics
    zero_means = np.mean(zero, axis=1)
    zero_sd = np.std(zero, axis=1)
    zero_meds = np.median(zero, axis=1)
    zero_skews = sp.stats.skew(zero, axis=1)
    zero_kurts = sp.stats.kurtosis(zero, axis=1)

    # Calculate absolute magnitude
    abs_mag = np.abs(y)

    # Calculate absolute magnitude aggregate statistics
    abs_mean = np.mean(abs_mag).reshape((1))
    abs_sd = np.std(abs_mag).reshape((1))
    abs_meds = np.median(abs_mag).reshape((1))
    abs_skews = np.asfarray(sp.stats.skew(abs_mag)).reshape((1))
    abs_kurts = np.asfarray(sp.stats.kurtosis(abs_mag)).reshape((1))
    abs_coefv = (abs_sd / abs_mean).reshape((1))

    # Calculate root mean square energy
    rmse = librosa.feature.rmse(y=y, frame_length=frame_length, hop_length=hop_length)

    # Calculate root mean square energy aggregate statistics
    rmse_m = np.mean(rmse, axis=1)
    rmse_sd = np.std(rmse, axis=1)
    rmse_med = np.median(rmse, axis=1)
    rmse_skew = sp.stats.skew(rmse, axis=1)
    rmse_kurt = sp.stats.kurtosis(rmse, axis=1)

    df = np.concatenate((leng, mfcc_means, mfcc_sd, mfcc_meds, mfcc_skews, mfcc_kurts,
                         chroma_m, chroma_sd, chroma_med, chroma_skew, chroma_kurt,
                         cent_means, cent_sd, cent_meds, cent_skews, cent_kurts,
                         band_means, band_sd, band_meds, band_skews, band_kurts,
                         cont_means, cont_sd, cont_meds, cont_skews, cont_kurts,
                         flat_means, flat_sd, flat_meds, flat_skews, flat_kurts,
                         roll_means, roll_sd, roll_meds, roll_skews, roll_kurts,
                         zero_means, zero_sd, zero_meds, zero_skews, zero_kurts,
                         abs_mean, abs_sd, abs_meds, abs_skews, abs_kurts,
                         rmse_m, rmse_sd, rmse_med, rmse_skew, rmse_kurt, abs_coefv))

    return (df)


def audioViterbiSegment(y):
    """
    We can use RMSE and Viterbi to split each sound clip represented by a 1-D array into silence and signal.

    :param y:  A 1-D numpy array of the audio file (represents one channel)
    :return: 2-D array with timestamps on top and silence/signal on bottom
    """

    # Calculate RMSE and get timestamps
    rmse = librosa.feature.rmse(y=y)[0]
    times = librosa.frames_to_time(np.arrange(len(rmse)))

    # Normalize RMSE by sigma to expand range of probability vector to calculate silence v. signal
    r_normalized = (rmse - 0.01) / (no.std(rmse))
    p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

    # Viterbi algorithm to improve threshold bifuraction
    transition = librosa.sequence.transition_loop(2, [0.5, 0.6])
    full_p = np.vstack([1 - p, p])
    states = librosa.sequence.viterbi_discriminative(full_p, transition)

    return (states, times)


def audioNormalSegments(y, smoothing=30, p_thresh=0.5):
    """
    Represents a more naive approach than the Viterbi algorithm. Often best used if there is an issue due to the
    threshold of Viterbi being too low and prematurely splitting audio files

    :param y: A 1-D numpy array of the audio file (represents one channel)
    :param smoothing: Smoothing parameter to smooth out the spectrogram and reduce noise when building threshold at
     which to accept whether a given moment represents silence or signal
    :param p_thresh: Threshold at which a moment is categorized as either silence or signal (ranges from 0 to 1)
    :return: A 2-D array with timestamps on top and silence or signal on bottom
    """

    # Calculate RMSE and get timestamps
    rmse = librosa.feature.rmse(y=y)[0]
    times = librosa.frames_to_time(np.arrange(len(rmse)))

    # Normalize RMSE by sigma to expand range of probability vector to calculate silence v. signal
    r_normalized = (rmse - 0.01) / (np.std(rmse))
    p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

    # Smooth window of probabilities to remove spikes
    p_smoothed = np.convolve(p, np.ones((smoothing,)) / smoothing, mode='valid')

    # Convert p_smoothed to binary based on threshold
    binary = np.where(p_smoothed > p_thresh, 1, 0)

    return (binary, times)


def audioSOXCleaner(in_file, out_file):
    """
    Programmatically convert raw wave files to 8000 Hz, reduce bass, and increase treble (for voice clarity - best
    used for phone calls). SOX needs to be installed and placed in system PATH for this equation to be functional

    :param in_file: Full path of input file
    :param out_file: Full path of output file
    :return: Nothing. File is written to chosen path.
    """

    # Create the transformer
    tfm = sox.Transformer()

    # Build out characteristics
    tfm.bass(-15.)  # Reduce bass by 15 Hz
    tfm.treble(15.)  # Increase treble by 15 Hz
    tfm.contrast(amount=80)  # Modifies an audio signal to make it louder
    tfm.rate(samplerate=8000, quality='h')  # Convert input file to 8000 Hz
    tfm.build(in_file, out_file)  # Perform operations and write file out to location given


def audioWAVSplitter(file):
    """
    Function to read a stereo WAV file and return two split channels

    :param file: Full locaiton of file
    :return: Numpy array of shape (length, 2)

    DEMO
    file = os.path.join(os.getcwd(), 'filename.wav')
    left, right = audioWAVSplitter(file)
    """

    # Open a raw WAV file (as 16-bit integer)
    with open(file, 'rb') as f:
        data, samplerate = sf.read(f, dtype='float32')

    return (data, samplerate)


def audioTimestamps(state, time):
    """
    Function to return the timestamps (used in segmentation function)

    :param state: Binary numpy array (result of segmentation function)
    :param time: Numpy array containing time (in seconds) corresponding to state
    :return: List containing two lists (starting and ending timestamps for signals)
    """

    # Initialize lists
    start = []
    end = []

    # If the state starts with 1, use it as starting time
    if state[0] == 1:
        start.append(time[0])

    # Loop through and tag start and end times (append to lists)
    for i in range(len(state)):
        if state[i - 1] == 0 and state[i] == 1:
            start.append(time[i])
        elif state[i - 1] == 1 and state[i] == 0:
            end.append(time[i - 1])

    # If state ends with 1, use it as ending time
    if state[len(state) - 1] == 1:
        end.append(time[len(state) - 1])

    # Return a list of the starting and ending times
    return ([start, end])
