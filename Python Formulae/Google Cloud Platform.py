# Import packages
import pandas as pd
import numpy as np
import codecs, json
import os

from google.protobuf.json_format import MessageToDict
from google.cloud import storage, speech
from google.cloud.speech import enums, types


def googleStorageUpload(file_path, json_credentials_path, bucket_name):
    """
    Funciton to upload files to Google Cloud Storage (GCS)

    :param file_path: Path of file to be uploaded to GCS
    :param json_credentials_path: Path of JSON file containing the credentials for Google Cloud Storage API
    :param bucket_name: Name of bucket to which files will be uploaded
    :return: Nothing. Uploads file to GCS
    """

    # Initiate connection to Google Cloud Storage
    os.environ["GOOGLE_APPLICATION_CREDDENTIALS"] = (json_credentials_path)
    gcs = storage.Client()
    bucket = gcs.get_bucket(bucket_name)  # Bucket is folder within a project

    # Upload file to GCS
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)


def googleTranscribe(wav, hertz=8000):
    """
    Function to transcribe file that already exists in a GCS bucket

    :param wav: Path of the wave file that exists in GCS (e.g. 'gs://project/filename.wav')
    :param hertz: Default hertz (is 8000)
    :return: Result of Google Speech-to-Text transcription
    """

    # Initiate a client
    client = speech.SpeechClient()

    # Configure transcription parameters
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=hertz, language_code='en-US', profanity_filter=False,
        enable_word_time_offset=True, model='phone_call'
    )

    # Load audio
    audio = types.RecognitionAudio(uri=wav)

    # Transcribe
    operation = client.long_running_recognize(config, audio)
    print("Waiting for ", wav, "to complete transcription")
    response = operation.result(timeout=90)

    return (response)


def googleTranscribe_AverageConfidence(transcription):
    """
    Function to get average confidence of transcribed file

    :param transcription: Transcription returned by googleTranscribe()
    :return: Average confidence of transcription
    """

    # Return mean of confidence numbers of transcription
    confidence = []
    for result in transcription.results:
        alternative = result.alternatives[0]
        confidence.append(alternative.confidence)

    return (np.mean(pd.to_numeric(confidence)))


def googleTranscribe_2JSON(transcription, json_output):
    """
    Function to write the Google Transcription to a JSON file

    :param transcription: Transcription from googleTranscribe()
    :param json_output: Path of JSON output
    :return: Nothing. JSON file created in path of output
    """

    df = MessageToDict(transcription)
    json.dump(df, codecs.open(json_output, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def googleTranscribe_2String(transcription):
    """
    Function to grab just the transcription output

    :param transcription: Transcription from googleTranscribe()
    :return: String transcription of .wav file
    """

    # Pull transcription into list
    trans = []
    for result in transcription.results:
        alternative = result.alternatives[0]
        trans.append(alternative.transcript)

    # Collapse list into one string
    trans = ''.join(trans)

    return (trans)
