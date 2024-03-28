# pip install moviepy speechrecognition pydub google-cloud-speech
#
# spleeter splits voices from audio file
# install spleeter (Tested mac os X with m1 cpu)
#
# pip install numba==0.56.2
# pip install numpy==1.23.5
# pip install llvmlite=0.39.1
# pip install tensorflow==2.15.1
#
# environment variables
# for naver speech to text
# - NAVER_CLIENT_ID - naver cloud client id
# - NAVER_CLIENT_SECRET - naver cloud application client secret
# for google cloud speech to text
# - GOOGLE_APPLICATION_CREDENTIALS - application credential json file path
# 

import sys
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import glob
import requests

from spleeter.separator import Separator


def convert_video_to_audio(video_path, audio_path="temp_audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()  # It's a good practice to close the clip to free resources
    return audio_path


def add_padding_to_chunks(chunks, padding_duration_ms=5000):
    """
    Adds padding to each chunk by extending the start of each chunk with audio from the
    end of the previous chunk.
    
    :param padding_duration_ms: Duration of the padding in milliseconds. Each chunk will
    be extended by this amount at the start with audio from the end of the previous chunk.
    :param chunks: A list of AudioSegment objects.
    :param padding_duration_ms: Duration
    of the padding in milliseconds. Each chunk will be extended by this amount at the
    start with audio from the end of the previous chunk.
    :return: A list of padded AudioSegment objects.
    """
    padded_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            padded_chunk = AudioSegment.silent(duration=padding_duration_ms) + chunk
        else:
            previous_chunk = chunks[i - 1]
            # Extract the last 'padding_duration_ms' milliseconds from the previous chunk
            overlap = previous_chunk[-min(len(previous_chunk), padding_duration_ms):]
            padded_chunk = overlap + chunk

        padded_chunks.append(padded_chunk)

    return padded_chunks


def split_audio_by_silence(audio_path, min_silence_len=1000, silence_thresh=-30,
                           max_chunk_length=45000, overlap_ms=500):
    """
    Splits the audio file into chunks at silent sections, ensuring each chunk is less
    than 60 seconds. If no suitable silence is detected within a chunk, it will be
    split at the maximum length without silence detection.

    :param overlap_ms:
    :param audio_path: Path to the audio file to split.
    :param min_silence_len: Minimum length of silence in milliseconds to consider
    as a split point.
    :param silence_thresh: Silence threshold in dB. Lower values mean more
    silence is detected.
    :param max_chunk_length: Maximum length of each chunk in
    milliseconds.
    :return: A list of AudioSegment chunks.
    """
    sound_file = AudioSegment.from_wav(audio_path)
    initial_chunks = split_on_silence(sound_file,
                                      min_silence_len=min_silence_len,
                                      silence_thresh=silence_thresh,
                                      keep_silence=1000)
    if not initial_chunks:
        initial_chunks = [sound_file]
    final_chunks = []
    for chunk in initial_chunks:
        if len(chunk) <= max_chunk_length:
            final_chunks.append(chunk)
        else:
            # If the chunk is too long and doesn't have suitable silence, split it into
            # smaller parts
            num_subchunks = len(chunk) // max_chunk_length + 1
            for i in range(num_subchunks):
                start_ms = i * max_chunk_length
                end_ms = min((i + 1) * max_chunk_length, len(chunk))
                sub_chunk = chunk[start_ms:end_ms]
                final_chunks.append(sub_chunk)

    return add_padding_to_chunks(final_chunks, overlap_ms)


def recognize_audio_chunks(audio_chunks, vendor, language="ko-KR"):
    """
    Recognizes speech from audio chunks using Google Cloud Speech-to-Text.

    :param audio_chunks: List of AudioSegment chunks to process.
    """
    recognizer = sr.Recognizer()

    for i, chunk in enumerate(audio_chunks):
        # Export chunk to a temporary WAV file
        chunk_file_path = f"chunk{i}.wav"
        chunk.export(chunk_file_path, format="wav")

        # Recognize the chunk
        recognize_audio(recognizer, f"{i}: ", chunk_file_path, vendor, language)


def transcribe_audio_naver(audio_path, client_id = os.environ.get('NAVER_CLIENT_ID'),
                           client_secret = os.environ.get('NAVER_CLIENT_SECRET')):
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"
    headers = {
        "Content-Type": "application/octet-stream",
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
    }

    with open(audio_path, 'rb') as audio_file:
        response = requests.post(url, headers=headers, data=audio_file)

    if response.status_code == 200:
        return response.json().get('text', '')
    else:
        print(f"Error Code: {response.status_code}")
        return None


def recognize_audio(recognizer, prefix, audio_file, vendor, language="ko-KR"):
    with sr.AudioFile(audio_file) as source:
        recognizer.adjust_for_ambient_noise(source)  # This line helps with noise
        audio_data = recognizer.record(source)
        # Attempt to recognize the speech in the audio
        try:
            if vendor == "google-cloud":
                # Specify the credentials explicitly only if you didn't set the
                # environment variable google_cloud_credentials =
                # r'path_to_your_service_account_json_file.json' response =
                # recognizer.recognize_google_cloud(audio_data, credentials_json=open(
                # google_cloud_credentials, 'r').read()) If you've set the
                # GOOGLE_APPLICATION_CREDENTIALS environment variable, you can omit the
                # credentials_json argument
                text = recognizer.recognize_google_cloud(audio_data, language=language)
            elif vendor == "naver":
                text = transcribe_audio_naver(audio_file)
            elif vendor == "whisper":
                text = recognizer.recognize_whisper(audio_data, language=language,
                                                    model='large')
            else:
                text = recognizer.recognize_google(audio_data, language=language)
            print(f"{prefix}{text}")
        except sr.UnknownValueError as e:
            print(f"{prefix} {vendor} Recognition could not understand audio; {e}")
        except sr.RequestError as e:
            print(f"{prefix} Could not request results from {vendor} Recognition service; {e}")


def find_existing_chunks(audio_path):
    """
    Finds and loads existing audio chunks based on the naming convention 'chunkX.wav'.
    
    :param audio_path: Path where the chunks are expected to be.
    :return: A list of AudioSegment objects if chunk files are found, else an empty list.
    """
    directory = os.path.dirname(audio_path)
    chunk_files = glob.glob(os.path.join(directory, 'chunk*.wav'))
    chunk_files_sorted = sorted(chunk_files, key=lambda x: int(
        os.path.splitext(os.path.basename(x))[0].replace('chunk', '')))

    audio_chunks = [AudioSegment.from_wav(chunk_file) for chunk_file in
                    chunk_files_sorted]
    return audio_chunks

def separate_vocals(audio_path, output_path):
    """
    Separates vocals from the background in an audio file using spleeter.

    :param audio_path: Path to the input audio file.
    :param output_path: Directory where the separated audio files will be saved.
    """
    # Use spleeter's '2stems' model to separate vocals and accompaniment
    separator = Separator('spleeter:2stems')

    # The vocals will be saved as 'output_path/filename/vocals.wav'
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(output_path, base_filename, 'vocals.wav')
    if not os.path.exists(vocals_path):
        # Perform separation
        separator.separate_to_file(audio_path, output_path)
    return vocals_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {__file__} <video_file_path> <speech-to-text vendor> <language>")
        sys.exit(1)

    video_path = sys.argv[1]  # Get video file path from script argument
    audio_path = "temp_audio.wav"
    vendor = 'google' if len(sys.argv) < 3 else sys.argv[2]
    language = 'ko-KR' if len(sys.argv) < 4 else sys.argv[3]
    if not os.path.exists(audio_path):
        audio_path = convert_video_to_audio(video_path, audio_path)
        audio_path = separate_vocals(audio_path, "audio_output")
        audio_chunks = split_audio_by_silence(audio_path)
    else:
        audio_chunks = find_existing_chunks(audio_path)
    recognize_audio_chunks(audio_chunks, vendor, language)
