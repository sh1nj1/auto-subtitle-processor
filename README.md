# autogen-subtitles

## Description

This script generates subtitles from video file using speech to text api.
This is just test, all these process can be done by [whisper](https://github.com/openai/whisper) so you better use whisper.

## Requirements

python 3.9

pip install moviepy speechrecognition pydub google-cloud-speech

spleeter splits voices from audio file
install spleeter (Tested on mac os X with m1 cpu)

pip install numba==0.56.2
pip install numpy==1.23.5
pip install llvmlite=0.39.1
pip install tensorflow==2.15.1
pip install spleeter==2.3.2

environment variables

for naver speech to text

- NAVER_CLIENT_ID - naver cloud client id
- NAVER_CLIENT_SECRET - naver cloud application client secret

for google cloud speech to text

- GOOGLE_APPLICATION_CREDENTIALS - application credential json file path

## Run script

python autogen-subtitles.py input.mp4 google

## 

combine these three auto-generated subtitles to guess best possible original text from original audio.

