import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
from gtts import gTTS
import speech_recognition as sr
import pygame


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector



def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  # return cosine similarity


def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def load_convo():
    files = os.listdir('nexus')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json('nexus/%s' % file)
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return ordered


def summarize_memories(memories):
    sorted_memories = sorted(memories, key=lambda d: d['time'])
    block = '\n\n'.join(mem['message'] for mem in sorted_memories)
    identifiers = [mem['uuid'] for mem in sorted_memories]
    timestamps = [mem['time'] for mem in sorted_memories]
    prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block)
    notes = gpt3_completion(prompt)
    vector = gpt3_embedding(block)
    info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()), 'vector': vector, 'time': time()}
    filename = f'notes_{time()}.json'
    save_json(f'internal_notes/{filename}', info)
    return notes


def get_last_messages(conversation, limit):
    short = conversation[-limit:] if len(conversation) >= limit else conversation
    return '\n\n'.join([i['message'] for i in short])


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
    max_retry = 5
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    for retry in range(max_retry):
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = f'{time()}_gpt3.txt'
            logs_dir = 'gpt3_logs'
            os.makedirs(logs_dir, exist_ok=True)
            save_file(f'{logs_dir}/{filename}', f'{prompt}\n\n==========\n\n{text}')
            return text
        except Exception as oops:
            if retry + 1 >= max_retry:
                return f"GPT3 error: {oops}"
            print(f'Error communicating with OpenAI: {oops}')


def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except Exception as e:
            print("Sorry, I didn't catch that. Could you please repeat?")
            return ""


def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit()

def modify_code(code):
    new_code = code.replace(
        'os.system("start response.mp3")',
        'play_audio("response.mp3")'
    )
    return new_code

if __name__ == '__main__':
    openai.api_key = open_file('openaiapikey.txt')
    prompt_template = open_file('prompt_response.txt')
    while True:
        a = get_speech_input()
        timestamp = time()
        vector = gpt3_embedding(a)
        timestring = timestamp_to_datetime(timestamp)
        message = f'{timestring}: {a}'
        info = {
            'speaker': 'USER',
            'time': timestamp,
            'vector': vector,
            'message': message,
            'uuid': str(uuid4()),
            'timestring': timestring
        }
        filename = f'log_{timestamp}_USER.json'
        save_json(f'nexus/{filename}', info)

        conversation = load_convo()
        memories = fetch_memories(vector, conversation, 10)
        notes = summarize_memories(memories)
        recent = get_last_messages(conversation, 4)
        prompt = prompt_template.replace('<<NOTES>>', notes).replace('<<CONVERSATION>>', recent)

        output = gpt3_completion(prompt)
        timestamp = time()
        vector = gpt3_embedding(output)
        timestring = timestamp_to_datetime(timestamp)
        message = f'{timestring}: {output}'
        info = {
            'speaker': 'RAVEN',
            'time': timestamp,
            'vector': vector,
            'message': message,
            'uuid': str(uuid4()),
            'timestring': timestring
        }
        filename = f'log_{timestamp}_RAVEN.json'
        save_json(f'nexus/{filename}', info)

        tts = gTTS(text=output, lang='en', tld='us')
        tts.save("response.mp3")
        play_audio("response.mp3")
