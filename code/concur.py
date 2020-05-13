# Translate sentences to English and save mapping dicitionary as json.

# Based on https://github.com/ruanchaves/assin/blob/master/sources/translate.py


from concurrent.futures import ThreadPoolExecutor, as_completed
from sys import getsizeof
from math import ceil
import json
import os

import json
from subprocess import PIPE, run
from tqdm import tqdm, trange

import six
from google.cloud import translate_v2 as translate


MAXSIZE = 2500
MAXWORKERS = 9

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './key.json'
translate_client = translate.Client()

def translation(text):
    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    try:
        result = translate_client.translate(text,'pt-br')
        return result['translatedText']
    except Exception as e:
        tqdm.write(e)
        return None
    
def check_translation(text,  translations):
    dic = {'original': text, 'translation': None}

    dic['translation'] = translations[text] \
    if text in translations.keys() else translation(text)

    return dic

def execparallel(sample, func, dic):
    missing = list()

    with ThreadPoolExecutor(MAXWORKERS) as executor:
        translations = {
            executor.submit(func, text, dic): 
            text for text in sample
        }

        for future in as_completed(translations):
            result = future.result()

            if not result['translation']:
                missing.append(result['original'])
                continue

            dic[result['original']] = result['translation']

    return missing


def translate2dictpath(sentences, dictpath):
    if not os.path.isfile(dictpath):
        with open(dictpath, 'w') as f:
            json.dump({}, f)

    with open(dictpath, 'r') as f:
        translations = json.load(f)

    batch = ceil(getsizeof(sentences)/MAXSIZE)
    batch_size = ceil(len(sentences)/batch)

    for idx in tqdm(range(batch), "Translating"):
        error = 0
        sample = sentences[idx*batch_size:(idx+1)*batch_size]

        while sample:
            sample = execparallel(sample, check_translation, translations)

            with open(dictpath, 'w+') as f:
                json.dump(translations, f)

            if sample:
                if error > 3:
                    break

                error = error + 1
