# pip install transformers --upgrade
# pip install mosestokenizer

# Translate sentences to English and save mapping dicitionary as json.
# Based on https://github.com/ruanchaves/assin/blob/master/sources/translate.py

import json
import math
import os

import torch
import tqdm
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to("cuda")


def decode(text):
    decoded = tokenizer.decode(
        text,
        skip_special_tokens=True,
    )

    return decoded


def needs_translation(sample, translations):
    needs = [text for text in sample if text not in translations.keys()]

    return needs


def translation(sample):
    src_txts = [f'>>pt_BR<< {text}' for text in sample]

    #translation_sample = tokenizer.prepare_translation_batch(src_txts)
    #translated = model.generate(translation_sample)

    model_inputs = tokenizer.prepare_translation_batch(src_txts).to("cuda")
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        num_beams=2
    )

    translated = [decode(text) for text in generated_ids]

    return translated


def translate2dict(sentences, dictpath, batch_size):
    if not os.path.isfile(dictpath):
        with open(dictpath, 'w') as f:
            json.dump({}, f)

    with open(dictpath) as f:
        translations = json.load(f)

    remaining = needs_translation(sentences, translations)
    batch = math.ceil(len(remaining)/batch_size)

    for idx in tqdm(range(batch), "Translating"):
        keys = remaining[idx*batch_size:(idx+1)*batch_size]
        values = translation(keys)

        new_translations = dict(zip(keys, values))
        translations.update(new_translations)
        torch.cuda.empty_cache()
        with open(dictpath, 'w+') as f:
            json.dump(translations, f)

