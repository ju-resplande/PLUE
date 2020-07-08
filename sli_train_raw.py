import json

import utils
import translate_opus

train_list, sentences = utils.get_sentences_csv('SNLI/train_raw.tsv', sentences_idx) 

with open('dictionaries/snli.json') as infile:
    dictionary = json.load(infile)

translations = set(dictionary.values())
missing = sentences - translations
missing = list(missing)

with open('missing.json', 'w') as f:
    json.dump(missing, f)

translate_opus.translate2dict(missing, 'dictionaries/snli.json', 20)