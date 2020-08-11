import subprocess
import shutil
import json
import os

import pandas as pd
import nltk

import utils

DATASET = 'SNLI'

stanford_parser_file = "stanford-parser-2010-11-30.tgz"
stanford_parser_link = f"https://nlp.stanford.edu/software/{stanford_parser_file}"
lx_parser_file = "cintil.ser.gz"
lx_parser_link = f"http://lxcenter.di.fc.ul.pt/tools/download/{lx_parser_file}"

parser_folder = os.path.join(os.getcwd(), "dependency parsing")
if not os.path.isdir(parser_folder):
    print('Creating dependency parsing folder')
    os.mkdir(parser_folder)
print('Changing current directory to dependency parsing')
os.chdir(parser_folder)

print('Downloading and extracting parser')
utils.download_file(stanford_parser_link, stanford_parser_file)
subprocess.run(["wget", lx_parser_link]) #for some reason request does not get it zipped


shutil.unpack_archive(stanford_parser_file)

nltk.download('punkt')

print("Tokenizing DATASET...")
sentences = list()
DATASET_folder = f'../{DATASET}'
for f in os.listdir(DATASET_folder):
    if f == 'train_raw.tsv':
        continue

    table = pd.read_csv(f'{DATASET_folder}/{f}', sep='\t', quoting=3, error_bad_lines=False)
    for idx in ['sentence1', 'sentence2']:
        sentences.extend(list(table[idx].astype('str')))

if 'nan' in sentences:
    sentences.remove('nan')

sentences = list(set(sentences))
tokenized_sentences = list(map(utils.tokenize, sentences))
tokenized_sentences = '\n'.join(tokenized_sentences)
with open(f'{DATASET}_tokenized.txt', 'w') as f:
    f.write(tokenized_sentences)

    sentences = '\n'.join(sentences)
    with open(f'{DATASET}.txt', 'w') as f:
        f.write(sentences)

print("Runing parser")
parser_cmd = [
    "java",
    "-Xmx10000m",
    "-cp",
    "stanford-parser-2010-11-30/stanford-parser.jar",
    "edu.stanford.nlp.parser.lexparser.LexicalizedParser",
    "-tokenized",
    "-sentences",
    "newline",
    "-outputFormat",
    "oneline",
    "-MAX_ITEMS",
    "20000000",
    "-maxLength",
    "120000",
    "-writeOutputFiles",
    "-uwModel",
    "edu.stanford.nlp.parser.lexparser.BaseUnknownWordModel",
    "cintil.ser.gz",
    f'{DATASET}_tokenized.txt',
]

subprocess.run(parser_cmd)

input_lines = utils.read_lines(f'{DATASET}.txt')
output_lines = utils.read_lines(f'{DATASET}_tokenized.txt.stp')
output_lines = output_lines[:-1] 

print(len(input_lines), len(output_lines))

dependency_dict = dict(zip(input_lines, output_lines))
with open(f'dependency_{DATASET}.json', 'w') as f:
    json.dump(dependency_dict, f)

binary_lines = [utils.binarize(nltk.Tree.fromstring(sentence)) for sentence in output_lines]
binary_dict = dict(zip(input_lines, binary_lines))

with open(f'binary_{DATASET}.json', 'w') as f:
    json.dump(binary_dict, f)
