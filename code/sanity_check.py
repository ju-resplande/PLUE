import os
import pprint
import subprocess
import utils
import importlib
importlib.reload(utils)


# Download GLUE datasets
glue_v1 = 'GLUE_v1'
glue_v1_download_link = 'https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/download_glue_data.py'
glue_v1_script = 'glue_v1.py'
glue_v1_cmd = [
    'python',
    glue_v1_script,
    '--data_dir',
    glue_v1,
    '--tasks',
    'CoLA,SST,QQP,STS,MNLI,SNLI,QNLI,RTE,WNLI',  # MRPC not available
]

glue_v2 = 'GLUE_v2'
glue_v2_download_link = 'https://raw.githubusercontent.com/nyu-mll/jiant/master/scripts/download_glue_data.py'
glue_v2_script = 'glue_v2.py'
glue_v2_cmd = [
    'python',
    glue_v2_script,
    '--data_dir',
    glue_v2,
]

utils.download_file(glue_v2_download_link, glue_v2_script)
utils.download_file(glue_v1_download_link, glue_v1_script)

print('Downloading GLUE v1 and v2 data...')

subprocess.run(glue_v1_cmd)
subprocess.run(glue_v2_cmd)

print('Differ both GLUE versions')

different_datasets = list()
for dataset in os.listdir(glue_v1):
    dataset_v1 = f'{glue_v1}/{dataset}'
    dataset_v2 = f'{glue_v2}/{dataset}'

    data_files = [f for f in os.listdir(dataset_v1) if f.endswith('.tsv')]

    equal = True
    for data in data_files:
        equal = equal and utils.is_equal(
            f'{dataset_v1}/{data}', f'{dataset_v2}/{data}')

    if not equal:
        different_datasets.append(dataset)

print(f'difference between {glue_v1} and {glue_v2}: {different_datasets}')


print('Downloading Scitail data...')

scitail_zip = 'SciTailV1.1.zip'
scitail_download_link = f'http://data.allenai.org.s3.amazonaws.com/downloads/{scitail_zip}'
scitail_unzip = [
    'unzip',
    scitail_zip,
]

utils.download_file(scitail_download_link, scitail_zip)
subprocess.run(scitail_unzip)

#Tasks and sentences_idx

dataset_sentences = {
    'CoLA': [
        'sentence',  # test: sentence; train and dev: 3
    ],
    'MNLI': [
        'sentence1',
        'sentence2',
        'sentence1_parse',
        'sentence2_parse',
        'sentence1_binary_parse',
        'sentence2_binary_parse',

    ],
    'MRPC': [
        '#1 String',
        '#2 String',
    ],
    'QNLI': [
        'question',
        'sentence',
    ],
    'QNLI_v2': [
        'question',
        'sentence',
    ],
    'QQP_v2': [
        'question1',
        'question2',
    ],
    'RTE': [
        'sentence1',
        'sentence2',
    ],
    'SNLI': [
        'sentence1',
        'sentence2',
        'sentence1_parse',
        'sentence2_parse',
        'sentence1_binary_parse',
        'sentence2_binary_parse',
    ],
    'SST-2': [
        'sentence',
    ],
    'STS-B': [
        'sentence1',
        'sentence2',
    ],
    'WNLI': [
        'sentence1',
        'sentence2',
    ],
    'SciTail': [
        0,
        1,
    ]
}

glue_version = glue_v2


def is_without_header(dataset: str, split_file: str):
    is_cola = (dataset == 'CoLA' and split_file in ['dev.tsv', 'train.tsv'])
    is_scitail = dataset == 'SciTail'

    return is_cola or is_scitail


print('Assert dimensions, non_null data and check translations...')

for dataset, sentences_idx in dataset_sentences.items():
    translation_folder = dataset if dataset != 'SciTail' else os.path.join('SciTail', 'tsv_format')
    translation_folder = os.path.join('datasets', translation_folder)
    original_folder = os.path.join(glue_v2, dataset)

    print(dataset)

    if dataset == 'QNLI':
        original_folder = os.path.join(glue_v1, dataset)
    elif dataset == 'QNLI_v2':
        original_folder = os.path.join(glue_v2, 'QNLI')
    elif dataset == 'QQP_v2':
        original_folder = os.path.join(glue_v2, 'QQP')
    elif dataset == 'SciTail':
        original_folder = os.path.join('SciTailV1.1', 'tsv_format')

    for split_file in os.listdir(translation_folder):
        if split_file == 'train_raw.tsv' or not split_file.endswith('.tsv'):
            continue

        print(split_file)

        has_not_header = is_without_header(dataset, split_file)

        if has_not_header and dataset == 'CoLA':
            sentences_idx = [3]
        elif dataset == 'CoLA':
            sentences_idx = ['sentence']

        fin, original = utils.get_sentences_df(
            os.path.join(original_folder, split_file), sentences_idx, has_not_header)
        fout, translated = utils.get_sentences_df(
            os.path.join(translation_folder, split_file), sentences_idx, has_not_header)

        assert fin.shape == fout.shape
        if not translated.notna().values.all():
            print(translated.notna().all(axis='index'))
        #assert translated.notna().values.all()

        if dataset in ['SNLI', 'MNLI']:
            print('ROOT' in translated[['sentence1_parse']])
            print('ROOT' in translated[['sentence2_parse']])
            assert not 'ROOT' in translated[['sentence1_binary_parse', 'sentence2_binary_parse']]

        original_set = set(utils.flatten(original.values.tolist()))
        translated_set = set(utils.flatten(translated.values.tolist()))

        not_translated = translated_set.intersection(original_set)
        not_translated -= set(translated.columns)

        pprint.pprint({'not translated': not_translated})

# Initially, not_translated in SNLI:
# Guy in Art Deco Glass Home. -> Cara na casa de vidro de Art Deco.
# man making golf club -> homem fazendo taco de golfe
# Andre the Giant wrestling Hulk Hogan. -> Andre, o gigante lutador Hulk Hogan.
# Don'g do drungs -> Não use drogas
# kid bowling -> garoto jogando boliche
# Palominos pantomime. ->  Pantomima de Palominos.
# The Riverview High School Kiltie Band on Parade. -> A banda de Kiltie da High School de Riverview na parada.
# Ballet dancers shake pompoms. -> Dançarinos de balé agitam pompons.
# One Direction sux. -> One Direction é uma merda
# play box. -> caixa de jogo.

print('Analysing train_raw.tsv...')

translation_file = os.path.join('SNLI', 'train_raw.tsv')
original_file = os.path.join(glue_v2, 'SNLI', 'train.tsv')
sentences_idx = range(4, 9)

fin, original = utils.get_sentences_csv(original_file, sentences_idx)
fout, translated = utils.get_sentences_csv(translation_file, sentences_idx)

assert utils.get_shape(fin) == utils.get_shape(fout)
#assert all([not "" in row for row in translated])


not_translated = set(original).intersection(set(translated))
pprint.pprint(not_translated)

# Initially, not_translated in SNLI train_raw:
# Boris Yeltzin raps. -> Boris Yeltzin bate.
