import os
from typing import List
import requests

import pandas as pd
import unicodecsv

# pip install unicodecsv


def download_file(url: str, filename: str):
    r = requests.get(url)
    open(filename, 'wb').write(r.content)


def get_sentences_df(filename: str, sentences_idx: list, has_not_header: bool):
    header = None if has_not_header else 'infer'
    df = pd.read_csv(filename, sep='\t', quoting=3,
                     header=header, error_bad_lines=False)
    sentences = df[sentences_idx]
    return df, sentences


def get_sentences_csv(filename: str, sentences_idx: list):
    with open(filename, 'rbU') as csvfile:
        reader = unicodecsv.reader(csvfile, delimiter="\t")
        csv = list(reader)

    sentences = list()
    for s_idx in sentences_idx:
        sentence_list = [line[s_idx]
                         for (idx, line) in enumerate(csv) if idx != 0]
        sentences.extend(sentence_list)

    return csv, sentences


def read_lines(filepath: str):
    with open(filepath) as f:
        text = f.read()
        lines = text.split('\n')
    return lines


def get_shape(matrix: List[list]):
    shape = list(map(len, matrix))
    return shape


def null_count(lst: List[str]):
    null = lst.count('')
    return null


def flatten(lst: List[list]) -> list:
    lst = [item for sublist in lst for item in sublist]
    return lst


def is_equal(filepath1: str, filepath2: str):
    import filecmp
    return filecmp.cmp(filepath1, filepath2)
