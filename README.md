# PortugueseGLUE
Portuguese translation of the [GLUE benchmark](https://gluebenchmark.com/) using [OPUS-MT model](https://github.com/Helsinki-NLP/OPUS-MT) and [Google Cloud Translation](https://cloud.google.com/translate/docs). 

Smaller datasets, such as CoLA, MRPC, RTE, SST-2, STS-B, and WNLI, were translated using Cloud Translation Free Tier. SNLI, MNLI, QNLI and QQP were translated using OPUS-MT.

[LX parser](http://lxcenter.di.fc.ul.pt/tools/en/LXParserEN.html), [Binarizer code](http://lascam.facom.ufu.br:8080/cookbooks/cookbook.jsp?api=nltk#ex11) and [NLTK word tokenizer](https://www.nltk.org/_modules/nltk/tokenize.html#word_tokenize) were used to create dependency parsings for SNLI and MNLI datasets.

## Code requirements



- Gcloud Translation
  - google-cloud-translate
- Opus Translation
  - transformers <= 2
  - mosestokenizer
  - tqdm
  - pytorch
- Additional Tools
  - Remove HTML marks
    - ftfy
  - Table manipulation
    - pandas
    - unicodecsv (snli train ragged matrix)
  - Dependency parsing
    - pandas
    - nltk
    - unicodecsv (snli train ragged matrix)
    - ~~LX-parser (downloaded in dependency_parsing.py)~~



## Observations

There are two original GLUE data versions: [first version](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py) and [second version](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py). We noticed the versions differs in QNLI and QQP datasets, where we made QNLI available in both versions and QQP in the newest version. 

SNLI train split is a ragged matrix, so we made available two version of the data: train_raw.tsv contains irregular lines and train.tsv excludes those lines. 

Dependency parsing code is provided using SNLI as an example. Although MNLI contains same number of sentences as SNLI, parsing SNLI takes about minutes while MNLI takes about a week because MNLI sentences structures are complex.

Manual translation were made on 12 sentences in SNLI where original sentences and their translations remained the same, that is, were not translated and 5 sentences in MNLI in which the binary parse returned error. 