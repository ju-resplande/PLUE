# PortugueseGLUE
Portuguese translation of the [GLUE benchmark](https://gluebenchmark.com/) and [Scitail dataset](https://allenai.org/data/scitail) using [OPUS-MT model](https://github.com/Helsinki-NLP/OPUS-MT) and [Google Cloud Translation](https://cloud.google.com/translate/docs). 

Smaller GLUE datasets, such as CoLA, MRPC, RTE, SST-2, STS-B, and WNLI, were translated using Cloud Translation Free Tier. Other GLUE dataset (SNLI, MNLI, QNLI and QQP) and Scitail were translated using OPUS-MT. 

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

## Installation (by Pierre Guillou)
    1. Install cabextract
    2. Create PortugueseGLUE folder (`mkdir`)
    3. Download `glue_v1.py`.
    4. Get MRPC dataset
    
    ```bash
    mkdir glue_data
    cd glue_data
    wget https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi
    mkdir MRPC
    cabextract MSRParaphraseCorpus.msi -d MRPC
    cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
    cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
    rm MRPC/_*
    rm MSRParaphraseCorpus.msi
    cd ..
    ```

5. Get and extract Portuguese GLUE v1

   ```bash
   python glue_v1.py --path_to_mrpc glue_data/MRPC
   ```


Created folders: CoLA, MNLI, MRPC, QNLI, QQP, RTE, SNLI, SST-2, STS-B, WNLI, diagnostic.


## Observations

There are two original GLUE data versions: [first version](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py) and [second version](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py). We noticed the versions differs in QNLI and QQP datasets, where we made QNLI available in both versions and QQP in the newest version. 

SNLI train split is a ragged matrix, so we made available two version of the data: train_raw.tsv contains irregular lines and train.tsv excludes those lines. 

Dependency parsing code is provided using SNLI as an example. Although MNLI contains same number of sentences as SNLI, parsing SNLI takes about minutes while MNLI takes about a week because MNLI sentences structures are complex.

Manual translation were made on 12 sentences in SNLI where original sentences and their translations remained the same, that is, were not translated and 5 sentences in MNLI in which the binary parse returned error.
