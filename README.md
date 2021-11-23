<div id="top">
  <p align="center">
    <em>(Files are at https://drive.google.com/drive/u/1/folders/1WXdarHqxlJqKm30uD2LubwuLfpAGYZX0 ...)</em>
  </p>
  <p align="center">
    <em>(I'll organize this repo within the next months)</em>
  </p>
</div>
<hr>
<br />
<div align="center">
    <h3 align="center">PLUE: Portuguese Language Understanding Evaluation</h3>
    <img src="https://user-images.githubusercontent.com/28462295/140660705-e39c001f-e311-4024-aa7a-a7e1c69268fc.png" alt="[https://fairytail.fandom.com/wiki/Plue" width="250">
  <p align="center">
    Portuguese translation of the <a href="https://gluebenchmark.com/">GLUE benchmark</a> and <a href=https://allenai.org/data/scitail> Scitail</a> using <a href=https://github.com/Helsinki-NLP/OPUS-MT>OPUS-MT model</a> and <a href=https://cloud.google.com/translate/docs>Google Cloud Translation</a>.
  </p>
</div>


## Getting Started


| Datasets | Translation Tool |
| --- | --- |
| CoLA, MRPC, RTE, SST-2, STS-B, and WNLI  | Google Cloud Translation |
| SNLI, MNLI, QNLI, QQP, and SciTail  |  OPUS-MT |

[LX parser](http://lxcenter.di.fc.ul.pt/tools/en/LXParserEN.html), [Binarizer code](http://lascam.facom.ufu.br:8080/cookbooks/cookbook.jsp?api=nltk#ex11) and [NLTK word tokenizer](https://www.nltk.org/_modules/nltk/tokenize.html#word_tokenize) were used to create dependency parsings for SNLI and MNLI datasets.

### Structure

```bash
├── code _____________________ # translation code and dependency parsing  
├── datasets
│   ├── CoLA
│   ├── MNLI
│   ├── MRPC
│   ├── QNLI
│   ├── QNLI_v2
│   ├── QQP_v2
│   ├── RTE
│   ├── SciTail
│   │   └── tsv_format
│   ├── SNLI
│   ├── SST-2
│   ├── STS-B
│   └── WNLI
└── pairs _____________________ # translation pairs as JSON dictionary
```

## Observations

- GLUE provides two versions: [first](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py) and [second](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py). We noticed the versions only differs in QNLI and QQP datasets, where we made QNLI available in both versions and QQP in the newest version. 
- SNLI train split is a ragged matrix, so we made available two version of the data: train_raw.tsv contains irregular lines and train.tsv excludes those lines. 
- Manual translation were made on 12 sentences due to translation errors.
- Our translation code is outdated. [We recommend using from others](https://github.com/unicamp-dl/mMARCO/blob/main/scripts/translate.py).


## Citing

```bibtex
@misc{Gomes2020,
  author = {GOMES, J. R. S.},
  title = {Portuguese Language Understanding Evaluation},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jubs12/PLUE}},
  commit = {CURRENT_COMMIT}
}
```

## Acknowledgments
- Deep Learning Brasil/CEIA
- Cyberlabs
