# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# https://github.com/huggingface/datasets/blob/master/datasets/glue/glue.py
# https://github.com/huggingface/datasets/blob/master/datasets/scitail/scitail.py
"""The General Language Understanding Evaluation (GLUE) benchmark."""


import csv
import os
import textwrap

import numpy as np

import datasets


_PLUE_CITATION = """\
@misc{Gomes2020,
  author = {GOMES, J. R. S.},
  title = {Portuguese Language Understanding Evaluation},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/jubs12/PLUE}},
  commit = {CURRENT_COMMIT}
}

@inproceedings{wang2019glue,
  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  note={In the Proceedings of ICLR.},
  year={2019}
}
"""

_PLUE_DESCRIPTION = """\
PLUE: Portuguese Language Understanding Evaluationis a Portuguese translation of 
the GLUE benchmark and Scitail using OPUS-MT model and Google Cloud Translation.
"""

MNLI_URL = "https://github.com/ju-resplande/PLUE/releases/download/v1.0.0/MNLI.zip"
SNLI_URL = "https://github.com/ju-resplande/PLUE/releases/download/v1.0.0/SNLI.zip"

_MNLI_BASE_KWARGS = dict(
    text_features={"premise": "sentence1", "hypothesis": "sentence2",},
    label_classes=["entailment", "neutral", "contradiction"],
    label_column="gold_label",
    data_dir="MNLI",
    citation=textwrap.dedent(
        """\
      @InProceedings{N18-1101,
        author = "Williams, Adina
                  and Nangia, Nikita
                  and Bowman, Samuel",
        title = "A Broad-Coverage Challenge Corpus for
                 Sentence Understanding through Inference",
        booktitle = "Proceedings of the 2018 Conference of
                     the North American Chapter of the
                     Association for Computational Linguistics:
                     Human Language Technologies, Volume 1 (Long
                     Papers)",
        year = "2018",
        publisher = "Association for Computational Linguistics",
        pages = "1112--1122",
        location = "New Orleans, Louisiana",
        url = "http://aclweb.org/anthology/N18-1101"
      }
      @article{bowman2015large,
        title={A large annotated corpus for learning natural language inference},
        author={Bowman, Samuel R and Angeli, Gabor and Potts, Christopher and Manning, Christopher D},
        journal={arXiv preprint arXiv:1508.05326},
        year={2015}
      }"""
    ),
    url="http://www.nyu.edu/projects/bowman/multinli/",
)


class PlueConfig(datasets.BuilderConfig):
    """BuilderConfig for GLUE."""

    def __init__(
        self,
        text_features,
        label_column,
        data_dir,
        citation,
        url,
        label_classes=None,
        process_label=lambda x: x,
        **kwargs,
    ):
        """BuilderConfig for GLUE.

        Args:
          text_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the tsv file
          label_column: `string`, name of the column in the tsv file corresponding
            to the label
          data_url: `string`, url to download the zip file from
          data_dir: `string`, the path to the folder containing the tsv files in the
            downloaded zip
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(PlueConfig, self).__init__(
            version=datasets.Version("1.0.2", ""), **kwargs
        )
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = (
            "https://github.com/ju-resplande/PLUE/archive/refs/tags/v1.0.1.zip"
        )
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label


class Plue(datasets.GeneratorBasedBuilder):
    """The General Language Understanding Evaluation (GLUE) benchmark."""

    BUILDER_CONFIGS = [
        PlueConfig(
            name="cola",
            description=textwrap.dedent(
                """\
            The Corpus of Linguistic Acceptability consists of English
            acceptability judgments drawn from books and journal articles on
            linguistic theory. Each example is a sequence of words annotated
            with whether it is a grammatical English sentence."""
            ),
            text_features={"sentence": "sentence"},
            label_classes=["unacceptable", "acceptable"],
            label_column="is_acceptable",
            data_dir="PLUE-1.0.1/datasets/CoLA",
            citation=textwrap.dedent(
                """\
            @article{warstadt2018neural,
              title={Neural Network Acceptability Judgments},
              author={Warstadt, Alex and Singh, Amanpreet and Bowman, Samuel R},
              journal={arXiv preprint arXiv:1805.12471},
              year={2018}
            }"""
            ),
            url="https://nyu-mll.github.io/CoLA/",
        ),
        PlueConfig(
            name="sst2",
            description=textwrap.dedent(
                """\
            The Stanford Sentiment Treebank consists of sentences from movie reviews and
            human annotations of their sentiment. The task is to predict the sentiment of a
            given sentence. We use the two-way (positive/negative) class split, and use only
            sentence-level labels."""
            ),
            text_features={"sentence": "sentence"},
            label_classes=["negative", "positive"],
            label_column="label",
            data_dir="PLUE-1.0.1/datasets/SST-2",
            citation=textwrap.dedent(
                """\
            @inproceedings{socher2013recursive,
              title={Recursive deep models for semantic compositionality over a sentiment treebank},
              author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew and Potts, Christopher},
              booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
              pages={1631--1642},
              year={2013}
            }"""
            ),
            url="https://datasets.stanford.edu/sentiment/index.html",
        ),
        PlueConfig(
            name="mrpc",
            description=textwrap.dedent(
                """\
            The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of
            sentence pairs automatically extracted from online news sources, with human annotations
            for whether the sentences in the pair are semantically equivalent."""
            ),  # pylint: disable=line-too-long
            text_features={"sentence1": "", "sentence2": ""},
            label_classes=["not_equivalent", "equivalent"],
            label_column="Quality",
            data_dir="PLUE-1.0.1/datasets/MRPC",
            citation=textwrap.dedent(
                """\
            @inproceedings{dolan2005automatically,
              title={Automatically constructing a corpus of sentential paraphrases},
              author={Dolan, William B and Brockett, Chris},
              booktitle={Proceedings of the Third International Workshop on Paraphrasing (IWP2005)},
              year={2005}
            }"""
            ),
            url="https://www.microsoft.com/en-us/download/details.aspx?id=52398",
        ),
        PlueConfig(
            name="qqp",
            description=textwrap.dedent(
                """\
            The Quora Question Pairs2 dataset is a collection of question pairs from the
            community question-answering website Quora. The task is to determine whether a
            pair of questions are semantically equivalent."""
            ),
            text_features={"question1": "question1", "question2": "question2",},
            label_classes=["not_duplicate", "duplicate"],
            label_column="is_duplicate",
            data_dir="PLUE-1.0.1/datasets/QQP_v2",
            citation=textwrap.dedent(
                """\
          @online{WinNT,
            author = {Iyer, Shankar and Dandekar, Nikhil and Csernai, Kornel},
            title = {First Quora Dataset Release: Question Pairs},
            year = {2017},
            url = {https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs},
            urldate = {2019-04-03}
          }"""
            ),
            url="https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        ),
        PlueConfig(
            name="stsb",
            description=textwrap.dedent(
                """\
            The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a collection of
            sentence pairs drawn from news headlines, video and image captions, and natural
            language inference data. Each pair is human-annotated with a similarity score
            from 1 to 5."""
            ),
            text_features={"sentence1": "sentence1", "sentence2": "sentence2",},
            label_column="score",
            data_dir="PLUE-1.0.1/datasets/STS-B",
            citation=textwrap.dedent(
                """\
            @article{cer2017semeval,
              title={Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation},
              author={Cer, Daniel and Diab, Mona and Agirre, Eneko and Lopez-Gazpio, Inigo and Specia, Lucia},
              journal={arXiv preprint arXiv:1708.00055},
              year={2017}
            }"""
            ),
            url="http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark",
            process_label=np.float32,
        ),
        PlueConfig(
            name="snli",
            description=textwrap.dedent(
                """\
            The SNLI corpus (version 1.0) is a collection of 570k human-written English
            sentence pairs manually labeled for balanced classification with the labels
            entailment, contradiction, and neutral, supporting the task of natural language
            inference (NLI), also known as recognizing textual entailment (RTE).
            """
            ),
            text_features={"premise": "sentence1", "hypothesis": "sentence2",},
            label_classes=["entailment", "neutral", "contradiction"],
            label_column="gold_label",
            data_dir="SNLI",
            citation=textwrap.dedent(
                """\
            @inproceedings{snli:emnlp2015,
                Author = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher, and Manning, Christopher D.},
                Booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
                Publisher = {Association for Computational Linguistics},
                Title = {A large annotated corpus for learning natural language inference},
                Year = {2015}
            }
            """
            ),
            url="https://nlp.stanford.edu/projects/snli/",
        ),
        PlueConfig(
            name="mnli",
            description=textwrap.dedent(
                """\
            The Multi-Genre Natural Language Inference Corpus is a crowdsourced
            collection of sentence pairs with textual entailment annotations. Given a premise sentence
            and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis
            (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are
            gathered from ten different sources, including transcribed speech, fiction, and government reports.
            We use the standard test set, for which we obtained private labels from the authors, and evaluate
            on both the matched (in-domain) and mismatched (cross-domain) section. We also use and recommend
            the SNLI corpus as 550k examples of auxiliary training data."""
            ),
            **_MNLI_BASE_KWARGS,
        ),
        PlueConfig(
            name="mnli_mismatched",
            description=textwrap.dedent(
                """\
          The mismatched validation and test splits from MNLI.
          See the "mnli" BuilderConfig for additional information."""
            ),
            **_MNLI_BASE_KWARGS,
        ),
        PlueConfig(
            name="mnli_matched",
            description=textwrap.dedent(
                """\
          The matched validation and test splits from MNLI.
          See the "mnli" BuilderConfig for additional information."""
            ),
            **_MNLI_BASE_KWARGS,
        ),
        PlueConfig(
            name="qnli",
            description=textwrap.dedent(
                """\
            The Stanford Question Answering Dataset is a question-answering
            dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn
            from Wikipedia) contains the answer to the corresponding question (written by an annotator). We
            convert the task into sentence pair classification by forming a pair between each question and each
            sentence in the corresponding context, and filtering out pairs with low lexical overlap between the
            question and the context sentence. The task is to determine whether the context sentence contains
            the answer to the question. This modified version of the original task removes the requirement that
            the model select the exact answer, but also removes the simplifying assumptions that the answer
            is always present in the input and that lexical overlap is a reliable cue."""
            ),  # pylint: disable=line-too-long
            text_features={"question": "question", "sentence": "sentence",},
            label_classes=["entailment", "not_entailment"],
            label_column="label",
            data_dir="PLUE-1.0.1/datasets/QNLI",
            citation=textwrap.dedent(
                """\
            @article{rajpurkar2016squad,
              title={Squad: 100,000+ questions for machine comprehension of text},
              author={Rajpurkar, Pranav and Zhang, Jian and Lopyrev, Konstantin and Liang, Percy},
              journal={arXiv preprint arXiv:1606.05250},
              year={2016}
            }"""
            ),
            url="https://rajpurkar.github.io/SQuAD-explorer/",
        ),
        PlueConfig(
            name="rte",
            description=textwrap.dedent(
                """\
            The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual
            entailment challenges. We combine the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim
            et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009).4 Examples are
            constructed based on news and Wikipedia text. We convert all datasets to a two-class split, where
            for three-class datasets we collapse neutral and contradiction into not entailment, for consistency."""
            ),  # pylint: disable=line-too-long
            text_features={"sentence1": "sentence1", "sentence2": "sentence2",},
            label_classes=["entailment", "not_entailment"],
            label_column="label",
            data_dir="PLUE-1.0.1/datasets/RTE",
            citation=textwrap.dedent(
                """\
            @inproceedings{dagan2005pascal,
              title={The PASCAL recognising textual entailment challenge},
              author={Dagan, Ido and Glickman, Oren and Magnini, Bernardo},
              booktitle={Machine Learning Challenges Workshop},
              pages={177--190},
              year={2005},
              organization={Springer}
            }
            @inproceedings{bar2006second,
              title={The second pascal recognising textual entailment challenge},
              author={Bar-Haim, Roy and Dagan, Ido and Dolan, Bill and Ferro, Lisa and Giampiccolo, Danilo and Magnini, Bernardo and Szpektor, Idan},
              booktitle={Proceedings of the second PASCAL challenges workshop on recognising textual entailment},
              volume={6},
              number={1},
              pages={6--4},
              year={2006},
              organization={Venice}
            }
            @inproceedings{giampiccolo2007third,
              title={The third pascal recognizing textual entailment challenge},
              author={Giampiccolo, Danilo and Magnini, Bernardo and Dagan, Ido and Dolan, Bill},
              booktitle={Proceedings of the ACL-PASCAL workshop on textual entailment and paraphrasing},
              pages={1--9},
              year={2007},
              organization={Association for Computational Linguistics}
            }
            @inproceedings{bentivogli2009fifth,
              title={The Fifth PASCAL Recognizing Textual Entailment Challenge.},
              author={Bentivogli, Luisa and Clark, Peter and Dagan, Ido and Giampiccolo, Danilo},
              booktitle={TAC},
              year={2009}
            }"""
            ),
            url="https://aclweb.org/aclwiki/Recognizing_Textual_Entailment",
        ),
        PlueConfig(
            name="wnli",
            description=textwrap.dedent(
                """\
            The Winograd Schema Challenge (Levesque et al., 2011) is a reading comprehension task
            in which a system must read a sentence with a pronoun and select the referent of that pronoun from
            a list of choices. The examples are manually constructed to foil simple statistical methods: Each
            one is contingent on contextual information provided by a single word or phrase in the sentence.
            To convert the problem into sentence pair classification, we construct sentence pairs by replacing
            the ambiguous pronoun with each possible referent. The task is to predict if the sentence with the
            pronoun substituted is entailed by the original sentence. We use a small evaluation set consisting of
            new examples derived from fiction books that was shared privately by the authors of the original
            corpus. While the included training set is balanced between two classes, the test set is imbalanced
            between them (65% not entailment). Also, due to a data quirk, the development set is adversarial:
            hypotheses are sometimes shared between training and development examples, so if a model memorizes the
            training examples, they will predict the wrong label on corresponding development set
            example. As with QNLI, each example is evaluated separately, so there is not a systematic correspondence
            between a model's score on this task and its score on the unconverted original task. We
            call converted dataset WNLI (Winograd NLI)."""
            ),
            text_features={"sentence1": "sentence1", "sentence2": "sentence2",},
            label_classes=["not_entailment", "entailment"],
            label_column="label",
            data_dir="PLUE-1.0.1/datasets/WNLI",
            citation=textwrap.dedent(
                """\
            @inproceedings{levesque2012winograd,
              title={The winograd schema challenge},
              author={Levesque, Hector and Davis, Ernest and Morgenstern, Leora},
              booktitle={Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning},
              year={2012}
            }"""
            ),
            url="https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html",
        ),
        PlueConfig(
            name="scitail",
            description=textwrap.dedent(
                """\
            The SciTail dataset is an entailment dataset created from multiple-choice science exams and web sentences. Each question
and the correct answer choice are converted into an assertive statement to form the hypothesis. We use information
retrieval to obtain relevant text from a large text corpus of web sentences, and use these sentences as a premise P. We
crowdsource the annotation of such premise-hypothesis pair as supports (entails) or not (neutral), in order to create
the SciTail dataset. The dataset contains 27,026 examples with 10,101 examples with entails label and 16,925 examples
with neutral label"""
            ),
            text_features={"premise": "premise", "hypothesis": "hypothesis",},
            label_classes=["entails", "neutral"],
            label_column="label",
            data_dir="PLUE-1.0.1/datasets/SciTail",
            citation=""""\
            inproceedings{scitail,
                Author = {Tushar Khot and Ashish Sabharwal and Peter Clark},
                Booktitle = {AAAI},
                Title = {{SciTail}: A Textual Entailment Dataset from Science Question Answering},
                Year = {2018}
            }
            """,
            url="https://gluebenchmark.com/diagnostics",
        ),
    ]

    def _info(self):
        features = {
            text_feature: datasets.Value("string")
            for text_feature in self.config.text_features.keys()
        }
        if self.config.label_classes:
            features["label"] = datasets.features.ClassLabel(
                names=self.config.label_classes
            )
        else:
            features["label"] = datasets.Value("float32")
        features["idx"] = datasets.Value("int32")
        return datasets.DatasetInfo(
            description=_PLUE_DESCRIPTION,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _PLUE_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "mnli":
            data_url = MNLI_URL
        elif self.config.name == "snli":
            data_url = SNLI_URL
        else:
            data_url = self.config.data_url

        dl_dir = dl_manager.download_and_extract(data_url)
        data_dir = os.path.join(dl_dir, self.config.data_dir)

        train_split = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "data_file": os.path.join(data_dir or "", "train.tsv"),
                "split": "train",
            },
        )
        if self.config.name == "mnli":
            return [
                train_split,
                _mnli_split_generator(
                    "validation_matched", data_dir, "dev", matched=True
                ),
                _mnli_split_generator(
                    "validation_mismatched", data_dir, "dev", matched=False
                ),
                _mnli_split_generator("test_matched", data_dir, "test", matched=True),
                _mnli_split_generator(
                    "test_mismatched", data_dir, "test", matched=False
                ),
            ]
        elif self.config.name == "mnli_matched":
            return [
                _mnli_split_generator("validation", data_dir, "dev", matched=True),
                _mnli_split_generator("test", data_dir, "test", matched=True),
            ]
        elif self.config.name == "mnli_mismatched":
            return [
                _mnli_split_generator("validation", data_dir, "dev", matched=False),
                _mnli_split_generator("test", data_dir, "test", matched=False),
            ]
        else:
            return [
                train_split,
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir or "", "dev.tsv"),
                        "split": "dev",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir or "", "test.tsv"),
                        "split": "test",
                    },
                ),
            ]

    def _generate_examples(self, data_file, split):
        if self.config.name in ["mrpc", "scitail"]:
            if self.config.name == "mrpc":
                examples = self._generate_example_mrpc_files(
                    data_file=data_file, split=split
                )
            elif self.config.name == "scitail":
                examples = self._generate_example_scitail_files(
                    data_file=data_file, split=split
                )

            for example in examples:
                yield example["idx"], example

        else:
            process_label = self.config.process_label
            label_classes = self.config.label_classes

            # The train and dev files for CoLA are the only tsv files without a
            # header.
            is_cola_non_test = self.config.name == "cola" and split != "test"

            with open(data_file, encoding="utf8") as f:
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                if is_cola_non_test:
                    reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

                for n, row in enumerate(reader):
                    if is_cola_non_test:
                        row = {
                            "sentence": row[3],
                            "is_acceptable": row[1],
                        }

                    example = {
                        feat: row[col]
                        for feat, col in self.config.text_features.items()
                    }
                    example["idx"] = n

                    if self.config.label_column in row:
                        label = row[self.config.label_column]
                        # For some tasks, the label is represented as 0 and 1 in the tsv
                        # files and needs to be cast to integer to work with the feature.
                        if label_classes and label not in label_classes:
                            label = int(label) if label else None
                        example["label"] = process_label(label)
                    else:
                        example["label"] = process_label(-1)

                    # Filter out corrupted rows.
                    for value in example.values():
                        if value is None:
                            break
                    else:
                        yield example["idx"], example

    def _generate_example_mrpc_files(self, data_file, split):
        print(data_file)

        with open(data_file, encoding="utf8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for idx, row in enumerate(reader):
                label = row["Quality"] if split != "test" else -1

                yield {
                    "sentence1": row["#1 String"],
                    "sentence2": row["#2 String"],
                    "label": int(label),
                    "idx": idx,
                }

    def _generate_example_scitail_files(self, data_file, split):
        with open(data_file, encoding="utf8") as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
                fieldnames=["premise", "hypothesis", "label"],
            )
            for idx, row in enumerate(reader):
                label = row["label"] if split != "test" else -1

                yield {
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": label,
                    "idx": idx,
                }


def _mnli_split_generator(name, data_dir, split, matched):
    return datasets.SplitGenerator(
        name=name,
        gen_kwargs={
            "data_file": os.path.join(
                data_dir, "%s_%s.tsv" % (split, "matched" if matched else "mismatched")
            ),
            "split": split,
        },
    )

