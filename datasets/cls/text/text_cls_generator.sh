#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


SAVE_DIR='/home/donny/DataSet/Text'
WORD2VEC_MODEL='/home/donny/DataSet/word2vec/GoogleNews-vectors-negative300.bin'
SENTENCE_FILE='/home/donny/DataSet/stanfordSentimentTreebank/datasetSentences.txt'
SPLIT_FILE='/home/donny/DataSet/stanfordSentimentTreebank/datasetSplit.txt'
DICT_FILE='/home/donny/DataSet/stanfordSentimentTreebank/dictionary.txt'
SENTIMENT_FILE='/home/donny/DataSet/stanfordSentimentTreebank/sentiment_labels.txt'


python2.7 text_cls_generator.py --save_dir $SAVE_DIR \
                                --vec_model $WORD2VEC_MODEL \
                                --sentence_file $SENTENCE_FILE \
                                --split_file $SPLIT_FILE \
                                --dict_file $DICT_FILE \
                                --sentiment_file $SENTIMENT_FILE
