#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# NLP test data generator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import nltk
import json
import numpy as np
from gensim.models.word2vec import Word2Vec as w


IMAGE_DIR = 'image'


class TextClsGenerator(object):

    def __init__(self, args, image_dir=IMAGE_DIR):
        self.args = args
        self.model = w.load_word2vec_format(self.args.vec_model, binary=True)
        self.train_lines = list()
        self.train_labels = list()
        self.val_lines = list()
        self.val_labels = list()
        self.test_lines = list()
        self.test_labels = list()
        self.text_score_dict = dict()

        self.train_json_file = os.path.join(self.args.save_dir, 'train/label.json')
        self.val_json_file = os.path.join(self.args.save_dir, 'val/label.json')
        self.test_json_file = os.path.join(self.args.save_dir, 'test/label.json')
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        self.train_image_dir = os.path.join(self.args.save_dir, 'train', image_dir)
        self.val_image_dir = os.path.join(self.args.save_dir, 'val', image_dir)
        self.test_image_dir = os.path.join(self.args.save_dir, 'test', image_dir)
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)

        if not os.path.exists(self.test_image_dir):
            os.makedirs(self.test_image_dir)

    def _get_text_score_dict(self):
        dict_stream = open(self.args.dict_file, 'r')
        sentiment_stream = open(self.args.sentiment_file, 'r')
        lines1 = dict_stream.readlines()
        lines2 = sentiment_stream.readlines()
        id_text = {}
        for i in range(0, len(lines1)):
            s = lines1[i].strip().split("|")
            id_text[s[1]] = s[0]

        for j in range(1, len(lines2)):
            k = lines2[j].strip().split("|")
            if k[0] in id_text:
                self.text_score_dict[id_text[k[0]]] = float(k[1])

        dict_stream.close()
        sentiment_stream.close()

    def _get_split_data(self, sentences_file, split_file):
        raw_train_lines = list()
        raw_val_lines= list()
        raw_test_lines = list()
        sentence_stream = open(sentences_file, 'r')
        split_stream = open(split_file, 'r')
        lines1 = sentence_stream.readlines()
        lines2 = split_stream.readlines()
        assert len(lines1) == len(lines2)
        for i in range(1, len(lines1)):
            t1 = lines1[i].replace("-LRB-", "(")
            t2 = t1.replace("-RRB-", ")")
            k = lines2[i].strip().split(",")
            t = t2.strip().split('\t')
            if k[1] == '1':
                raw_train_lines.append(t[1])
            elif k[1] == '2':
                raw_val_lines.append(t[1])
            elif k[1] == '3':
                raw_test_lines.append(t[1])

        return raw_train_lines, raw_val_lines, raw_test_lines

    def _score_to_label(self, score):
        if score >= 0.0 and score <= 0.2:
            return 0

        elif score <= 0.4:
            return 1

        elif score <= 0.6:
            return 2

        elif score <= 0.8:
            return 3

        elif score <= 1.0:
            return 4

        else:
            print('Score Error: '.format(score))
            exit(1)

    def _get_label(self):
        self._get_text_score_dict()
        raw_train_lines, raw_val_lines, raw_test_lines = self._get_split_data(self.args.sentences_file,
                                                                              self.args.split_file)
        for line in raw_train_lines:
            if line in self.text_score_dict:
                self.train_lines.append(line)
                self.train_labels.append(self._score_to_label(self.text_score_dict[line]))

        for line in raw_val_lines:
            if line in self.text_score_dict:
                self.val_lines.append(line)
                self.val_labels.append(self._score_to_label(self.text_score_dict[line]))

        for line in raw_test_lines:
            if line in self.text_score_dict:
                self.test_lines.append(line)
                self.test_labels.append(self._score_to_label(self.text_score_dict[line]))

    def _filter_words(self, words):
        if words is None:
            return []

        return [word for word in words if word in self.model.vocab]

    def _text2vec(self, text):
        valid_words = self._filter_words(nltk.word_tokenize(text.lower()))
        print(valid_words)
        vec_list = []
        for word in valid_words:
            word_vec = self.model[word]
            vec_list.append(word_vec)

        return vec_list

    def generate_label(self):
        self._get_label()
        train_json_list = list()
        val_json_list = list()
        test_json_list = list()

        item_count = 1
        for line, label in zip(self.train_lines, self.train_labels):
            item_dict = dict()
            line_vec_list = self._text2vec(line)
            if len(line_vec_list) <= 0:
                continue

            np.save(os.path.join(self.train_image_dir, '{}.npy'.format(str(item_count).zfill(6))),
                    np.array(line_vec_list))

            item_dict['image_path'] = '{}/{}.npy'.format(IMAGE_DIR, str(item_count).zfill(6))
            item_dict['label'] = label
            item_dict['text'] = line
            item_dict['word_count'] = len(line_vec_list)
            train_json_list.append(item_dict)
            item_count += 1

        item_count = 1
        for line, label in zip(self.val_lines, self.val_labels):
            item_dict = dict()
            line_vec_list = self._text2vec(line)
            if len(line_vec_list) <= 0:
                continue

            np.save(os.path.join(self.val_image_dir, '{}.npy'.format(str(item_count).zfill(6))),
                    np.array(line_vec_list))

            item_dict['image_path'] = '{}/{}.npy'.format(IMAGE_DIR, str(item_count).zfill(6))
            item_dict['label'] = label
            item_dict['text'] = line
            item_dict['word_count'] = len(line_vec_list)
            val_json_list.append(item_dict)
            item_count += 1

        item_count = 1
        for line, label in zip(self.test_lines, self.test_labels):
            item_dict = dict()
            line_vec_list = self._text2vec(line)
            if len(line_vec_list) <= 0:
                continue

            np.save(os.path.join(self.test_image_dir, '{}.npy'.format(str(item_count).zfill(6))),
                    np.array(line_vec_list))

            item_dict['image_path'] = '{}/{}.npy'.format(IMAGE_DIR, str(item_count).zfill(6))
            item_dict['text'] = line
            item_dict['label'] = label
            item_dict['word_count'] = len(line_vec_list)
            test_json_list.append(item_dict)
            item_count += 1

        fw = open(self.train_json_file, 'w')
        fw.write(json.dumps(train_json_list))
        fw.close()

        fw = open(self.val_json_file, 'w')
        fw.write(json.dumps(val_json_list))
        fw.close()

        fw = open(self.test_json_file, 'w')
        fw.write(json.dumps(test_json_list))
        fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--vec_model', default=None, type=str,
                        dest='vec_model', help='The directory of the image data.')
    parser.add_argument('--dict_file', default=None, type=str,
                        dest='dict_file', help='The dictionary of treebank.')
    parser.add_argument('--sentiment_file', default=None, type=str,
                        dest='sentiment_file', help='The sentiment labels of treebank.')
    parser.add_argument('--sentence_file', default=None, type=str,
                        dest='sentences_file', help='The sentences of treebank.')
    parser.add_argument('--split_file', default=None, type=str,
                        dest='split_file', help='The train, val, test split of treebank.')

    args = parser.parse_args()

    text_cls_generator = TextClsGenerator(args)

    text_cls_generator.generate_label()
