# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
import sys
import copy
import math
import json
import glob
import torch
import wandb
import random
import logging
import argparse
import numpy as np

from tqdm import tqdm, trange

def token_frequency(token_freq_dict, token_cons_dict, words, labels):
    temp_token_cons_dict = {}
    for word in words:
        if word not in token_freq_dict:
            token_freq_dict[word] = 0
        token_freq_dict[word] += 1

    for word, label in zip(words, labels):
        if word not in temp_token_cons_dict:
            temp_token_cons_dict[word] = {0:0, 1:0, 2:0, 3:0, 4:0}
        temp_token_cons_dict[word][label] += 1

    for key, val in temp_token_cons_dict.items():
        if key not in token_cons_dict:
            token_cons_dict[key] = []

        total_cnt = 0
        temp_list = []
        for label_id, label_cnt in val.items():
            total_cnt += label_cnt

        temp_cnt, temp_cnt2 = 0, 0
        for label_id, label_cnt in val.items():
            if label_id == 0:
                temp_list.append(label_cnt / total_cnt)
            elif label_id == 1:
                temp_cnt += label_cnt
            elif label_id == 2:
                temp_cnt += label_cnt
                temp_list.append(temp_cnt / total_cnt)
            elif label_id == 3:
                temp_cnt2 += label_cnt
            elif label_id == 4:
                temp_cnt2 += label_cnt
                temp_list.append(temp_cnt2 / total_cnt)
            
        token_cons_dict[key].append(temp_list)

    return token_freq_dict, token_cons_dict

def entity_frequency(entity_freq_dict, entity_density_list, entity_cons_dict, words, labels):
    entity = ""
    in_entity = ""
    in_entity_freq_dict = {}
    def cnt_function(entity_freq_dict, entity):
        entity = entity.strip()
        if entity not in entity_freq_dict:
            entity_freq_dict[entity] = 0
        entity_freq_dict[entity] += 1
        entity = ""
        return entity_freq_dict, entity

    for idx, (word, label) in enumerate(zip(words, labels)):
        if label != 0:
            if label % 2 == 1:
                entity += word + " "
                in_entity += word + " "
                try:
                    if labels[idx+1] == 0 or labels[idx+1] == 1:
                        entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                        in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
                    else:
                        continue
                except:
                    entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                    in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
            else:
                entity += word + " "
                in_entity += word + " "
                try:
                    if labels[idx+1] != 0:
                        continue
                    else:
                        entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                        in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
                except:
                    entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                    in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
                
    # get a density of entity per document
    doc_len = len(words)
    temp_entity_length_dict = {key:len(key.split()) for key in in_entity_freq_dict.keys()}
    cnt_len = 0
    for val in temp_entity_length_dict.values():
        cnt_len += val
    entity_density = cnt_len / doc_len
    entity_density_list.append(entity_density)

    # get a consistency of entity per document
    
    sentence = " ".join([word for word in words])
    for key in in_entity_freq_dict.keys():
        orig_key = copy.deepcopy(key)
        specialChars = '~`!@#$%^&*()_-+=[{]}:;\'",<.>/?|'
        for char in specialChars:
            key = key.replace(char, '\%c'%char)

        key_list = re.findall(key, sentence)
        if orig_key not in entity_cons_dict:
            entity_cons_dict[orig_key] = []

        entity_cons_dict[orig_key].append(in_entity_freq_dict[orig_key] / len(key_list))

    return entity_freq_dict, entity_density_list, entity_cons_dict

def entity_length(entity_freq_dict):
    entity_len_dict = {key:len(key.split()) for key in entity_freq_dict.keys()}
    return entity_len_dict

def document_length(doc_len, words):
    doc_len.append(len(words))
    return doc_len

def out_of_density(train_entity_freq_dict, test_entity_freq_dict, out_of_dens_dict, test_doc_len, test_data):
    train_entity_set = set([key for key in train_entity_freq_dict.keys()])
    test_entity_set = set([key for key in test_entity_freq_dict.keys()])

    diff_set = train_entity_set - test_entity_set

    for data_idx, data_inst in tqdm(enumerate(test_data), desc='Out of Density'):
        words = data_inst['str_words']
        labels = data_inst['tags']

        sentence = " ".join([word for word in words])

        for key in list(diff_set):
            orig_key = copy.deepcopy(key)
            specialChars = '~`!@#$%^&*()_-+=[{]}:;\'",<.>/?|'
            for char in specialChars:
                key = key.replace(char, '\%c'%char)

            key_list = re.findall(key, sentence)

            if key_list:
                if key not in out_of_dens_dict:
                    out_of_dens_dict[key] = []
                out_of_dens_dict[key].append(len(key.split()) / test_doc_len[data_idx])

    return out_of_dens_dict

def main():
    data_dir = '/ssd1/minbyul/docner/data/low-resource'
    entity_list = ['ncbi-disease', 'bc5cdr', 'anatem', 'gellus']
    file_list = ['doc_train.json', 'doc_dev.json', 'doc_test.json']
    # entity_name = 'ncbi-disease' 
    for entity_name in entity_list:
        out_of_dens_dict = {}
        for file_name in file_list:
            with open(data_dir+'/'+entity_name+'/'+'from_rawdata'+'/'+file_name, 'r') as fp:
                data = json.load(fp)
                token_freq_dict, entity_freq_dict = {}, {}
                token_cons_dict, entity_cons_dict = {}, {}
                doc_len,entity_density_list = [], []

                for data_idx, data_inst in tqdm(enumerate(data), desc='Total Run'):
                    words = data_inst['str_words']
                    labels = data_inst['tags']
                    
                    token_freq_dict, token_cons_dict = token_frequency(token_freq_dict, token_cons_dict, words, labels)
                    entity_freq_dict, entity_density_list, entity_cons_dict = entity_frequency(entity_freq_dict, entity_density_list, entity_cons_dict, words, labels)
                    doc_len = document_length(doc_len, words)

                entity_len_dict = entity_length(entity_freq_dict)

            if 'train' in file_name:
                train_entity_freq_dict = copy.deepcopy(entity_freq_dict)
                
            if 'test' in file_name:
                test_data = copy.deepcopy(data)
                test_entity_freq_dict = copy.deepcopy(entity_freq_dict)
                test_doc_len = copy.deepcopy(doc_len)

        # get out of density through a set of training entites 
        out_of_dens_dict = out_of_density(train_entity_freq_dict, test_entity_freq_dict, out_of_dens_dict, test_doc_len, test_data)
        

                    
if __name__ == "__main__":
    main()
