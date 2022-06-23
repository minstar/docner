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

def token_frequency(token_freq_dict, words, labels):
    for word in words:
        if word not in token_freq_dict:
            token_freq_dict[word] = 0
        token_freq_dict[word] += 1
    return token_freq_dict

def entity_frequency(entity_freq_dict, entity_density_list, words, labels):
    entity = ""
    in_entity_freq_dict = []
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
                try:
                    if labels[idx+1] == 0 or labels[idx+1] == 1:
                        entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                        in_entity_freq_dict, entity = cnt_function(in_entity_freq_dict, entity)
                    else:
                        continue
                except:
                    entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                    in_entity_freq_dict, entity = cnt_function(in_entity_freq_dict, entity)
            else:
                entity += word + " "
                try:
                    if labels[idx+1] != 0:
                        continue
                    else:
                        entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                        in_entity_freq_dict, entity = cnt_function(in_entity_freq_dict, entity)
                except:
                    entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                    in_entity_freq_dict, entity = cnt_function(in_entity_freq_dict, entity)
                
    # get a density of entity per document
    doc_len = len(words)
    temp_entity_length_dict = {key:len(key.split()) for key in in_entity_freq_dict.keys()}
    cnt_len = 0
    for val in temp_entity_length_dict.values():
        cnt_len += val
    entity_density = cnt_len / doc_len
    entity_density_list.append(entity_density)
    return entity_freq_dict, entity_density_list

def token_consistency():
    return None

def entity_length(entity_freq_dict):
    entity_len_dict = {key:len(key.split()) for key in entity_freq_dict.keys()}
    return entity_len_dict

def document_length(doc_len, words):
    doc_len.append(len(words))
    return doc_len

def out_of_density():
    return None

def main():
    data_dir = '/ssd1/minbyul/docner/data/low-resource'
    entity_list = ['ncbi-disease', 'bc5cdr', 'anatem', 'gellus']
    file_list = ['doc_train.json', 'doc_dev.json', 'doc_test.json']
    # entity_name = 'ncbi-disease' 
    for entity_name in entity_list:
        for file_name in file_list:
            with open(data_dir+'/'+entity_name+'/'+'from_rawdata'+'/'+file_name, 'r') as fp:
                data = json.load(fp)
                token_freq_dict = {}
                entity_freq_dict = {}
                doc_len = []
                entity_density_list = []
                for data_idx, data_inst in tqdm(enumerate(data), desc='Total Run'):
                    words = data_inst['str_words']
                    labels = data_inst['tags']
                    
                    token_freq_dict = token_frequency(token_freq_dict, words, labels)
                    entity_freq_dict, entity_density_list = entity_frequency(entity_freq_dict, entity_density_list, words, labels)
                    doc_len = document_length(doc_len, words)
                entity_len_dict = entity_length(entity_freq_dict)
                    
if __name__ == "__main__":
    main()
