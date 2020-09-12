#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: 00-calculate_number_of_classification.py
@time: 2020/9/9 上午12:24
@desc:
'''
import os, json
data_dir = "/Users/daiyizheng/Documents/my-project/entity-relationship-extraction-of-chinese-medical-texts/corpus"
output_dir = "./predicate_classification/"
file_name = "53_schemas.json"
output_name = "calssification_labels.txt"
input_path = os.path.join(data_dir, file_name)
output_path = os.path.join(output_dir, output_name)
predicate = set()
with open(input_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line =line.strip()
        if line:
            line_json = json.loads(line)
            predicate.add(line_json['subject_type']+"|||"+line_json['predicate']+"|||"+line_json['object_type'])
with open(output_path, "w", encoding="utf-8") as fw:
    fw.write("\n".join(list(predicate)))