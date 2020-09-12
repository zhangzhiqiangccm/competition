#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: 01-predicate_data.py.py
@time: 2020/9/8 下午11:19
@desc:
'''
import os,json
data_dir = "/media/daiyizheng/SSD/data/CCKS/2020/ccks2020中文医学文本实体关系抽取/"
output_dir = "/home/daiyizheng/文档/NLP-competition/Entity-relationship-extraction-of-Chinese-medical-texts/input/predicate_classification/"
input_file_name = ["train_data.json", "val_data.json", "test1.json"]
output_file_name = ["train.txt", "dev.txt", "test.txt"]
max_length = 0
for index in range(len(input_file_name)):
    input_path = os.path.join(data_dir,input_file_name[index])
    output_path = os.path.join(output_dir, output_file_name[index])
    fw = open(output_path, "w", encoding="utf-8")
    countrow = 0
    with open(input_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    json_line = json.loads(line)
                    """
                    {'text': '另一种形式的IgA称为分泌型IgA（SIgA），存在于人的外分泌物中，如唾液、眼泪、肠内分泌物以及初乳中。',
                     'spo_list': [{'Combined': False, 'predicate': '同义词', 'subject': '分泌型IgA', 'subject_type': '疾病', 'object': {'@value': 'SIgA'}, 'object_type': {'@value': '疾病'}}]}
                    """
                    text = json_line["text"]
                    if len(text)>max_length:
                        max_length = len(text)
                    if input_file_name[index] != "test1.json":
                        spo_list = json_line['spo_list']
                        predicate_list = set()
                        for item in spo_list:
                            predicate_list.add(item['subject_type']+"|||"+item['predicate']+"|||"+item['object_type']['@value'])
                        predicate_str = " ".join(predicate_list)
                        fw.write(str(countrow)+"\t"+text+"\t"+predicate_str)
                    else:
                        fw.write(str(countrow)+"\t"+text)
                    countrow = countrow+1
                    fw.write("\n")
            except Exception as e:
                print(e)
    fw.close()
    print(input_file_name[index], "countrow:", countrow)
print("max_length:", max_length) ## 300


