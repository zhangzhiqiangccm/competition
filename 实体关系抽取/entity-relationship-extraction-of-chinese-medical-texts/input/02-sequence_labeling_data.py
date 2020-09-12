#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: 02-sequence_labeling_data.py
@time: 2020/9/9 下午6:21
@desc:
'''
import os,json

# 在列表 k 中确定列表 q 的位置
def _index_q_list_in_k_list(q_list, k_list):
    """Known q_list in k_list, find index(first time) of q_list in k_list"""
    q_list_length = len(q_list)
    k_list_length = len(k_list)
    for idx in range(k_list_length - q_list_length + 1):
        t = [q == k for q, k in zip(q_list, k_list[idx: idx + q_list_length])]
        # print(idx, t)
        if all(t):
            # print(idx)
            idx_start = idx
            return idx_start



# data_dir = "/media/daiyizheng/SSD/data/CCKS/2020/ccks2020中文医学文本实体关系抽取/"
# output_dir = "/home/daiyizheng/文档/NLP-competition/Entity-relationship-extraction-of-Chinese-medical-texts/input/subject_object_labeling/"
# input_file_name = ["train_data.json", "val_data.json", "test1.json"]
# output_file_name = ["train.txt", "dev.txt", "test.txt"]
# subject_object_labeling_error = "subject_object_labeling_error.txt"
# max_length = 0
# error_path = os.path.join(output_dir,subject_object_labeling_error)
# error_f = open(error_path, 'w', encoding="utf-8")
# for index in range(len(input_file_name)):
#     countrow = 0
#     input_path = os.path.join(data_dir, input_file_name[index])
#     output_path = os.path.join(output_dir, output_file_name[index])
#     error_f.write(input_file_name[index]+"\n")
#     data_f = open(output_path, 'w', encoding="utf-8")
#     with open(input_path, 'r', encoding="utf-8") as f:
#         for line in f.readlines():
#             line = line.strip()
#             if line:
#                 line_json = json.loads(line)
#                 text = line_json["text"].replace(" ", "")
#                 text_list = list(text)
#                 label_list = ["O"]*len(text_list)
#                 if input_file_name[index]!="test1.json":
#                     sop_list = line_json['spo_list']
#                     for item in sop_list:
#                         subject_ = item.get("subject", "")
#                         object_ = item.get("object", {}).get("@value","")
#                         if subject_ and object_:
#                             s_start_id = _index_q_list_in_k_list(q_list=list(subject_), k_list=list(text))
#                             o_start_id = _index_q_list_in_k_list(q_list=list(object_), k_list=list(text))
#                             if not s_start_id is None and not o_start_id is None:
#                                 label_list[o_start_id] = "B-" + "OBJ"
#                                 if len(object_)==2:
#                                     label_list[o_start_id+1] = "I-" + "OBJ"
#                                 else:
#                                     label_list[o_start_id + 1:o_start_id+len(object_)] = ["I-" + "OBJ"]*(len(object_)-1)
#
#                                 label_list[s_start_id] = "B-" + "SUB"
#                                 if len(subject_) == 2:
#                                     label_list[s_start_id + 1] = "I-" + "SUB"
#                                 else:
#                                     label_list[s_start_id + 1:s_start_id + len(subject_)] = ["I-" + "SUB"] * (len(subject_) - 1)
#                             else:
#                                 error_f.write("no start position" + subject_+" @@ "+object_ + " @@ " + text + "\n")
#                         else:
#                             error_f.write(line)
#                             continue
#                 assert len(text_list) == len(label_list), "文本长度与label长度不一致！"
#                 for idxs, item in enumerate(zip(text_list, label_list)):
#                     data_f.write(str(idxs) + "\t" + item[0] + "\t" + item[1] + "\n")
#                 data_f.write("\n")
#
#                 if len(text_list) > max_length:
#                     max_length = len(text_list)
#                 countrow += 1
#
#     data_f.close()
#     print(input_file_name[index], countrow)
#
# error_f.close()
# print("max_length:", max_length)

def generate_bio():
    data_dir = "/media/daiyizheng/SSD/data/CCKS/2020/ccks2020中文医学文本实体关系抽取/"
    output_dir = "/home/daiyizheng/文档/NLP-competition/Entity-relationship-extraction-of-Chinese-medical-texts/input/subject_object_labeling/"
    input_file_name = ["train_data.json", "val_data.json", "test1.json"]
    output_file_name = ["train.txt", "dev.txt", "test.txt"]
    subject_object_labeling_error = "subject_object_labeling_error.txt"
    max_length = 0
    error_path = os.path.join(output_dir,subject_object_labeling_error)
    error_f = open(error_path, 'w', encoding="utf-8")
    for index in range(len(input_file_name)):
        countrow = 0
        input_path = os.path.join(data_dir, input_file_name[index])
        output_path = os.path.join(output_dir, output_file_name[index])
        error_f.write(input_file_name[index]+"\n")
        data_f = open(output_path, 'w', encoding="utf-8")
        with open(input_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    line_json = json.loads(line)
                    text = line_json["text"].replace(" ", "")
                    text_list = list(text)
                    if input_file_name[index]!="test1.json":
                        sop_list = line_json['spo_list']
                        for item in sop_list:
                            label_list = ["O"] * len(text_list)
                            subject_ = item.get("subject", "")
                            object_ = item.get("object", {}).get("@value","")
                            predicate = item.get("predicate", "")
                            subject_type = item.get("subject_type", "")
                            object_type = item.get("object_type", {}).get("@value","")
                            predicate_list = list(subject_type) + list(predicate) + list(object_type)
                            if subject_ and object_:
                                s_start_id = _index_q_list_in_k_list(q_list=list(subject_), k_list=list(text))
                                o_start_id = _index_q_list_in_k_list(q_list=list(object_), k_list=list(text))
                                if not s_start_id is None and not o_start_id is None:
                                    label_list[o_start_id] = "B-" + "OBJ"
                                    if len(object_)==2:
                                        label_list[o_start_id+1] = "I-" + "OBJ"
                                    else:
                                        label_list[o_start_id + 1:o_start_id+len(object_)] = ["I-" + "OBJ"]*(len(object_)-1)
                                    label_list[s_start_id] = "B-" + "SUB"
                                    if len(subject_) == 2:
                                        label_list[s_start_id + 1] = "I-" + "SUB"
                                    else:
                                        label_list[s_start_id + 1:s_start_id + len(subject_)] = ["I-" + "SUB"] * (len(subject_) - 1)
                                    data_line = str(countrow)+"\t"+ " ".join(text_list)+"\t"+" ".join(label_list) +"\t"+" ".join(predicate_list)
                                    countrow+=1
                                    if (len(text_list)+len(predicate_list))>max_length:
                                        max_length = (len(text_list)+len(predicate_list))
                                    data_f.write(data_line+"\n")
                                else:
                                    error_f.write("no start position" + subject_+" @@ "+object_ + " @@ " + text + "\n")

                            else:
                                error_f.write(line)
                                continue
                    else:
                        data_line = str(countrow) + "\t" + " ".join(text_list)
                        countrow += 1
                        if (len(text_list) + len(predicate_list)) > max_length:
                            max_length = (len(text_list) + len(predicate_list))
                        data_f.write(data_line + "\n")
        data_f.close()
        print(input_file_name[index], countrow)
    error_f.close()
    print("max_length:", max_length) ## 305
generate_bio()
# with open(".//subject_object_labeling/train.txt") as f:
#     a = f.readlines()
#     for line in a:
#         line = line.strip()
#         if line:
#             line_list = line.split("\t")
#             if 4 !=len(line_list):
#                 print(line)