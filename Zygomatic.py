'''
Author: letianyu letianyu@tencent.com
Date: 2024-11-28 13:30:12
LastEditors: letianyu letianyu@tencent.com
LastEditTime: 2024-11-29 19:27:43
FilePath: \weibo-crawler\Zygomatic.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import json
import weibo
import csv
import re
from text2vec import SentenceModel
#from difflib import get_close_matches
#from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
#import torch
#from simcse import SimCSE
import numpy as np
import weibo_follow
import progressbar
import time

# 加载 BERT 模型和分词器
#tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
#model = BertModel.from_pretrained("bert-base-chinese")
#simcse_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
model = SentenceModel("shibing624/text2vec-base-chinese", device="cpu")

cfg_file_path = "config.json"
crawled_users = "./weibo/users.csv"
output_searchedPages_path = "./weibo/searchedPages.csv"
output_userid_path = "./weibo/searchedUsers.txt"
output_nosearched_users = "./weibo/noSearchedUsers.txt"
output_graph_path = "./weibo/searchedGraph.txt"
init_userid = "5576424769"
avoid_bigv = True
#init_userid = "7808280970"
crawle_key_words = ["颧骨","颧骨内推","下垂","垂","pz","zzy","gj","郭军","颧弓","ljq","李金清","轮廓整形","颧骨手术","郑康在"]

relevant_threshold = 0.7
follower_tracking_threshold = 2

def write_array_to_txt_incremental(file_name, array):
    """
    将数组写入 TXT 文件，增量式写入（避免重复）。
    :param file_name: str, 输出的 TXT 文件名
    :param array: list, 要写入的数组，每个元素表示一行
    """
    # 检查输入数据
    if not array:
        print("输入数组为空，未写入任何内容。")
        return

    # 检查文件是否存在
    existing_items = set()
    if os.path.exists(file_name):
        # 读取现有文件内容到集合中
        with open(file_name, mode="r", encoding="utf-8") as file:
            existing_items = set(line.strip() for line in file)

    # 过滤掉已存在的元素
    new_items = [item for item in array if item not in existing_items]

    if not new_items:
        print("没有新数据需要写入，文件未更新。")
        return

    # 追加写入新数据
    with open(file_name, mode="a", encoding="utf-8") as file:
        for item in new_items:
            file.write(f"{item}\n")

def write_array_to_txt(file_name, array):
    # 使用 'w' 模式覆盖写入
    with open(file_name, mode="w", encoding="utf-8") as file:
        for item in array:
            file.write(f"{item}\n")

def write_page_incremental(file_name, data):
    dict_page_path = file_name.replace(".csv", ".txt")
    existing_page_list = []
    if not os.path.exists(dict_page_path):
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                page_list_data = list(reader)
            for row in page_list_data:
                if row["bid"].strip() not in existing_page_list:
                    existing_page_list.append(row["bid"].strip())
    else:
        with open(dict_page_path, 'r') as file:
            lines = file.readlines()
        for li in lines:
            existing_page_list.append(li.strip())
    valid_pages = []
    for row in data:
        if row["bid"].strip() not in existing_page_list:
            valid_pages.append(row)
            existing_page_list.append(row["bid"].strip())
    write_array_to_txt(dict_page_path, existing_page_list)
    write_dict_to_csv_incremental(file_name, valid_pages)

def write_dict_to_csv_incremental(file_name, data):
    """
    将字典列表写入 CSV 文件，增量式写入（避免重复）。
    :param file_name: str, 输出的 CSV 文件名
    :param data: list, 字典列表，每个字典表示一行数据
    """
    if len(data) == 0:
        print("输入数据为空，未写入任何内容。")
        return
    
    # 提取表头
    fieldnames = data[0].keys()

    # 检查文件是否存在
    existing_data = []
    if os.path.exists(file_name):
        # 读取现有文件内容
        with open(file_name, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_data = [row for row in reader]
    
    # 过滤掉已存在的数据
    filtered_data = [row for row in data if row not in existing_data]

    if len(filtered_data) == 0:
        print("没有新数据需要写入，文件未更新。")
        return

    # 写入新数据（追加模式）
    with open(file_name, mode="a" if existing_data else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # 如果文件之前不存在，则写入表头
        if not existing_data:
            writer.writeheader()
        
        # 写入新数据行
        writer.writerows(filtered_data)

# 清理文本中的无用符号
def clean_text(text):
    text = re.sub(r"[\n\r\t]", " ", text)  # 替换换行符和制表符为空格
    text = re.sub(r"\s+", " ", text)       # 合并多余的空格
    text = text.strip()                    # 去掉首尾空格
    return text

# 拆分长句为子句
def split_sentences(text):
    text = clean_text(text)
    sentences = re.split(r"[，。！？；：,.!?]", text)
    return [s.strip() for s in sentences if s.strip()]

# 主函数：匹配关键词并计算相似度
def match_sentences_with_keywords(text, candidates):
    # 拆分子句
    total_score = 0
    sub_sentences = split_sentences(text)
    sub_vecs = model.encode(sub_sentences)  # 子句向量
    candidate_vecs = model.encode(candidates)  # 候选词向量

    # 初始化候选词最大相似度为 0
    max_scores = np.zeros(len(candidates))
    for sub_vec in sub_vecs:
        # 计算子句与候选词的相似度
        cosine_sim = cosine_similarity([sub_vec], candidate_vecs).flatten()
        max_scores = np.maximum(max_scores, cosine_sim)  # 更新最大相似度

    # 再次遍历子句，检查是否包含关键词
    final_scores = []
    for i, candidate in enumerate(candidates):
        # 默认相似度为 0
        score = 0
        for sub_sentence in sub_sentences:
            if candidate in sub_sentence:  # 如果子句包含关键词
                score = max_scores[i]
                break
        final_scores.append((candidate, score))
        total_score = total_score + score

    # 按相似度排序
    results = sorted(final_scores, key=lambda x: x[1], reverse=True)
    return results, total_score

# 编码文本为向量
def vector_encode(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings

def SetupConfig(user_id, need_cookie):
    with open(cfg_file_path, encoding="utf-8") as f:
        crawle_cfg = json.loads(f.read())
    crawle_cfg["user_id_list"] = [user_id]
    if need_cookie :
        crawle_cfg["cookie"] = crawle_cfg["cookie_temp"]
    else:
        crawle_cfg["cookie"] = "your cookie"
    with open(cfg_file_path, "w") as file:
        json.dump(crawle_cfg, file, indent=4)

def RunCrawler(user_id):
    print("Set User To Crawle :" + user_id)
    SetupConfig(user_id, False)
    print("Run Crawler")
    weibo.main()
    print("Crawle End for :" + user_id)

def SearchFollower(user_id):
    SetupConfig(user_id, True)
    got_follower = weibo_follow.main()
    ret_arr = []
    try:
        follower_pack = got_follower[0]["data"]
        for item in follower_pack:
            if item["uri"] not in ret_arr:
                ret_arr.append(item["uri"])
    except TypeError:
        pass
    return ret_arr

def AnalyseID(user_id, keywords):
    out_ids = []
    out_pages = []
    track_followers = True
    if not os.path.exists(crawled_users):
        return out_ids, out_pages, track_followers
    with open(crawled_users, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        user_list_data = list(reader)
    # 逐行读取内容
    target_row = ""
    for row in user_list_data:
        if row["\ufeff用户id"] == user_id:
            target_row = row
    if target_row == "":
        return out_ids, out_pages, track_followers
    print("Analyse User:" + target_row["昵称"])
    target_csv_path = "./weibo/" + target_row["昵称"] + "/" + user_id + ".csv"
    target_json_path = "./weibo/" + target_row["昵称"] + "/" + user_id + ".json"
    if not os.path.exists(target_csv_path):
        return out_ids, out_pages, track_followers
    if not os.path.exists(target_json_path):
        return out_ids, out_pages, track_followers

    with open(target_csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        user_data = list(reader)

    with open(target_json_path, encoding="utf-8") as f:
        user_cfg = json.loads(f.read())

    if avoid_bigv:
        track_followers = not user_cfg['user']["verified"]
    if track_followers:
        for row in user_data:
            main_text = str(row["源微博正文"]) + "。" + str(row["正文"])
            if main_text != "":
                results, total_score = match_sentences_with_keywords(main_text, keywords)
                if total_score > relevant_threshold :
                    row_to_writeout = row
                    row_to_writeout["uid"] = user_id
                    out_pages.append(row_to_writeout)
                    for candidate, score in results:
                        print(f"- {candidate}: 最大相似度 {score:.4f}")
                    print(main_text + "\n\n")
                    if row["是否原创"] == "False":
                        if row["源用户id"] not in out_ids:
                            out_ids.append(row["源用户id"])
    else:
        print("Jumpover BigV")
    print("Analyse End for User:" + target_row["昵称"])
    return out_ids, out_pages, track_followers

def load_or_init():
    ran_users = []
    new_users = []
    graph_datas = []
    if os.path.exists(output_userid_path):
        with open(output_userid_path, 'r') as file:
            lines = file.readlines()
        for li in lines:
            ran_users.append(li.strip())

    if os.path.exists(output_nosearched_users):
        with open(output_nosearched_users, 'r') as file:
            lines = file.readlines()
        for li in lines:
            new_users.append(li.strip())
    else:
        new_users.append(init_userid)

    if os.path.exists(output_graph_path):
        with open(output_graph_path, 'r') as file:
            lines = file.readlines()
        for li in lines:
            graph_datas.append(li.replace("\n",""))
    return ran_users, new_users, graph_datas

def crawle_and_analyse():
    ran_users, new_users, graph_datas = load_or_init()
    bar = progressbar.ProgressBar(max_value=100)
    while len(new_users) > 0:
        user_id = new_users.pop(0)
        bar.update((1.0 - (len(new_users)/(len(new_users) + len(ran_users))))*100)
        if user_id.isdigit():
            if user_id not in ran_users:
                graph_datas.append("node:" + user_id)
                RunCrawler(user_id)
                analyse_ids, analyse_pages, analyse_track_follower = AnalyseID(user_id, crawle_key_words)
                new_users.extend(analyse_ids)
                for it in analyse_ids:
                    graph_datas.append("redirect:" + user_id + " " + it)
                ran_users.append(user_id)
                write_page_incremental(output_searchedPages_path, analyse_pages)
                write_array_to_txt_incremental(output_userid_path, ran_users)
                if len(analyse_pages) > follower_tracking_threshold :
                    if analyse_track_follower:
                        get_follower = SearchFollower(user_id)
                        new_users.extend(get_follower)
                        for it in get_follower :
                            graph_datas.append("follower:" + it + " " + user_id)
        write_array_to_txt(output_nosearched_users, new_users)
        write_array_to_txt_incremental(output_graph_path, graph_datas)
    print("Done!!")

crawle_and_analyse()
