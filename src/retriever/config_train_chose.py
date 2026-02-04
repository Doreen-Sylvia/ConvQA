# config_train_chose.py
import torch
import logging
import math
import os
import numpy as np
import scipy.sparse as ss
from collections import defaultdict

log = logging.getLogger()

class Config(object):
    def __init__(self, args=None):
        # 基本路径配置
        self.entity_dict_path = '../../data/preprocessed/dialogue_kg/entities.dict'
        self.relation_dict_path = '../../data/preprocessed/dialogue_kg/relations.dict'
        self.kg_path = '../../data/preprocessed/dialogue_kg/train.txt'
        self.entity_emb_path = '../../models/kg_embeddings/preprocess/dialogue_kg/E.npy'
        self.relation_emb_path = '../../models/kg_embeddings/preprocess/dialogue_kg/E.npy'

        # 数据路径
        self.conversation_path = 'D:/Project/ConvQA/data/merged_dialogues/comprehensive_merged_dialogues.json'
        self.conversation_valid_path = 'D:/Project/ConvQA/data/merged_dialogues/valid.json'
        self.conversation_test_path = 'D:/Project/ConvQA/data/merged_dialogues/test.json'

        # 训练参数
        self.batch_size = 12
        self.test_batch_size = 8
        self.sample_strategy = 'kmeans--'
        self.sample_size = 300000
        self.cluster_num = 5
        self.num_workers = 4

        # 主动学习参数
        self.al_epochs = 20
        self.active_round = 3

        # 模型参数
        self.embedding_dim = 200
        self.freeze = False
        self.device = 0
        self.gpu = self.device
        self.entdrop = 0.0
        self.reldrop = 0.0
        self.scoredrop = 0.0
        self.l3_reg = 0.001
        self.ls = 0.05
        self.do_batch_norm = 1
        self.use_cuda = True
        self.decay = 1.0
        self.lr = 0.00002

        # 其他配置
        self.model_save_path = "./src_reformulate/best_model_3E_fuxian.pt"
        self.save_model = True
        self.load_model = False
        self.load_lstm_model = True
        self.parallel = False
        self.reranking = False

        # 从args更新配置
        if args:
            for key, value in vars(args).items():
                if value is not None:
                    setattr(self, key, value)

        # 加载实体和关系字典
        self.entity2id, self.id2entity = self.read_dict(self.entity_dict_path)
        self.relation2id, self.id2relation = self.read_dict(self.relation_dict_path)
        self.adj_list = self.get_adj_map_with_rel()
        self.num_entities = len(self.entity2id)

        # 加载关系映射
        self.ht2relation = self.get_h_t_to_rel()
        self.entity2neighbor_relation = self.get_entity2neighbor_relation()

        # 主题感知配置
        self.use_topic_aware = getattr(args, 'use_topic_aware', False)
        self.topic_model_type = getattr(args, 'topic_model_type', 'bertopic')
        self.num_topics = getattr(args, 'num_topics', 15)

    def read_dict(self, file_path):
        name2id = {}
        id2name = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                if len(line) >= 2:
                    n = line[0]  # 实体名称在第一列
                    id = line[1]  # ID在第二列
                    try:
                        name2id[n] = int(id)
                        id2name[int(id)] = n
                    except ValueError:
                        print(f"警告: 无法解析ID '{id}' 对应实体 '{n}'")
                        continue
        return name2id, id2name


    def get_adj_map_with_rel(self):
        """实现邻接映射逻辑"""
        adj_map = {}
        try:
            with open(self.kg_path, 'r') as f:
                for line in f.readlines():
                    ws = line.strip().split("\t")
                    if len(ws) < 3:
                        continue

                    h = ws[0]
                    r = ws[1]
                    t = ws[2]

                    if h not in adj_map:
                        adj_map[h] = {}
                    if t not in adj_map:
                        adj_map[t] = {}

                    if r not in adj_map[h]:
                        adj_map[h][r] = set()
                    if r not in adj_map[t]:
                        adj_map[t][r] = set()

                    adj_map[h][r].add(t)
                    adj_map[t][r].add(h)
        except Exception as e:
            print(f"Error reading adjacency map: {e}")

        return adj_map

    def get_h_t_to_rel(self):
        """实现头尾实体到关系的映射逻辑"""
        ht2relation = {}
        try:
            with open(self.kg_path, 'r') as f:
                for line in f.readlines():
                    ws = line.strip().split("\t")
                    if len(ws) < 3:
                        continue

                    h = ws[0]
                    r = ws[1]
                    t = ws[2]

                    key1 = (h, t)
                    key2 = (t, h)

                    if key1 not in ht2relation:
                        ht2relation[key1] = []
                    if key2 not in ht2relation:
                        ht2relation[key2] = []

                    ht2relation[key1].append(r)
                    ht2relation[key2].append(r)
        except Exception as e:
            print(f"Error reading h-t to relation map: {e}")

        return ht2relation

    def get_entity2neighbor_relation(self):
        """实现实体到邻居关系的映射逻辑"""
        entity2neighbor = defaultdict(set)
        try:
            with open(self.kg_path, 'r') as f:
                for line in f.readlines():
                    ws = line.strip().split("\t")
                    if len(ws) < 3:
                        continue

                    h = ws[0]
                    r = ws[1]
                    t = ws[2]

                    entity2neighbor[h].add(r)
                    entity2neighbor[t].add(r)
        except Exception as e:
            print(f"Error reading entity to neighbor relation map: {e}")

        return entity2neighbor
