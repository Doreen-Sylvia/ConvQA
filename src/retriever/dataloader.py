import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from tqdm.utils import RE_ANSI
from transformers import *
from transformers import AutoTokenizer

# KG and NLP use two different emb modules
# input data should be
# [topic_entity, Q1, answer1, Q2, answer2, ...]
# entity2idx for KG will map topic_entity -> ID, answer1 -> ID, ...
# tokenizer is a pretrained bert
class DatasetConversation(Dataset):
    def __init__(self, data, entity2idx, relation2idx, node_rels_map, ht2relation, topic_extractor=None):
        self.data = data
        self.entity2idx = entity2idx
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = 'roberta-base'

        self.tokenizer = AutoTokenizer.from_pretrained("D:/Project/ConvQA/model/roberta-base")
        self.pad_sequence_max_len = 64
        self.node_rels_map = node_rels_map
        self.relation2idx = relation2idx
        self.ht2relation = ht2relation
        self.num_questions=10

        # ============ 新增：主题提取器 ============
        self.topic_extractor = topic_extractor
        self.use_topic_aware = topic_extractor is not None
        self.topic_cache = {}  # 缓存主题信息

        # ============ 新增：预计算主题信息 ============
        if self.use_topic_aware:
            self._precompute_topics()

    def __len__(self):
        return len(self.data)

    def _precompute_topics(self):
        """预计算所有问题的主题信息"""
        print("Precomputing topic information...")
        all_questions = []

        for data_point in tqdm(self.data):
            # 收集所有问题
            questions = []
            for i in range(self.num_questions):
                question_idx = 1 + i * 2  # 问题在数据点中的索引位置
                if len(data_point) > question_idx and data_point[question_idx]:
                    question_text = data_point[question_idx][0] if data_point[question_idx] else ""
                    questions.append(question_text)
                else:
                    questions.append("")

            all_questions.extend(questions)

        # 过滤空问题
        all_questions = [q for q in all_questions if q.strip()]

        # 训练主题模型
        if all_questions:
            self.topic_extractor.fit(all_questions)
            print("Topic model training completed.")

    def _extract_topic_info(self, question_text):
        """提取问题主题信息"""
        if not self.use_topic_aware or not question_text.strip():
            return {
                'label': 'general',
                'vector': torch.zeros(384),  # 默认向量维度
                'confidence': 1.0
            }

        if question_text in self.topic_cache:
            return self.topic_cache[question_text]

        try:
            topic_features = self.topic_extractor([question_text])
            topic_info = {
                'label': f"topic_{topic_features['topics'][0]}",
                'vector': topic_features['embeddings'][0],
                'confidence': topic_features['probabilities'][0]
            }
        except Exception as e:
            # 如果主题提取失败，返回默认值
            topic_info = {
                'label': 'general',
                'vector': torch.zeros(384),
                'confidence': 1.0
            }

        self.topic_cache[question_text] = topic_info
        return topic_info

    def _calculate_topic_similarity(self, current_question, historical_questions):
        """计算当前问题与历史问题的主题相似度"""
        if not self.use_topic_aware or not historical_questions:
            return 0.0

        current_topic = self._extract_topic_info(current_question)
        similarities = []

        for hist_question in historical_questions[-3:]:  # 只看最近3轮
            hist_topic = self._extract_topic_info(hist_question)

            # 计算余弦相似度
            vec1 = current_topic['vector'].cpu().numpy()
            vec2 = hist_topic['vector'].cpu().numpy()

            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            similarities.append(cosine_sim)

        return np.mean(similarities) if similarities else 0.0

    def _get_conversation_context(self, data_point):
        """
        从数据点中提取对话上下文
        """
        # 检查数据点类型
        if isinstance(data_point, dict):
            # 字典格式访问 - 支持10个问题
            topic_node = data_point.get('anchor', '')
            questions = [
                data_point.get('q1', [''])[0],
                data_point.get('q2', [''])[0],
                data_point.get('q3', [''])[0],
                data_point.get('q4', [''])[0],
                data_point.get('q5', [''])[0],
                data_point.get('q6', [''])[0],
                data_point.get('q7', [''])[0],
                data_point.get('q8', [''])[0],
                data_point.get('q9', [''])[0],
                data_point.get('q10', [''])[0]
            ]
            answers = [
                data_point.get('a1', ''),
                data_point.get('a2', ''),
                data_point.get('a3', ''),
                data_point.get('a4', ''),
                data_point.get('a5', ''),
                data_point.get('a6', ''),
                data_point.get('a7', ''),
                data_point.get('a8', ''),
                data_point.get('a9', ''),
                data_point.get('a10', '')
            ]
        else:
            # 列表/元组格式访问 - 支持10个问题
            # 确保数据点有足够的元素
            if len(data_point) >= 21:  # 10个问题需要21个元素
                topic_node = data_point[0]
                questions = [
                    data_point[1][0], data_point[3][0], data_point[5][0], data_point[7][0], data_point[9][0],
                    data_point[11][0], data_point[13][0], data_point[15][0], data_point[17][0], data_point[19][0]
                ]
                answers = [
                    data_point[2], data_point[4], data_point[6], data_point[8], data_point[10],
                    data_point[12], data_point[14], data_point[16], data_point[18], data_point[20]
                ]
            else:
                # 如果数据点元素不足，用空值填充
                topic_node = data_point[0] if len(data_point) > 0 else ''
                questions = [data_point[i][0] if len(data_point) > i else '' for i in range(1, 20, 2)]
                answers = [data_point[i] if len(data_point) > i else '' for i in range(2, 21, 2)]

        return topic_node, questions, answers

        return topic_node, questions, answers

    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        if num_to_add < 0:
            return arr[0:max_len]
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def relToOneHot2(self, indices):
        num_rel = len(indices)
        indices = torch.LongTensor(indices)
        vec_len = len(self.relation2idx) + 1  # plus one for dummy relation
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def sampleRelation(self, indices):
        sample_size = 1
        if len(indices) >= sample_size:
            indices = indices[0:sample_size]
            sampled_neighbors = indices
        else:
            lef_len = sample_size - len(indices)
            tmp = np.random.choice(list(indices), size=lef_len, replace=len(indices) < sample_size)
            indices.extend(tmp)
            sampled_neighbors = indices

        return torch.LongTensor(sampled_neighbors)

    def get_relation_of_ht(self, head, tail):
        rel_ids = []
        key = (head, tail)
        rels = []
        not_exist = False
        if key in self.ht2relation:
            rels = self.ht2relation[key]
        elif (tail, head) in self.ht2relation:
            rels = self.ht2relation[(tail, head)]
        else:
            not_exist = True

        if not not_exist:
            for rel_name in rels:
                rel_name = rel_name.strip()
                if rel_name in self.relation2idx:
                    rel_ids.append(self.relation2idx[rel_name])
                else:
                    error = 0
        else:
            rel_ids.append(len(self.relation2idx))
        rel_score = self.sampleRelation(rel_ids)
        return rel_score

    def __getitem__(self, index):
        data_point = self.data[index]

        # ============ 新增：提取主题相关信息 ============
        topic_node, questions, answers = self._get_conversation_context(data_point)

        # 提取每个问题的主题信息
        topic_infos = []
        for i, question in enumerate(questions):
            if question.strip():
                topic_info = self._extract_topic_info(question)
                topic_infos.append(topic_info)
            else:
                topic_infos.append({
                    'label': 'general',
                    'vector': torch.zeros(384),
                    'confidence': 1.0
                })

        # 计算主题相似度（用于决定是否使用历史信息）
        topic_similarities = []
        for i in range(1, len(questions)):
            if questions[i].strip() and questions[i - 1].strip():
                similarity = self._calculate_topic_similarity(questions[i], questions[:i])
                topic_similarities.append(similarity)
            else:
                topic_similarities.append(0.0)

        # 填充相似度列表到10个元素
        while len(topic_similarities) < self.num_questions:
            topic_similarities.append(0.0)

        # ============ 处理10个问题的tokenization ============
        question_tokenized_list = []
        attention_mask_list = []
        answer_ids = []

        # 锚点实体
        topic_node_id = self.entity2idx.get(topic_node.strip(), 0)

        # 处理每个问题
        for i in range(10):  # 10个问题
            question_idx = 1 + i * 2
            answer_idx = 2 + i * 2

            # 问题文本
            if len(data_point) > question_idx and data_point[question_idx]:
                question_texts = data_point[question_idx]
            else:
                question_texts = [""]  # 默认空问题

            # 答案
            if len(data_point) > answer_idx and data_point[answer_idx]:
                answer_text = data_point[answer_idx].strip()
                if answer_text in self.entity2idx:
                    answer_id = self.entity2idx[answer_text]
                else:
                    answer_id = 0  # 默认ID
            else:
                answer_id = 0

            answer_ids.append(answer_id)

            # Tokenize问题
            q_tokenized = []
            q_masks = []
            for q_txt in question_texts:
                question_tokenized, attention_mask = self.tokenize_question(q_txt)
                q_tokenized.append(question_tokenized)
                q_masks.append(attention_mask)

            if q_tokenized:
                question_tokenized_list.append(torch.stack(q_tokenized, dim=0))
                attention_mask_list.append(torch.stack(q_masks, dim=0))
            else:
                # 处理空问题的情况
                dummy_tokenized, dummy_mask = self.tokenize_question("")
                question_tokenized_list.append(torch.stack([dummy_tokenized], dim=0))
                attention_mask_list.append(torch.stack([dummy_mask], dim=0))

    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, self.pad_sequence_max_len)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)


# ============ 新增：主题感知的数据加载器 ============
class TopicAwareDataLoader(DataLoader):
    """支持主题感知的数据加载器"""

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        if collate_fn is None:
            collate_fn = self._topic_aware_collate_fn

        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                         num_workers, collate_fn, pin_memory, drop_last,
                         timeout, worker_init_fn)

    def _topic_aware_collate_fn(self, batch):
        """处理包含主题信息的批次数据"""
        # 解构批次数据
        (topic_nodes, q1_tokens, q1_masks, a1_ids,
         q2_tokens, q2_masks, a2_ids,
         q3_tokens, q3_masks, a3_ids,
         q4_tokens, q4_masks, a4_ids,
         q5_tokens, q5_masks, a5_ids,
         q1_texts, q2_texts, q3_texts, q4_texts, q5_texts,
         *other_data,
         topic_vectors, topic_similarities, topic_labels) = zip(*batch)

        # 处理基本数据
        batch_data = (
            torch.stack(topic_nodes),
            torch.stack(q1_tokens),
            torch.stack(q1_masks),
            torch.stack(a1_ids),
            torch.stack(q2_tokens),
            torch.stack(q2_masks),
            torch.stack(a2_ids),
            torch.stack(q3_tokens),
            torch.stack(q3_masks),
            torch.stack(a3_ids),
            torch.stack(q4_tokens),
            torch.stack(q4_masks),
            torch.stack(a4_ids),
            torch.stack(q5_tokens),
            torch.stack(q5_masks),
            torch.stack(a5_ids),
            q1_texts, q2_texts, q3_texts, q4_texts, q5_texts,
            *[torch.stack(data) for data in other_data]
        )

        # 添加主题信息
        batch_data += (
            torch.stack(topic_vectors),
            torch.stack(topic_similarities),
            topic_labels
        )

        return batch_data


