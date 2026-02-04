import logging
import os
import numpy as np
import random
import json
from collections import defaultdict

from sklearn.cluster import KMeans

# 配置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ActiveDataSelector():
    """主动学习数据选择器基类 - 简化版本"""

    def __init__(self, conversation_path, config):
        self.conversation_path = conversation_path
        self.sampled_data = []
        self.remaining_data = []
        self.config = config
        self._init()

    def next(self):
        """获取下一批样本数据"""
        pass

    def _init(self):
        """初始化数据"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.conversation_path):
                log.warning(f"文件不存在，使用空数据: {self.conversation_path}")
                self.remaining_data = []
                return

            with open(self.conversation_path, "r", encoding='utf-8') as data:
                loaded_data = json.load(data)

            # 如果数据为空，使用空列表
            if not loaded_data:
                log.warning(f"文件为空: {self.conversation_path}")
                self.remaining_data = []
                return

            # 确保数据是列表格式
            if isinstance(loaded_data, dict):
                self.remaining_data = [loaded_data]
            else:
                self.remaining_data = list(loaded_data)

            log.info(f"成功加载对话数据，共 {len(self.remaining_data)} 条对话")

        except Exception as e:
            log.warning(f"加载数据失败，使用空数据: {e}")
            self.remaining_data = []

    def process(self, data):
        """处理10个问题的对话数据"""
        processed_data = []

        for conv in data:
            processed_conv = {
                'conv_id': conv['conv_id'],
                'anchor': conv.get('seed_entity_text', ''),
                'questions': []
            }

            # 处理10个问题
            for i in range(10):  # 修改为10个问题
                if i < len(conv['questions']):
                    q = conv['questions'][i]
                    question_info = {
                        'question': q['question'],
                        'answer': q.get('gold_answer_text', ''),
                        'reformulations': [r['reformulation'] for r in q.get('reformulations', [])]
                    }
                else:
                    # 如果问题不足10个，填充空问题
                    question_info = {
                        'question': '',
                        'answer': '',
                        'reformulations': []
                    }
                processed_conv['questions'].append(question_info)

            processed_data.append(processed_conv)

        return processed_data


class ActiveRandomSelector(ActiveDataSelector):
    """随机选择策略 - 简化版本"""

    def __init__(self, conversation_path, sample_size, config):
        super().__init__(conversation_path, config)
        self.sample_size = sample_size

    def next(self):
        """随机选择下一批样本"""
        if len(self.remaining_data) == 0:
            log.warning("没有更多数据可供选择，返回空列表")
            return []

        # 确保不超过剩余数据量
        actual_sample_size = min(self.sample_size, len(self.remaining_data))

        # 随机选择
        selected_indices = random.sample(range(len(self.remaining_data)), actual_sample_size)
        current_sample = [self.remaining_data[i] for i in selected_indices]

        # 更新数据池
        for i in sorted(selected_indices, reverse=True):
            self.remaining_data.pop(i)

        self.sampled_data.extend(current_sample)
        log.info(f"随机选择 {len(current_sample)} 条对话，已选择总数: {len(self.sampled_data)}")

        return self.process(current_sample)


class ActiveKMeansSelector(ActiveDataSelector):
    """基于K-means聚类的选择策略 - 简化版本"""

    def __init__(self, conversation_path, entity_emb, entity_dict, sample_size, cluster_num, config):
        self.sample_size = sample_size
        self.cluster_num = cluster_num
        self.entity_dict = entity_dict
        self.kg_node_embeddings = entity_emb
        self.clusters = defaultdict(list)
        self.config = config

        super().__init__(conversation_path, config)

    def _init(self):
        """初始化聚类结构"""
        super()._init()

        if not self.remaining_data:
            log.warning("无数据可用，跳过聚类初始化")
            return

        # 构建对话ID到对话的映射
        self.convID_2_conv = {}
        for conv in self.remaining_data:
            conv_id = conv.get('conv_id', conv.get('id', f'conv_{random.randint(1000, 9999)}'))
            self.convID_2_conv[conv_id] = conv

        # 构建聚类
        self.build_clusters()

    def _make_conversation_embedding(self):
        """生成对话的嵌入表示 - 简化版本"""
        conv_emb_list = []
        idx_to_convID = {}

        for idx, conv in enumerate(self.remaining_data):
            try:
                # 提取对话中的所有实体
                entities = []

                if 'seed_entity_text' in conv:
                    entities.append(conv['seed_entity_text'])
                elif 'anchor' in conv:
                    entities.append(conv['anchor'])

                if 'questions' in conv:
                    for q in conv['questions']:
                        if 'gold_answer_text' in q:
                            entities.append(q['gold_answer_text'])
                        elif 'answer' in q:
                            entities.append(q['answer'])

                # 计算实体嵌入的平均值
                entity_embeddings = []
                for entity in entities:
                    if entity in self.entity_dict:
                        entity_id = self.entity_dict[entity]
                        entity_embeddings.append(self.kg_node_embeddings[entity_id])

                if entity_embeddings:
                    conv_emb = np.mean(entity_embeddings, axis=0)
                    conv_emb_list.append(conv_emb)
                    conv_id = conv.get('conv_id', conv.get('id', f'conv_{idx}'))
                    idx_to_convID[len(conv_emb_list) - 1] = conv_id

            except Exception as e:
                log.warning(f"生成对话嵌入时跳过对话 {idx}: {e}")
                continue

        return idx_to_convID, conv_emb_list

    def build_clusters(self):
        """构建对话聚类"""
        if not self.remaining_data:
            log.warning("无数据可用，跳过聚类构建")
            return

        log.info("开始构建对话聚类...")

        idx_to_convID, conversation_embeddings = self._make_conversation_embedding()

        if len(conversation_embeddings) == 0:
            log.error("无法生成对话嵌入，聚类失败")
            return

        # 执行聚类
        convID_to_cluster = self.do_clusterize(conversation_embeddings, self.cluster_num)

        # 将对话分配到对应的聚类中
        for emb_idx, cluster_id in convID_to_cluster.items():
            conv_id = idx_to_convID[emb_idx]
            if conv_id in self.convID_2_conv:
                self.clusters[cluster_id].append(self.convID_2_conv[conv_id])

        log.info(f"聚类完成，共 {len(self.clusters)} 个聚类")

    def do_clusterize(self, embeddings, cluster_num):
        """执行K-means聚类"""
        if len(embeddings) < cluster_num:
            cluster_num = len(embeddings)

        kmeans = KMeans(n_clusters=cluster_num, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        return {idx: int(label) for idx, label in enumerate(cluster_labels)}

    def next(self):
        """获取下一批样本（聚类策略）"""
        if not self.remaining_data:
            log.warning("无数据可用，返回空列表")
            return []

        if not hasattr(self, 'clusters_initialized') or not self.clusters_initialized:
            self.clusters_initialized = True
            sample = self._initial_sampling()
        else:
            sample = self._update_sampling()

        self.sampled_data.extend(sample)
        return self.process(sample)

    def _initial_sampling(self):
        """初始聚类采样"""
        if not self.clusters:
            log.warning("无聚类信息，使用随机采样")
            return random.sample(self.remaining_data,
                                 min(self.sample_size, len(self.remaining_data)))

        sample = []
        conversations_per_cluster = max(1, self.sample_size // len(self.clusters))

        for cluster_id, cluster_data in self.clusters.items():
            random.shuffle(cluster_data)
            sample_count = min(conversations_per_cluster, len(cluster_data))
            sample.extend(cluster_data[:sample_count])

            if len(cluster_data) > sample_count:
                self.clusters[cluster_id] = cluster_data[sample_count:]
            else:
                del self.clusters[cluster_id]

            if len(sample) >= self.sample_size:
                sample = sample[:self.sample_size]
                break

        return sample

    def _update_sampling(self):
        """更新阶段采样"""
        if not self.clusters:
            log.warning("无聚类信息，返回空列表")
            return []

        sample = []
        total_remaining = sum(len(data) for data in self.clusters.values())

        for cluster_id, cluster_data in self.clusters.items():
            cluster_ratio = len(cluster_data) / total_remaining
            sample_count = max(1, int(cluster_ratio * self.sample_size))
            sample_count = min(sample_count, len(cluster_data))

            random.shuffle(cluster_data)
            sample.extend(cluster_data[:sample_count])

            if len(cluster_data) > sample_count:
                self.clusters[cluster_id] = cluster_data[sample_count:]
            else:
                del self.clusters[cluster_id]

        return sample


# 选择器工厂函数
def create_selector(selector_type, conversation_path, config, **kwargs):
    """创建数据选择器工厂函数"""

    if selector_type.lower() == "random":
        sample_size = kwargs.get('sample_size', 100)
        return ActiveRandomSelector(conversation_path, sample_size, config)

    elif selector_type.lower() == "kmeans":
        entity_emb = kwargs.get('entity_emb')
        entity_dict = kwargs.get('entity_dict')
        sample_size = kwargs.get('sample_size', 100)
        cluster_num = kwargs.get('cluster_num', 10)

        if entity_emb is None or entity_dict is None:
            raise ValueError("KMeans选择器需要实体嵌入和字典")

        return ActiveKMeansSelector(conversation_path, entity_emb, entity_dict,
                                    sample_size, cluster_num, config)

    else:
        raise ValueError(f"未知的选择器类型: {selector_type}")