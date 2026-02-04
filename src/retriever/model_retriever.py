import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from transformers import *
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformer import TransformerEncoder
from collections import defaultdict


# ============ 新增：主题提取模块 ============
class NeuralTopicExtractor(nn.Module):
    """基于神经网络的现代主题提取模块"""

    def __init__(self, config, device):
        super(NeuralTopicExtractor, self).__init__()
        self.config = config
        self.device = device

        # 加载预训练语言模型
        # 使用本地模型路径
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer(
                './model/all-MiniLM-L6-v2',  # 本地路径
                device=device
            )
            print("✓ 成功加载本地SentenceTransformer模型")
        except Exception as e:
            print(f"✗ 加载本地模型失败: {e}")
            self.sentence_model = None

        # 初始化BERTopic
        try:
            from bertopic import BERTopic
            self.topic_model = BERTopic(
                embedding_model=self.sentence_model,
                min_topic_size=getattr(config, 'min_topic_size', 5),
                verbose=False
            )
        except ImportError:
            self.topic_model = None
            print("BERTopic not available, using simple embedding-based topics")

        self.is_trained = False
        self.topic_centers = {}

    def forward(self, texts):
        """前向传播，提取主题特征"""
        if self.sentence_model is None:
            # 退化：返回零向量，保证不崩
            dummy = torch.zeros(len(texts), 384, device=self.device)
            return {"topics": [0] * len(texts), "probabilities": [1.0] * len(texts), "embeddings": dummy}

        if (not self.is_trained) or (self.topic_model is None):
            emb = self.sentence_model.encode(texts)
            emb_tensor = torch.as_tensor(emb, device=self.device)
            return {"topics": [0] * len(texts), "probabilities": [1.0] * len(texts), "embeddings": emb_tensor}

        topics, probabilities = self.topic_model.transform(texts)
        topic_embeddings = []

        for i, text in enumerate(texts):
            topic_id = topics[i]
            if topic_id in self.topic_centers:
                embedding = self.topic_centers[topic_id]  # torch.Tensor
            else:
                # sentence_model.encode 可能返回 numpy；统一转 torch
                emb = self.sentence_model.encode(text)
                embedding = torch.as_tensor(emb, device=self.device)

            topic_embeddings.append(embedding)

        emb_tensor = torch.stack(
            [e if isinstance(e, torch.Tensor) else torch.as_tensor(e, device=self.device) for e in topic_embeddings],
            dim=0
        ).to(self.device)

        return {
            "topics": topics,
            "probabilities": probabilities,
            "embeddings": emb_tensor,
        }

    def _extract_untrained_topics(self, texts):
        """在模型未训练时的主题提取"""
        emb = self.sentence_model.encode(texts)
        emb_tensor = torch.as_tensor(emb, device=self.device)
        return {
            "topics": [0] * len(texts),
            "probabilities": [1.0] * len(texts),
            "embeddings": emb_tensor,
        }

    def fit(self, texts):
        """训练主题模型"""
        if texts and self.topic_model is not None:
            self.topic_model.fit(texts)
            self.is_trained = True
            self._calculate_topic_centers(texts)

    def _calculate_topic_centers(self, texts):
        """计算每个主题的中心向量（torch 版本）"""
        topics = self.topic_model.transform(texts)[0]
        emb = self.sentence_model.encode(texts)
        emb_tensor = torch.as_tensor(emb, device=self.device)

        topic_to_embs = {}
        for i, topic_id in enumerate(topics):
            topic_to_embs.setdefault(topic_id, []).append(emb_tensor[i])

        for topic_id, emb_list in topic_to_embs.items():
            stacked = torch.stack(emb_list, dim=0)  # [n, d]
            self.topic_centers[topic_id] = stacked.mean(dim=0)  # [d]


class CosineSimilarityRetriever:
    """基于余弦相似度的检索器（纯 torch，不用 numpy）"""

    def __init__(self, config, entity_embeddings, relation_embeddings, device):
        self.config = config
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.device = device
        self.top_k = getattr(config, "top_k", 10)

    @staticmethod
    def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: [B, D], x2: [N, D] -> [B, N]
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        return x1 @ x2.t()

    def get_topk_entities(self, query_embedding: torch.Tensor, k: int | None = None):
        if k is None:
            k = self.top_k
        sims = self.cosine_similarity(query_embedding, self.entity_embeddings.weight)  # [B, num_ent]
        scores, indices = torch.topk(sims, k=k, dim=-1, largest=True, sorted=True)
        return indices, scores

    def get_topk_relations(self, query_embedding: torch.Tensor, k: int | None = None):
        if k is None:
            k = self.top_k
        sims = self.cosine_similarity(query_embedding, self.relation_embeddings.weight)  # [B, num_rel]
        scores, indices = torch.topk(sims, k=k, dim=-1, largest=True, sorted=True)
        return indices, scores


class RelationExtractor(nn.Module):
    def __init__(self, config, embedding_dim, num_entities, relation_emb, pretrained_embeddings, freeze, device,
                 entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0, ls=0.0, do_batch_norm=True):
        super().__init__()
        self.config = config
        self.device = device
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm

        self.topic_extractor = NeuralTopicExtractor(config, device)
        self.use_topic_aware = getattr(config, "use_topic_aware", True)

        # RoBERTa
        self.roberta_pretrained_weights = "./model/roberta-base"
        self.roberta_model = RobertaModel.from_pretrained(
            self.roberta_pretrained_weights,
            local_files_only=True
        ).to(self.device)
        for p in self.roberta_model.parameters():
            p.requires_grad = True

        self.hidden_dim = 768
        self.num_entities = num_entities

        # KG embeddings
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=self.freeze).to(self.device)
        self.relation_emb = nn.Embedding.from_pretrained(relation_emb, freeze=False).to(self.device)

        # 设定检索空间维度 D：最简单取 entity embedding 的维度
        self.retrieval_dim = self.embedding.weight.shape[1]

        # 关键：768 -> D
        self.query_proj = nn.Linear(self.hidden_dim, self.retrieval_dim).to(self.device)

        # 关键：384 -> D（MiniLM 一般 384；做成 Lazy 更稳，避免你以后换模型维度）
        self.topic_proj = nn.LazyLinear(self.retrieval_dim).to(self.device)

        self.topic_alpha = float(getattr(config, "topic_alpha", 0.7))

        # retriever（注意：现在要求 query_embedding 是 D 维）
        self.retriever = CosineSimilarityRetriever(config, self.embedding, self.relation_emb, device)

        self.transformer = TransformerEncoder(self.hidden_dim, self.hidden_dim, 0.01, 0.01, 16, 4)

        # \[新增\] 是否启用“历史聚合”
        self.use_history_agg = bool(getattr(config, "use_history_agg", True))
        self.history_lstm_hidden = int(getattr(config, "history_lstm_hidden", self.hidden_dim))
        self.history_lstm_layers = int(getattr(config, "history_lstm_layers", 1))
        self.history_lstm_dropout = float(getattr(config, "history_lstm_dropout", 0.0))

        if self.use_history_agg:
            self.history_lstm = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.history_lstm_hidden,
                num_layers=self.history_lstm_layers,
                batch_first=True,
                dropout=self.history_lstm_dropout if self.history_lstm_layers > 1 else 0.0,
                bidirectional=False,
            ).to(self.device)
            # 把 LSTM hidden 映射回 query_proj 需要的 768 维（或你也可直接改 query_proj 的输入维）
            self.history_to_hidden = nn.Linear(self.history_lstm_hidden, self.hidden_dim).to(self.device)

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self._klloss = torch.nn.KLDivLoss(reduction="sum")

    def kge_loss(self, scores, targets):
        # targets: \[B, num_entities\] one\-hot/multi\-hot
        return self._klloss(F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1))


    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        last_hidden = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        return last_hidden[:, 0, :]

    def get_topic_enhanced_embedding(self, question_tokenized, attention_mask, question_text=None):
        """
        输出：query_embedding（维度 D），以及 topic_info
        """
        base_768 = self.getQuestionEmbedding(question_tokenized, attention_mask)  # [B, 768]
        base_D = self.query_proj(base_768)  # [B, D]

        if (not self.use_topic_aware) or (question_text is None):
            return base_D, None

        topic_features = self.topic_extractor([question_text])
        topic_vec = topic_features["embeddings"][0].to(self.device)  # [topic_dim]，通常 384
        topic_vec = topic_vec.unsqueeze(0)  # [1, topic_dim]
        topic_D = self.topic_proj(topic_vec)  # [1, D]（LazyLinear 自动推断 topic_dim）

        # 融合到同一空间 D
        alpha = self.topic_alpha
        enhanced_D = alpha * base_D + (1.0 - alpha) * topic_D  # [B, D]（利用广播）

        topic_info = {
            "topic_id": int(topic_features["topics"][0]) if "topics" in topic_features else 0,
            "confidence": float(topic_features["probabilities"][0]) if "probabilities" in topic_features else 1.0,
        }
        return enhanced_D, topic_info

    def cosine_retrieval(self, query_embedding_D, head_entity=None, top_k=10):
        # query_embedding_D: [B, D]
        top_entities, entity_scores = self.retriever.get_topk_entities(query_embedding_D, top_k)
        top_relations, relation_scores = self.retriever.get_topk_relations(query_embedding_D, top_k)
        return {
            "entities": (top_entities, entity_scores),
            "relations": (top_relations, relation_scores),
        }

    def _sample_pos_from_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        targets: [B, num_entities] (one-hot 或 multi-hot, float/long 都可)
        返回: pos_ids [B] long。multi-hot 时从正例集合里随机采样一个。
        """
        if targets.dtype != torch.float32:
            targets = targets.float()
        # 避免全 0：若某行没有正例，会导致 multinomial 报错
        row_sum = targets.sum(dim=1, keepdim=True)  # [B,1]
        safe = torch.where(row_sum > 0, targets / row_sum, torch.zeros_like(targets))
        # 对于没有正例的行，退化成随机正例（不理想，但保证不崩；数据应保证有正例）
        no_pos = (row_sum.squeeze(1) <= 0)
        if no_pos.any():
            safe[no_pos] = 1.0 / safe.size(1)
        pos_ids = torch.multinomial(safe, num_samples=1).squeeze(1)  # [B]
        return pos_ids.long()

    def infonce_loss_b(
        self,
        query_D: torch.Tensor,
        targets: torch.Tensor,
        k: int | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        方案B：InfoNCE（负采样）
        query_D: [B, D]
        targets: [B, num_entities] one-hot/multi-hot
        """
        if k is None:
            k = int(getattr(self.config, "infonce_k", 64))
        if temperature is None:
            temperature = float(getattr(self.config, "infonce_temp", 0.07))

        B = query_D.size(0)
        num_entities = self.embedding.weight.size(0)

        # 1) 正例 id（multi-hot 时随机取一个正例）
        pos_ids = self._sample_pos_from_targets(targets)  # [B]

        # 2) 负例采样：从全实体采 K 个，尽量排除正例
        #    这里用全局均匀采样；更复杂可用 hard negative（比如 topk）但别 detach/numpy。
        neg_ids = torch.randint(low=0, high=num_entities, size=(B, k), device=query_D.device)  # [B,K]
        # 若碰撞到正例，重采一次（简单做法；k 不大时足够）
        collide = neg_ids.eq(pos_ids.unsqueeze(1))  # [B,K]
        if collide.any():
            resample = torch.randint(low=0, high=num_entities, size=(B, k), device=query_D.device)
            neg_ids = torch.where(collide, resample, neg_ids)

        # 3) 构造候选集合：[pos, negs] => [B, 1+K]
        cand_ids = torch.cat([pos_ids.unsqueeze(1), neg_ids], dim=1)  # [B, 1+K]
        cand_emb = self.embedding(cand_ids)  # [B, 1+K, D]

        # 4) 余弦相似度 logits（可微）
        q = F.normalize(query_D, p=2, dim=-1)                 # [B,D]
        c = F.normalize(cand_emb, p=2, dim=-1)                # [B,1+K,D]
        logits = torch.einsum("bd,bkd->bk", q, c)              # [B,1+K]
        logits = logits / max(temperature, 1e-8)

        # 5) label 永远是 0（正例在第 0 位）
        labels = torch.zeros(B, dtype=torch.long, device=query_D.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def _encode_turns(self, question_tokenized_list, attention_mask_list) -> torch.Tensor:
        """
        question_tokenized_list: list\[Tensor\[B,L\]\] 长度 T
        attention_mask_list:     list\[Tensor\[B,L\]\] 长度 T
        return: turn_embs \[B, T, 768\]
        """
        turn_embs = []
        for qt, am in zip(question_tokenized_list, attention_mask_list):
            turn_embs.append(self.getQuestionEmbedding(qt, am))  # \[B,768\]
        return torch.stack(turn_embs, dim=1)  # \[B,T,768\]

    def _aggregate_history(self, turn_embs: torch.Tensor) -> torch.Tensor:
        """
        turn_embs: \[B,T,768\]
        return: agg \[B,768\]
        """
        if not self.use_history_agg:
            return turn_embs[:, -1, :]  # 直接用最后一轮

        out, (h_n, _) = self.history_lstm(turn_embs)  # h_n: \[layers,B,H\]
        last_h = h_n[-1]  # \[B,H\]
        agg = self.history_to_hidden(last_h)  # \[B,768\]
        return agg

    def full_softmax_logits(self, query_D: torch.Tensor) -> torch.Tensor:
        """
        query_D: \[B,D\]
        return logits: \[B,num_entities\]
        """
        q = F.normalize(query_D, p=2, dim=-1)
        e = F.normalize(self.embedding.weight, p=2, dim=-1)  # \[N,D\]
        logits = q @ e.t()
        return logits

    @torch.inference_mode()
    def get_score_ranked(
            self,
            question_tokenized,
            attention_mask,
            question_texts=None,
            return_per_turn: bool = True,
    ):
        """
        Step 5: 推理阶段 get_score_ranked
        \- 返回 logits（scores），外部再做 topk
        \- 兼容旧版：多轮输入时可返回“每轮 [B, num_entities]”的 scores 列表

        Args:
            question_tokenized:
                \- Tensor[B,L] 单轮
                \- 或 list/tuple[Tensor[B,L]] 多轮（长度T）
            attention_mask:
                \- Tensor[B,L] 单轮
                \- 或 list/tuple[Tensor[B,L]] 多轮（长度T）
            question_texts: 可选（若未来要接 topic 融合可用；当前未用）
            return_per_turn:
                \- True: 多轮时返回 List[Tensor[B,num_entities]]（每轮一个）
                \- False: 多轮时返回最后聚合后的 Tensor[B,num_entities]（与 forward 推理一致）

        Returns:
            \- 单轮输入: Tensor[B, num_entities]
            \- 多轮输入:
                \- return_per_turn=True -> List[Tensor[B, num_entities]]
                \- return_per_turn=False -> Tensor[B, num_entities]
        """
        # 多轮：每轮单独出分（旧版兼容）
        if isinstance(question_tokenized, (list, tuple)):
            if return_per_turn:
                scores_per_turn = []
                for qt, am in zip(question_tokenized, attention_mask):
                    base_768 = self.getQuestionEmbedding(qt, am)  # [B,768]
                    query_D = self.query_proj(base_768)  # [B,D]
                    logits = self.full_softmax_logits(query_D)  # [B,num_entities]
                    scores_per_turn.append(logits)
                return scores_per_turn

            # 多轮：按当前实现聚合历史后出分（与 forward(mode!=train) 对齐）
            turn_embs = self._encode_turns(question_tokenized, attention_mask)  # [B,T,768]
            base_768 = self._aggregate_history(turn_embs)  # [B,768]
            query_D = self.query_proj(base_768)  # [B,D]
            return self.full_softmax_logits(query_D)  # [B,num_entities]

        # 单轮
        base_768 = self.getQuestionEmbedding(question_tokenized, attention_mask)  # [B,768]
        query_D = self.query_proj(base_768)  # [B,D]
        return self.full_softmax_logits(query_D)  # [B,num_entities]

    def forward(
            self,
            question_tokenized,
            attention_mask,
            targets=None,
            tail_id=None,
            question_texts=None,
            mode: str = "train",
            loss_type: str = "full",  # "infonce" 或 "full"
            k: int | None = None,
            temperature: float | None = None,
    ):
        if isinstance(question_tokenized, (list, tuple)):
            turn_embs = self._encode_turns(question_tokenized, attention_mask)
            base_768 = self._aggregate_history(turn_embs)
        else:
            base_768 = self.getQuestionEmbedding(question_tokenized, attention_mask)

        query_D = self.query_proj(base_768)

        if mode != "train":
            return self.full_softmax_logits(query_D)

        # Step 4: 训练默认走 full-softmax + KL/CE
        if loss_type == "full":
            logits = self.full_softmax_logits(query_D)
            if tail_id is not None:
                return F.cross_entropy(logits, tail_id.long())
            if targets is not None:
                return self.kge_loss(logits, targets)
            raise ValueError("训练模式下需要提供 tail_id 或 targets")

        # 仍保留 infonce（但不再默认）
        if loss_type == "infonce":
            if targets is None:
                raise ValueError("loss_type=infonce 时必须提供 targets")
            return self.infonce_loss_b(query_D, targets, k=k, temperature=temperature)

        raise ValueError(f"未知 loss_type: {loss_type}")

# ============ 配置类 ============
class CosineSimilarityConfig:
    def __init__(self):
        self.top_k = 10
        self.use_topic_aware = True
        self.topic_model_name = 'all-MiniLM-L6-v2'
        self.min_topic_size = 5
        self.rollout_num = 1  # 为了兼容性保留


# ============ 工具函数 ============
def safe_log(x):
    """安全对数计算"""
    return torch.log(torch.clamp(x, min=1e-10))
