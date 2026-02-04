import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import *
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformer import TransformerEncoder


# ============ 新增：余弦相似度检索器 ============
class CosineSimilarityRetriever:
    """基于余弦相似度的检索器"""

    def __init__(self, config, entity_embeddings, relation_embeddings, device):
        self.config = config
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.device = device
        self.top_k = getattr(config, 'top_k', 10)

    def get_topk_entities(self, query_embedding, k=None):
        """通过余弦相似度找到top-k相关实体"""
        if k is None:
            k = self.top_k

        # 使用 PyTorch 计算余弦相似度
        query_normalized = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        entity_emb_normalized = F.normalize(self.entity_embeddings.weight, p=2, dim=1)

        similarities = torch.mm(query_normalized, entity_emb_normalized.transpose(0, 1))[0]

        # 获取top-k实体
        topk_scores, topk_indices = torch.topk(similarities, k)

        return topk_indices.cpu().numpy(), topk_scores.cpu().numpy()

    def get_topk_relations(self, query_embedding, k=None):
        """通过余弦相似度找到top-k相关关系"""
        if k is None:
            k = self.top_k

        query_normalized = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        relation_emb_normalized = F.normalize(self.relation_embeddings.weight, p=2, dim=1)

        similarities = torch.mm(query_normalized, relation_emb_normalized.transpose(0, 1))[0]

        topk_scores, topk_indices = torch.topk(similarities, k)

        return topk_indices.cpu().numpy(), topk_scores.cpu().numpy()


# ============ 新增：简化版主题提取器 ============
class SimpleTopicExtractor:
    """简化版主题提取器"""

    def __init__(self, config=None, device='cpu'):
        self.config = config
        self.device = device
        self.is_trained = False

    def fit(self, texts):
        """模拟训练"""
        self.is_trained = True
        print(f"Simple topic extractor trained on {len(texts)} texts")

    def __call__(self, texts):
        """模拟主题提取"""
        if not isinstance(texts, list):
            texts = [texts]

        topics = []
        probabilities = []
        embeddings = []

        for text in texts:
            # 简单基于文本长度的模拟主题
            topic_id = len(text) % 5
            topics.append(topic_id)
            probabilities.append(0.9)
            # 随机生成嵌入向量
            embedding = torch.randn(384)
            embeddings.append(embedding)

        return {
            'topics': topics,
            'probabilities': probabilities,
            'embeddings': torch.stack(embeddings) if len(embeddings) > 1 else embeddings[0]
        }


"""
This model is a pretrained model, it is pretrained to choose the current topic entity according to
the conversation history and current question.
According to the experiment, this model will choose the global topic entity as the current topic entity
most of the time.
"""


class MulticlassClassification(nn.Module):
    def __init__(self, input_dim, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_2 = nn.Linear(input_dim, int(input_dim * 1.2))
        self.layer_3 = nn.Linear(int(input_dim * 1.2), int(num_class * 0.8))
        self.layer_out = nn.Linear(int(num_class * 0.8), num_class)

        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.xavier_uniform_(self.layer_3.weight)
        nn.init.xavier_uniform_(self.layer_out.weight)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(input_dim)
        self.batchnorm2 = nn.BatchNorm1d(int(input_dim * 1.2))
        self.batchnorm3 = nn.BatchNorm1d(int(num_class * 0.8))
        self.m = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        x = self.m(x)

        return x


class RelationExtractor(nn.Module):
    def __init__(self, config, embedding_dim, num_entities, relation_emb, pretrained_embeddings, freeze, device,
                 entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0, ls=0.0, do_batch_norm=True):
        super(RelationExtractor, self).__init__()
        self.device = device
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm

        # ============ 新增：主题感知和余弦相似度功能 ============
        self.use_topic_aware = getattr(config, 'use_topic_aware', False)
        self.use_cosine_similarity = True

        if not self.do_batch_norm:
            print('Not doing batch norm')

        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained("D:/Project/ConvQA/model/roberta-base", local_files_only=True)
        for param in self.roberta_model.parameters():
            param.requires_grad = True

        multiplier = 2
        self.getScores = self.ComplEx

        self.hidden_dim = 768
        self.num_entities = num_entities
        self.loss = self.kge_loss

        # dropout layers
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.fcnn_dropout = torch.nn.Dropout(0.1)

        print('Frozen:', self.freeze)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=self.freeze).to(self.device)
        print(self.embedding.weight.shape)

        self.lstm_layer_size = 2
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, self.lstm_layer_size, batch_first=True).to(
            self.device)

        relation_dim = self.embedding.weight.shape[1]
        self.relation_dim = relation_dim

        self.mid1 = 512
        self.mid2 = 512
        self.mid3 = 512
        self.mid4 = 512

        self.hidden2rel = nn.Linear(self.hidden_dim, self.relation_dim)
        self.hidden2rel_base = nn.Linear(self.mid2, self.relation_dim)

        self.bn0 = torch.nn.BatchNorm1d(multiplier)
        self.bn2 = torch.nn.BatchNorm1d(multiplier)

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self._klloss = torch.nn.KLDivLoss(reduction='sum')

        self.relation_emb = nn.Embedding.from_pretrained(relation_emb, freeze=False).to(self.device)

        # ============ 移除RL相关组件，添加余弦相似度检索器 ============
        self.retriever = CosineSimilarityRetriever(config, self.embedding, self.relation_emb, device)

        self.transformer = TransformerEncoder(self.hidden_dim, self.hidden_dim, 0.01, 0.01, 16, 4)

        # ============ 新增：主题提取器 ============
        if self.use_topic_aware:
            self.topic_extractor = SimpleTopicExtractor(config, device)
        else:
            self.topic_extractor = None

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.head_chose_network = MulticlassClassification(self.relation_dim, 2)
        for name, param in self.head_chose_network.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

        self.head_chose_loss = torch.nn.BCELoss(reduction='sum')

        self.config = config

    def set_bn_eval(self):
        self.bn0.eval()
        self.bn2.eval()

    def kge_loss(self, scores, targets):
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def applyNonLinear(self, outputs):
        outputs = self.hidden2rel(outputs)
        return outputs

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn0(head)

        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)

        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        pred = score
        return pred

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1, 0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        return question_embedding

    # ============ 新增：主题增强的问题嵌入方法 ============
    def get_topic_enhanced_embedding(self, question_tokenized, attention_mask, question_text=None):
        """获取结合主题信息的问题嵌入"""
        base_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)

        if not self.use_topic_aware or question_text is None:
            return base_embedding, None

        # 提取主题特征
        topic_features = self.topic_extractor([question_text])
        topic_embedding = topic_features['embeddings'][0]

        # 加权融合
        if base_embedding.dim() == 1:
            base_embedding = base_embedding.unsqueeze(0)

        alpha = 0.7  # 基础嵌入权重
        enhanced_embedding = alpha * base_embedding + (1 - alpha) * topic_embedding.unsqueeze(0)

        topic_info = {
            'topic_id': topic_features['topics'][0],
            'confidence': topic_features['probabilities'][0],
            'label': f"topic_{topic_features['topics'][0]}"
        }

        return enhanced_embedding.squeeze(), topic_info

    # ============ 新增：余弦相似度检索方法 ============
    def cosine_retrieval(self, question_embedding, head_entity=None, top_k=10):
        """基于余弦相似度的检索方法"""
        top_entities, entity_scores = self.retriever.get_topk_entities(question_embedding, top_k)
        top_relations, relation_scores = self.retriever.get_topk_relations(question_embedding, top_k)

        return {
            'entities': (top_entities, entity_scores),
            'relations': (top_relations, relation_scores)
        }

    def get_final_answer(self, retrieval_results, method='combined'):
        """根据检索结果确定最终答案"""
        entities, entity_scores = retrieval_results['entities']
        relations, relation_scores = retrieval_results['relations']

        if method == 'entity_only':
            best_idx = np.argmax(entity_scores)
            return entities[best_idx], entity_scores[best_idx]
        elif method == 'relation_only':
            best_idx = np.argmax(relation_scores)
            return relations[best_idx], relation_scores[best_idx]
        else:
            combined_scores = (entity_scores + relation_scores) / 2
            best_idx = np.argmax(combined_scores)
            return entities[best_idx], combined_scores[best_idx]

    def head_choose(self, question_embedding):
        predict = self.head_chose_network(question_embedding)
        return predict

    # ============ 修改forward方法，支持主题信息 ============
    def forward(self, p_head, question_tokenized1, attention_mask1, p_tail1,
                question_tokenized2, attention_mask2, p_tail2,
                question_tokenized3, attention_mask3, p_tail3,
                question_tokenized4, attention_mask4, p_tail4,
                question_tokenized5, attention_mask5, p_tail5,
                startpoint1, startpoint2, startpoint3, startpoint4, startpoint5,
                question_texts=None):  # 新增参数

        head = p_head

        question_shape = question_tokenized1.shape
        mask_shape = attention_mask1.shape

        question_tokenized1 = question_tokenized1.view(-1, question_shape[-1])
        attention_mask1 = attention_mask1.view(-1, mask_shape[-1])
        question_tokenized2 = question_tokenized2.view(-1, question_shape[-1])
        attention_mask2 = attention_mask2.view(-1, mask_shape[-1])
        question_tokenized3 = question_tokenized3.view(-1, question_shape[-1])
        attention_mask3 = attention_mask3.view(-1, mask_shape[-1])
        question_tokenized4 = question_tokenized4.view(-1, question_shape[-1])
        attention_mask4 = attention_mask4.view(-1, mask_shape[-1])
        question_tokenized5 = question_tokenized5.view(-1, question_shape[-1])
        attention_mask5 = attention_mask5.view(-1, mask_shape[-1])

        # ============ 修改：使用主题增强的嵌入 ============
        if question_texts and self.use_topic_aware:
            question_embedding1, _ = self.get_topic_enhanced_embedding(
                question_tokenized1, attention_mask1, question_texts[0])
            question_embedding2, _ = self.get_topic_enhanced_embedding(
                question_tokenized2, attention_mask2, question_texts[1])
            question_embedding3, _ = self.get_topic_enhanced_embedding(
                question_tokenized3, attention_mask3, question_texts[2])
            question_embedding4, _ = self.get_topic_enhanced_embedding(
                question_tokenized4, attention_mask4, question_texts[3])
            question_embedding5, _ = self.get_topic_enhanced_embedding(
                question_tokenized5, attention_mask5, question_texts[4])
        else:
            question_embedding1 = self.getQuestionEmbedding(question_tokenized1, attention_mask1)
            question_embedding2 = self.getQuestionEmbedding(question_tokenized2, attention_mask2)
            question_embedding3 = self.getQuestionEmbedding(question_tokenized3, attention_mask3)
            question_embedding4 = self.getQuestionEmbedding(question_tokenized4, attention_mask4)
            question_embedding5 = self.getQuestionEmbedding(question_tokenized5, attention_mask5)

        # 使用multi-head attention
        question_shape = list(question_shape)
        question_shape[-1] = -1
        question_embedding1 = question_embedding1.view(question_shape)
        question_embedding2 = question_embedding2.view(question_shape)
        question_embedding3 = question_embedding3.view(question_shape)
        question_embedding4 = question_embedding4.view(question_shape)
        question_embedding5 = question_embedding5.view(question_shape)

        question_embedding1 = self.transformer(question_embedding1)
        question_embedding2 = self.transformer(question_embedding2)
        question_embedding3 = self.transformer(question_embedding3)
        question_embedding4 = self.transformer(question_embedding4)
        question_embedding5 = self.transformer(question_embedding5)

        question_embedding1 = question_embedding1[:, 0, :]
        question_embedding2 = question_embedding2[:, 0, :]
        question_embedding3 = question_embedding3[:, 0, :]
        question_embedding4 = question_embedding4[:, 0, :]
        question_embedding5 = question_embedding5[:, 0, :]

        question_embedding1 = question_embedding1.unsqueeze(1)
        question_embedding2 = question_embedding2.unsqueeze(1)
        question_embedding3 = question_embedding3.unsqueeze(1)
        question_embedding4 = question_embedding4.unsqueeze(1)
        question_embedding5 = question_embedding5.unsqueeze(1)

        h_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim,
                              requires_grad=False).to(question_embedding1.device)
        c_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim,
                              requires_grad=False).to(question_embedding1.device)

        # LSTM处理
        output1, (h1, c1) = self.lstm(question_embedding1, (h_state, c_state))
        output2, (h2, c2) = self.lstm(question_embedding2, (h1, c1))
        output3, (h3, c3) = self.lstm(question_embedding3, (h2, c2))
        output4, (h4, c4) = self.lstm(question_embedding4, (h3, c3))
        output5, (h5, c5) = self.lstm(question_embedding5, (h4, c4))

        rel_embedding1 = self.applyNonLinear(output1.squeeze(1))
        rel_embedding2 = self.applyNonLinear(output2.squeeze(1))
        rel_embedding3 = self.applyNonLinear(output3.squeeze(1))
        rel_embedding4 = self.applyNonLinear(output4.squeeze(1))
        rel_embedding5 = self.applyNonLinear(output5.squeeze(1))

        # 头部选择预测
        pred1 = self.head_choose(rel_embedding1)
        pred2 = self.head_choose(rel_embedding2)
        pred3 = self.head_choose(rel_embedding3)
        pred4 = self.head_choose(rel_embedding4)
        pred5 = self.head_choose(rel_embedding5)

        # 计算损失
        path_ground_truth1 = F.one_hot(startpoint1, num_classes=2)
        path_ground_truth2 = F.one_hot(startpoint2, num_classes=2)
        path_ground_truth3 = F.one_hot(startpoint3, num_classes=2)
        path_ground_truth4 = F.one_hot(startpoint4, num_classes=2)
        path_ground_truth5 = F.one_hot(startpoint5, num_classes=2)

        path_loss1 = self.head_chose_loss(pred1, path_ground_truth1.float())
        path_loss2 = self.head_chose_loss(pred2, path_ground_truth2.float())
        path_loss3 = self.head_chose_loss(pred3, path_ground_truth3.float())
        path_loss4 = self.head_chose_loss(pred4, path_ground_truth4.float())
        path_loss5 = self.head_chose_loss(pred5, path_ground_truth5.float())

        loss = path_loss1 + path_loss2 + path_loss3 + path_loss4 + path_loss5
        loss = loss.to(self.device)
        return loss

    # ============ 修改推理方法，支持余弦相似度检索 ============
    def get_score_ranked(self, head, question_tokenized1, attention_mask1,
                         question_tokenized2, attention_mask2,
                         question_tokenized3, attention_mask3,
                         question_tokenized4, attention_mask4,
                         question_tokenized5, attention_mask5,
                         startpoint1, startpoint2, startpoint3, startpoint4, startpoint5,
                         question_texts=None):  # 新增参数

        question_shape = question_tokenized1.shape
        mask_shape = attention_mask1.shape

        question_tokenized1 = question_tokenized1.view(-1, question_shape[-1])
        attention_mask1 = attention_mask1.view(-1, mask_shape[-1])
        question_tokenized2 = question_tokenized2.view(-1, question_shape[-1])
        attention_mask2 = attention_mask2.view(-1, mask_shape[-1])
        question_tokenized3 = question_tokenized3.view(-1, question_shape[-1])
        attention_mask3 = attention_mask3.view(-1, mask_shape[-1])
        question_tokenized4 = question_tokenized4.view(-1, question_shape[-1])
        attention_mask4 = attention_mask4.view(-1, mask_shape[-1])
        question_tokenized5 = question_tokenized5.view(-1, question_shape[-1])
        attention_mask5 = attention_mask5.view(-1, mask_shape[-1])

        # ============ 修改：使用主题增强的嵌入 ============
        if question_texts and self.use_topic_aware:
            question_embedding1, _ = self.get_topic_enhanced_embedding(
                question_tokenized1, attention_mask1, question_texts[0])
            question_embedding2, _ = self.get_topic_enhanced_embedding(
                question_tokenized2, attention_mask2, question_texts[1])
            question_embedding3, _ = self.get_topic_enhanced_embedding(
                question_tokenized3, attention_mask3, question_texts[2])
            question_embedding4, _ = self.get_topic_enhanced_embedding(
                question_tokenized4, attention_mask4, question_texts[3])
            question_embedding5, _ = self.get_topic_enhanced_embedding(
                question_tokenized5, attention_mask5, question_texts[4])
        else:
            question_embedding1 = self.getQuestionEmbedding(question_tokenized1, attention_mask1)
            question_embedding2 = self.getQuestionEmbedding(question_tokenized2, attention_mask2)
            question_embedding3 = self.getQuestionEmbedding(question_tokenized3, attention_mask3)
            question_embedding4 = self.getQuestionEmbedding(question_tokenized4, attention_mask4)
            question_embedding5 = self.getQuestionEmbedding(question_tokenized5, attention_mask5)

        # Transformer处理
        question_shape = list(question_shape)
        question_shape[-1] = -1
        question_embedding1 = question_embedding1.view(question_shape)
        question_embedding2 = question_embedding2.view(question_shape)
        question_embedding3 = question_embedding3.view(question_shape)
        question_embedding4 = question_embedding4.view(question_shape)
        question_embedding5 = question_embedding5.view(question_shape)

        question_embedding1 = self.transformer(question_embedding1)
        question_embedding2 = self.transformer(question_embedding2)
        question_embedding3 = self.transformer(question_embedding3)
        question_embedding4 = self.transformer(question_embedding4)
        question_embedding5 = self.transformer(question_embedding5)

        question_embedding1 = question_embedding1[:, 0, :]
        question_embedding2 = question_embedding2[:, 0, :]
        question_embedding3 = question_embedding3[:, 0, :]
        question_embedding4 = question_embedding4[:, 0, :]
        question_embedding5 = question_embedding5[:, 0, :]

        question_embedding1 = question_embedding1.unsqueeze(1)
        question_embedding2 = question_embedding2.unsqueeze(1)
        question_embedding3 = question_embedding3.unsqueeze(1)
        question_embedding4 = question_embedding4.unsqueeze(1)
        question_embedding5 = question_embedding5.unsqueeze(1)

        h_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim,
                              requires_grad=False).to(question_embedding1.device)
        c_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim,
                              requires_grad=False).to(question_embedding1.device)

        # LSTM处理
        output1, (h1, c1) = self.lstm(question_embedding1, (h_state, c_state))
        output2, (h2, c2) = self.lstm(question_embedding2, (h1, c1))
        output3, (h3, c3) = self.lstm(question_embedding3, (h2, c2))
        output4, (h4, c4) = self.lstm(question_embedding4, (h3, c3))
        output5, (h5, c5) = self.lstm(question_embedding5, (h4, c4))

        rel_embedding1 = self.applyNonLinear(output1.squeeze(1))
        rel_embedding2 = self.applyNonLinear(output2.squeeze(1))
        rel_embedding3 = self.applyNonLinear(output3.squeeze(1))
        rel_embedding4 = self.applyNonLinear(output4.squeeze(1))
        rel_embedding5 = self.applyNonLinear(output5.squeeze(1))

        # ============ 新增：余弦相似度检索 ============
        answers = []
        scores = []

        for i, rel_embedding in enumerate(
                [rel_embedding1, rel_embedding2, rel_embedding3, rel_embedding4, rel_embedding5]):
            retrieval = self.cosine_retrieval(rel_embedding, head)
            answer, score = self.get_final_answer(retrieval)
            answers.append(answer)
            scores.append(score)

        # 原有的头部选择预测（保持兼容性）
        pred1 = self.head_choose(rel_embedding1)
        pred2 = self.head_choose(rel_embedding2)
        pred3 = self.head_choose(rel_embedding3)
        pred4 = self.head_choose(rel_embedding4)
        pred5 = self.head_choose(rel_embedding5)

        pred_top1 = pred1.argmax(1)
        pred_top2 = pred2.argmax(1)
        pred_top3 = pred3.argmax(1)
        pred_top4 = pred4.argmax(1)
        pred_top5 = pred5.argmax(1)

        def calculate_error(list1, list2):
            error = 0
            for i in range(list1.shape[0]):
                a = list1[i]
                b = list2[i]
                if b == 1:
                    error += 1
            return error

        b1 = calculate_error(startpoint1.cpu().detach().numpy(), pred_top1.cpu().detach().numpy())
        b2 = calculate_error(startpoint2.cpu().detach().numpy(), pred_top2.cpu().detach().numpy())
        b3 = calculate_error(startpoint3.cpu().detach().numpy(), pred_top3.cpu().detach().numpy())
        b4 = calculate_error(startpoint4.cpu().detach().numpy(), pred_top4.cpu().detach().numpy())
        b5 = calculate_error(startpoint5.cpu().detach().numpy(), pred_top5.cpu().detach().numpy())

        # 返回检索结果和原有的误差
        return answers, scores, b1 + b2 + b3 + b4 + b5