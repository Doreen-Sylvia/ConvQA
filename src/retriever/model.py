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

"""
This model has the same code as model_LSTM.py
"""
class RelationExtractor(nn.Module):

    def __init__(self, embedding_dim, num_entities, pretrained_embeddings, freeze, device, 
    entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0, ls=0.0, do_batch_norm=True):
        super(RelationExtractor, self).__init__()
        self.device = device
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm
        if not self.do_batch_norm:
            print('Not doing batch norm')
        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
    
        multiplier = 2
        self.getScores = self.ComplEx
        
        self.hidden_dim = 768
        
        self.num_entities = num_entities
        self.loss = self.kge_loss

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.fcnn_dropout = torch.nn.Dropout(0.1)

        # self.pretrained_embeddings = pretrained_embeddings
        # random.shuffle(pretrained_embeddings)
        # print(pretrained_embeddings[0])
        print('Frozen:', self.freeze)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=self.freeze)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=self.freeze)
        print(self.embedding.weight.shape)

        self.lstm_layer_size = 2
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, self.lstm_layer_size, batch_first=True).to(self.device)

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


    def set_bn_eval(self):
        self.bn0.eval()
        self.bn2.eval()

    def kge_loss(self, scores, targets):
        # loss = torch.mean(scores*targets)
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def applyNonLinear(self, outputs):
        outputs = self.hidden2rel(outputs)
        # outputs = self.hidden2rel_base(outputs)
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
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)

        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        # pred = torch.sigmoid(score)
        pred = score
        return pred

    
    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding

    def forward(self, p_head, question_tokenized1, attention_mask1, p_tail1,
        question_tokenized2, attention_mask2, p_tail2,
        question_tokenized3, attention_mask3, p_tail3,
        question_tokenized4, attention_mask4, p_tail4,
        question_tokenized5, attention_mask5, p_tail5):
        head = p_head
        question_embedding1 = self.getQuestionEmbedding(question_tokenized1, attention_mask1)
        question_embedding2 = self.getQuestionEmbedding(question_tokenized2, attention_mask2)
        question_embedding3 = self.getQuestionEmbedding(question_tokenized3, attention_mask3)
        question_embedding4 = self.getQuestionEmbedding(question_tokenized4, attention_mask4)
        question_embedding5 = self.getQuestionEmbedding(question_tokenized5, attention_mask5)

        question_embedding1 = question_embedding1.unsqueeze(1)
        question_embedding2 = question_embedding2.unsqueeze(1)
        question_embedding3 = question_embedding3.unsqueeze(1)
        question_embedding4 = question_embedding4.unsqueeze(1)
        question_embedding5 = question_embedding5.unsqueeze(1)

        h_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim, requires_grad=False).to(self.device)
        c_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim, requires_grad=False).to(self.device)

        # output shoule be the same as h, check
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

        p_head = self.embedding(p_head)
        pred1 = self.getScores(p_head, rel_embedding1)
        pred2 = self.getScores(p_head, rel_embedding2)
        pred3 = self.getScores(p_head, rel_embedding3)
        pred4 = self.getScores(p_head, rel_embedding4)
        pred5 = self.getScores(p_head, rel_embedding5)
        #actual = F.one_hot(p_tail, num_classes=self.num_entities)
        actual1 = p_tail1
        if self.label_smoothing:
            actual1 = ((1.0-self.label_smoothing)*actual1) + (1.0/actual1.size(1)) 
        loss1 = self.loss(pred1, actual1)

        actual2 = p_tail2
        if self.label_smoothing:
            actual2 = ((1.0-self.label_smoothing)*actual2) + (1.0/actual2.size(1)) 
        loss2 = self.loss(pred2, actual2)

        actual3 = p_tail3
        if self.label_smoothing:
            actual3 = ((1.0-self.label_smoothing)*actual3) + (1.0/actual3.size(1)) 
        loss3 = self.loss(pred3, actual3)

        actual4 = p_tail4
        if self.label_smoothing:
            actual4 = ((1.0-self.label_smoothing)*actual4) + (1.0/actual4.size(1)) 
        loss4 = self.loss(pred4, actual4)

        actual5 = p_tail5
        if self.label_smoothing:
            actual5 = ((1.0-self.label_smoothing)*actual5) + (1.0/actual5.size(1)) 
        loss5 = self.loss(pred5, actual5)

        loss = loss1 + loss2 + loss3 + loss4 + loss5
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        
        return loss
    

    def get_score_ranked(self, head, question_tokenized1, attention_mask1,
        question_tokenized2, attention_mask2,
        question_tokenized3, attention_mask3,
        question_tokenized4, attention_mask4,
        question_tokenized5, attention_mask5):

        question_embedding1 = self.getQuestionEmbedding(question_tokenized1, attention_mask1)
        question_embedding2 = self.getQuestionEmbedding(question_tokenized2, attention_mask2)
        question_embedding3 = self.getQuestionEmbedding(question_tokenized3, attention_mask3)
        question_embedding4 = self.getQuestionEmbedding(question_tokenized4, attention_mask4)
        question_embedding5 = self.getQuestionEmbedding(question_tokenized5, attention_mask5)

        # a = torch.cat([question_tokenized1, question_tokenized2, question_tokenized3, question_tokenized4, question_tokenized5], dim=0)
        # b = torch.cat([attention_mask1, attention_mask2, attention_mask3, attention_mask4, attention_mask5])

        # question_embedding_all = self.getQuestionEmbedding(a, b)
        # question_embedding_all_chunk = question_embedding_all.chunk(5)

        # question_embedding1 = question_embedding_all_chunk[0].unsqueeze(1)
        # question_embedding2 = question_embedding_all_chunk[1].unsqueeze(1)
        # question_embedding3 = question_embedding_all_chunk[2].unsqueeze(1)
        # question_embedding4 = question_embedding_all_chunk[3].unsqueeze(1)
        # question_embedding5 = question_embedding_all_chunk[4].unsqueeze(1)

        question_embedding1 = question_embedding1.unsqueeze(1)
        question_embedding2 = question_embedding2.unsqueeze(1)
        question_embedding3 = question_embedding3.unsqueeze(1)
        question_embedding4 = question_embedding4.unsqueeze(1)
        question_embedding5 = question_embedding5.unsqueeze(1)

        h_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim, requires_grad=False).to(self.device)
        c_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim, requires_grad=False).to(self.device)

        # output shoule be the same as h, check
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


        
        head = self.embedding(head)

        scores1 = self.getScores(head, rel_embedding1)
        scores2 = self.getScores(head, rel_embedding2)
        scores3 = self.getScores(head, rel_embedding3)
        scores4 = self.getScores(head, rel_embedding4)
        scores5 = self.getScores(head, rel_embedding5)


        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        return scores1, scores2, scores3, scores4, scores5



