import torch
import logging
import math
import os
import numpy as np
import scipy.sparse as ss


log = logging.getLogger()

# number of training samples after preprocessing

class Config(object):
    
    
    def read_dict(self, file_path):
        name2id = {}
        id2name = {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                id = line[0]
                n = line[1]
                name2id[n] = int(id)
                id2name[int(id)] = n
        return name2id, id2name

    def get_adj_map(self):
        adj_map_no_rel = {}
        with open(self.kg_path, 'r') as f:
            for line in f.readlines():
                ws = line.strip().split("\t")
                if len(ws) < 3:
                    continue
                
                h = ws[0]
                p = ws[1]
                t = ws[2]
               
                if h not in adj_map_no_rel:
                    adj_map_no_rel[h] = set()
                if t not in adj_map_no_rel:
                    adj_map_no_rel[t] = set()
                
                adj_map_no_rel[h].add(t)
                adj_map_no_rel[t].add(h)
    
        f.close()
        self.adj_map_no_rel = adj_map_no_rel
        # make matrix
        all_row = []
        all_col = []
        all_value = []
        for k, v in adj_map_no_rel.items():
            row = []
            col = []
            r = self.entity2id[k]
            for l in v:
                row.append(r)
                col.append(self.entity2id[l])
                all_value.append(1)
            all_row.extend(row)
            all_col.extend(col)

        #all_row_indices = torch.tensor(all_row)
        #all_col_indices = torch.tensor(all_col)
        #all_value = torch.tensor(all_value)
        self.adj_matrix = ss.csc_matrix((all_value,(all_row,all_col)))
        #self.adj_matrix = torch.sparse_csr_tensor(all_row_indices, all_col_indices, all_value).to(self.device)

        return

    def get_adj_map_with_rel(self):
        self.triple_num = 0
        adj_map_no_rel = {}
        adj_map_rel = {}
        with open(self.kg_path, 'r') as f:
            for line in f.readlines():
                ws = line.strip().split("\t")
                if len(ws) < 3:
                    continue
                self.triple_num += 1
                h = ws[0]
                p = ws[1]
                t = ws[2]
                p_reverse = p+"_reverse"
                h = self.entity2id[h]
                t = self.entity2id[t]
                p = self.relation2id[p]
                p_reverse = self.relation2id[p_reverse]

                if h not in adj_map_rel:
                    adj_map_rel[h] = {}
                if t not in adj_map_rel:
                    adj_map_rel[t] = {}
                if p not in adj_map_rel[h]:
                    adj_map_rel[h][p] = set()
                
                if  p_reverse not in adj_map_rel[t]:
                    adj_map_rel[t][p_reverse] = set()
                adj_map_rel[h][p].add(t)
                adj_map_rel[t][p_reverse].add(h)
    
        f.close()
        # make matrix

        return adj_map_rel

    def get_h_t_to_rel(self):
        h_t_to_p = {}
        with open(self.kg_path, 'r') as f:
            for line in f.readlines():
                ws = line.strip().split("\t")
                ws[0] = self.entity2id[ws[0]]
                ws[2] = self.entity2id[ws[2]]
                ws[1] = self.relation2id[ws[1]]

                key = (ws[0], ws[2])
                if key not in h_t_to_p:
                    h_t_to_p[key] = set()
                h_t_to_p[key].add(ws[1])
        self.h_t_to_rel = h_t_to_p
        return h_t_to_p
    
    def get_entity2neighbor_relation(self):
        adj_map_rel = {}
        with open(self.kg_path, 'r') as f:
            for line in f.readlines():
                ws = line.strip().split("\t")
                if len(ws) < 3:
                    continue
                h = ws[0]
                p = ws[1]
                t = ws[2]
                if h not in adj_map_rel:
                    adj_map_rel[h] = set()
                if t not in adj_map_rel:
                    adj_map_rel[t] = set()
                adj_map_rel[h].add(p)
                adj_map_rel[t].add(p)
        return adj_map_rel

    def __init__(self, args):
        self.entity_emb_path = "kg_embeddings/wikidata/E.npy"
        self.relation_emb_path = "kg_embeddings/wikidata/R.npy"
        self.entity_dict_path = "kg_embeddings/wikidata/entities.dict"
        self.relation_dict_path = "kg_embeddings/wikidata/relations.dict"

        self.conversation_path = 'data/CONQUER/ConvRef_trainset_processed.json'
        self.conversation_valid_path = 'data/CONQUER/ConvRef_devset_processed.json'
        self.conversation_test_path = 'data/CONQUER/ConvRef_testset_processed.json'

        # self.conversation_path = 'bart_ref/ConvRef_trainset_processed.json'
        # self.conversation_valid_path = 'bart_ref/ConvRef_devset_processed.json'
        # self.conversation_test_path = 'bart_ref/ConvRef_testset_processed.json'

        #self.conversation_path = 'bart_ref_wo_ans/ConvRef_trainset_processed.json'
        #self.conversation_valid_path = 'bart_ref_wo_ans/ConvRef_devset_processed.json'
        #self.conversation_test_path = 'bart_ref_wo_ans/ConvRef_testset_processed.json'

        # self.conversation_path = 'bart_ref_wo_ans_dev/ConvRef_trainset_processed.json'g
        # self.conversation_valid_path = 'bart_ref_wo_ans_dev/ConvRef_devset_processed.json'
        # self.conversation_test_path = 'bart_ref_wo_ans_dev/ConvRef_testset_processed.json'
        
        
        # self.conversation_path = 'gpt2_ref_wo_ans/ConvRef_trainset_processed.json'
        # self.conversation_valid_path = 'gpt2_ref_wo_ans/ConvRef_devset_processed.json'
        # self.conversation_test_path = 'gpt2_ref_wo_ans/ConvRef_testset_processed.json'

        self.kg_path = 'data/wikidata/train.txt'
        self.batch_size = 12 # the experiment is done when batch_size = 12
        self.rollout_num = 20

        self.test_batch_size = 8

        self.sample_strategy = 'kmeans--'
        self.sample_size = 300000 # 20
        self.cluster_num = 5
        self.num_workers = 4

        # active learning
        self.al_epochs = 20  # inner epoch
        self.active_round = 3  # outer epoch

        self.embedding_dim = 200
        self.freeze = False

        self.device = 1
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
        self.entity2id, self.id2entity = self.read_dict(self.entity_dict_path)
        self.relation2id, self.id2relation = self.read_dict(self.relation_dict_path)
        self.adj_list = self.get_adj_map_with_rel()
        self.num_entities = len(self.entity2id)

        self.bucket_interval = 10
        # change it late
        self.page_rank_file = "data/wikidata/page_rank_score.txt"
        self.use_action_space_bucketing = True
        self.beta=0.02
        
        self.model_save_path = "./src_reformulate/best_model_3E_fuxian.pt"
        self.save_model = True
        self.load_model = False
        self.load_lstm_model = True
        self.other_model = None
        self.parallel = False
        self.reranking = False
        if self.load_lstm_model:
            self.get_adj_map()

        # for contrastive learning
        self.n_views = 2
        self.temperature = 0.1
        self.training = False

        self.ht2relation = self.get_h_t_to_rel()
        self.entity2neighbor_relation = self.get_entity2neighbor_relation()

        self.is_cal_p1 = False
        self.cal_p1_step = 4

        
