from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from tqdm import tqdm
import os

    
class Experiment:
# 初始化模型中的训练参数
    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., outfile='tucker.model', valid_steps=1, loss_type='BCE', do_batch_norm=1,
                 dataset='', model='Rotat3', l3_reg = 0.0, load_from = ''):
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = False
        self.outfile = outfile
        self.valid_steps = valid_steps
        self.model = model
        self.l3_reg = l3_reg
        self.loss_type = loss_type
        self.load_from = load_from
        if do_batch_norm == 1:
            do_batch_norm = True
        else:
            do_batch_norm = False
        #dropout正则化是一种防止过拟合的正则化技术
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "model": model, "loss_type": loss_type,
                       "do_batch_norm": do_batch_norm, "l3_reg": l3_reg}

#数据索引映射，将文本三元组转换成数字索引的形式
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

#构建ER词汇表，创建（头实体，关系）到尾实体的映射
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab


#批次获取数据
    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = torch.zeros([len(batch), len(d.entities)], dtype=torch.float32)
        if self.cuda:
            targets = targets.cuda(self.cuda_num)
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return np.array(batch), targets


#模型评估
    def evaluate(self, model, data):
        model.eval()#设置评估模式
        hits = [[] for _ in range(10)]#存储不同级别的命中结果
        ranks = []#存储排名结果
        test_data_idxs = self.get_data_idxs(data)#转换为索引形式
        er_vocab = self.get_er_vocab(test_data_idxs)#构建ER词汇表

        print("Number of data points: %d" % len(test_data_idxs))
        for i in tqdm(range(0, len(test_data_idxs), self.batch_size)):
            data_batch = np.array(test_data_idxs[i: i+self.batch_size])
            e1_idx = torch.tensor(data_batch[:,0])#批次处理中，获取其中的头实体e1,关系r,尾实体e2索引
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda(self.cuda_num)
                r_idx = r_idx.cuda(self.cuda_num)
                e2_idx = e2_idx.cuda(self.cuda_num)
            predictions = model.forward(e1_idx, r_idx)#获取预测分数

            # following lines commented means RAW evaluation (not filtered)
            for j in range(data_batch.shape[0]):#过滤式评估，将训练集中已经存在的三元组分数置零
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]#获取所有有效尾实体
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0# 将所有有效尾实体分数置零
                predictions[j, e2_idx[j]] = target_value
            #排序和统计
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)
                #统计不同级别命中的情况
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)#命中
                    else:
                        hits[hits_level].append(0.0)#未命中

        hitat10 = np.mean(hits[9])#指标计算，分别对应Hit@10,Hit@3,Hit@1,以及平均排名和平均倒数排名
        hitat3 = np.mean(hits[2])
        hitat1 = np.mean(hits[0])
        meanrank = np.mean(ranks)
        mrr = np.mean(1./np.array(ranks))
        print('Hits @10: {0}'.format(hitat10))
        print('Hits @3: {0}'.format(hitat3))
        print('Hits @1: {0}'.format(hitat1))
        print('Mean rank: {0}'.format(meanrank))
        print('Mean reciprocal rank: {0}'.format(mrr))
        return [mrr, meanrank, hitat10, hitat3, hitat1]

    def write_embedding_files(self, model):#嵌入文件保存
        model.eval()
        model_folder = "../../models/kg_embeddings/%s/" % self.dataset
        data_folder = "../../data/preprocessed/dialogue_kg/" 
        embedding_type = self.model
        if os.path.exists(model_folder) == False:
            os.makedirs(model_folder)
        R_numpy = model.R.weight.data.cpu().numpy()
        E_numpy = model.E.weight.data.cpu().numpy()
        bn_list = []
        for bn in [model.bn0, model.bn1, model.bn2]:
            bn_weight = bn.weight.data.cpu().numpy()
            bn_bias = bn.bias.data.cpu().numpy()
            bn_running_mean = bn.running_mean.data.cpu().numpy()
            bn_running_var = bn.running_var.data.cpu().numpy()
            bn_numpy = {}
            bn_numpy['weight'] = bn_weight#批归一化参数 (bn0.npy, bn1.npy, bn2.npy): 权重、偏置、运行均值和方差
            bn_numpy['bias'] = bn_bias
            bn_numpy['running_mean'] = bn_running_mean
            bn_numpy['running_var'] = bn_running_var
            bn_list.append(bn_numpy)
            
        if embedding_type == 'TuckER':
            W_numpy = model.W.detach().cpu().numpy()
            
        np.save(model_folder +'/E.npy', E_numpy)
        np.save(model_folder +'/R.npy', R_numpy)
        for i, bn in enumerate(bn_list):
            np.save(model_folder + '/bn' + str(i) + '.npy', bn)

        if embedding_type == 'TuckER':
            np.save(model_folder +'/W.npy', W_numpy)

        f = open(data_folder + '/entities.dict', 'r')
        f2 = open(model_folder + '/entities.dict', 'w')
        ents = {}
        idx2ent = {}
        for line in f:
            line = line.rstrip().split('\t')
            name = line[0]
            id = int(line[1])
            ents[name] = id
            idx2ent[id] = name
            f2.write(str(id) + '\t' + name + '\n')
        f.close()
        f2.close()
        f = open(data_folder + '/relations.dict', 'r')
        f2 = open(model_folder + '/relations.dict', 'w')
        rels = {}
        idx2rel = {}
        for line in f:
            line = line.strip().split('\t')
            name = line[0]
            id = int(line[1])
            rels[name] = id
            idx2rel[id] = name
            f2.write(str(id) + '\t' + name + '\n')
        f.close()
        f2.close()


    def train_and_eval(self):#主训练循环
        torch.set_num_threads(2)#设置线程数
        best_valid = [0, 0, 0, 0, 0]#存储最佳验证结果[MRR,MR,H@10,H@3,H@1]
        best_test = [0, 0, 0, 0, 0]#存储对应的测试结果
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        f = open('../../data/preprocessed/dialogue_kg/entities.dict', 'w')#保存字典文件
        for key, value in self.entity_idxs.items():
            f.write(key + '\t' + str(value) +'\n')
        f.close()
        f = open('../../data/preprocessed/dialogue_kg/relations.dict', 'w')
        for key, value in self.relation_idxs.items():
            f.write(key + '\t' + str(value) +'\n')
        f.close()
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        print('Entities: %d' % len(self.entity_idxs))
        print('Relations: %d' % len(self.relation_idxs))
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model.init()
        if self.load_from != '':#加载预训练模型
            fname = self.load_from
            checkpoint = torch.load(fname)
            model.load_state_dict(checkpoint)
        if self.cuda:#移动到GPU
            model.cuda(self.cuda_num)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:#学习率衰减
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")

        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()#梯度清零
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda(self.cuda_num)
                    r_idx = r_idx.cuda(self.cuda_num)
                predictions = model.forward(e1_idx, r_idx)#前向传播
                if self.label_smoothing:#标签平滑
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)#计算损失和反向传播
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            if it%100 == 0:
                print('Epoch', it, ' Epoch time', time.time()-start_train, ' Loss:', np.mean(losses))
            model.eval()
            
            with torch.no_grad():#验证和保存
                if it % self.valid_steps == 0:
                    start_test = time.time()
                    print("Validation:")#验证集评估
                    valid = self.evaluate(model, d.valid_data)
                    print("Test:")#测试集评估
                    test = self.evaluate(model, d.test_data)
                    valid_mrr = valid[0]
                    test_mrr = test[0]
                    if valid_mrr >= best_valid[0]:#若MRR提升，则保存最佳模型
                        best_valid = valid
                        best_test = test
                        print('Validation MRR increased.')
                        print('Saving model...')
                        self.write_embedding_files(model)
                        print('Model saved!')    
                    
                    print('Best valid:', best_valid)
                    print('Best Test:', best_test)
                    print('Dataset:', self.dataset)
                    print('Model:', self.model)

                    print(time.time()-start_test)
                    print('Learning rate %f | Decay %f | Dim %d | Input drop %f | Hidden drop 2 %f | LS %f | Batch size %d | Loss type %s | L3 reg %f' % 
                        (self.learning_rate, self.decay_rate, self.ent_vec_dim, self.kwargs["input_dropout"], 
                         self.kwargs["hidden_dropout2"], self.label_smoothing, self.batch_size,
                         self.loss_type, self.l3_reg))        
           

        
## --data MetaQA_half --num_iterations 1000 --batch_size 256 \ --lr 0.005 --dr 1.0 --edim 200 --rdim 200 --input_dropout 0.2 \ --hidden_dropout1 0.2
## --hidden_dropout2 0.3 --label_smoothing 0.1 \ --valid_steps 10 --model ComplEx \ --loss_type BCE --do_batch_norm 1 --l3_reg 0.0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="preprocess/dialogue_kg", nargs="?",
                    help="Which data to use: dialogue_kg.")
    parser.add_argument("--num_iterations", type=int, default=100, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=256, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--model", type=str, default='ComplEx', nargs="?",
                    help="Model.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.2, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.2, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.3, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--outfile", type=str, default='ComplEx.model', nargs="?",
                    help="File to save")
    parser.add_argument("--valid_steps", type=int, default=10, nargs="?",
                    help="Epochs before u validate")
    parser.add_argument("--loss_type", type=str, default='BCE', nargs="?",
                    help="Loss type")
    parser.add_argument("--do_batch_norm", type=int, default=1, nargs="?",
                    help="Do batch norm or not (0, 1)")
    parser.add_argument("--l3_reg", type=float, default=0.0, nargs="?",
                    help="l3 reg hyperparameter")
    parser.add_argument("--load_from", type=str, default='', nargs="?",
                    help="load from state dict")

    args = parser.parse_args()
    dataset = args.data
    data_dir = "../../data/preprocessed/dialogue_kg/"
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, outfile=args.outfile,
                            valid_steps=args.valid_steps, loss_type=args.loss_type, do_batch_norm=args.do_batch_norm,
                            dataset=args.data, model=args.model, l3_reg=args.l3_reg, load_from=args.load_from)
    experiment.train_and_eval()
                

