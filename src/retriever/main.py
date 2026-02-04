# coding=utf-8
from dataloader import *
from model_retriever import *
import argparse
from torch.optim.lr_scheduler import ExponentialLR
from config_train import Config
from active_selector_CONQUER import *
import sys

sys.path.append("./")

from model_LSTM import *

from logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--al-epochs", dest="al_epochs", type=int, help="iterations of active learning")
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--dataset", dest="dataset")
    parser.add_argument("--embedding-dim", dest="embedding_dim", type=int)
    parser.add_argument("--early-stop-threshold", dest="early_stop_threshold", type=int)
    parser.add_argument("--eval-rate", dest="eval_rate", type=int, help="make evaluation each n epochs")
    parser.add_argument("--inner-lr", dest="inner_learning_rate", type=int)
    parser.add_argument("--lr", dest="learning_rate", type=float)
    parser.add_argument("--lr-decay", dest="learning_rate_decay", type=float)
    parser.add_argument("--model", dest="model_name", choices=["ConvE", "MLP"])
    parser.add_argument("--n-clusters", dest="n_clusters", type=int)
    parser.add_argument("--sample-size", dest="sample_size", type=int, help="number of training examples per one AL iteration")
    parser.add_argument("--sampling-mode", dest="sampling_mode", choices=["random", "uncertainty", "structured", "structured-uncertainty"])
    parser.add_argument("--training-mode", dest="training_mode", choices=["retrain", "incremental", "meta-incremental"])
    parser.add_argument("--window-size", dest="window_size", type=int)

    return parser.parse_args()


# 'What date was the 2001 film The Fast and the Furious released in theaters?'

def validate_v2(config, model, data_loader, device, num_entities, output_rank=False):
    model.eval()
    config.training = False

    hits = [[] for _ in range(10)]
    ranks = []

    top_results = []

    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    for i_batch, data in enumerate(loader):
        head = data[0].to(device)
        q1 = data[1].to(device)
        m1 = data[2].to(device)
        a1 = data[3]
        q2 = data[4].to(device)
        m2 = data[5].to(device)
        a2 = data[6]
        q3 = data[7].to(device)
        m3 = data[8].to(device)
        a3 = data[9]
        q4 = data[10].to(device)
        m4 = data[11].to(device)
        a4 = data[12]
        q5 = data[13].to(device)
        m5 = data[14].to(device)
        a5 = data[15]
        q_text1 = data[16]
        q_text2 = data[17]
        q_text3 = data[18]
        q_text4 = data[19]
        q_text5 = data[20]
        answers = model.get_score_ranked(head, q1, m1,
        q2, m2,
        q3, m3,
        q4, m4,
        q5, m5
        )

        batch_size = head.shape[0]

        if config.other_model is not None:
            all_scores = config.other_model.get_score_ranked(head, q1[:,0,:], m1[:,0,:],
            q2[:,0,:], m2[:,0,:],
            q3[:,0,:], m3[:,0,:],
            q4[:,0,:], m4[:,0,:],
            q5[:,0,:], m5[:,0,:]
            )
            mask = torch.zeros(batch_size, num_entities).to(device)
            idx = torch.range(0, batch_size-1).long()
            mask[idx, head] = 1

            # adj matrix
            adjmatrix = config.adj_matrix[head.cpu().numpy(),:].todense()
            adjmatrix = torch.tensor(adjmatrix).to(device)
            
       
        all_a = [a1, a2, a3, a4, a5]
        all_q = [q_text1, q_text2, q_text3, q_text4, q_text5]
        for idx_, anw in enumerate(answers):
            # for other model
            if config.other_model is not None:
                scores = all_scores[idx_]
                new_scores = scores + adjmatrix * 1000
                new_scores = new_scores - (mask*99999)
                # make the answer of RL socre very low
                new_scores = new_scores.scatter(dim=1, index=anw.type(torch.int64), value=-1)
                sorted, indices = torch.sort(new_scores, descending=True)
                indices = indices.cpu().numpy()
            
            anw = anw.cpu().numpy()
            head_numpy = head.cpu().numpy()
            for j in range(batch_size):
                #################
                # head_rank = np.where(anw[j]==head_numpy[j])
                # head_rank = head_rank[0]
                # if head_rank.size != 0:
                #     head_rank = head_rank[0]
                #     kk = anw[j].tolist()
                #     del kk[head_rank]
                #     kk.append(head_numpy[j])
                #     anw[j] = np.array(kk)
                #################
                rank = np.where(anw[j]==all_a[idx_][j].item())
                rank = rank[0]
                if rank.size == 0:
                    rank = np.where(indices[j]==all_a[idx_][j].item())[0][0]
                    #rank = 999
                    aaa = 0
                    # read score of other model
                    # make the answer of RL socre very low
                    # sort according to the embedding score distance [complEx] to head node
                else:
                    rank = rank[0]
                ans_list = [config.id2entity[x] for x in anw[j]]
                if output_rank:
                    if rank != 0 and rank < 3:
                        # output head, answer, results_predicted by model.
                        a = 0
                        #print([all_q[idx_][j], config.id2entity[head[j].item()], config.id2entity[all_a[idx_][j].item()], ans_list])
                
                pred_res = [all_q[idx_][j], config.id2entity[head[j].item()], config.id2entity[all_a[idx_][j].item()], ans_list]
                top_results.append(pred_res)
                a = 0
                ranks.append(rank+1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

    hitat8 = np.mean(hits[7])
    hitat5 = np.mean(hits[4])
    hitat3 = np.mean(hits[2])
    hitat2 = np.mean(hits[1])
    hitat1 = np.mean(hits[0])
    meanrank = np.mean(ranks)
    mrr = np.mean(1./np.array(ranks))
    print('Hits @8: {0}'.format(hitat8))
    print('Hits @5: {0}'.format(hitat5))
    print('Hits @3: {0}'.format(hitat3))
    print('Hits @2: {0}'.format(hitat2))
    print('Hits @1: {0}'.format(hitat1))
    print('Mean rank: {0}'.format(meanrank))
    print('Mean reciprocal rank: {0}'.format(mrr))

    # store file
    output_rank = False
    if output_rank:
        if config.is_cal_p1:
            with open(config.conversation_test_path + ".needRerank.step" + str(config.cal_p1_step), 'wb') as f:
                pickle.dump(top_results, f)
        else:
            with open(config.conversation_test_path + ".needRerank", 'wb') as f:
                pickle.dump(top_results, f)
    return [mrr, meanrank, hitat8, hitat5, hitat3]

def output_top(config, model, data_loader, device, num_entities):
    validate_v2(config=config, model=model, data_loader = data_loader, device=device, num_entities=num_entities, output_rank=True)

def train_again(config, model, optimizer, scheduler, dataloader, valid_data_loader, test_data_loader, device, num_entities, max_acc):
    #validate_v2(config=config, model=model, data_loader = test_data_loader, device=device, num_entities=num_entities, output_rank=True)

    best_hit1 = max_acc
    best_res = None
    config.training = True
    for epoch in range(config.al_epochs):
        log.info("{} iteration of active learning: started".format(epoch + 1))
        log.info("Train model: started")

        model.train()
        loader = tqdm(dataloader, total=len(dataloader), unit="batches")
        running_loss = 0
        for i_batch, data in enumerate(loader):
            model.zero_grad()
            loss = model(data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device),
            data[4].to(device), data[5].to(device), data[6].to(device),
            data[7].to(device), data[8].to(device), data[9].to(device),
            data[10].to(device), data[11].to(device), data[12].to(device),
            data[13].to(device), data[14].to(device), data[15].to(device),
            data[21].to(device), data[22].to(device), data[23].to(device), data[24].to(device), data[25].to(device)
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*config.batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, config.al_epochs))
            loader.update()
        scheduler.step()

        if epoch % 3 == 0 and epoch != 0:
            # evaluate
            model.eval()
            print("evaluate")
            valid_res = validate_v2(config=config, model=model, data_loader = valid_data_loader, device=device, num_entities=num_entities)
            if best_hit1 < valid_res[-1]: # here the valid_res will also effect the model results. Try to use different index
                best_hit1 = valid_res[-1]
                best_res = valid_res
                # run test
                print("-----------------TEST--------------------------")
                test_res = validate_v2(config=config, model=model, data_loader = test_data_loader, device=device, num_entities=num_entities)
                print(test_res)
                # save model
                if config.save_model:
                    torch.save(model.state_dict(), config.model_save_path)


    
    # print final results
    print("*******************************************************************")
    print(best_res)
    return best_hit1

          
def main():
    args = parse_args()

    config = Config(args)

    # load data
    log.info("Initializing training sample streamer")

    # how many rounds
    
    kg_node_embeddings = np.load(config.entity_emb_path)
    relation_embeddings = np.load(config.relation_emb_path)
    
    print(config.batch_size)
    print(config.conversation_path)

    # this place should be deleted
    if config.sample_strategy == 'kmeans':
        train_batcher = ActiveKMeansSelector(config.conversation_path, kg_node_embeddings, config.entity2id, 
        config.sample_size, config.cluster_num, config)
    else:
        train_batcher = ActiveRandomSelector(config.conversation_path, config.sample_size, config)
    
    # use a extremaly large value to get all
    valid_batcher = ActiveRandomSelector(config.conversation_valid_path, 999999, config)
    valid_data = valid_batcher.next()
    valid_conv_dataset = DatasetConversation(valid_data, config.entity2id, config.relation2id, config.entity2neighbor_relation, config.ht2relation)
    valid_data_loader = DataLoader(valid_conv_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

    test_batcher = ActiveRandomSelector(config.conversation_test_path, 999999, config)
    test_data = test_batcher.next()
    test_conv_dataset = DatasetConversation(test_data, config.entity2id, config.relation2id, config.entity2neighbor_relation, config.ht2relation)
    test_data_loader = DataLoader(test_conv_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)
    
    log.info("Initializing test_rank streamer")

    # initialize model
    kg_node_embeddings = torch.tensor(kg_node_embeddings)
    relation_embeddings = torch.tensor(relation_embeddings)

    if config.parallel:
        device = torch.device("cuda:0")
    else:
        device = torch.device(config.gpu if config.use_cuda else "cpu")

    config.device = device

    ########### read lstm model
    if config.load_lstm_model:
        rl_lstm_model = RL_LSTM(embedding_dim=config.embedding_dim, num_entities = len(config.entity2id), pretrained_embeddings=kg_node_embeddings, 
        freeze=config.freeze, device=config.device, entdrop = config.entdrop, reldrop = config.reldrop, scoredrop = config.scoredrop, 
        l3_reg = config.l3_reg, ls = config.ls, do_batch_norm=config.do_batch_norm)
        rl_lstm_model.load_state_dict(torch.load("./pretrain_QA_model/best_model.pt"))
        config.other_model = rl_lstm_model
        if config.use_cuda:
            config.other_model.to(device)
    ###############################################

    model = RelationExtractor(config, embedding_dim=config.embedding_dim, num_entities = len(config.entity2id), relation_emb = relation_embeddings, 
        pretrained_embeddings=kg_node_embeddings, 
        freeze=config.freeze, device=config.device, entdrop = config.entdrop, reldrop = config.reldrop, scoredrop = config.scoredrop, 
        l3_reg = config.l3_reg, ls = config.ls, do_batch_norm=config.do_batch_norm)
    
    if config.load_model:
        model.load_state_dict(torch.load(config.model_save_path))

    if torch.cuda.device_count() > 1 and config.parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    if config.use_cuda:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ExponentialLR(optimizer, config.decay)
    optimizer.zero_grad()

    # each round use action learning to get some new data

    if config.reranking:
        train_batcher = ActiveRandomSelector(config.conversation_test_path, 999999)
        new_data = train_batcher.next()
        if len(new_data) <= 0:
            print("end")
            return
        conv_dataset = DatasetConversation(new_data, config.entity2id, config.relation2id, config.entity2neighbor_relation, config.ht2relation)
        data_loader = DataLoader(conv_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        output_top(config, model, data_loader, device, len(config.entity2id))
    else:
        max_acc = 0
        for i in range(config.active_round):
            print("###################################################")
            new_data = train_batcher.next()
            if len(new_data) <= 0:
                print("end")
                break
            conv_dataset = DatasetConversation(new_data, config.entity2id, config.relation2id, config.entity2neighbor_relation, config.ht2relation)
            # maybe should set shuffle=False here?
            data_loader = DataLoader(conv_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
            max_acc = train_again(config, model, optimizer, scheduler, data_loader, valid_data_loader, test_data_loader, device, len(config.entity2id), max_acc)
            #max_acc = 0
    
    # test data
        

main()



