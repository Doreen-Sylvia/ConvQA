# coding=utf-8
from dataloader import *
from model_chose import *
import argparse
from torch.optim.lr_scheduler import ExponentialLR
from config_train_chose import Config
from active_selector_CONQUER import *
import sys

sys.path.append("./")

from model_LSTM import *

from logger import setup_logger

import os
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['SENTENCE_TRANSFORMERS_HOME']='./model'
os.environ['CURL_CA_BUNDLE'] = ''

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("D:/Project/ConvQA/model/roberta-base")

"""
This one is used to pretrain the topic entity selection model (model_LSTM.py)
But because the model will choose the global topic entity most of the time, 
in RL based model, we simplely use the global topic entity for each input query
"""


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
    parser.add_argument("--sample-size", dest="sample_size", type=int,
                        help="number of training examples per one AL iteration")
    parser.add_argument("--sampling-mode", dest="sampling_mode",
                        choices=["random", "uncertainty", "structured", "structured-uncertainty"])
    parser.add_argument("--training-mode", dest="training_mode", choices=["retrain", "incremental", "meta-incremental"])
    parser.add_argument("--window-size", dest="window_size", type=int)
    # ============ 新增参数 ============
    parser.add_argument("--use-topic-aware", dest="use_topic_aware", action="store_true",
                        help="enable topic-aware features")
    parser.add_argument("--topic-model-type", dest="topic_model_type", default="bertopic",
                        choices=["bertopic", "lda", "keyword"])
    parser.add_argument("--num-topics", dest="num_topics", type=int, default=15)

    return parser.parse_args()


def validate_v2(config, model, data_loader, device, num_entities, output_rank=False):
    model.eval()
    config.training = False

    hits = [[] for _ in range(10)]
    ranks = []

    top_results = []

    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    total_error = 0

    # ============ 新增：检索性能指标 ============
    all_recalls = {1: [], 5: [], 10: []}
    all_mrrs = []

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

        # ============ 修改：添加主题信息 ============
        if len(data) > 26:  # 如果有主题信息
            topic_vectors = data[26].to(device) if len(data) > 26 else None
            topic_similarities = data[27].to(device) if len(data) > 27 else None
            topic_labels = data[28] if len(data) > 28 else None
        else:
            topic_vectors = None
            topic_similarities = None
            topic_labels = None

        # ============ 修改：调用新的推理方法 ============
        answers, scores = model.get_score_ranked(
            head, q1, m1, q2, m2, q3, m3, q4, m4, q5, m5,
            question_texts=[q_text1, q_text2, q_text3, q_text4, q_text5]
        )

        # ============ 新增：计算检索指标 ============
        true_answers = [a1, a2, a3, a4, a5]
        batch_recalls, batch_mrr = calculate_retrieval_metrics(answers, true_answers)

        for k in [1, 5, 10]:
            all_recalls[k].extend(batch_recalls[k])
        all_mrrs.extend(batch_mrr)

        # 原有误差计算逻辑
        batch_error = calculate_batch_error(answers, true_answers)
        total_error = total_error + batch_error

    print(f"Total Error: {total_error}")

    # ============ 新增：输出检索性能 ============
    if output_rank:
        print("\n=== 检索性能指标 ===")
        for k in [1, 5, 10]:
            recall = np.mean(all_recalls[k]) if all_recalls[k] else 0
            print(f"Recall@{k}: {recall:.4f}")
        mrr = np.mean(all_mrrs) if all_mrrs else 0
        print(f"MRR: {mrr:.4f}")

    return total_error


def calculate_retrieval_metrics(predictions, true_answers):
    """计算检索性能指标"""
    recalls = {1: [], 5: [], 10: []}
    mrrs = []

    for pred_list, true_answer in zip(predictions, true_answers):
        if not isinstance(pred_list, list):
            pred_list = [pred_list]

        # 计算Recall@K
        for k in [1, 5, 10]:
            if true_answer in pred_list[:k]:
                recalls[k].append(1.0)
            else:
                recalls[k].append(0.0)

        # 计算MRR
        if true_answer in pred_list:
            rank = pred_list.index(true_answer) + 1
            mrrs.append(1.0 / rank)
        else:
            mrrs.append(0.0)

    return recalls, mrrs


def calculate_batch_error(predictions, true_answers):
    """计算批次误差"""
    error = 0
    for pred_list, true_answer in zip(predictions, true_answers):
        if not isinstance(pred_list, list):
            pred_list = [pred_list]

        if true_answer not in pred_list:
            error += 1
    return error


def output_top(config, model, data_loader, device, num_entities):
    validate_v2(config=config, model=model, data_loader=data_loader, device=device,
                num_entities=num_entities, output_rank=True)


def train_again(config, model, optimizer, scheduler, dataloader, valid_data_loader, test_data_loader, device,
                num_entities, max_acc):
    # ============ 修改：添加主题信息输出 ============
    print("开始验证测试集性能...")
    test_res = validate_v2(config=config, model=model, data_loader=test_data_loader,
                           device=device, num_entities=num_entities, output_rank=True)

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

            # ============ 修改：处理主题信息 ============
            if len(data) > 26:  # 如果有主题信息
                topic_vectors = data[26].to(device)
                topic_similarities = data[27].to(device)
                topic_labels = data[28]
                question_texts = [data[16], data[17], data[18], data[19], data[20]]
            else:
                topic_vectors = None
                topic_similarities = None
                topic_labels = None
                question_texts = None

            loss = model(
                data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device),
                data[4].to(device), data[5].to(device), data[6].to(device),
                data[7].to(device), data[8].to(device), data[9].to(device),
                data[10].to(device), data[11].to(device), data[12].to(device),
                data[13].to(device), data[14].to(device), data[15].to(device),
                data[21].to(device), data[22].to(device), data[23].to(device),
                data[24].to(device), data[25].to(device),
                question_texts=question_texts  # 新增参数
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss / ((i_batch + 1) * config.batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, config.al_epochs))
            loader.update()

        scheduler.step()

        if epoch % 3 == 0 and epoch != 0:
            # evaluate
            model.eval()
            print("开始验证...")
            valid_res = validate_v2(config=config, model=model, data_loader=valid_data_loader,
                                    device=device, num_entities=num_entities, output_rank=True)

            if best_hit1 > valid_res:
                best_hit1 = valid_res
                best_res = valid_res
                # run test
                print("-----------------测试集性能 --------------------------")
                test_res = validate_v2(config=config, model=model, data_loader=test_data_loader,
                                       device=device, num_entities=num_entities, output_rank=True)
                print(f"测试集误差: {test_res}")

                # save model
                if config.save_model:
                    torch.save(model.state_dict(), config.model_save_path)
                    print(f"模型已保存到: {config.model_save_path}")

    # print final results
    print("*******************************************************************")
    print(f"最佳验证误差: {best_res}")
    return best_hit1


def main():
    setup_logger()  # 添加这行来初始化日志
    args = parse_args()
    args = parse_args()

    config = Config(args)

    # ============ 新增：初始化主题提取器 ============
    if config.use_topic_aware:
        print("初始化主题提取器...")
        from model_retriever import NeuralTopicExtractor
        topic_extractor = NeuralTopicExtractor(config, config.device)
    else:
        topic_extractor = None
        print("未启用主题感知功能")

    # load data
    log.info("Initializing training sample streamer")

    # how many rounds

    kg_node_embeddings = np.load(config.entity_emb_path,allow_pickle=True)
    relation_embeddings = np.load(config.relation_emb_path,allow_pickle=True)

    if config.sample_strategy == 'kmeans':
        train_batcher = ActiveKMeansSelector(config.conversation_path, kg_node_embeddings, config.entity2id,
                                             config.sample_size, config.cluster_num, config)
    else:
        train_batcher = ActiveRandomSelector(config.conversation_path, config.sample_size, config)

    # use a extremaly large value to get all
    valid_batcher = ActiveRandomSelector(config.conversation_valid_path, 999999, config)
    valid_data = valid_batcher.next()
    # ============ 修改：添加主题提取器到数据集 ============
    valid_conv_dataset = DatasetConversation(valid_data, config.entity2id, config.relation2id,
                                             config.entity2neighbor_relation, config.ht2relation,
                                             topic_extractor=topic_extractor)
    valid_data_loader = DataLoader(valid_conv_dataset, batch_size=config.test_batch_size, shuffle=False,
                                   num_workers=config.num_workers)

    test_batcher = ActiveRandomSelector(config.conversation_test_path, 999999, config)
    test_data = test_batcher.next()
    # ============ 修改：添加主题提取器到数据集 ============
    test_conv_dataset = DatasetConversation(test_data, config.entity2id, config.relation2id,
                                            config.entity2neighbor_relation, config.ht2relation,
                                            topic_extractor=topic_extractor)
    test_data_loader = DataLoader(test_conv_dataset, batch_size=config.test_batch_size, shuffle=False,
                                  num_workers=config.num_workers)

    log.info("Initializing test_rank streamer")

    # initialize model
    kg_node_embeddings = torch.tensor(kg_node_embeddings)
    relation_embeddings = torch.tensor(relation_embeddings)

    if config.parallel:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    config.device = device

    ########### read lstm model
    if config.load_lstm_model:
        """rl_lstm_model = RL_LSTM(embedding_dim=config.embedding_dim, num_entities=len(config.entity2id),
                                pretrained_embeddings=kg_node_embeddings,
                                freeze=config.freeze, device=config.device, entdrop=config.entdrop,
                                reldrop=config.reldrop, scoredrop=config.scoredrop,
                                l3_reg=config.l3_reg, ls=config.ls, do_batch_norm=config.do_batch_norm)
        rl_lstm_model.load_state_dict(torch.load("./src_active_learning/best_model.pt"))"""
        config.other_model = None
        if config.use_cuda:
            config.other_model=None
    ###############################################

    # ============ 修改：初始化RelationExtractor ============
    model = RelationExtractor(config, embedding_dim=config.embedding_dim, num_entities=len(config.entity2id),
                              relation_emb=relation_embeddings,
                              pretrained_embeddings=kg_node_embeddings,
                              freeze=config.freeze, device=config.device, entdrop=config.entdrop,
                              reldrop=config.reldrop, scoredrop=config.scoredrop,
                              l3_reg=config.l3_reg, ls=config.ls, do_batch_norm=config.do_batch_norm)

    if config.load_model:
        model.load_state_dict(torch.load(config.model_save_path))
        print(f"已加载预训练模型: {config.model_save_path}")

    if torch.cuda.device_count() > 1 and config.parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if config.use_cuda:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ExponentialLR(optimizer, config.decay)
    optimizer.zero_grad()

    # ============ 新增：训练主题模型 ============
    if config.use_topic_aware and topic_extractor:
        print("开始训练主题模型...")
        # 使用训练数据训练主题模型
        train_batcher_temp = ActiveRandomSelector(config.conversation_path, 999999, config)
        train_data_temp = train_batcher_temp.next()
        all_questions = []
        for data_point in train_data_temp:
            questions = [data_point[1][0] if data_point[1] else "",
                         data_point[3][0] if data_point[3] else "",
                         data_point[5][0] if data_point[5] else "",
                         data_point[7][0] if data_point[7] else "",
                         data_point[9][0] if data_point[9] else ""]
            all_questions.extend([q for q in questions if q.strip()])

        topic_extractor.fit(all_questions)
        print("主题模型训练完成")

    # each round use action learning to get some new data

    if config.reranking:
        train_batcher = ActiveRandomSelector(config.conversation_test_path, 999999, config)
        new_data = train_batcher.next()
        if len(new_data) <= 0:
            print("end")
            return
        # ============ 修改：添加主题提取器 ============
        conv_dataset = DatasetConversation(new_data, config.entity2id, config.relation2id,
                                           config.entity2neighbor_relation, config.ht2relation,
                                           topic_extractor=topic_extractor)
        data_loader = DataLoader(conv_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers)
        output_top(config, model, data_loader, device, len(config.entity2id))
    else:
        max_acc = 99999
        for i in range(config.active_round):
            print("###################################################")
            print(f"第 {i + 1} 轮主动学习")
            new_data = train_batcher.next()
            if len(new_data) <= 0:
                print("end")
                break
            # ============ 修改：添加主题提取器 ============
            conv_dataset = DatasetConversation(new_data, config.entity2id, config.relation2id,
                                               config.entity2neighbor_relation, config.ht2relation,
                                               topic_extractor=topic_extractor)
            data_loader = DataLoader(conv_dataset, batch_size=config.batch_size, shuffle=True,
                                     num_workers=config.num_workers)
            max_acc = train_again(config, model, optimizer, scheduler, data_loader, valid_data_loader, test_data_loader,
                                  device, len(config.entity2id), max_acc)

    print("训练完成!")


if __name__ == "__main__":
    main()