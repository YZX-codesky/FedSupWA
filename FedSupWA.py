import argparse
import os
import os.path as osp
import random
import numpy as np
# import sys
import time
from datetime import timedelta
import collections

import torch.nn.functional as F
import torch
from torch import nn
from torch.backends import cudnn
from reid.models.memory import MemoryClassifier

from reid import models
from reid.server import FedDomainMemoTrainer
from reid.evaluators import Evaluator, extract_features
# from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.tools import get_test_loader, get_data,get_train_loaders,get_train_loader
from reid import datasets
import copy
from collections import OrderedDict


start_epoch = best_mAP = best_R1 = former_R5 = former_R10 = former_mAP = former_Rank1 = 0


def create_model(args, num_cls=0):
    # we only use triplet loss, remember to turn off 'norm'
    # model = models.create(# resnet50
    #     args.arch, num_features=args.features, norm=True,
    #     dropout=args.dropout, num_classes=num_cls
    # )
    model = models.create( #resnet50_snr
        args.arch, norm=True,num_classes=num_cls
    )
    # use CUDA
    model = model.cuda()
    model = nn.DataParallel(model) if args.is_parallel else model
    return model


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP, best_R1, former_mAP, former_Rank1, former_R5, former_R10
    start_time = time.monotonic()

    cudnn.benchmark = True
    all_datasets = datasets.names()
    test_set_name = args.test_dataset
    all_datasets.remove(test_set_name)
    
    if args.exclude_dataset is not '':
        exclude_set_name = args.exclude_dataset.split(',')
        [all_datasets.remove(name) for name in exclude_set_name]
    train_sets_name = sorted(all_datasets)
    
    print("==========\nArgs:{}\n==========".format(args))
    # Create datasets
    print("==> Building Datasets")
    test_set = get_data(args) #获取指定目标域数据
    test_loader = get_test_loader(test_set, args.height, args.width,  #创建目标域数据加载器
                                  args.batch_size, args.workers)
    
    train_sets = get_data(args, train_sets_name) #获取训练数据集
    num_classes1 = train_sets[0].num_train_pids
    num_classes2 = train_sets[1].num_train_pids
    num_classes3 = train_sets[2].num_train_pids
    num_classes4 = train_sets[3].num_train_pids
    num_classes = [num_classes1, num_classes2, num_classes3,num_classes4]
    # num_classes = [num_classes1, num_classes2]
    print('number classes = ', num_classes)
    num_users = len(train_sets) #计算训练集数量（即联邦学习的用户数）

   

    # Create model
    model = create_model(args)  
    # sub models on different servers
    sub_models = [create_model(args) for key in range(num_users)]
    aug_mods = [
        models.create('aug', num_features=3, width=args.width, height=args.height).cuda() 
        for idx in range(num_users)
    ]
    
    print("==> Initialize source-domain class centroids and memorys ")
    memories = []
    for dataset_i in range(len(train_sets)):#遍历每个源域数据集
        dataset_source = train_sets[dataset_i] 
        #获取数据集对应的测试数据
        sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                              args.batch_size, args.workers,testset=sorted(dataset_source.train)) #mixdata还没设计
        source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
        sour_fea_dict = collections.defaultdict(list) #默认字典 用于存储每个类别的特征
        
        for f, pid, _ in sorted(dataset_source.train):
            sour_fea_dict[pid].append(source_features[f].unsqueeze(0)) #将同一身份的特征聚合在一起

        source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in sorted(sour_fea_dict.keys())] #计算每个身份标识的类中心
        source_centers = torch.stack(source_centers, 0)  ## pid,2048 将所有类中心堆叠起来)
        print(source_centers.shape)

        source_centers = F.normalize(source_centers, dim=1).cuda() # 沿着第一个维度就进行归一化处理
        print('the num of source centers is:',source_centers.shape[0])
        # 自定义的内存分类器
        curMemo = MemoryClassifier(2048, source_centers.shape[0], #类中心的数量
                                   temp=args.temp, momentum=args.momentumm).cuda()
        curMemo.features = source_centers
        curMemo.labels = torch.arange(num_classes[dataset_i]).cuda()
        curMemo = nn.DataParallel(curMemo) #并行模式
        
        memories.append(curMemo) #将配置好的分类器添加到其中

        del source_centers, sour_cluster_loader, sour_fea_dict
    # Evaluator
    evaluator = Evaluator(model)

    trainer = FedDomainMemoTrainer(args, train_sets, model,memory=memories,snr=True)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        for idx in range(num_users):
            # sub_models[idx].load_state_dict(checkpoint['state_dict'])
            sub_models[idx].load_state_dict(checkpoint['sub_models'][idx])
            # trainer.classifier[idx].load_state_dict(checkpoint['cls_params'][idx])
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['state_dict'])
        
    if args.evaluate:
        evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
        return
        
    # start training
    for epoch in range(start_epoch, args.epochs):  # number of epochs
        w_locals = []
        torch.cuda.empty_cache()
        w_origin = model.state_dict()
        delta_accs = []
        if epoch == 0:
            w_difference = [OrderedDict((key, 1) for key in w_origin.keys()) for _ in range(num_users)]
        F_news = [[],[],[],[]]
        Labels = [[],[],[],[]]
        for index in range(num_users):
            w,delta_acc = trainer.train_dps(
                memories, sub_models[index], model, w_difference[index],aug_mods[index], 
                epoch, index, op_type='sgd',F_news=F_news,Labels=Labels
            )
            delta_accs.append(delta_acc)
            w_locals.append(w)
        #将acc进行归一化，作为聚合的依据
        max_value =max(delta_accs)
        delta_accs = [x - max_value for x in delta_accs]
        delta_accs = [np.exp(x) for x in delta_accs]
        delta_accs = [ x / np.sum(delta_accs) for x in delta_accs]
        w_global = trainer.fed_dwa(w_locals,delta_accs)
        for i in range(num_users):
            w_d = OrderedDict()
            for key in w_global.keys():
                
                if key in w_locals[i]:
                    #修改为分层归一化
                    layer_diff = abs(w_global[key] - w_locals[i][key])
                    min_val = layer_diff.min()
                    max_val = layer_diff.max()
                    if max_val == min_val:
                        w_d[key] = torch.ones_like(layer_diff)  # 无差异时权重为1
                    else:
                        w_d[key] = 1 - ((layer_diff - min_val) / (max_val - min_val + 1e-4))
                else:
                    w_d[key] = torch.ones_like(w_global[key])  # 如果本地模型没有该参数，则权重为1
            w_difference[i] = w_d
        model.load_state_dict(w_global)
        
        cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query,test_set.gallery, cmc_flag=True)
        if rank1[9] >= former_R10:
            former_R10 = rank1[9]
            print('sever Performance better with this epoch of style images,update the memory!')
            # for index in selected_users: 
            for index in range(num_users): 
                for lenth in range(len(Labels[index])):
                    memories[index].module.MomentumUpdate(F_news[index][lenth], Labels[index][lenth])
        save_checkpoint({
                    'state_dict': w_global,
                    'cls_params': [cls_layer.state_dict() for cls_layer in trainer.classifier],
                    'sub_models': [mod.state_dict() for mod in sub_models],
                    'epoch': epoch + 1, 'best_mAP': best_mAP,
                }, is_best=(cur_map > best_mAP), fpath=osp.join(args.logs_dir, f'checkpoint_update_w_{epoch}.pth.tar'))
        if cur_map > best_mAP:
            print('best model saved!')
            best_mAP = cur_map
      

    end_time = time.monotonic()
    print('bese mAP:',best_mAP)
    print('best rank1:',best_R1)
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Domain-level Fed Learning")
    # data
    parser.add_argument('-td', '--test-dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-ed', '--exclude-dataset', type=str, default='')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50_snr',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lam', type=float, default=1)
    # optimizer
    parser.add_argument('--temp', type=float, default=0.05, help="temperature")
    parser.add_argument('--rho', type=float, default=0.05, help="rho")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="momentum to update model")
    parser.add_argument('--momentumm', type=float, default=0.9,
                        help="momentumm to update memory")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    
    parser.add_argument('--milestones', nargs='+', type=int, 
                        default=[20, 40], help='milestones for the learning rate decay')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate")
    
    parser.add_argument('--epochs', type=int, default=61)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--is_parallel', type=int, default=1)
    main()
