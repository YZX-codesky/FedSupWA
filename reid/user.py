from torch.utils.data import DataLoader
from .utils.data import IterLoader, Preprocessor
import torch
from torch.cuda import amp
from .models.resnet import UBS
from .utils.data.sampler import RandomMultipleGallerySampler
from .utils.tools import get_entropy, get_auth_loss, ScaffoldOptimizer, cn_op_2ins_space_chan, freeze_model, inception_score
from .loss.triplet import TripletLoss
from .loss.triplet_loss import TripletLoss as Tri_clip
from .loss.softmax_loss import CrossEntropyLabelSmooth
from .loss.make_loss import make_loss
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import os
import copy
from torchvision.utils import save_image
# from pytorch_msssim import ssim
import numpy as np
from .loss.supcontrast import SupConLoss
from reid.lr_scheduler import WarmupMultiStepLR
from sklearn.metrics import average_precision_score

# trainers in user side
class DomainLocalUpdate(object):
    def __init__(self, args, dataset=None, trans=None,memory = None,client_id = None):
        self.args = args
        self.trans = trans
        self.memory = memory
        self.client_id = client_id
        # only for non-qaconv algos
        if dataset is not None:
            if not isinstance(dataset, list):
                self.local_train = IterLoader(DataLoader(
                    Preprocessor(dataset.train, transform=trans, root=None),
                    batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    sampler=RandomMultipleGallerySampler(
                        dataset.train, args.num_instances),
                    pin_memory=False, num_workers=self.args.num_workers
                ), length=None)
                self.set_name = dataset.__class__.__name__
            else:
                self.local_train = [IterLoader(DataLoader(
                    Preprocessor(cur_set.train, transform=trans, root=None),
                    batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    sampler=RandomMultipleGallerySampler(
                        cur_set.train, args.num_instances),
                    pin_memory=False, num_workers=self.args.num_workers
                ), length=None) for cur_set in dataset]
                pid_list = [user.num_train_pids for user in dataset]
                self.padding = np.cumsum([0, ]+pid_list)
        self.max_iter = args.max_iter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tri_loss = TripletLoss(margin=0.5, is_avg=True)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduce='mean')
        self.dataset = dataset

    def handle_set(self, dataset):
        cur_loader = IterLoader(DataLoader(
            Preprocessor(dataset.train, transform=self.trans, root=None),
            batch_size=self.args.batch_size, shuffle=False, drop_last=True,
            sampler=RandomMultipleGallerySampler(
                dataset.train, self.args.num_instances),
            pin_memory=True, num_workers=self.args.num_workers
        ), length=None)
        return cur_loader

    def get_optimizer(self, nets, epoch, optimizer_type='sgd'):
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                [{'params': sub_net.parameters()} for sub_net in nets],
                lr=self.args.lr, weight_decay=self.args.weight_decay,
                momentum=self.args.momentum
            )
            lr_scheduler = MultiStepLR(
                optimizer, milestones=self.args.milestones, gamma=0.5)
        elif optimizer_type.lower() == 'scaffold':
            optimizer = ScaffoldOptimizer(
                [{'params': sub_net.parameters()} for sub_net in nets],
                lr=self.args.lr, weight_decay=self.args.weight_decay
            )
            lr_scheduler = MultiStepLR(optimizer,
                                       milestones=self.args.milestones, gamma=0.5)
        lr_scheduler.step(epoch)
        return optimizer
    
    def train_dps(self, net, avg_net,w_d, aug_mod,
                        global_epoch, client_id,cls_layer,
                        fc, fc1, fc2, fc3, F_news,Labels,op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        memory = copy.deepcopy(self.memory)

        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[avg_net, fc, fc1, fc2, fc3, aug_mod],
            epoch=global_epoch, optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        avg_net_origin = copy.deepcopy(avg_net)
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_image = (images-cur_mean).div(cur_var.sqrt()+1e-8)

            #原始图像训练本地模型
            local_features, local_features_norm, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = net(images)
            
            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            loss_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob), get_entropy(x_1_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob), get_entropy(x_2_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                    x_3_useful_prob), get_entropy(x_3_useless_prob))
            # score_avg = fc(local_features) #消融
            # loss_id_local = self.ce_loss(score_avg, labels) #消融
            
            loss_id_local = memory[client_id](local_features_norm, labels).mean()
            loss = loss_causality + loss_id_local
            optimizer_local.zero_grad()
            loss.backward()
            optimizer_local.step()

            #原始图像训练本地侧全局模型
            feature_avg, feature_avg_norm, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = avg_net(
                    images)
            
            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            score_avg = fc(feature_avg)
            # Causality loss for avg model:
            loss_causality_avg = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob), get_entropy(x_1_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob), get_entropy(x_2_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                    x_3_useful_prob), get_entropy(x_3_useless_prob))
            
            # score_avg = fc(feature_avg) #消融
            # loss_id_avg = self.ce_loss(score_avg, labels) #消融

            loss_id_avg = memory[client_id](feature_avg_norm, labels).mean() 
            loss_aug, loss_aux_avg = 0, 0
            aug_image = aug_mod(norm_image)

            with torch.no_grad():
                f_new = avg_net(aug_image)[1]
                
                #消融实验
                # f_new = feature_avg_norm

                F_news[client_id].append(f_new)
                Labels[client_id].append(labels)

                #消融实验
                memory[client_id].module.MomentumUpdate(f_new, labels)

            # aug avg model
            if global_epoch > 0:

                aug_feature_avg, aug_feature_avg_norm,x_IN_1_pool_aug, x_1_useful_pool_aug, x_1_useless_pool_aug, \
                    x_IN_2_pool_aug, x_2_useful_pool_aug, x_2_useless_pool_aug, \
                    x_IN_3_pool_aug, x_3_useful_pool_aug, x_3_useless_pool_aug = avg_net(
                        aug_image)
                x_IN_1_prob_aug = F.softmax(fc1(x_IN_1_pool_aug))
                x_1_useful_prob_aug = F.softmax(fc1(x_1_useful_pool_aug))
                x_1_useless_prob_aug = F.softmax(fc1(x_1_useless_pool_aug))
                x_IN_2_prob_aug = F.softmax(fc2(x_IN_2_pool_aug))
                x_2_useful_prob_aug = F.softmax(fc2(x_2_useful_pool_aug))
                x_2_useless_prob_aug = F.softmax(fc2(x_2_useless_pool_aug))
                x_IN_3_prob_aug = F.softmax(fc3(x_IN_3_pool_aug))
                x_3_useful_prob_aug = F.softmax(fc3(x_3_useful_pool_aug))
                x_3_useless_prob_aug = F.softmax(fc3(x_3_useless_pool_aug))
                loss_aug_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob_aug), get_entropy(x_1_useful_prob_aug), get_entropy(x_1_useless_prob_aug)) + \
                    0.01 * get_auth_loss(get_entropy(x_IN_2_prob_aug), get_entropy(x_2_useful_prob_aug), get_entropy(x_2_useless_prob_aug)) + \
                    0.01 * get_auth_loss(get_entropy(x_IN_3_prob_aug), get_entropy(
                        x_3_useful_prob_aug), get_entropy(x_3_useless_prob_aug))

                aug_feature_local = net(aug_image)[0]
                aug_score_avg, aug_score_local = fc(
                    aug_feature_avg), fc(aug_feature_local)
                # loss to disentangle, fL(I) < fA(I) < fA(I') < fL(I')
                loss_aux_avg = get_auth_loss(
                    get_entropy(F.softmax(aug_score_avg)),
                    get_entropy(F.softmax(score_avg)),
                    get_entropy(F.softmax(aug_score_local))
                )
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels) + loss_aug_causality

            # optimize avg model, sahre across domains
            loss = loss_id_avg + loss_causality_avg + loss_aug + self.args.lam * loss_aux_avg #消融

            # loss = loss_id_avg + loss_causality_avg + self.args.lam * loss_aux_avg #消融

            # loss = loss_id_avg + loss_causality_avg #消融
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx+1 == self.max_iter:
                #全局引导的更新,也可以放在客户端的每轮更新中:
                with torch.no_grad():
                    before_weight = avg_net_origin.state_dict()
                    after_weight = avg_net.state_dict()
                    for key in before_weight.keys():
                        after_weight[key] = before_weight[key]+(after_weight[key] - before_weight[key])*w_d[key]
                    avg_net.load_state_dict(after_weight)
                    #精度引导的聚合
                    feature_origin, feature_update = avg_net_origin(images)[0], avg_net(images)[0]
                    score_origin, score_update = cls_layer(feature_origin), cls_layer(feature_update)
                    #使用top-1作为依据
                    top_k_scores_origin, top_k_indices_origin = torch.topk(score_origin, k=1, dim=1)
                    correct_predictions_top_1_origin = (top_k_indices_origin == labels.view(-1, 1)).sum().item()
                    top_k_scores_update, top_k_indices_update = torch.topk(score_update, k=1, dim=1)
                    correct_predictions_top_1_update = (top_k_indices_update == labels.view(-1, 1)).sum().item()
                    total_predictions = labels.size(0)
                    top1_origin = correct_predictions_top_1_origin / total_predictions
                    top1_update = correct_predictions_top_1_update / total_predictions
                    top1_acc = top1_update - top1_origin
                    print("top1_origin",top1_origin)
                    print("top1_update",top1_update)
                    print("top1_acc",top1_acc)
                    #计算mAP
                    # 计算每个类别的AP
                    print("feature_origin",feature_origin.shape)
                    print(type(feature_origin)) 
                    def calculate_mAP(features, labels, cam_ids=None):
                        # 计算余弦相似度矩阵[4,5](@ref)
                        features_norm = features / features.norm(dim=1, keepdim=True)
                        sim_matrix = torch.mm(features_norm, features_norm.T)  # [N, N]
                        
                        aps = []
                        num_samples = labels.size(0)
                        
                        for i in range(num_samples):
                            query_label = labels[i]
                            query_cam = cam_ids[i] if cam_ids is not None else None
                            # 构建有效样本掩码[1](@ref)
                            mask = torch.ones(num_samples, dtype=torch.bool, device=features.device)
                            mask[i] = False  # 排除自身
                            if cam_ids is not None:
                                mask &= (cam_ids != query_cam)  # 跨摄像头排除    
                            # 提取有效分数和匹配标签
                            scores = sim_matrix[i][mask]
                            matches = (labels[mask] == query_label).long()  # 转换为0/1
                            # 降序排序[3](@ref)
                            sorted_indices = torch.argsort(scores, descending=True)
                            sorted_matches = matches[sorted_indices]
                            # 计算正样本位置
                            pos_ranks = torch.nonzero(sorted_matches, as_tuple=True)[0] + 1  # 1-based索引
                            if pos_ranks.numel() == 0:
                                aps.append(0.0)
                                continue
                            # 计算平均精度[9](@ref)
                            precisions = torch.arange(1, len(pos_ranks)+1, device=pos_ranks.device).float() / pos_ranks.float()
                            aps.append(precisions.mean().item())
                        return torch.tensor(aps).mean().item()
                    mAP_origin = calculate_mAP(feature_origin, labels)
                    mAP_update = calculate_mAP(feature_update, labels)
                    mAP_acc=mAP_update-mAP_origin
                    # mAP_acc = 0
                    print("mAP_origin",mAP_origin)
                    print("mAP_update",mAP_update)
                    print("mAP_update-mAP_origan",mAP_acc)
                    # lamda = 0.4
                    delta_acc=mAP_acc+top1_acc
                    # delta_acc=lamda*top1_acc+(1-lamda)*mAP_acc
                    print("accuracy_update-accuracy_origan",delta_acc)
            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                  f'LossID: {loss_id_avg.item():.2f}, LossAux:{float(loss_aux_avg):.2f})')

        return avg_net.state_dict(),delta_acc
    