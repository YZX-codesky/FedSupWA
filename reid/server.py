import os
from tkinter import Image
import torch
from collections import OrderedDict
import copy
from .utils.data import transforms as T 
from .user import DomainLocalUpdate
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
from .utils.tools import plotTSNE

# trainers in server side, used for federated setting
class FedDomainMemoTrainer(object):
    def __init__(self, args, user_sets, model,memory=None, snr=False,
                 cls_params=None, one_cls=False, clipcls=False, feature_dim=2048):
        super(FedDomainMemoTrainer, self).__init__()
        self.args = args
        self.model = model
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.train_transformer = T.Compose([
            T.Resize((args.height, args.width), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10), T.RandomCrop((args.height, args.width)),
            T.ToTensor(), normalizer,
            T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
        self.test_trans = T.Compose([
            T.Resize((args.height, args.width), interpolation=3),
            T.ToTensor(), normalizer
        ])
        self.user_sets = user_sets
        # number of pids in the dataset
        self.pid_list = [user.num_train_pids for user in user_sets]
        self.max_iter = args.max_iter
        # for fedpav
        if not snr:
            if one_cls: #措
                # use a global classifier
                self.classifier = []
                self.classifier = nn.Linear(
                    feature_dim, sum(self.pid_list)).to(self.device)
            elif clipcls:
                if args.backbone == 'ViT-B-16':
                    self.in_planes = 768
                    self.in_planes_proj = 512
                elif args.backbone == 'RN50':
                    self.in_planes = 2048
                    self.in_planes_proj = 1024
                self.classifier = []
                self.classifier_proj = []
                for idx in range(len(user_sets)):
                    cur_ids = user_sets[idx].num_train_pids
                    cur_linear = nn.Linear(
                            self.in_planes, cur_ids, bias=False).to(self.device)  
                    cur_linear_proj = nn.Linear(
                            self.in_planes_proj, cur_ids, bias=False).to(self.device) 
                    self.classifier.append(cur_linear)
                    self.classifier_proj.append(cur_linear_proj) 
            else:
                # indipendently stored in clients
                self.classifier = []
                for idx in range(len(user_sets)):
                    cur_ids = user_sets[idx].num_train_pids
                    cur_linear = nn.Linear(
                        feature_dim, cur_ids, bias=False).to(self.device)
                    if cls_params is not None:
                        cur_linear.load_state_dict(cls_params[idx])
                    self.classifier.append(cur_linear)
        else:
            self.fc, self.fc1, self.fc2, self.fc3,self.classifier= [], [], [], [],[]
            if one_cls:
                cur_ids = sum(
                    [cur_set.num_train_pids for cur_set in user_sets])
                self.fc = nn.Linear(feature_dim, cur_ids).to(self.device)
                self.fc1 = nn.Linear(256, cur_ids).to(self.device)
                self.fc2 = nn.Linear(512, cur_ids).to(self.device)
                self.fc3 = nn.Linear(1024, cur_ids).to(self.device)
                self.classifier=[]
                self.classifier = nn.Linear(
                    feature_dim, sum(self.pid_list)).to(self.device)
            else:
                self.classifier=[]
                for idx in range(len(user_sets)):
                    cur_ids = user_sets[idx].num_train_pids
                    cur_linear = nn.Linear(
                        feature_dim, cur_ids, bias=False).to(self.device)
                    if cls_params is not None:
                        cur_linear.load_state_dict(cls_params[idx])
                    self.classifier.append(cur_linear)
                    self.fc.append(
                        nn.Linear(feature_dim, cur_ids).to(self.device))
                    self.fc1.append(nn.Linear(256, cur_ids).to(self.device))
                    self.fc2.append(nn.Linear(512, cur_ids).to(self.device))
                    self.fc3.append(nn.Linear(1024, cur_ids).to(self.device))

    def fed_dwa(self, w, accs, weights=None, exclude_set=None):
        print("accs:",accs)
        # filter out unused sets
        if exclude_set is not None:
            w = [we for (idx, we) in enumerate(w) if idx not in exclude_set]
        # avgeraging
        # sample_num = sum(sample_count)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * (accs[0]) if weights is None else w_avg[k] * weights[0]
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] = w_avg[k] + w[i][k] * accs[i] \
                    if weights is None else w_avg[k] + w[i][k] * weights[i]  # weighted model
        return w_avg
    def save_images(self, aug_mod, epoch):
        file_list = [
            './data/msmt17/MSMT17_V1/train/1027/1027_000_12_0114afternoon_0586_0.jpg',
            './data/msmt17/MSMT17_V1/train/0965/0965_001_01_0114afternoon_0686_3_ex.jpg',
            './data/msmt17/MSMT17_V1/train/0938/0938_000_12_0114noon_1460_0.jpg',
            './data/msmt17/MSMT17_V1/train/0913/0913_000_01_0114noon_0995_1.jpg'
        ]
        # file_name = [os.path.join(self.args.data_dir, fname) for fname in file_list]
        file_name = file_list
        for fname in file_name:
            cur_img = self.test_trans(Image.open(fname).convert('RGB')).to(self.device)
            novel_img = aug_mod(cur_img)
            # # denorm
            norm_stat = self.test_trans.transforms[2]
            stat_mean, stat_std = torch.as_tensor(norm_stat.mean).float().to(cur_img.device).view(-1,1,1), \
                torch.as_tensor(norm_stat.std).float().to(cur_img.device).view(-1,1,1)
            novel_img.mul_(stat_std).add_(stat_mean)
            novel_img-=novel_img.min()
            novel_img/=novel_img.max()
            # cur_img.mul_(stat_std).add_(stat_mean)
            
            save_image(cur_img, os.path.join(self.args.logs_dir, os.path.basename(fname)))
            # save_image(novel_img, os.path.join(self.args.logs_dir, 
            #                                 f"{os.path.splitext(os.path.basename(fname))[0]}_{epoch}.jpg"))
        



    def train_dps(self, memory, sub_model, avg_net, w_d,aug_mod,
                        epoch, client_id,F_news,Labels, op_type='sgd'):
        sub_model.train(True)
        cur_avg = copy.deepcopy(avg_net)
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer,memory = memory)
        w,accs = local.train_dps(net=sub_model, avg_net=cur_avg, w_d=w_d,aug_mod=aug_mod,
                                  global_epoch=epoch, client_id=client_id,
                                  fc=self.fc[client_id], fc1=self.fc1[client_id],
                                  fc2=self.fc2[client_id], fc3=self.fc3[client_id],
                                  cls_layer=self.classifier[client_id],
                                  op_type=op_type,F_news=F_news,Labels=Labels)
        return w,accs
    def get_pth(self, client_id,file_path = None):
        
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer,memory=self.memory)
        (images, _, _, _, _) = local.local_train.next()
        torch.save(images,file_path)
