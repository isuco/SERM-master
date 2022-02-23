"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numpy.ma as ma
from model.SoftCrossEntropy import SoftCrossEntropyLoss
import json
from model.aggcn import GCNClassifier
from utils import torch_utils
from utils import constant


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            #checkpoint = torch.load(filename,map_location={'cuda:1':'cuda:0','cuda:0':'cuda:1'})
            checkpoint = torch.load(filename,map_location=torch.device('cpu'))
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()

        self.model.load_state_dict(checkpoint['model'])
        #self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.alpha=checkpoint['alpha']
        # self.beta=checkpoint['beta']
        #self.loss_matrix = checkpoint['misclass'].cuda()
        self.opt = checkpoint['config']
        #print(self.loss_matrix)

    def save(self, filename, epoch):
        #getmaxn(self.loss_matrix,10)
        params = {
                'model': self.model.state_dict(),
                # 'alpha':self.alpha,
                # 'beta':self.beta,
                #'misclass':self.loss_matrix,
                'optimizer':self.optimizer.state_dict(),
                'config': self.opt,
                }

        # print("model saved to {}".format(filename))
        try:
            torch.save(params, filename)
            # torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

def getmaxn(tensor,n):
    tlen=tensor.shape[-1]
    tensor=tensor.reshape(-1)
    idx=tensor.argsort(descending=True)[:n]
    value = tensor[idx]
    maxn=torch.cat((torch.true_divide(idx,tlen).float().unsqueeze(0),(torch.floor_divide(idx,tlen)).float().unsqueeze(0),value.unsqueeze(0)),dim=0)
    #print(maxn)


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:19]]
        labels = Variable(batch[19].cuda())
        distance=batch[23].cuda()
    else:
        inputs = [Variable(b) for b in batch[:19]]
        labels = Variable(batch[19])
        distance=batch[23]
    tokens = batch[0]
    ids=batch[20]
    iscross=batch[21]
    #head = batch[5]
    #subj_pos = batch[6]
    #obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, distance ,ids,iscross

def calchis(his):
    his=ma.array(data=his,mask=his<0,dtype=np.float32).compressed()
    if len(his)==0:
        return 0.
    return his.mean()


class GCNTrainer(Trainer):
    def __init__(self, opt,lbstokens=None,emb_matrix=None,cls=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        if cls is None:
            self.model = GCNClassifier(opt,emb_matrix=emb_matrix, lbstokens=lbstokens)
        else:
            self.model=GCNClassifier(opt,emb_matrix=emb_matrix,lbstokens=lbstokens,cls=cls)
        self.cls=cls
        self.rel_types=len(constant.LABEL_TO_ID)
        self.loss_matrix = torch.zeros((self.rel_types, self.rel_types),requires_grad=False)
        self.miss_matrix = torch.zeros((self.rel_types, self.rel_types),requires_grad=False)
        # self.alpha=torch.full((1,),0.1,requires_grad=True)
        # self.beta=torch.full((1,),0.1,requires_grad=True)
        #self.model = nn.DataParallel(GCNClassifier(opt, emb_matrix=emb_matrix),device_ids=[0,1])

        self.model=GCNClassifier(opt,emb_matrix=emb_matrix)
        #self.model.half()
        print(self.get_parameter_number(self.model))
        # self.soft_criterion = SoftCrossEntropyLoss()
        self.criterion= nn.CrossEntropyLoss()
        # self.bert=self.model.gcn_model.gcn.module.albert
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
            self.loss_matrix=self.loss_matrix.cuda()
            self.miss_matrix=self.miss_matrix.cuda()
        albert_params_ids = list(map(id, self.model.gcn_model.gcn.module.albert.parameters()))
        #decay_base_params=filter(lambda p: p.requires_grad and id(p) not in albert_params_ids, self.model.parameters())
        no_decay=["bias", "LayerNorm.weight","norm.weight"]
        decay_base_params = filter(lambda p: p[1].requires_grad and id(p[1]) not in albert_params_ids and not any(nd in p[0] for nd in no_decay), self.model.named_parameters())
        nodecay_base_params = filter(lambda p: p[1].requires_grad and id(p[1]) not in albert_params_ids and any(nd in p[0] for nd in no_decay), self.model.named_parameters())
        decay_bert_params = filter(
            lambda p: p[1].requires_grad and id(p[1]) in albert_params_ids and not any(nd in p[0] for nd in no_decay), self.model.named_parameters())
        nodecay_bert_params = filter(lambda p: p[1].requires_grad and id(p[1]) in albert_params_ids and any(nd in p[0] for nd in no_decay), self.model.named_parameters())
        decay_base_params=[param[1] for param in decay_base_params]
        nodecay_base_params=[param[1] for param in nodecay_base_params]
        decay_bert_params = [param[1] for param in decay_bert_params]
        nodecay_bert_params = [param[1] for param in nodecay_bert_params]
        self.base_params=list(filter(lambda p: p.requires_grad and id(p) not in albert_params_ids, self.model.parameters()))

        self.optimizer = torch_utils.get_optimizer(opt['optim'],
                                                   [{'params':decay_base_params,'weight_decay':opt['pooling_l2']},
                                                    {'params':nodecay_base_params,'weight_decay':0.0},
                                                    {'params':decay_bert_params,'lr':opt['bert_lr'],'weight_decay':opt['pooling_l2']},
                                                    {'params':nodecay_bert_params,'weight_decay':0.0,'lr':opt['bert_lr']}], opt['lr'])
        #self.lr_scheduler=torch_utils._create_scheduler(self.optimizer,opt)

    def get_parameter_number(self,net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    def update(self, batch,acc_loss=[],isbackward=True,is_soft=False):
        inputs, labels, tokens, distance,ids,iscross= unpack_batch(batch, self.opt['cuda'])
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(inputs)

        batch_size = labels.shape[0]
        rel_types= len(constant.LABEL_TO_ID)
        # indices = torch.cat((labels.unsqueeze(0), logits.argmax(dim=-1).unsqueeze(0)),dim=0)
        #
        # extra_loss =torch.from_numpy(np.apply_along_axis(calchis, -1, self.loss_matrix.cpu().numpy()))
        # extra_loss =F.softmax(extra_loss.reshape(-1)).reshape(rel_types,rel_types)[tuple(indices.cpu().detach().numpy().tolist())]


        #calc&update history loss
        # margin=self.loss_matrix[labels]

        #indices = torch.cat((torch.linspace(0, batch_size - 1, batch_size).unsqueeze(0).long().cuda(), labels.unsqueeze(0)))
        # gold_mask = torch.sparse_coo_tensor(indices, torch.ones(batch_size),
        #                                     (batch_size, rel_types)).to_dense().cuda()
        # self.logits=(1+(distance / 14.0)).unsqueeze(-1).mul(logits)
        # logits = logits + 0.15*torch.abs(logits.mul(margin))
        #indices = torch.cat((torch.linspace(0, batch_size - 1, batch_size).unsqueeze(0).long(), labels.unsqueeze(0)))
                # gold_logits = torch.masked_select(logits, gold_mask == 1)
        # dif = (logits.max(dim=-1)[0] - gold_logits).detach()
        # extra_loss=dif.mul(extra_loss.float()).sum().squeeze()
        # loss+=extra_loss
        if is_soft:
            # if self.cls is not None:
            #     labels=torch.FloatTansor([[1-l[self.cls],l[self.cls]] for l in labels])
            #     if self.opt['cuda']:
            #         labels=labels.cuda()
            labels=labels.float()
            loss=self.soft_criterion(logits,labels)
        else:
            # if self.cls is not None:
            #     labels =torch.LongTensor([l if l in self.cls else 0 for l in labels])
            #     if self.opt['cuda']:
            #         labels=labels.cuda()
            loss = self.criterion(logits, labels)
        # l2 decay on all conv layers
        # if self.opt.get('conv_l2', 0) > 0:
        #     loss += self.model.conv_l2() * self.opt['conv_l2']
        # # l2 penalty on output representations
        # if self.opt.get('pooling_l2', 0) > 0:
        #     #loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        #     loss += self.opt['pooling_l2'] * (self.model.parameters().sum()self.base_params ** 2).sum()
        #
        #loss += self.calculate_l2()
        loss_val = loss.item()
        # print(((predictions) * (labels != 0) * 1).sum())
        # indices = torch.cat((labels.unsqueeze(0), logits.argmax(dim=-1).unsqueeze(0)), dim=0).detach()
        # mismat = torch.sparse_coo_tensor(indices, dif,
        #                                  (rel_types, rel_types)).to_dense()
        # mismask = torch.sparse_coo_tensor(indices, torch.ones(batch_size),
        #                                    (rel_types, rel_types)).to_dense().cuda()
        #mismask = mismask - 1
        # mismask = mismask.masked_fill(mismask >= 0, 0)
        # mismat += mismask
        # self.miss_matrix += mismask
        # # # backward
        # with torch.autograd.detect_anomaly():
        if isbackward:
            loss = (sum(acc_loss) + loss) / (len(acc_loss) + 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
            self.optimizer.step()
            #self.lr_scheduler.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, distance,ids,iscross = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[22]
        batch_size = labels.shape[0]
        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        # labels_map = dict(zip(self.cls, list(range(1, 1 + len(self.cls)))))

        # if self.cls is not None:
        #     labels=labels.cpu().detach().numpy().tolist()
        #     labels =torch.LongTensor([labels_map[l] if l in self.cls else 0 for l in labels])
        #     if self.opt['cuda']:
        #         labels=labels.cuda()
        #raw_predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        # logits[:,1:]=logits[:,1:].mul((iscross==0).unsqueeze(-1).float().cuda())

        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=-1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

        samples = []
        for i in range(batch_size):
            if predictions[i] != labels[i]:
                samples.append(ids[i])
        NER2ID=constant.NER_TO_ID
        ID2NER=dict([(v, k) for k, v in NER2ID.items()])
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        ids = [str(i) for i in ids]
        probs = dict(zip(ids, probs))
        return predictions, probs, loss.item(),samples


    def updatemisclass(self):
        self.loss_matrix = self.miss_matrix.clone()
        self.loss_matrix = self.loss_matrix/(self.loss_matrix.sum(dim=-1).unsqueeze(-1))
        np.save('missclass.npy',self.miss_matrix.cpu().detach().numpy())
        self.loss_matrix = self.loss_matrix.masked_fill(torch.eye(self.rel_types).cuda() == 1, 0)
        self.miss_matrix = self.miss_matrix.zero_()
