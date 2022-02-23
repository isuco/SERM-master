"""
GCN model for relation extraction.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from transformers.modeling_albert import AlbertModel
from transformers.configuration_albert import AlbertConfig
import torch.nn.init as init
from utils import constant, torch_utils

from model.tree import head_to_tree, tree_to_adj
from model.Swish import Swish




class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None,lbstokens=None,cls=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix,lbstokens=None)
        self.lab_emb=nn.Parameter(torch.cuda.FloatTensor(opt['hidden_dim'],len(constant.LABEL_TO_ID)),requires_grad=True)
        self.bias = nn.Parameter(torch.cuda.FloatTensor(1,len(constant.LABEL_TO_ID)),requires_grad=True)
        init.xavier_normal_(self.bias)
        init.xavier_normal_(self.lab_emb)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        # if cls is None:
        #     self.classifier = nn.Linear(in_dim, opt['num_class'])
        #     # self.W_L = nn.Linear(in_dim, 30)
        #     # self.W_I = nn.Linear(in_dim, 300)
        # else:
        #     self.classifier=nn.Linear(in_dim,len(cls)+1)
        self.opt = opt

    def forward(self, inputs):
        words, masks, pos, subj_mask, obj_mask, ner, depmap, adj, rel, resrel, deprel, domain,sdp_domain ,domain_subj, domain_obj,bias_mm,order_mm, sdp_mask, batch_map = inputs
        # torch.cuda().empty_cache()
        outputs, pooling_output= self.gcn_model(inputs)
        # logits = self.classifier(outputs)
        # logits=self.classifier(outputs)
        # lab_emb = self.W_L(lab_emb)
        # outputs = self.W_I(outputs)
        # # #
        #logits = self.classifier(outputs)
        # batch_map=(batch_map==0).unsqueeze(-1)
        # logits = logits.unsqueeze(0).repeat(batch_map.shape[0], 1, 1).masked_fill(batch_map, -constant.INFINITY_NUMBER)
        # #gold_logits=torch.zeros(logits.shape[0],logits.shape[-1])
        #
        #
        # gold_logits=logits[:,:,1:].max(dim=-1)[0]
        # no_logits=logits[:,:,0]
        # dif=gold_logits-no_logits
        # dif=dif.masked_fill(batch_map.squeeze(),-constant.INFINITY_NUMBER)
        # dif_mask=(dif==(dif.max(dim=-1)[0]).unsqueeze(-1))
        # logits=logits.masked_fill(dif_mask.unsqueeze(-1)==0,-constant.INFINITY_NUMBER).max(dim=1)[0]
        # #
        # logits= (logits/(outputs.unsqueeze(-1)*lab_emb.norm(dim=-1)))

        logits=outputs.mm(self.lab_emb)+self.bias
        return logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, opt,emb_matrix=None,lbstokens=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.lbstokens=lbstokens
        # create embedding layers
        #self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.rel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt['hidden_dim'])
        # self.dir_emb = nn.Embedding(2, 6)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        # self.lab_emb = nn.Parameter(torch.cuda.FloatTensor(len(constant.LABEL_TO_ID),opt['hidden_dim']),requires_grad=True)
        # init.xavier_normal_(self.lab_emb)
        # self.subj_emb= nn.Embedding(constant.MAX_DIS + 1, opt['pos_dim'])
        # self.obj_emb = nn.Embedding(constant.MAX_DIS + 1, opt['pos_dim'])
        # embeddings = (self.emb, self.pos_emb, self.ner_emb, self.rel_emb)
        embeddings = (self.pos_emb, self.ner_emb, self.rel_emb)
        # self.init_embeddings()
        # self.initmask()
        # gcn layer
        self.gcn = nn.DataParallel(AGGCN(opt, embeddings),device_ids=[0,1,2,3])
        #self.gcn = AGGCN(opt, embeddings)
        self.reason_layer = nn.LSTM(opt['rnn_hidden'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                           dropout=opt['rnn_dropout'], bidirectional=True)

        # mlp output layer
        # self.keep=10
        in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.relproj = nn.Sequential(*layers)
        self.global_rel_proj=nn.Sequential(*[nn.Linear(in_dim, opt['hidden_dim']),nn.ReLU()])
        self.mlp=nn.Sequential(*[nn.Linear(opt['hidden_dim']*2, opt['hidden_dim']), nn.ReLU()])
        self.reldropout=nn.Dropout(opt['input_dropout'])
        self.reasondropout = nn.Dropout(opt['rnn_dropout'])
        # self.lab_proj = nn.Linear(opt['hidden_dim'],opt['hidden_dim'])
        #self.global_attn = GlobalAttention(opt['hidden_dim'])
        #self.relproj=nn.Linear(3*opt['hidden_dim'],opt['hidden_dim'])

        #self.keep = 10
        # self.query = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
        # self.key = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])

    # def init_embeddings(self):
    #     if self.emb_matrix is None:
    #         self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
    #     else:
    #         self.emb_matrix = torch.from_numpy(self.emb_matrix)
    #         self.emb.weight.data.copy_(self.emb_matrix)
    #     # decide finetuning
    #     if self.opt['topn'] <= 0:
    #         print("Do not finetune word embedding layer.")
    #         self.emb.weight.requires_grad = False
    #     elif self.opt['topn'] < self.opt['vocab_size']:
    #         print("Finetune top {} word embeddings.".format(self.opt['topn']))
    #         self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
    #     else:
    #         print("Finetune all embeddings.")

    def initmask(self):
        self.lbstokens=self.lbstokens.cuda() if self.opt['cuda'] else self.lbstokens
        mask = (self.lbstokens != 0).float()
        mask[:,0]= 0.5
        mask[:,1:] = (0.5*mask[:,1:])/mask[:,1:].sum(dim=-1).unsqueeze(-1)
        self.lab_mask=mask

    # def getlabEmbed(self):
    #     lab_emb=self.emb(self.lbstokens)
    #     lab_emb=(lab_emb.mul(self.lab_mask.unsqueeze(-1))).sum(dim=1)
    #     lab_emb=self.lab_proj(lab_emb)
    #     return lab_emb

    # def sentattn(self,inputs,subj_mask,obj_mask,mask):
    #     batch_size=inputs.shape[0]
    # #     seq_len=inputs.shape[1]
    #     subj = pool(inputs, subj_mask, type="max")
    #     obj= pool(inputs, obj_mask, type="max")
    #     subj_score=self.query(subj).unsqueeze(1).mul(self.key(inputs))
    #     obj_score = self.query(obj).unsqueeze(1).mul(self.key(inputs))
    #     entity_mask=(~((subj_mask).squeeze().mul(obj_mask.squeeze()))|mask.squeeze())
    #     m_indice=(subj_score+obj_score).max(dim=-1)[0].masked_fill(entity_mask,-1e9).argsort(dim=-1,descending=True)[:,:self.keep].reshape(-1)
    #     #m_atten = (subj_score + obj_score).sum(dim=-1).masked_fill(entity_mask.squeeze(), -1e9).max(dim=-1)
    #     k_indices=torch.cat([(torch.linspace(0,batch_size-1, batch_size).unsqueeze(-1).mm(torch.ones(1,self.keep))).reshape(-1).unsqueeze(0).long(),m_indice.unsqueeze(0).cpu().long()],dim=0)
    #     keyword=inputs.masked_fill(mask,-1e9)[tuple(k_indices.numpy().tolist())].reshape(batch_size,self.keep,-1)
    #     w_h=keyword.max(dim=1)[0]
    #     return w_h,subj,obj

    def reason_with_rnn(self, rel_inputs, masks, batch_size):
        #tensor_len = rel_inputs.shape[1]
        domain_len_mask=((masks.sum([1,2]).sort(descending=True)[0])==1).unsqueeze(-1)
        seq_lens=list(masks.sum([1,2]).sort(descending=True)[0])
        #seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1))
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'],bidirectional=True)
        rel_inputs = nn.utils.rnn.pack_padded_sequence(rel_inputs, seq_lens, batch_first=True)
        self.reason_layer.flatten_parameters()
        rnn_outputs, (ht, ct) = self.reason_layer(rel_inputs, (h0, c0))
        # rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        # rnn_outputs = F.pad(rnn_outputs, (0, 0, 0, tensor_len - rnn_outputs.shape[1]), 'constant', 0)
        # outputs=ht[0].masked_fill(domain_len_mask,-constant.INFINITY_NUMBER)
        return ht[0],domain_len_mask

    def forward(self, inputs):
        words, masks, pos,subj_mask,obj_mask,ner, depmap, adj,rel,resrel,deprel, domain,sdp_domain,domain_subj, domain_obj,bias_mm,order_mm,sdp_mask,batch_map= inputs  # unpack
        # l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = words.shape[1]
        batchsize = words.shape[0]

        # subj_ids= [i.nonzero().squeeze().numpy().reshape(-1) for i in ((subj_pos == 0).mul(~masks))]
        # obj_ids = [i.nonzero().squeeze().numpy().reshape(-1) for i in ((obj_pos == 0).mul(~masks))]
        # #l:[sent_size,words_size]
        # def inputs_to_tree_reps(head,deprel,l):
        #     trees = [head_to_tree(head[i],deprel[i],l[i],subj_ids[i],obj_ids[i]) for i in range(len(l))]
        #     adj=[]
        #     rel=[]
        #     domain=[]
        #     domain_id=[]
        #     s_pos=[]
        #     o_pos=[]
        #     sdp=[]
        #     dep=[]
        #     for tree in trees:
        #         #d, a, r, s, o, sd = tree_to_adj(maxlen, tree, directed=False)
        #         d, a, r ,do,di= tree_to_adj(maxlen, tree, directed=False)
        #         dep.append(d.reshape(1, maxlen, maxlen))
        #         adj.append(a.reshape(1, maxlen, maxlen))
        #         rel.append(r.reshape(1, maxlen, maxlen))
        #         domain.append(do.reshape(1,maxlen,maxlen))
        #         domain_id.append(di.reshape(1,maxlen,maxlen))
        #         #s_pos.append(s.reshape(1, maxlen))
        #         #o_pos.append(o.reshape(1, maxlen))
        #         #sdp.append(sd.reshape(1, maxlen))
        #     dep = np.concatenate(dep, axis=0)
        #     dep = torch.from_numpy(dep)
        #     adj = np.concatenate(adj, axis=0)
        #     adj = torch.from_numpy(adj)
        #     # adj = torch.from_numpy(adj.repeat(len(range(torch.cuda.device_count()))),1)
        #     rel = np.concatenate(rel, axis=0)
        #     rel = torch.from_numpy(rel)
        #     domain = np.concatenate(domain, axis=0)
        #     domain = torch.from_numpy(domain)
        #     domain_id = np.concatenate(domain_id, axis=0)
        #     domain_id = torch.from_numpy(domain_id)
        #     #s_pos = np.concatenate(s_pos, axis=0)
        #     #s_pos = torch.from_numpy(s_pos)
        #     #o_pos = np.concatenate(o_pos, axis=0)
        #     #o_pos = torch.from_numpy(o_pos)
        #     #sdp = np.concatenate(sdp, axis=0)
        #     #sdp = torch.from_numpy(sdp)
        #     # rel = torch.from_numpy(rel.repeat(len(range(torch.cuda.device_count()))),1)
        #     if self.opt['cuda']:
        #         #return dep.cuda(), adj.cuda(), rel.cuda(), s_pos.cuda(), o_pos.cuda(), sdp.cuda()
        #         return dep.cuda(), adj.cuda(), rel.cuda(),domain.cuda(),domain_id.cuda()
        #     else:
        #         #return dep, adj, rel, s_pos, o_pos, sdp
        #         return dep, adj, rel,domain,domain_id
        #
        #
        #
        #
        # def adj2globaladj(adj, maxlen):
        #     local_adj = adj
        #     for i in range(maxlen):
        #         temp = local_adj.bmm(adj)
        #         local_mask = (local_adj == 0).mul(temp != 0)
        #         if local_mask.sum() == 0:
        #             return local_adj
        #         else:
        #             local_adj = local_adj.masked_fill(local_mask, i)
        #
        # def geodistribute(p, adj):
        #     return torch.pow(p, adj.double()).float()

        #dep, adj, rel, s_pos, o_pos, sdp = inputs_to_tree_reps(head.data, deprel.data, l)
        # dep, adj, rel,domain,domain_id= inputs_to_tree_reps(head.data, deprel.data, l)
        # del head, deprel
        # batchsize=words.shape[0]
        # outputs=[]
        # inputsf=[]
        # i=0
        # sdp_mask = (sdp_mask != 1).unsqueeze(-1)
        # max_size=64
        # while(batchsize>max_size*i):
        #     inputsf.append(tuple([f[max_size*i:min(max_size*(i+1),batchsize)] for f in list(inputs[:-1])]))
        #     i=i+1
        # for inputf in inputsf:
        #     words, _, _, _, _, _, _, adj, _, _, _, _, _, _, _ = inputf
        #     mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        #     aspect_h, pool_mask, embedding_output=self.gcn(inputf,mask)
        #     outputs.append(aspect_h)
        #
        # aspect_h=torch.cat(outputs,dim=0)
        # no connection,not in the dependency tree
        input = inputs[:-1]
        # sdp_mask = (sdp_mask != 0).unsqueeze(1)
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)  #
        # lab_emb=self.getlabEmbed()# no connection,not in the dependency tree
        aspect_h, pool_mask, embedding_output = self.gcn(input, mask)
        aspect_subj_mask = ((subj_mask != 0).unsqueeze(-1)) | mask
        aspect_obj_mask = ((obj_mask != 0).unsqueeze(-1)) | mask
        # aspect_subj_mask = ((subj_mask!= 0).unsqueeze(-1)) | sdp_mask
        # aspect_obj_mask = ((obj_mask!= 0).unsqueeze(-1)) | sdp_mask
        # aspect_subj = pool(aspect_h, aspect_subj_mask, type="max")
        # aspect_obj = pool(aspect_h, aspect_obj_mask, type="max")

        # globaladj = adj2globaladj(adj, maxlen)
        # geoadj = geodistribute(0.5, globaladj)
        # print('classifier:'+str(torch.cuda.memory_allocated()))



        # poolingkk
        pool_type = self.opt['pooling']
        # subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask

        #sdp_mask=(sdp_mask&subj_mask&obj_mask)|mask
        # subj_mask = subj_mask | mask
        # obj_mask = obj_mask | mask
        # subj_mask = (~pool_mask).mul(~subj_mask)
        # obj_mask = (~pool_mask).mul(~obj_mask)

        #lab_eff,lab_num=self.calceffect(embedding_output,h,clspara,mlppara,labels,pool_mask)
        # imps=[]
        # for i in range(batchsize):
        #     imps.append(self.analyseimporttance(clspara,mlppara,embedding_output,labels,pool_mask,i))
        aspect_h_pool = aspect_h.masked_fill(mask, -constant.INFINITY_NUMBER).max(dim=1)[0]
        #aspect_h_pool = torch.where(aspect_h_pool<-1e11,torch.zeros_like(aspect_h_pool),aspect_h_pool)
        aspect_subj = aspect_h.masked_fill(aspect_subj_mask, -constant.INFINITY_NUMBER).max(dim=1)[0]
        aspect_obj = aspect_h.masked_fill(aspect_obj_mask, -constant.INFINITY_NUMBER).max(dim=1)[0]
        aspect_output = self.reldropout(self.global_rel_proj(torch.cat((aspect_h_pool, aspect_subj, aspect_obj), dim=-1)))
        #aspect_output=F.relu(aspect_direct_output)
        #aspect_output = self.relproj(torch.cat((aspect_h_pool, aspect_subj, aspect_obj), dim=-1))

        aspect_dom_h = aspect_h.unsqueeze(2).repeat(1,1,domain.shape[-1],1).masked_fill(domain.unsqueeze(-1)==0,-constant.INFINITY_NUMBER).max(dim=1)[0]
        #aspect_dom_h = torch.where(aspect_h_pool<-1e11,torch.zeros_like(aspect_h_pool),aspect_h_pool).unsqueeze(1).repeat(1,domain.shape[-1],1)
        #aspect_dom_h = torch.where(laspect_dom_h < -1e11, torch.zeros_like(aspect_dom_h), aspect_dom_h)
        #aspect_dom_h_pool=aspect_dom_h.mean(dim=1)
        #aspect_dom_h = aspect_h_pool.unsqueeze(dim=1).repeat(1,domain.shape[-1],1)
        aspect_dom_subj = aspect_h.unsqueeze(2).repeat(1, 1,domain_subj.shape[-1], 1).masked_fill(domain_subj.unsqueeze(-1)==0,-constant.INFINITY_NUMBER).max(dim=1)[0]
        aspect_dom_obj = aspect_h.unsqueeze(2).repeat(1, 1,domain_obj.shape[-1], 1).masked_fill(domain_obj.unsqueeze(-1)==0,-constant.INFINITY_NUMBER).max(dim=1)[0]
        aspect_rel_input = self.relproj(torch.cat((aspect_dom_h,aspect_dom_subj,aspect_dom_obj),dim=-1))
        aspect_rel_input = self.reldropout(aspect_rel_input)
        aspect_rel_input = sdp_domain.bmm(aspect_rel_input)
        aspect_rel_input = order_mm.mm(aspect_rel_input.reshape(batchsize,-1)).reshape(batchsize,-1,self.opt['hidden_dim'])
        aspect_reason_output,domain_len_mask = self.reason_with_rnn(aspect_rel_input,sdp_domain,batchsize)
        aspect_reason_output = self.reasondropout(aspect_reason_output)
        aspect_reason_output = order_mm.transpose(-1,0).mm(aspect_reason_output.reshape(batchsize,-1)).reshape(batchsize,self.opt['hidden_dim'])
        # domain_len_mask=order_mm.transpose(-1,0).mm(domain_len_mask.float())
        # aspect_h_pool=pool(aspect_h,mask)
        # aspect_subj=pool(aspect_h,aspect_subj_mask)
        # aspect_obj=pool(aspect_h,aspect_obj_mask)



        # sdp_mask=(sdp_mask.unsqueeze(-1)==0)
        # h_out=aspect_h.masked_fill(sdp_mask,-constant.INFINITY_NUMBER).max(dim=1)[0]
        # h_out = absMaxpool(aspect_h,sdp_mask,dim=1)
        # h_abs = torch.abs(aspect_h.masked_fill(sdp_mask.unsqueeze(-1) == 0, 0)).max(dim=1)[0]
        # h_out=torch.where(h_abs>h_max,-h_abs,h_max)
        # batch_map=(batch_map==0).unsqueeze(-1)
        #aspect_reason_output = aspect_reason_output.masked_fill(domain_len_mask==1, -constant.INFINITY_NUMBER)
        #aspect_fuse_output = torch.cat((aspect_output.unsqueeze(0),aspect_reason_output.unsqueeze(0)),dim=0).max(dim=0)[0]
        aspect_fuse_output=self.mlp(torch.cat((aspect_reason_output,aspect_output),dim=-1))
        # aspect_fuse_output = self.mlp(aspect_fuse_output)
        aspect_fuse_output=aspect_output+aspect_fuse_output
        batch_map = (batch_map == 0).unsqueeze(-1)
        h_out = aspect_fuse_output.unsqueeze(0).repeat(batch_map.shape[0],1,1).masked_fill(batch_map,-constant.INFINITY_NUMBER).max(dim=1)[0]


        # h_out=absMaxpool(h_out,batch_map,dim=1)
        # masked_fill(batch_map,-constant.INFINITY_NUMBER).max(dim=1)[0]
        # h_out_abs = torch.abs(h_out).unsqueeze(0).repeat(batch_map.shape[0], 1, 1).masked_fill(batch_map, -constant.INFINITY_NUMBER).max(dim=1)[0]
        # h_out = torch.where(h_out_abs > h_out_max, -h_out_abs, h_out_max)
        #h_out = batch_map.mm(aspect_h.reshape(batch_map.shape[-1],-1)).reshape(batch_map.shape[0],maxlen,-1)
        # subj=batch_map.mm(aspect_subj.reshape(batch_map.shape[-1],-1)).reshape(batch_map.shape[0],-1)
        # obj=batch_map.mm(aspect_obj.reshape(batch_map.shape[-1],-1)).reshape(batch_map.shape[0],-1)
        #sdp_mask=(batch_map.mm(sdp_mask.float())==0).unsqueeze(-1)
        #sdp_mask = (batch_map.mm(sdp_mask.float()) == 1).unsqueeze(1)
        #attn=self.global_attn(h,self.lab_emb,sdp_mask).unsqueeze(-1)
        #h_out = pool(h, sdp_mask, type='max')
        #w_h, subj, obj = self.sentattn(h, subj_mask, obj_mask, pool_mask)
        # h_out=attn.mul(h).sum(dim=1)
        # outputs = torch.cat([h_out, subj, obj], dim=1)
        # outputs = self.out_mlp(h_out)
        # logits=outputs.mm(self.lab_emb.transpose(-1,-2))
        return h_out,aspect_h_pool


class AGGCN(nn.Module):
    def __init__(self, opt, embeddings):
        super().__init__()
        self.opt = opt
        self.in_dim = opt['emb_dim']+opt['pos_dim']+opt['ner_dim']
        #self.in_dim = opt['emb_dim']
        #self.in_dim = opt['emb_dim']
        self.pos_emb, self.ner_emb, self.rel_emb = embeddings
        self.use_cuda = opt['cuda']
        self.mem_dim = opt['hidden_dim']
        albert_config=AlbertConfig.from_pretrained(opt['config_path'])
        self.albert=AlbertModel.from_pretrained(opt['albert_path'],config=albert_config)
        self.albert.resize_token_embeddings(opt['vocab_size'])
        # rnn layer
        # if self.opt.get('rnn', False):
        #     self.input_W_R = nn.Linear(self.in_dim, opt['rnn_hidden'])
        #     self.rnn = nn.LSTM(opt['rnn_hidden'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
        #                        dropout=opt['rnn_dropout'], bidirectional=True)
        #     self.in_dim = opt['rnn_hidden'] * 2
        #     self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)
        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.num_layers = opt['num_layers']
        # self.proj=nn.Linear(opt['hidden_dim'],opt['hidden_dim']//opt['num_class'])

        self.layers = nn.ModuleList()

        self.heads = opt['heads']
        self.sublayer_first = opt['sublayer_first']
        self.sublayer_second = opt['sublayer_second']
        # self.gate=GraphGate(opt,self.mem_dim)
        #self.layernorm=nn.LayerNorm(self.mem_dim,elementwise_affine=True)

        # gcn layer
        for i in range(self.num_layers):
            self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_first, self.heads))
            self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_second, self.heads))

            # if i == 0:
            #     self.layers.append(GraphConvLayer(opt, self.mem_dim, self.sublayer_first))
            #     self.layers.append(GraphConvLayer(opt, self.mem_dim, self.sublayer_second))
            # else:
            #     self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_first, self.heads))
            #     self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_second, self.heads))

        self.aggregate_W = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

        self.attn = MultiHeadAttention(self.heads, self.mem_dim).cuda()
        #self.global_attn=GlobalAttention(self.mem_dim).cuda()

    # def encode_with_rnn(self, rnn_inputs, masks, batch_size):
    #     tensor_len = rnn_inputs.shape[1]
    #     seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1))
    #     h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
    #     rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
    #     self.rnn.flatten_parameters()
    #     rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
    #     rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
    #     rnn_outputs = F.pad(rnn_outputs, (0, 0, 0, tensor_len - rnn_outputs.shape[1]), 'constant', 0)
    #     return rnn_outputs

    # def inferatten(self,inputs,mask):
    #     dcgcn_output = inputs.masked_fill(mask, -1e4)
    #     weight = self.weight_attn(stdzscore(torch.max(dcgcn_output, -1)[0], mask, dim=-1).masked_fill(mask.squeeze(-1), -1e4))
    #     return weight
    #

    def forward(self,inputs,mask):#adj matrix and other all information about the sentence
        words, masks, pos,subj_pos,obj_pos,ner, dep,adj,rel,resrel,deprel, domain,sdp_domain,domain_subj,domain_obj,bias_mm,order_mm,sdp_mask= inputs # unpack
        if len(words.shape)<2:
            words=words.unsqueeze(0)
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)
        # word_embs = self.emb(words)
        # rel=torch.abs(rel)
        word_embs=self.albert(words,attention_mask=(src_mask.squeeze()))[0]
        #word_embs=torch.ones((batchsize,words.size(1),1024)).cuda()
        word_embs=bias_mm.bmm(word_embs)
        rel_embs = self.rel_emb(deprel)
        # front_dir = self.dir_emb(torch.zeros(rel_embs.shape[0], rel_embs.shape[1]).long().cuda())
        # back_dir = self.dir_emb(torch.ones(rel_embs.shape[0], rel_embs.shape[1]).long().cuda())
        # front_rel_embs = torch.cat([front_dir, rel_embs], dim=-1)
        # back_rel_embs = torch.cat([back_dir, rel_embs], dim=-1)
        embs = [word_embs]
        # domain = domain.unsqueeze(-1)==0

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
            # embs +=[self.subj_emb(s_pos)]
            # embs +=[self.obj_emb(o_pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        # if self.opt.get('rnn', False):
        # embs = self.input_W_R(embs)
        #gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.shape[0]))
        # else:
        #     gcn_inputs = embs
        gcn_inputs = self.input_W_G(embs)
        # emb-rnn+fnn->gcn-input
        layer_list = []  # layer_output
        outputs = gcn_inputs
        # for i in range(len(self.layers)):
        #     if i < 2:
        #         outputs = self.layers[i](adj, outputs)
        #         layer_list.append(outputs)
        #     else:
        #         attn_tensor = self.attn(outputs, outputs, adj,src_mask)
        #         attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        #         outputs = self.layers[i](attn_adj_list, outputs)
        #         layer_list.append(outputs)
        #
        # aggregate_out = torch.cat(layer_list, dim=2)
        # dcgcn_output = self.aggregate_W(aggregate_out)
        # print('aggcn:'+str(torch.cuda.memory_allocated()))
        domain=domain.float()

        domain_mask=domain.bmm(domain.transpose(-1,-2))!=0
        domain_mask=domain_mask.masked_fill(torch.eye(domain_mask.shape[-1]).unsqueeze(0).cuda()==1,0)
        for i in range(len(self.layers)):
            # outputs = self.layers[i](adj, outputs)  # gcl
            # layer_list.append(outputs)
            # if i < 2:
            #     outputs = self.layers[i](adj, outputs)#gcl
            #     layer_list.append(outputs)
            # else:#mgcl6y
            # print(torch.cuda.memory_cached())
            attn_tensor = self.attn(outputs, domain_mask, rel_embs, mask)  # 每步一个attention
            attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
            #t_output=outputs.clone()
            #global_attn = self.global_attn(outputs, lab_emb, src_mask).unsqueeze(-1)
            outputs = self.layers[i](attn_adj_list,outputs,mask,domain_mask)
            # global_attn = self.global_attn(torch.cat([t_output.unsqueeze(0),outputs.unsqueeze(0)],dim=0),lab_emb,src_mask)
            # outputs = (global_attn.unsqueeze(-1)*outputs).sum(dim=0)
            # outputs=(1-global_attn).mul(outputs)+global_attn.mul(t_output)
            #outputs=self.layernorm(self.proj((outputs.unsqueeze(-2)+t_outputs.unsqueeze(1)).transpose(-2,-3)).view(batchsize,-1,self.mem_dim))

            #self.analyseimporttance(outputs, mask, 0)
            #layer_list.append(outputs)
        # aggregate_out = torch.cat(layer_list, dim=2)

        # dcgcn_output = self.aggregate_W(aggregate_out)
        return outputs, mask,gcn_inputs


# class GraphGate(nn.Module):
#     def __init__(self,opt,mem_dim):
#         super().__init__()
#         self.reset_gate = nn.Sequential(
#             nn.Linear(mem_dim * 2, mem_dim),
#             nn.Sigmoid()
#         )
#         self.update_gate = nn.Sequential(
#             nn.Linear(mem_dim * 2, mem_dim),
#             nn.Sigmoid()
#         )
#         self.transform = nn.Sequential(
#             nn.Linear(mem_dim * 2, mem_dim),
#             nn.Tanh()
#         ).cuda()
#
#     def forward(self,state_out,state_in):
#         zt = self.reset_gate(torch.cat((state_out,state_in),dim=-1))
#         rt = self.update_gate(torch.cat((state_out,state_in),dim=-1))
#         hhat= self.transform(torch.cat((state_out,rt.mul(state_in)),dim=-1))
#         output=(1-zt).mul(state_in)+zt.mul(hhat)
#         return output


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])
        # self.gru_drop = nn.Dropout(opt['gru_dropout'])

        # linear transformation
        self.linear_output = nn.Linear(self.layers * self.mem_dim, self.mem_dim)
        # self.domain_gate = self.message_gate = nn.Sequential(
        #     nn.Linear(mem_dim, mem_dim),
        #     nn.ReLU()
        # )
        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim), self.mem_dim))
            self.active_list.append(nn.PReLU())


        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

    def forward(self, adj,domain,domain_id,redomain_id,frontrel,backrel,depmap,rel,resrel,gcn_inputs,mask):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        # cache_list = [outputs]
        output_list = []
        seq_len = gcn_inputs.shape[1]
        batch_size = gcn_inputs.shape[0]
        domains = domain.shape[2]
        domain_mask = domain_id.sum(dim=-1).unsqueeze(-1) == 1
        redomain_mask = redomain_id.sum(dim=-1).unsqueeze(-1) == 1
        frontadj = adj.masked_select(depmap == 1).reshape(batch_size, seq_len)
        backadj = adj.transpose(-1, -2).masked_select(depmap == 1).reshape(batch_size, seq_len)
        denom = adj.sum(2).unsqueeze(2) + 1

        for l in range(self.layers):
            index = l
            delta = torch.zeros_like(outputs)
            frontoutputs = outputs.masked_fill(domain_mask, 0)
            backoutputs = rel.float().bmm(outputs)
            backoutputs = backoutputs.masked_fill(redomain_mask, 0)
            # if l == self.layers - 1:
            domain_outputs = self.domain_gate(outputs.unsqueeze(2).repeat(1,1,domains,1).masked_fill(domain,-1e4).max(dim=1)[0].squeeze())
            # domain_outputs = outputs.unsqueeze(2).repeat(1, 1, domains, 1).masked_fill(domain, -1e4).max(dim=1)[
            #        0].squeeze()
            frontoutputs = frontoutputs + domain_id.bmm(domain_outputs)
            backoutputs = backoutputs + redomain_id.bmm(domain_outputs)
            frontor = torch.cat([frontoutputs, frontrel], dim=-1)
            frontdepadj = self.message_gate(frontor)
            fadj = frontadj.unsqueeze(-1).mul(frontdepadj)
            delta += rel.transpose(-1, -2).float().bmm(fadj.mul(frontoutputs))
            # delta += rel.transpose(-1, -2).float().bmm(fadj.mul(frontoutputs))
            backor = torch.cat([backoutputs, backrel], dim=-1)
            backdepadj = self.message_gate(backor)
            badj = backadj.unsqueeze(-1).mul(backdepadj)
            delta += badj.mul(backoutputs)
            Ax = adj.bmm(outputs.mul(rel))

            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            outputs = self.active_list[l](AxW)
            # cache_list.append(gAxW)
            # outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(outputs))

        gcn_outputs = torch.cat(output_list, dim=2)

        out = self.linear_output(gcn_outputs)
        out = out + gcn_inputs
        # # gcn layer
        # denom = adj.sum(2).unsqueeze(2) + 1
        #
        # outputs = gcn_inputs
        # cache_list = [outputs]
        # output_list = []
        # #weighted_z,weughted_r=weighted
        # for l in range(self.layers):
        #     Ax = adj.bmm(outputs)#adj word emb count
        #     # zt = self.reset_gate(torch.cat((Ax,outputs),dim=-1))
        #     # rt = self.update_gate(torch.cat((Ax,outputs),dim=-1))
        #     # hhat= self.gru_drop(self.transform(torch.cat((Ax,rt.mul(outputs)),dim=-1)))
        #     # outputs=(1-zt).mul(outputs)+zt.mul(hhat)
        #     AxW = self.weight_list[l](Ax)#all linear
        #     AxW = AxW + self.weight_list[l](outputs)  # self loop AxW=linear(adj.outputs)+linear(outputs)
        #     AxW = AxW / denom#normalization
        #     gAxW = F.relu(AxW)
        #     cache_list.append(gAxW)
        #     outputs = torch.cat(cache_list, dim=2)
        #     output_list.append(self.gcn_drop(gAxW))
        #
        # gcn_outputs = torch.cat(output_list, dim=2)
        # gcn_outputs = gcn_outputs + gcn_inputs#resnet
        #
        # out = self.linear_output(gcn_outputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        # self.gate=gate
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads*self.layers, self.mem_dim)
        self.weight_list = nn.ModuleList()
        # self.domain_gate=self.message_gate = nn.Sequential(
        #     nn.Linear(mem_dim, mem_dim),
        #     nn.ReLU()
        # )


        self.active_list= nn.ModuleList()
        #self.layer_aggre=nn.Linear(self.mem_dim*self.layers*self.heads,self.mem_dim)
        # self.message_gate = nn.Sequential(
        #     nn.Linear(mem_dim * 2, mem_dim),
        #     nn.Tanh()
        # )

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim, self.mem_dim))
                self.active_list.append(nn.ReLU())
        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()
        self.active_list=self.active_list.cuda()

    # def keyword(self,inputs,mask):
    #     indices=torch.cat((torch.linspace(0,inputs.shape[0]-1,inputs.shape[0]).long().unsqueeze(0),inputs.masked_fill(mask,-1e4).max(dim=-1)[0].argmax(dim=1).unsqueeze(0)),dim=0).cpu().numpy().tolist()
    #     return inputs[tuple(indices)]

    def forward(self,adj_list,gcn_inputs,mask,domain_mask):

        multi_head_list = []
        seq_len = gcn_inputs.shape[1]
        batch_size = gcn_inputs.shape[0]
        # domains=domain.shape[2]
        # domain_mask=domain_id.sum(dim=-1).unsqueeze(-1)==1
        # redomain_mask=redomain_id.sum(dim=-1).unsqueeze(-1)==1
        for i in range(self.heads):
            adj = adj_list[i]
            # frontadj = adj.masked_select(depmap == 1).reshape(batch_size, seq_len)
            # backadj = adj.transpose(-1, -2).masked_select(depmap == 1).reshape(batch_size, seq_len)
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            # cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                # print('gcn:'+str(torch.cuda.memory_allocated()))
                index = i * self.layers + l
                # delta = torch.zeros_like(outputs)
                # frontoutputs = outputs.masked_fill(domain_mask, 0)
                # #frontoutputs=outputs
                # backoutputs = rel.float().bmm(outputs)
                # backoutputs = backoutputs.masked_fill(redomain_mask, 0)
                # if l==0:
                #     #domain_outputs = absMaxpool(outputs.unsqueeze(2).repeat(1,1,domains,1),domain,dim=1)
                #     domain_outputs = outputs.unsqueeze(2).repeat(1, 1, domains, 1).masked_fill(domain,-constant.INFINITY_NUMBER).max(dim=1)[0]
                #     #domain_outputs = outputs.unsqueeze(2).repeat(1, 1, domains, 1).masked_fill(domain, -1e4).max(dim=1)[0].squeeze()
                #     frontoutputs = frontoutputs + domain_id.bmm(domain_outputs)
                #     backoutputs = backoutputs + redomain_id.bmm(domain_outputs)
                # frontor=torch.cat([frontoutputs,frontrel],dim=-1)
                # frontdepadj = self.message_gate(frontor)
                # fadj = frontadj.unsqueeze(-1).mul(frontdepadj)
                # #delta += resrel.transpose(-1,-2).float().bmm(fadj.mul(frontoutputs))
                # delta += rel.transpose(-1, -2).float().bmm(fadj.mul(frontoutputs))
                # backor = torch.cat([backoutputs, backrel], dim=-1)
                # backdepadj = self.message_gate(backor)
                # badj = backadj.unsqueeze(-1).mul(backdepadj)
                # delta += badj.mul(backoutputs)
                #delta += resrel.float().bmm(badj.mul(backoutputs))
                # if self.training:
                #     drop_mask = (torch.rand((batch_size, seq_len)) > 0.5) * 1
                #     delta = delta.mul(drop_mask.unsqueeze(-1))

                #Ax = (adj.unsqueeze(-1).mul(outputs.unsqueeze(1)).masked_fill(domain_mask.unsqueeze(-1)==0,-constant.INFINITY_NUMBER)).max(dim=-2)[0]
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                outputs = self.active_list[i](AxW)
                # cache_list.append(gAxW)
                # outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(outputs))

            gcn_outputs = torch.cat(output_list, dim=2)
            # gcn_outputs = self.layer_aggre(gcn_outputs)
            # gcn_outputs = gcn_outputs + gcn_inputs

            multi_head_list.append(gcn_outputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)
        # keyword=self.keyword(out,mask)
        out = gcn_inputs + out

        # multi_head_list = []
        # for i in range(self.heads):
        #     adj = adj_list[i]
        #     denom = adj.sum(2).unsqueeze(2) + 1
        #     outputs = gcn_inputs
        #     cache_list = [outputs]
        #     output_list = []
        #     for l in range(self.layers):
        #         index = i * self.layers + l
        #         Ax = adj.bmm(outputs)
        #         #Ax = self.gate(Ax,outputs)
        #         AxW = self.weight_list[index](Ax)
        #         AxW = AxW + self.weight_list[index](outputs)  # self loop
        #         AxW = AxW / denom
        #         gAxW = F.relu(AxW)
        #         cache_list.append(gAxW)
        #         outputs = torch.cat(cache_list, dim=2)
        #         outputs=self.gcn_drop(gAxW)
        #
        #     gcn_ouputs = torch.cat(output_list, dim=2)
        #     gcn_ouputs = gcn_ouputs + gcn_inputs
        #
        #     multi_head_list.append(outputs)
        #
        # final_output = torch.cat(multi_head_list, dim=2)
        # out = self.Linear(final_output)

        return out


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def absMaxpool(h,mask,dim=1):
    h_max = h.masked_fill(mask, -constant.INFINITY_NUMBER).max(dim=dim)[0]
    h_abs = torch.abs(h.masked_fill(mask, 0)).max(dim=dim)[0]
    h_out=torch.where(h_abs>h_max,-h_abs,h_max)
    return h_out


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def zscore(data):
    min = torch.min(data)
    max = torch.max(data)
    mean = torch.mean(data)
    return (data - mean) / (max - min)

def mystd(data,mask,dim=[-1]):
    data = data.float()
    mdata = mymean(data, mask=mask,dim=dim)
    for d in dim:
        mdata=mdata.unsqueeze(d)
    sdata = torch.pow(data - mdata, 2).masked_fill(mask == 0, 0).sum(dim=dim)
    dd=torch.pow(mask.float().sum(dim=dim[-1]),len(dim))+1e-4
    if len(dim)==2:
        dd=dd.squeeze(-1)
    st = ((sdata / (dd))+1e-10).sqrt()
    return st

def mymean(data,mask,dim=[-1]):
    ds=data.masked_fill(mask==0,0).sum(dim=dim)
    if dim[-1]==-2:
        mask=mask.float().squeeze(-1)
    else:
        mask = mask.float().squeeze(-2)
    dd=torch.pow(mask.sum(dim=-1), len(dim))+1e-4
    if dd.dim()!=ds.dim():
        dd=(dd).unsqueeze(-1)
    dm=(ds/dd)
    return dm

def stdzscore(data,dim=[-1],mask=None):
    std = mystd(data, mask=mask, dim=dim)
    mean = mymean(data, mask=mask, dim=dim)
    for d in dim:
        std=std.unsqueeze(-1)
        mean=mean.unsqueeze(-1)
    std=std+1e-4
    res=(data-mean)/std
    #detachres=res.detach()
    # if torch.any(torch.isnan(detachres)):
    #     res=data
    res = res.masked_fill(mask==0,0)
    # res = res.masked_fill(mask.== 0, 0)
    return res



def weightsoftmax(data, adj=None,mask=None):
    data = data - data.max(dim=-1)[0].unsqueeze(-1)
    data_exp = torch.exp(data)
    if adj is not None:
        data_exp=data_exp.mul(adj.unsqueeze(1).float())
    if mask is not None:
        data_exp=data_exp.masked_fill(mask==0,0)
    data_exp = data_exp + 1e-10
    weight = (data_exp / (data_exp.sum(dim=-1).unsqueeze(dim=-1))).masked_fill(mask==0,0)
    return weight

def simplesoftmax(data,mask=None,dim=[-1],T=None):
    if mask is not None:
        data = data.masked_fill(mask == 0, -constant.INFINITY_NUMBER)
        # data=data.masked_fill(mask.transpose(-1,-2)==0,-constant.INFINITY_NUMBER)
    m = data
    for d in dim:
        m=m.max(dim=d)[0]
    for d in dim:
        m=m.unsqueeze(d)
    data = data - m
    data = data / torch.abs(T)
    data_exp = torch.exp(data)
    data_exp = data_exp + 1e-10
    if mask is not None:
        data_exp=data_exp.masked_fill(mask==0,0)
    des=data_exp.sum(dim)
    for d in dim:
        des=des.unsqueeze(d)
    weight = (data_exp / des).masked_fill(mask==0,0)
    return weight


def attention(query, key, adj, mask=None, dropout=None):
    # d_k = query.size(-1)
    #scores=torch.matmul(query, key.transpose(-2, -1))
    scores = stdzscore(torch.matmul(query, key.transpose(-2, -1)),mask=mask,dim=[-1])

    thresh_mask=scores<0
    scores=scores.masked_fill(thresh_mask,-1e9)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # scores = scores.masked_fill(adj.unsqueeze(dim=1)==0,-1e9)

    p_attn = weightsoftmax(scores, adj,mask)
    mask = ((mask==0)|thresh_mask|(adj==0).unsqueeze(1))
    # mask = ((mask == 0) | (adj == 0).unsqueeze(1))
    p_attn=p_attn.masked_fill(mask,0)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    # 多头注意力层，包含两层全连接表示的投影矩阵，维度正则化等，同样要进行dropout以及mask
    def __init__(self, h, d_model, dropout=0.3):  # heads,dim
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h  # dim of a head
        self.h = h
        self.W_K = nn.Linear(d_model, d_model)
        self.W_Q = nn.Linear(2*d_model, d_model)
        # self.linears = clones(nn.Linear(d_model, d_model), 2)#2 layers of linears
        # self.W_attn=nn.Linear(3*d_model,d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, adj, rel, mask=None):
        if mask is not None:
            mask = (mask==0).unsqueeze(1)
        nbatches = query.size(0)
        # maxlen=query.size(1)
        # index=torch.LongTensor([[[i,j] for i in range(maxlen)] for j in range(maxlen)])
        # input= torch.reshape(query[:,index],(nbatches,maxlen,maxlen,-1))
        # input=torch.cat((input,rel),dim=-1)
        # input=self.W_attn(input)
        # del adj,rel,query
        #
        # input= zscore(input)
        # #print(mask.shape)
        # input=input.masked_fill(mask.unsqueeze(-1)==0,-1e4)
        # #print(input.shape)
        # attn=F.softmax(input,dim=2)
        # del input,mask
        # print(attn.shape)
        key = query.clone()
        query=torch.cat((query,rel),dim=-1)
        query = self.W_Q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.W_K(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #                      for l, x in zip(self.linears, (query, key))]#*WQ,WK?
        attn = attention(query, key, adj, mask=mask, dropout=self.dropout)
        return attn

def global_attention(query,key,T=None,mask=None,dropout=None):
    scores=torch.matmul(query,key).transpose(-1,-2).sum(dim=-2)
    #selfmask = torch.eye(scores.shape[-1]).unsqueeze(0).repeat(scores.shape[0], 1, 1).cuda()
    # scores = stdzscore(scores,mask=mask,dim=[-1])
    # scores=scores.masked_fill(selfmask==1,-constant.INFINITY_NUMBER)
    p_attn = simplesoftmax(scores, mask,dim=[0],T=T)
    # p_attn = p_attn.sum(dim=1)
    # if dropout is not None:
    #     p_attn = dropout(p_attn)
    return p_attn


class GlobalAttention(nn.Module):
    # 多头注意力层，包含两层全连接表示的投影矩阵，维度正则化等，同样要进行dropout以及mask
    def __init__(self,d_model, dropout=0.3):  # heads,dim
        super(GlobalAttention, self).__init__()
        #self.W_K_global = nn.Linear(d_model, d_model)
        self.W_Q_global = nn.Linear(d_model, d_model)
        self.T=nn.Parameter(torch.cuda.FloatTensor([0.1]),requires_grad=True)

        # self.linears = clones(nn.Linear(d_model, d_model), 2)#2 layers of linears
        # self.W_attn=nn.Linear(3*d_model,d_model)
        self.d_model=d_model
        self.dropout = nn.Dropout(p=dropout)


    def forward(self,query,lab_emb, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        mask=mask.squeeze()
        nbatches = query.size(1)
        # maxlen=query.size(1)
        # index=torch.LongTensor([[[i,j] for i in range(maxlen)] for j in range(maxlen)])
        # input= torch.reshape(query[:,index],(nbatches,maxlen,maxlen,-1))
        # input=torch.cat((input,rel),dim=-1)
        # input=self.W_attn(input)
        # del adj,rel,query
        #
        # input= zscore(input)
        # #print(mask.shape)
        # input=input.masked_fill(mask.unsqueeze(-1)==0,-1e4)
        # #print(input.shape)
        # attn=F.softmax(input,dim=2)
        # del input,mask
        # print(attn.shape)
        query = self.W_Q_global(query).view(2,nbatches, -1,self.d_model)
        # key = self.W_K_global(lab_emb.transpose(-1,-2)).view(self.d_model,-1)
        key=lab_emb.view(self.d_model,-1)
        # lab_emb=lab_emb.transpose(-1,-2)
        # key = self.W_K(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #                      for l, x in zip(self.linears, (query, key))]#*WQ,WK?
        attn = global_attention(query, key,mask=mask, dropout=self.dropout,T=self.T)
        return attn
